import torch
import torch.nn as nn
import numpy as np
import copy

from common.adaptive_m import get_adaptive_m
from continuous import lunar_mdp
from training_utils import (
    DynamicLearningRateScheduler, AdaptiveRewardScaler, 
    AdaptiveEntropyScheduler, GradientMonitor, adaptive_value_epochs,
    compute_return_statistics
)

def compute_m_step_returns(rewards, values, gamma, m):
    T = len(rewards)
    returns = []

    for t in range(T):
        G = 0.0

        for k in range(m):
            if t + k < T:
                G += (gamma ** k) * rewards[t + k]

        if t + m < T:
            G += (gamma ** m) * values[t + m].item()

        returns.append(G)

    return returns

def compute_Tm_value(mdp, policy, value_net, gamma, m, K=3):
    """
    Recursive T^m Operator.
    K: Number of actions to sample at each node (Branching factor)
    m: Depth of the tree
    """
    if m == 0:
        state_t = torch.FloatTensor(mdp.state).unsqueeze(0)
        return value_net(state_t).item()

    checkpoint = mdp.get_checkpoint()
    current_state = mdp.state.copy()
    
    max_q = -float('inf')

    for _ in range(K):
        mdp.restore_checkpoint(checkpoint)
        mdp.state = current_state
        
        action, _ = policy.sample_action(mdp.state)
        action = np.asarray(action, dtype=np.float32)
        action += np.random.normal(0, 0.1, size=action.shape)
        action = np.clip(action, -1, 1)
        
        next_state, reward, done, _, _ = mdp.step(action)

        if done:
            q_val = reward
        else:
            q_val = reward + gamma * compute_Tm_value(
                mdp, policy, value_net, gamma, m - 1, K
            )

        if q_val > max_q:
            max_q = q_val

    mdp.restore_checkpoint(checkpoint)
    mdp.state = current_state
    return max_q

def compute_strided_Tm_returns(
    env, policy, value_net, gamma, m, states, rewards, dones, checkpoints, K=3, search_interval=4
):
    """
    Computes the T^m returns for an entire trajectory using a strided backward loop.
    Reduces computation by only performing the full tree search every `search_interval` steps.
    """
    T = len(states)
    returns = np.zeros(T, dtype=np.float32)

    for i in reversed(range(T)):
        # FIX 2: Check if the state is terminal. If it is, the return is just the reward.
        if dones[i]:
            returns[i] = rewards[i]
        # Condition 1: Anchor points (Last step or falls on the interval)
        elif i == T - 1 or i % search_interval == 0:
            env.restore_checkpoint(checkpoints[i])
            env.state = states[i]
            returns[i] = compute_Tm_value(env, policy, value_net, gamma, m, K=K)
            
        # Condition 2: Intermediate steps (Bootstrap backwards)
        else:
            returns[i] = rewards[i] + gamma * returns[i+1]
            
    return returns.tolist()

def update_ema(target, source, tau=0.05):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(tau * s_param.data + (1 - tau) * t_param.data)

def train_value_network(optimizer_v, value_net, states_t, targets_t, v_epochs=25):
    """Common Value Network update logic using Huber Loss and Grad Clipping."""
    for _ in range(v_epochs):
        optimizer_v.zero_grad()
        current_values = value_net(states_t).view(-1)
        targets_flat = targets_t.view(-1)
        v_loss = torch.nn.functional.huber_loss(current_values, targets_flat, delta=1.0)
        v_loss.backward()
        # Increased gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
        
        # Check for NaN before step
        valid = True
        for param in value_net.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                valid = False
                break
        
        if valid:
            optimizer_v.step()
        else:
            print("[WARNING] NaN gradient in value network, skipping update")

def train_policy_network(
    optimizer_p, policy, states_t, actions_t, advantages, 
    entropy_coef, log_probs_old=None, clip_epsilon=0.2
):
    """Common Policy update logic supporting both Vanilla PG and Lagging/PPO."""
    mean, std = policy(states_t)
    dist = torch.distributions.Normal(mean, std)
    new_log_probs = dist.log_prob(actions_t).sum(dim=-1)

    if log_probs_old is not None:
        # PPO/Lagging Style update
        log_probs_old = log_probs_old.view_as(new_log_probs)
        ratios = torch.exp(new_log_probs - log_probs_old)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
    else:
        # Vanilla Policy Gradient
        policy_loss = -(advantages * new_log_probs).mean()

    # Add Entropy Bonus
    policy_loss -= entropy_coef * dist.entropy().mean()

    optimizer_p.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer_p.step()

def get_advantages(returns_t, values_t):
    """Computes standardized advantages."""
    advantages = returns_t - values_t
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

def should_early_stop(
    rewards_history,
    min_episodes=300,
    check_every=50,
    window=100,
    plateau_delta=10.0,
    max_bad_checks=4,
    bad_check_count=0,
):
    """Return (stop_flag, updated_bad_check_count) using rolling-mean degradation checks."""
    n = len(rewards_history)
    required = max(min_episodes, 2 * window)

    if n < required or (n % check_every) != 0:
        return False, bad_check_count

    prev_mean = float(np.mean(rewards_history[-2 * window:-window]))
    curr_mean = float(np.mean(rewards_history[-window:]))
    improvement = curr_mean - prev_mean

    if improvement < -abs(plateau_delta):
        bad_check_count += 1
    else:
        bad_check_count = 0

    return bad_check_count >= max_bad_checks, bad_check_count

def evaluate_policy_mean(env, policy, episodes=5, deterministic=True):
    """Evaluate current policy on raw env rewards and return mean episodic return."""
    was_training = policy.training
    policy.eval()
    returns = []

    try:
        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                if deterministic:
                    with torch.no_grad():
                        state_t = torch.FloatTensor(state).unsqueeze(0)
                        mean, _ = policy(state_t)
                        action = mean.squeeze(0).cpu().tolist()
                else:
                    action, _ = policy.sample_action(state)

                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated

            returns.append(total_reward)
    finally:
        if was_training:
            policy.train()

    return float(np.mean(returns)) if len(returns) > 0 else -float("inf")

def run_pgts(
    env,
    policy,
    value_net,
    optimizer_p,
    optimizer_v,
    episodes=200,
    gamma=0.99,
    adaptive=False, 
    max_m=20,
    m=4,
    entropy_coef=0.1,
    use_lagging=False,
    clip_epsilon=0.2,
    tau=0.01,
    v_epochs=25,
    critic_warmup_episodes=300,
    critic_warmup_multiplier=2.0,
    K=3,
    eval_interval=100,
    eval_episodes=5,
    eval_deterministic=True,
    early_stop=False,
    early_stop_min_episodes=400,
    early_stop_check_every=50,
    early_stop_window=100,
    early_stop_delta=10.0,
    early_stop_patience=4,
):
    print(f"Running PGTS with m={m}, adaptive={adaptive}, lagging={use_lagging}")
    
    # Initialize adaptive components
    lr_scheduler = DynamicLearningRateScheduler(
        initial_lr_policy=5e-5, 
        initial_lr_value=1e-3
    )
    reward_scaler = AdaptiveRewardScaler(init_scale=0.01)
    entropy_scheduler = AdaptiveEntropyScheduler(initial_entropy_coef=entropy_coef, total_episodes=episodes)
    
    rewards_history = []
    lag_policy = None 
    bad_check_count = 0
    best_eval_mean = -float("inf")
    best_policy_state = copy.deepcopy(policy.state_dict())

    if use_lagging:
        lag_policy = copy.deepcopy(policy)
        lag_policy.eval()
    
    current_m = (1 if adaptive else m)

    for ep in range(episodes):
        if adaptive:
            current_m = get_adaptive_m(
                rewards_history, 
                ep, 
                current_m, 
                max_m=max_m, 
                min_m=1
            )
        
        # Get adaptive entropy coefficient
        current_entropy_coef = entropy_scheduler.get_entropy_coef(ep)
        
        if use_lagging and ep > 20:
            rollout_policy = lag_policy
        else:
            rollout_policy = policy
        
        # NOTE: Ensure your env.rollout returns a `dones` array/list
        states, actions, rewards, dones, log_probs, checkpoints = env.rollout(rollout_policy)
        
        # Use adaptive reward scaling
        scaled_rewards = [r * reward_scaler.get_scale() for r in rewards]
        reward_scaler.update_scale(scaled_rewards)

        returns = compute_strided_Tm_returns(
            env, policy, value_net, gamma, current_m, states, scaled_rewards, dones, checkpoints, 
            K=K, search_interval=4
        )

        returns_t = torch.FloatTensor(returns)
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(np.array(actions)) 

        # FIX 1: Calculate advantages BEFORE modifying the value network
        with torch.no_grad():
            advantages = get_advantages(returns_t, value_net(states_t).squeeze())

        # Adaptive value epochs
        v_epochs_current = adaptive_value_epochs(
            ep,
            episodes,
            base_epochs=v_epochs,
            warmup_episodes=critic_warmup_episodes,
            warmup_multiplier=critic_warmup_multiplier,
        )
        train_value_network(optimizer_v, value_net, states_t, returns_t, v_epochs=v_epochs_current)

        # Proceed with Policy update
        old_probs = torch.stack(log_probs).detach() if use_lagging else None
        train_policy_network(optimizer_p, policy, states_t, actions_t, advantages, 
                            entropy_coef=current_entropy_coef, log_probs_old=old_probs, clip_epsilon=clip_epsilon)

        if use_lagging:
            update_ema(lag_policy, policy, tau=tau)

        total_reward = float(np.sum(scaled_rewards))
        rewards_history.append(total_reward)
        
        # Update learning rates based on performance
        if len(rewards_history) >= 2:
            avg_reward = np.mean(rewards_history[-5:]) if len(rewards_history) >= 5 else np.mean(rewards_history)
            lr_policy, lr_value = lr_scheduler.get_learning_rates(ep, avg_reward)
            lr_scheduler.update_optimizer_lr(optimizer_p, optimizer_v, lr_policy, lr_value)

        if ep % 50 == 0:
            print(f"[PGTS] Ep {ep:4d} | Reward: {total_reward:8.6f} | m: {current_m} | "
                  f"Entropy: {current_entropy_coef:.3f} | Scale: {reward_scaler.get_scale():.4f} | "
                  f"V-Epochs: {v_epochs_current}")

        if eval_interval > 0 and (ep + 1) % eval_interval == 0:
            eval_mean = evaluate_policy_mean(env, policy, episodes=eval_episodes, deterministic=eval_deterministic)
            if eval_mean > best_eval_mean:
                best_eval_mean = eval_mean
                best_policy_state = copy.deepcopy(policy.state_dict())
            print(f"[PGTS][Eval] Ep {ep:4d} | mean_return: {eval_mean:8.3f} | best: {best_eval_mean:8.3f}")

        if early_stop:
            stop, bad_check_count = should_early_stop(
                rewards_history,
                min_episodes=early_stop_min_episodes,
                check_every=early_stop_check_every,
                window=early_stop_window,
                plateau_delta=early_stop_delta,
                max_bad_checks=early_stop_patience,
                bad_check_count=bad_check_count,
            )
            if stop:
                print(
                    f"[PGTS] Early stop at ep {ep} | "
                    f"No sustained improvement over recent windows."
                )
                break

    if best_policy_state is not None:
        policy.load_state_dict(best_policy_state)
        print(f"[PGTS] Loaded best checkpoint by eval mean: {best_eval_mean:.3f}")

    return rewards_history

def run_pgts_td(
    env, policy, value_net, optimizer_p, optimizer_v,
    episodes=200, gamma=0.99, adaptive=False, max_m=20, m=4,
    entropy_coef=0.1, use_lagging=False, clip_epsilon=0.2, 
    tau=0.01, v_epochs=4, critic_warmup_episodes=300, critic_warmup_multiplier=2.0,
    K=3, search_interval=1,
    eval_interval=100,
    eval_episodes=5,
    eval_deterministic=True,
    early_stop=False,
    early_stop_min_episodes=400,
    early_stop_check_every=50,
    early_stop_window=100,
    early_stop_delta=10.0,
    early_stop_patience=4,
):
    print(f"Running PGTS-TD with m={m}, adaptive={adaptive}, lagging={use_lagging}")
    rewards_history = []
    lag_policy = copy.deepcopy(policy) if use_lagging else None
    reward_scaler = AdaptiveRewardScaler(init_scale=0.01)
    bad_check_count = 0
    best_eval_mean = -float("inf")
    best_policy_state = copy.deepcopy(policy.state_dict())

    current_m = (1 if adaptive else m)

    for ep in range(episodes):
        if adaptive:
            current_m = get_adaptive_m(rewards_history, ep, current_m, max_m=max_m, min_m=1)
        
        rollout_policy = lag_policy if (use_lagging and ep > 20) else policy
        
        # 1. ROLLOUT: Still collect the trajectory first for efficiency
        states, actions, rewards, dones, log_probs, checkpoints = env.rollout(rollout_policy)
        raw_rewards = np.array(rewards, dtype=np.float32)
        reward_scaler.update_scale(raw_rewards)
        current_scale = reward_scaler.get_scale()
        scaled_rewards = [r * current_scale for r in raw_rewards]

        # 2. TD-STYLE TARGET COMPUTATION
        # Instead of bootstrapping backward, we treat each state as a TD-target source
        td_targets = []
        for i in range(len(states)):
            if dones[i]:
                td_targets.append(scaled_rewards[i])
            else:
                # TD(0) logic: Target = Tree Search Lookahead from current state
                env.restore_checkpoint(checkpoints[i])
                env.state = states[i]
                target_val = compute_Tm_value(env, policy, value_net, gamma, current_m, K=K)
                td_targets.append(current_scale * target_val)

        # 3. PREPARE TENSORS
        returns_t = torch.FloatTensor(td_targets)
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(np.array(actions))

        # 4. ADVANTAGE CALCULATION (On-policy TD-error)
        with torch.no_grad():
            advantages = get_advantages(returns_t, value_net(states_t).squeeze())

        # 5. VALUE UPDATE (Critic) with early warmup so critic can get ahead
        v_epochs_current = adaptive_value_epochs(
            ep,
            episodes,
            base_epochs=v_epochs,
            warmup_episodes=critic_warmup_episodes,
            warmup_multiplier=critic_warmup_multiplier,
            min_epochs=2,
        )
        train_value_network(optimizer_v, value_net, states_t, returns_t, v_epochs=v_epochs_current)

        # 6. POLICY UPDATE (Actor)
        old_probs = torch.stack(log_probs).detach() if use_lagging else None
        train_policy_network(optimizer_p, policy, states_t, actions_t, advantages, 
                            entropy_coef = entropy_coef, log_probs_old = old_probs, clip_epsilon = clip_epsilon)

        if use_lagging:
            update_ema(lag_policy, policy, tau=tau)

        raw_total_reward = float(np.sum(raw_rewards))
        scaled_total_reward = float(np.sum(scaled_rewards))
        rewards_history.append(raw_total_reward)

        if ep % 50 == 0:
            print(f"[PGTS-TD] Ep {ep:4d} | Raw: {raw_total_reward:8.6f} | Scaled: {scaled_total_reward:8.6f} | "
                  f"m: {current_m} | Scale: {current_scale:.4f} | V-Epochs: {v_epochs_current}")

        if eval_interval > 0 and (ep + 1) % eval_interval == 0:
            eval_mean = evaluate_policy_mean(env, policy, episodes=eval_episodes, deterministic=eval_deterministic)
            if eval_mean > best_eval_mean:
                best_eval_mean = eval_mean
                best_policy_state = copy.deepcopy(policy.state_dict())
            print(f"[PGTS-TD][Eval] Ep {ep:4d} | mean_return: {eval_mean:8.3f} | best: {best_eval_mean:8.3f}")

        if early_stop:
            stop, bad_check_count = should_early_stop(
                rewards_history,
                min_episodes=early_stop_min_episodes,
                check_every=early_stop_check_every,
                window=early_stop_window,
                plateau_delta=early_stop_delta,
                max_bad_checks=early_stop_patience,
                bad_check_count=bad_check_count,
            )
            if stop:
                print(
                    f"[PGTS-TD] Early stop at ep {ep} | "
                    f"No sustained improvement over recent windows."
                )
                break

    if best_policy_state is not None:
        policy.load_state_dict(best_policy_state)
        print(f"[PGTS-TD] Loaded best checkpoint by eval mean: {best_eval_mean:.3f}")

    return rewards_history

def run_pgts_online(env, policy, value_net, optimizer_p, optimizer_v, episodes=200, gamma=0.99, adaptive=False, max_m=20, m=4, entropy_coef=0.1, use_lagging=False, clip_epsilon=0.2, tau=0.01, v_epochs=5, critic_warmup_episodes=300, critic_warmup_multiplier=2.0, K=3, early_stop=False, early_stop_min_episodes=400, early_stop_check_every=50, early_stop_window=100, early_stop_delta=10.0, early_stop_patience=4, eval_interval=100, eval_episodes=5, eval_deterministic=True):
    
    print(f"Running PGTS Online with m={m}, adaptive={adaptive}, lagging={use_lagging}")

    # Initialize adaptive components
    lr_scheduler = DynamicLearningRateScheduler(
        initial_lr_policy=5e-5, 
        initial_lr_value=1e-3
    )
    reward_scaler = AdaptiveRewardScaler(init_scale=0.01)
    entropy_scheduler = AdaptiveEntropyScheduler(initial_entropy_coef=entropy_coef, total_episodes=episodes)
    grad_monitor = GradientMonitor()
    
    rewards_history = []
    lag_policy = copy.deepcopy(policy).eval() if use_lagging else None
    current_m = (1 if adaptive else m)
    bad_check_count = 0
    best_eval_mean = -float("inf")
    best_policy_state = copy.deepcopy(policy.state_dict())

    for ep in range(episodes):
        if adaptive: 
            current_m = get_adaptive_m(rewards_history, ep, current_m, max_m=max_m)
        
        # Get adaptive entropy coefficient (decay over time)
        current_entropy_coef = entropy_scheduler.get_entropy_coef(ep)
        
        state, _ = env.reset()
        total_reward = 0.0
        scaled_total_reward = 0.0
        done = False
        
        while not done:
            checkpoint = env.get_checkpoint()
            rollout_policy = lag_policy if (use_lagging and ep > 20) else policy
            action, log_prob = rollout_policy.sample_action(state)
            
            # Compute T^m Target
            env.restore_checkpoint(checkpoint)
            target_v = compute_Tm_value(env, policy, value_net, gamma, current_m, K=K)
            
            # Step Environment
            env.restore_checkpoint(checkpoint) 
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            raw_reward = float(reward)

            # Use adaptive reward scaling
            scaled_reward = raw_reward * reward_scaler.get_scale()
            
            state_t = torch.FloatTensor(state).unsqueeze(0)
            target_t = torch.tensor([target_v], dtype=torch.float32)
            action_t = torch.FloatTensor(action).unsqueeze(0)
            
            # Adaptive value update epochs with early warmup so critic can get ahead
            v_epochs_current = adaptive_value_epochs(
                ep,
                episodes,
                base_epochs=v_epochs,
                warmup_episodes=critic_warmup_episodes,
                warmup_multiplier=critic_warmup_multiplier,
                min_epochs=2,
            )
            train_value_network(optimizer_v, value_net, state_t, target_t, v_epochs=v_epochs_current)
            
            with torch.no_grad():
                adv_t = torch.tensor([target_v - value_net(state_t).item()], dtype=torch.float32)

            train_policy_network(
                optimizer_p, 
                policy, 
                state_t, 
                action_t, 
                adv_t, 
                current_entropy_coef,  # Use adaptive entropy
                (log_prob.detach() if use_lagging else None), 
                clip_epsilon
            )
            
            if use_lagging: 
                update_ema(lag_policy, policy, tau)
            
            state = next_state
            total_reward = total_reward + raw_reward
            scaled_total_reward = scaled_total_reward + scaled_reward
        
        rewards_history.append(total_reward)
        reward_scaler.update_scale([total_reward])
        
        # Update learning rates based on performance
        if len(rewards_history) >= 2:
            avg_reward = np.mean(rewards_history[-5:]) if len(rewards_history) >= 5 else np.mean(rewards_history)
            lr_policy, lr_value = lr_scheduler.get_learning_rates(ep, avg_reward)
            lr_scheduler.update_optimizer_lr(optimizer_p, optimizer_v, lr_policy, lr_value)
        
        if ep % 50 == 0:
            rolling_window = min(100, len(rewards_history))
            rolling_raw_mean = float(np.mean(rewards_history[-rolling_window:]))
            current_lr_p = float(optimizer_p.param_groups[0]["lr"])
            current_lr_v = float(optimizer_v.param_groups[0]["lr"])
            print(f"[PGTS-Online] Ep {ep:4d} | Raw: {total_reward:8.6f} | Scaled: {scaled_total_reward:8.6f} | m: {current_m} | "
                  f"Roll100: {rolling_raw_mean:8.3f} | LR_P: {current_lr_p:.2e} | LR_V: {current_lr_v:.2e} | "
                  f"Entropy: {current_entropy_coef:.3f} | Scale: {reward_scaler.get_scale():.4f}")

        if eval_interval > 0 and (ep + 1) % eval_interval == 0:
            eval_mean = evaluate_policy_mean(env, policy, episodes=eval_episodes, deterministic=eval_deterministic)
            if eval_mean > best_eval_mean:
                best_eval_mean = eval_mean
                best_policy_state = copy.deepcopy(policy.state_dict())
            print(f"[PGTS-Online][Eval] Ep {ep:4d} | mean_return: {eval_mean:8.3f} | best: {best_eval_mean:8.3f}")

        if early_stop:
            stop, bad_check_count = should_early_stop(
                rewards_history,
                min_episodes=early_stop_min_episodes,
                check_every=early_stop_check_every,
                window=early_stop_window,
                plateau_delta=early_stop_delta,
                max_bad_checks=early_stop_patience,
                bad_check_count=bad_check_count,
            )
            if stop:
                print(
                    f"[PGTS-Online] Early stop at ep {ep} | "
                    f"No sustained improvement over recent windows."
                )
                break

    if best_policy_state is not None:
        policy.load_state_dict(best_policy_state)
        print(f"[PGTS-Online] Loaded best checkpoint by eval mean: {best_eval_mean:.3f}")

    return rewards_history

def run_pg_mstep(
    env,
    policy,
    value_net,
    optimizer_p,
    optimizer_v,
    episodes=200,
    gamma=0.99,
    m=4,  # This is your 'n' in n-step bootstrapping
    adaptive = False,
    max_m = 20,
    entropy_coef=0.1,
    use_lagging=False,
    clip_epsilon=0.2,
    tau=0.01,
    v_epochs=25,
    eval_interval=100,
    eval_episodes=5,
    eval_deterministic=True,
    early_stop=False,
    early_stop_min_episodes=400,
    early_stop_check_every=50,
    early_stop_window=100,
    early_stop_delta=10.0,
    early_stop_patience=4,
):
    print(f"Running PG-MStep with m={m}, adaptive={adaptive}, lagging={use_lagging}")
    rewards_history = []
    lag_policy = None 
    bad_check_count = 0
    best_eval_mean = -float("inf")
    best_policy_state = copy.deepcopy(policy.state_dict())

    if use_lagging:
        lag_policy = copy.deepcopy(policy)
        lag_policy.eval()

    for ep in range(episodes):
        # 1. Rollout selection
        rollout_policy = lag_policy if (use_lagging and ep > 20) else policy
        
        # 2. Collect actual trajectory
        states, actions, rewards, dones, log_probs, _ = env.rollout(rollout_policy)
        
        # 3. Scale rewards (0.01) for numerical stability
        rewards = [r * 0.01 for r in rewards]

        # 4. COMPUTE M-STEP RETURNS (n-step bootstrapping)
        # We use the current Value Network to bootstrap after 'm' actual steps
        with torch.no_grad():
            states_t_all = torch.FloatTensor(np.array(states))
            values = value_net(states_t_all)
            
            # This uses your provided compute_m_step_returns logic
            returns = compute_m_step_returns(rewards, values, gamma, m, max_m=max_m if adaptive else m)
            returns_t = torch.FloatTensor(returns)

        # 5. ADVANTAGE CALCULATION
        with torch.no_grad():
            advantages = get_advantages(returns_t, values.squeeze())

        # 6. CRITIC UPDATE (Value Network)
        train_value_network(optimizer_v, value_net, states_t_all, returns_t, v_epochs=v_epochs)

        # 7. ACTOR UPDATE (Policy Network)
        old_probs = torch.stack(log_probs).detach() if use_lagging else None
        train_policy_network(optimizer_p, policy, states_t_all, torch.FloatTensor(np.array(actions)), advantages, 
                            entropy_coef = entropy_coef, log_probs_old = old_probs, clip_epsilon = clip_epsilon)
        
        if use_lagging:
            update_ema(lag_policy, policy, tau=tau)

        # Track unscaled rewards for the history
        total_reward = float(np.sum(rewards))
        rewards_history.append(total_reward)

        if ep % 50 == 0 or ep == 0 or ep == episodes - 1:
            print(f"[m-Step PG] Ep {ep:4d} | Reward: {total_reward:8.6f}")

        if eval_interval > 0 and (ep + 1) % eval_interval == 0:
            eval_mean = evaluate_policy_mean(env, policy, episodes=eval_episodes, deterministic=eval_deterministic)
            if eval_mean > best_eval_mean:
                best_eval_mean = eval_mean
                best_policy_state = copy.deepcopy(policy.state_dict())
            print(f"[m-Step PG][Eval] Ep {ep:4d} | mean_return: {eval_mean:8.3f} | best: {best_eval_mean:8.3f}")

        if early_stop:
            stop, bad_check_count = should_early_stop(
                rewards_history,
                min_episodes=early_stop_min_episodes,
                check_every=early_stop_check_every,
                window=early_stop_window,
                plateau_delta=early_stop_delta,
                max_bad_checks=early_stop_patience,
                bad_check_count=bad_check_count,
            )
            if stop:
                print(
                    f"[m-Step PG] Early stop at ep {ep} | "
                    f"No sustained improvement over recent windows."
                )
                break

    if best_policy_state is not None:
        policy.load_state_dict(best_policy_state)
        print(f"[m-Step PG] Loaded best checkpoint by eval mean: {best_eval_mean:.3f}")

    return rewards_history