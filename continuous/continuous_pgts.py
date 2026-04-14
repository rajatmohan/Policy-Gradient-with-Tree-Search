import torch
import torch.nn as nn
import numpy as np
import copy

from common.adaptive_m import get_adaptive_m
from continuous import lunar_mdp

def compute_gae(rewards, values, next_values, dones, gamma, lambd):
    """
    GAE calculation as seen in the GitHub repo.
    rewards, values, next_values, dones should be numpy arrays or lists.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0
    
    # Calculate deltas: r + gamma * V(s') - V(s)
    # Using 'dones' to mask transitions to terminal states
    deltas = np.array(rewards) + gamma * np.array(next_values) * (1 - np.array(dones)) - np.array(values)
    
    # Backward pass for GAE
    for t in reversed(range(T)):
        gae = deltas[t] + gamma * lambd * (1 - dones[t]) * gae
        advantages[t] = gae
        
    # TD Target = Advantage + Value
    targets = advantages + np.array(values)
    return advantages, targets

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

def compute_Tm_value(mdp, policy, value_net, gamma, m, K=3, device='cpu'):
    """
    Recursive T^m Operator.
    K: Number of actions to sample at each node (Branching factor)
    m: Depth of the tree
    """
    if m == 0:
        state_t = torch.FloatTensor(mdp.state).unsqueeze(0).to(device)
        return value_net(state_t).item()

    checkpoint = mdp.get_checkpoint()
    current_state = mdp.state.copy()
    
    max_q = -float('inf')

    for _ in range(K):
        mdp.restore_checkpoint(checkpoint)
        mdp.state = current_state
        
        action, _ = policy.sample_action(mdp.state)
        action += np.random.normal(0, 0.1, size=action.shape)
        action = np.clip(action, -1, 1)
        
        next_state, reward, done, _, _ = mdp.step(action)

        if done:
            q_val = reward
        else:
            q_val = reward + gamma * compute_Tm_value(
                mdp, policy, value_net, gamma, m - 1, K, device
            )

        if q_val > max_q:
            max_q = q_val

    mdp.restore_checkpoint(checkpoint)
    mdp.state = current_state
    return max_q

def compute_strided_Tm_returns(
    env, policy, value_net, gamma, m, states, rewards, dones, checkpoints, K=3, search_interval=4, device='cpu'
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
            returns[i] = compute_Tm_value(env, policy, value_net, gamma, m, K=K, device=device)
            
        # Condition 2: Intermediate steps (Bootstrap backwards)
        else:
            returns[i] = rewards[i] + gamma * returns[i+1]
            
    return returns.tolist()

def update_ema(target, source, tau=0.05):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(tau * s_param.data + (1 - tau) * t_param.data)

def train_value_network(optimizer_v, value_net, states_t, targets_t, v_epochs=25, batch_size=64, device='cpu'):
    """Common Value Network update logic using Huber Loss and Grad Clipping."""
    # states_t = states_t.to(device)
    # targets_t = targets_t.to(device)
    # for _ in range(v_epochs):
    #     optimizer_v.zero_grad()
    #     current_values = value_net(states_t).view(-1)
    #     targets_flat = targets_t.view(-1)
    #     v_loss = torch.nn.functional.huber_loss(current_values, targets_flat, delta=1.0)
    #     v_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
    #     optimizer_v.step()

    data_size = states_t.size(0)
    for _ in range(v_epochs):
        # Generate random indices to shuffle the trajectory
        perm = torch.randperm(data_size).to(device)
        
        for i in range(0, data_size, batch_size):
            indices = perm[i:i + batch_size]
            
            optimizer_v.zero_grad()
            current_values = value_net(states_t[indices]).view(-1)
            batch_targets = targets_t[indices].view(-1)
            
            v_loss = torch.nn.functional.huber_loss(current_values, batch_targets, delta=1.0)
            v_loss.backward()
            
            # Grad clipping is even more important with mini-batches
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            optimizer_v.step()

def train_policy_network(
    optimizer_p, policy, states_t, actions_t, advantages, 
    entropy_coef, log_probs_old=None, clip_epsilon=0.2, batch_size=64, device='cpu'
):
    """Common Policy update logic supporting both Vanilla PG and Lagging/PPO."""
    # states_t = states_t.to(device)
    # actions_t = actions_t.to(device)
    # advantages = advantages.to(device)
    # if log_probs_old is not None:
    #     log_probs_old = log_probs_old.to(device)
    # mean, std = policy(states_t)
    # dist = torch.distributions.Normal(mean, std)
    # new_log_probs = dist.log_prob(actions_t).sum(dim=-1)

    # if log_probs_old is not None:
    #     # PPO/Lagging Style update
    #     log_probs_old = log_probs_old.view_as(new_log_probs)
    #     ratios = torch.exp(new_log_probs - log_probs_old)
    #     surr1 = ratios * advantages
    #     surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    #     policy_loss = -torch.min(surr1, surr2).mean()
    # else:
    #     # Vanilla Policy Gradient
    #     policy_loss = -(advantages * new_log_probs).mean()

    # # Add Entropy Bonus
    # policy_loss -= entropy_coef * dist.entropy().mean()

    # optimizer_p.zero_grad()
    # policy_loss.backward()
    # torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    # optimizer_p.step()
    data_size = states_t.size(0)
    
    # We typically do fewer epochs for the policy to avoid moving too far from old policy
    for _ in range(10): 
        perm = torch.randperm(data_size).to(device)
        
        for i in range(0, data_size, batch_size):
            idx = perm[i:i + batch_size]
            
            # Get current distribution for this batch
            dist = policy.get_dist(states_t[idx])
            new_log_probs = dist.log_prob(actions_t[idx]).sum(dim=-1)
            
            if log_probs_old is not None:
                # PPO Clipping logic
                old_batch_log_probs = log_probs_old[idx].view_as(new_log_probs)
                ratio = torch.exp(new_log_probs - old_batch_log_probs)
                
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
            else:
                # Standard Policy Gradient
                policy_loss = -(advantages[idx] * new_log_probs).mean()

            # Entropy bonus for exploration
            policy_loss -= entropy_coef * dist.entropy().mean()

            optimizer_p.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer_p.step()

def get_standardized_advantages(advantages_np, device='cpu'):
    """Standardizes the advantage vector (Mean 0, Std 1)."""
    adv = torch.FloatTensor(advantages_np).to(device)
    if adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv

def get_advantages(returns_t, values_t):
    """Computes standardized advantages."""
    advantages = returns_t - values_t
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

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
    K=3,
    device='cpu',
    lambd=0.95,
    scheduler_p=None,
    scheduler_v=None,
):
    print(f"Running PGTS with m={m}, adaptive={adaptive}, lagging={use_lagging}")
    rewards_history = []
    lag_policy = None 

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
        if use_lagging and ep > 20:
            rollout_policy = lag_policy
        else:
            rollout_policy = policy
        
        # NOTE: Ensure your env.rollout returns a `dones` array/list
        states, actions, rewards, dones, log_probs, checkpoints = env.rollout(rollout_policy)
        
        rewards = [r * 0.01 for r in rewards]

        returns = compute_strided_Tm_returns(
            env, policy, value_net, gamma, current_m, states, rewards, dones, checkpoints, 
            K=K, search_interval=4, device=device
        )

        returns_t = torch.FloatTensor(returns).to(device)
        states_t = torch.FloatTensor(np.array(states)).to(device)
        actions_t = torch.FloatTensor(np.array(actions)).to(device) 

        with torch.no_grad():
            values = value_net(states_t).view(-1).cpu().numpy()

        adv_np, targets_np = compute_gae(rewards, values, returns, dones, gamma, lambd)

        advantages = get_standardized_advantages(adv_np, device=device)

        # FIX 1: Calculate advantages BEFORE modifying the value network
        # with torch.no_grad():
        #     advantages = get_advantages(returns_t, value_net(states_t).squeeze())

        train_value_network(optimizer_v, value_net, states_t, returns_t, v_epochs=v_epochs, device=device)

        # Proceed with Policy update
        old_probs = torch.stack(log_probs).detach() if use_lagging else None
        train_policy_network(optimizer_p, policy, states_t, actions_t, advantages, 
                            entropy_coef = entropy_coef, log_probs_old = old_probs, clip_epsilon = clip_epsilon, device=device)

        if use_lagging:
            update_ema(lag_policy, policy, tau=tau)

        rewards = np.array(rewards, dtype=np.float32)
        total_reward = float(np.sum(rewards))
        rewards_history.append(total_reward)

        if scheduler_p is not None:
            scheduler_p.step()
        if scheduler_v is not None:
            scheduler_v.step()

        if ep % 50 == 0:
            print(f"[PGTS] Ep {ep:4d} | Reward: {total_reward:8.6f} | m: {current_m}")

    return rewards_history

def run_pgts_td(
    env, policy, value_net, optimizer_p, optimizer_v,
    episodes=200, gamma=0.99, adaptive=False, max_m=20, m=4,
    entropy_coef=0.1, use_lagging=False, clip_epsilon=0.2, 
    tau=0.01, v_epochs=5, K=3, search_interval=1, device='cpu' ,
    scheduler_p=None, scheduler_v=None
):
    print(f"Running PGTS-TD with m={m}, adaptive={adaptive}, lagging={use_lagging}")
    rewards_history = []
    lag_policy = copy.deepcopy(policy) if use_lagging else None

    current_m = (1 if adaptive else m)

    for ep in range(episodes):
        if adaptive:
            current_m = get_adaptive_m(rewards_history, ep, current_m, max_m=max_m, min_m=1)
        
        rollout_policy = lag_policy if (use_lagging and ep > 20) else policy
        
        # 1. ROLLOUT: Still collect the trajectory first for efficiency
        states, actions, rewards, dones, log_probs, checkpoints = env.rollout(rollout_policy)
        rewards = [r * 0.01 for r in rewards] # Scale rewards

        # 2. TD-STYLE TARGET COMPUTATION
        # Instead of bootstrapping backward, we treat each state as a TD-target source
        td_targets = []
        for i in range(len(states)):
            if dones[i]:
                td_targets.append(rewards[i])
            else:
                # TD(0) logic: Target = Tree Search Lookahead from current state
                env.restore_checkpoint(checkpoints[i])
                env.state = states[i]
                target_val = compute_Tm_value(env, policy, value_net, gamma, current_m, K=K, device=device)
                td_targets.append(target_val)

        # 3. PREPARE TENSORS
        returns_t = torch.FloatTensor(td_targets).to(device)
        states_t = torch.FloatTensor(np.array(states)).to(device)
        actions_t = torch.FloatTensor(np.array(actions)).to(device)

        # 4. ADVANTAGE CALCULATION (On-policy TD-error)
        with torch.no_grad():
            advantages = get_advantages(returns_t, value_net(states_t).squeeze())

        # 5. VALUE UPDATE (Critic)
        # In TD style, we often do fewer epochs or use a smaller buffer
        train_value_network(optimizer_v, value_net, states_t, returns_t, v_epochs = v_epochs, device=device)

        # 6. POLICY UPDATE (Actor)
        old_probs = torch.stack(log_probs).detach() if use_lagging else None
        train_policy_network(optimizer_p, policy, states_t, actions_t, advantages, 
                            entropy_coef = entropy_coef, log_probs_old = old_probs, clip_epsilon = clip_epsilon, device=device)

        if use_lagging:
            update_ema(lag_policy, policy, tau=tau)

        total_reward = float(np.sum(rewards))
        rewards_history.append(total_reward)

        if scheduler_p is not None:
            scheduler_p.step()
        if scheduler_v is not None:
            scheduler_v.step()

        if ep % 50 == 0:
            print(f"[PGTS-TD-Batch] Ep {ep:4d} | Reward: {total_reward:8.6f} | m: {current_m}")

    return rewards_history

def run_pgts_online(env, policy, value_net, optimizer_p, optimizer_v, episodes=200, gamma=0.99, adaptive=False, max_m=20, m=4, entropy_coef=0.1, use_lagging=False, clip_epsilon=0.2, tau=0.01, v_epochs=1, K=3, device='cpu', scheduler_p=None, scheduler_v=None):
    
    print(f"Running PGTS Online with m={m}, adaptive={adaptive}, lagging={use_lagging}")

    rewards_history = []
    lag_policy = copy.deepcopy(policy).eval() if use_lagging else None
    current_m = (1 if adaptive else m)

    for ep in range(episodes):
        if adaptive: current_m = get_adaptive_m(rewards_history, ep, current_m, max_m=max_m)
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            checkpoint = env.get_checkpoint()
            rollout_policy = lag_policy if (use_lagging and ep > 20) else policy
            action, log_prob = rollout_policy.sample_action(state, device=device)
            
            # Compute T^m Target
            env.restore_checkpoint(checkpoint)
            target_v = compute_Tm_value(env, policy, value_net, gamma, current_m, K=K, device=device)
            
            # Step Environment
            env.restore_checkpoint(checkpoint) 
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            reward = 0.01 * reward  # Scale reward for stability
            
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            target_t = torch.tensor([target_v], dtype=torch.float32).to(device)
            action_t = torch.FloatTensor(action).unsqueeze(0).to(device)
            
            train_value_network(optimizer_v, value_net, state_t, target_t, v_epochs=v_epochs, device=device)
            with torch.no_grad():
                adv_t = torch.tensor([target_v - value_net(state_t).item()], dtype=torch.float32).to(device)

            train_policy_network(
                optimizer_p, 
                policy, 
                state_t, 
                action_t, 
                adv_t, 
                entropy_coef, 
                (log_prob.detach() if use_lagging else None), 
                clip_epsilon,
                device=device
            )
            if use_lagging: update_ema(lag_policy, policy, tau)
            state, total_reward = next_state, total_reward + reward
        
        rewards_history.append(total_reward)

        if scheduler_p is not None:
            scheduler_p.step()
        if scheduler_v is not None:
            scheduler_v.step()
        if ep % 50 == 0:
            print(f"[PGTS-ONLINE] Ep {ep:4d} | Reward: {total_reward:8.6f} | m: {current_m}")

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
    device='cpu',
):
    print(f"Running PG-MStep with m={m}, adaptive={adaptive}, lagging={use_lagging}")
    rewards_history = []
    lag_policy = None 

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
            states_t_all = torch.FloatTensor(np.array(states)).to(device)
            values = value_net(states_t_all)
            
            # This uses your provided compute_m_step_returns logic
            returns = compute_m_step_returns(rewards, values, gamma, m, max_m=max_m if adaptive else m)
            returns_t = torch.FloatTensor(returns).to(device)

        # 5. ADVANTAGE CALCULATION
        with torch.no_grad():
            advantages = get_advantages(returns_t, values.squeeze())

        # 6. CRITIC UPDATE (Value Network)
        train_value_network(optimizer_v, value_net, states_t_all, returns_t, v_epochs=v_epochs, device=device)

        # 7. ACTOR UPDATE (Policy Network)
        old_probs = torch.stack(log_probs).detach() if use_lagging else None
        train_policy_network(optimizer_p, policy, states_t_all, torch.FloatTensor(np.array(actions)).to(device), advantages, 
                            entropy_coef = entropy_coef, log_probs_old = old_probs, clip_epsilon = clip_epsilon, device=device)
        
        if use_lagging:
            update_ema(lag_policy, policy, tau=tau)

        # Track unscaled rewards for the history
        total_reward = float(np.sum(rewards))
        rewards_history.append(total_reward)

        if ep % 50 == 0 or ep == 0 or ep == episodes - 1:
            print(f"[m-Step PG] Ep {ep:4d} | Reward: {total_reward:8.6f}")

    return rewards_history