import torch
import torch.nn as nn
import numpy as np
import copy

from common.adaptive_m import get_adaptive_m
from continuous import lunar_mdp

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
    entropy_coef=0.05,
    use_lagging=False,
    clip_epsilon=0.2,
    tau=0.01,
    v_epochs=10,
    K=3,
):
    rewards_history = []
    mse = nn.MSELoss()
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
        
        returns = compute_strided_Tm_returns(
            env, policy, value_net, gamma, current_m, states, rewards, dones, checkpoints, 
            K=K, search_interval=4
        )

        returns_t = torch.FloatTensor(returns)
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(np.array(actions)) 

        # FIX 1: Calculate advantages BEFORE modifying the value network
        with torch.no_grad():
            old_values = value_net(states_t).squeeze()
            advantages = returns_t - old_values
            # Standardize advantages (ensure there is more than 1 item to avoid NaN)
            if advantages.shape[0] > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # NOW train the value network
        for _ in range(v_epochs):
            optimizer_v.zero_grad()
            current_values = value_net(states_t).squeeze()
            v_loss = mse(current_values, returns_t)
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
            optimizer_v.step()

        # Proceed with Policy update
        mean, std = policy(states_t)
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions_t).sum(dim=-1)

        rewards = np.array(rewards, dtype=np.float32)

        if use_lagging:
            log_probs_t = torch.stack(log_probs).detach()
            ratios = torch.exp(new_log_probs - log_probs_t)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
        else:
            policy_loss = -(advantages * new_log_probs).mean()

        # Entropy bonus for exploration
        entropy = dist.entropy().mean()
        policy_loss -= entropy_coef * entropy

        optimizer_p.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer_p.step()

        if use_lagging:
            update_ema(lag_policy, policy, tau=tau)

        total_reward = float(np.sum(rewards))
        rewards_history.append(total_reward)

        if ep % 50 == 0 or ep == 0 or ep == episodes - 1:
            print(f"[PGTS] Ep {ep:4d} | Reward: {total_reward:8.6f}")

    return rewards_history
