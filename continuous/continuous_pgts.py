import torch
import torch.nn as nn
import numpy as np


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
    entropy_coef=0.01
):

    rewards_history = []

    mse = nn.MSELoss()

    for ep in range(episodes):

        states, actions, rewards, log_probs = env.rollout(policy)

        rewards = np.array(rewards, dtype=np.float32)

        states_t = torch.FloatTensor(np.array(states))
        log_probs_t = torch.stack(log_probs)

        values = value_net(states_t)
        values_detached = values.detach()

        

        returns = compute_m_step_returns(
            rewards,
            values_detached,
            gamma,
            m
        )

        returns_t = torch.FloatTensor(returns)

        advantages = returns_t - values_detached
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        entropy = -(log_probs_t.exp() * log_probs_t).mean()

        policy_loss = -(log_probs_t * advantages).sum()
        policy_loss -= entropy_coef * entropy

        value_loss = mse(values, returns_t)

        optimizer_p.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer_p.step()

        optimizer_v.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
        optimizer_v.step()

        total_reward = float(np.sum(rewards))
        rewards_history.append(total_reward)

        if ep % 10 == 0:
            print(f"[PGTS] Ep {ep:4d} | Reward: {total_reward:8.2f}")

    return rewards_history
