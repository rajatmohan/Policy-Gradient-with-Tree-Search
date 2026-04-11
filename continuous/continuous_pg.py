import torch
import numpy as np


def compute_returns(rewards, gamma):
    returns = []
    G = 0

    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    return returns


def run_pg(env, policy, optimizer, episodes=200, gamma=0.99):

    rewards_history = []

    for ep in range(episodes):

        states, actions, rewards, log_probs = env.rollout(policy)

        returns = compute_returns(rewards, gamma)
        returns = torch.FloatTensor(returns)

        log_probs = torch.stack(log_probs)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -(log_probs * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        total_reward = sum(rewards)
        rewards_history.append(total_reward)

        if ep % 10 == 0:
            print(f"[PG] Ep {ep:4d} | Reward: {total_reward:8.2f}")

    return rewards_history
