import torch
import numpy as np

def compute_returns(rewards, gamma):
    returns = []
    G = 0
    # Monte Carlo return calculation: G_t = r_t + gamma * G_{t+1}
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def run_pg(env, policy, optimizer, episodes=200, gamma=0.99):
    rewards_history = []

    for ep in range(episodes):
        # Unpack the 6 variables now returned by your updated MDP rollout
        # states, actions, rewards, dones, log_probs, checkpoints
        states, actions, rewards, dones, log_probs, _ = env.rollout(policy)

        # Convert rewards to numpy for math operations
        rewards_np = np.array(rewards, dtype=np.float32)

        # Calculate Monte Carlo returns
        returns = compute_returns(rewards_np, gamma)
        returns = torch.FloatTensor(returns)

        # Stack log probabilities collected during rollout
        log_probs_t = torch.stack(log_probs)

        # Standardize returns to reduce variance (Standard PG trick)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # REINFORCE Loss: -sum(log_prob * G)
        # We use a sum here because we are updating once per trajectory
        loss = -(log_probs_t * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        total_reward = float(np.sum(rewards_np))
        rewards_history.append(total_reward)

        if ep % 50 == 0 or ep == 0 or ep == episodes - 1:
            print(f"[PG] Ep {ep:4d} | Reward: {total_reward:8.6f}")

    return rewards_history