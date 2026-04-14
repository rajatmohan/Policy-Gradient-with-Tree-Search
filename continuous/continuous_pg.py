import torch
import numpy as np
from training_utils import DynamicLearningRateScheduler, AdaptiveRewardScaler, AdaptiveEntropyScheduler

def compute_returns(rewards, gamma):
    returns = []
    G = 0
    # Monte Carlo return calculation: G_t = r_t + gamma * G_{t+1}
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def run_pg(env, policy, optimizer, episodes=200, gamma=0.99):
    # Initialize adaptive components
    lr_scheduler = DynamicLearningRateScheduler(
        initial_lr_policy=5e-5,
        min_lr_policy=1e-6,
        max_lr_policy=1e-3
    )
    reward_scaler = AdaptiveRewardScaler(init_scale=0.01)
    entropy_scheduler = AdaptiveEntropyScheduler(initial_entropy_coef=0.1, total_episodes=episodes)
    
    rewards_history = []

    for ep in range(episodes):
        # Get adaptive entropy and learning rate
        current_entropy_coef = entropy_scheduler.get_entropy_coef(ep)
        
        # Unpack the 6 variables now returned by your updated MDP rollout
        # states, actions, rewards, dones, log_probs, checkpoints
        states, actions, rewards, dones, log_probs, _ = env.rollout(policy)

        raw_rewards = np.array(rewards, dtype=np.float32)

        # Use adaptive reward scaling
        reward_scaler.update_scale(raw_rewards)
        scaled_rewards = [r * reward_scaler.get_scale() for r in raw_rewards]
        
        # Convert rewards to numpy for math operations
        rewards_np = np.array(scaled_rewards, dtype=np.float32)

        # Calculate Monte Carlo returns
        returns = compute_returns(rewards_np, gamma)
        returns = torch.FloatTensor(returns)

        # Stack log probabilities collected during rollout
        log_probs_t = torch.stack(log_probs)

        # Standardize returns to reduce variance (Standard PG trick)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy gradient with entropy bonus
        mean, std = policy(torch.FloatTensor(np.array(states)))
        dist = torch.distributions.Normal(mean, std)
        
        # REINFORCE Loss with entropy regularization
        policy_loss = -(log_probs_t * returns).sum()
        policy_loss -= current_entropy_coef * dist.entropy().sum()

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        raw_total_reward = float(np.sum(raw_rewards))
        scaled_total_reward = float(np.sum(rewards_np))
        rewards_history.append(raw_total_reward)
        
        # Update learning rates based on performance
        if len(rewards_history) >= 2:
            avg_reward = np.mean(rewards_history[-5:]) if len(rewards_history) >= 5 else np.mean(rewards_history)
            lr_policy, _ = lr_scheduler.get_learning_rates(ep, avg_reward)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_policy

        if ep % 50 == 0 or ep == 0 or ep == episodes - 1:
            print(f"[PG] Ep {ep:4d} | Raw: {raw_total_reward:8.6f} | Scaled: {scaled_total_reward:8.6f} | "
                f"Entropy: {current_entropy_coef:.3f} | Scale: {reward_scaler.get_scale():.4f}")

    return rewards_history