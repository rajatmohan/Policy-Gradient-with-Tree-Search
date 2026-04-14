import torch
import numpy as np

class DynamicLearningRateScheduler:
    """
    Adaptive learning rate scheduler based on reward progress.
    Increases LR when training stalls, decreases when unstable.
    """
    def __init__(self, 
                 initial_lr_policy=5e-5, 
                 initial_lr_value=1e-3,
                 min_lr_policy=1e-6,
                 max_lr_policy=1e-3,
                 min_lr_value=1e-4,
                 max_lr_value=5e-3):
        self.initial_lr_policy = initial_lr_policy
        self.initial_lr_value = initial_lr_value
        self.min_lr_policy = min_lr_policy
        self.max_lr_policy = max_lr_policy
        self.min_lr_value = min_lr_value
        self.max_lr_value = max_lr_value
        
        self.reward_history = []
        self.stale_count = 0
        self.volatile_count = 0
    
    def get_learning_rates(self, episode, avg_reward):
        """Compute adaptive learning rates based on training progress."""
        self.reward_history.append(avg_reward)
        
        # Check if training is staling (no improvement)
        if len(self.reward_history) > 10:
            recent_avg = np.mean(self.reward_history[-10:])
            older_avg = np.mean(self.reward_history[-20:-10])
            improvement = (recent_avg - older_avg) / (abs(older_avg) + 1e-8)
            
            # If improvement < 1%, we're staling
            if improvement < 0.01:
                self.stale_count += 1
                self.volatile_count = 0
                # Increase LR slightly to escape plateau
                policy_factor = 1.5
                value_factor = 1.3
            else:
                self.stale_count = 0
                self.volatile_count = 0
                policy_factor = 1.0
                value_factor = 1.0
        else:
            policy_factor = 1.0
            value_factor = 1.0
        
        # Check for volatility (large swings)
        if len(self.reward_history) > 5:
            recent_rewards = np.array(self.reward_history[-5:])
            volatility = np.std(recent_rewards) / (abs(np.mean(recent_rewards)) + 1e-8)
            
            if volatility > 0.5:  # High volatility
                self.volatile_count += 1
                self.stale_count = 0
                # Decrease LR to stabilize
                policy_factor = 0.8
                value_factor = 0.85
        
        # Warmup phase (first 50 episodes): use reduced LR
        if episode < 50:
            warmup_factor = (episode + 1) / 50.0
            policy_factor *= warmup_factor
            value_factor *= warmup_factor
        
        # Compute new LRs with bounds
        lr_policy = self.initial_lr_policy * policy_factor
        lr_value = self.initial_lr_value * value_factor
        
        lr_policy = np.clip(lr_policy, self.min_lr_policy, self.max_lr_policy)
        lr_value = np.clip(lr_value, self.min_lr_value, self.max_lr_value)
        
        return lr_policy, lr_value
    
    def update_optimizer_lr(self, optimizer_policy, optimizer_value, lr_policy, lr_value):
        """Update optimizer learning rates."""
        for param_group in optimizer_policy.param_groups:
            param_group['lr'] = lr_policy
        for param_group in optimizer_value.param_groups:
            param_group['lr'] = lr_value


class AdaptiveRewardScaler:
    """
    Learns optimal reward scaling factor during training.
    Keeps advantages and returns in reasonable ranges.
    """
    def __init__(self, init_scale=0.01, target_return_std=1.0):
        self.scale = init_scale
        self.target_return_std = target_return_std
        self.return_history = []
    
    def update_scale(self, returns):
        """Adjust scale to keep returns normalized."""
        self.return_history.extend(returns)
        
        if len(self.return_history) > 100:
            # Keep only recent history
            self.return_history = self.return_history[-500:]
            
            returns_array = np.array(self.return_history)
            returns_std = np.std(returns_array)
            
            if returns_std > 0:
                # Scale factor to achieve target std
                scale_adjustment = self.target_return_std / (returns_std + 1e-8)
                # Smooth update to avoid sudden changes
                self.scale = 0.9 * self.scale + 0.1 * scale_adjustment * self.scale
                self.scale = np.clip(self.scale, 0.001, 0.1)
    
    def get_scale(self):
        return self.scale


class AdaptiveEntropyScheduler:
    """
    Dynamically schedules entropy coefficient.
    High entropy early for exploration, reduce for convergence.
    """
    def __init__(self, initial_entropy_coef=0.1, min_coef=0.01, max_coef=0.3, total_episodes=500):
        self.initial_entropy_coef = initial_entropy_coef
        self.min_coef = min_coef
        self.max_coef = max_coef
        self.total_episodes = total_episodes
    
    def get_entropy_coef(self, episode):
        """Linear schedule: high early, low later."""
        progress = episode / self.total_episodes
        # Decay from max to min
        entropy_coef = self.max_coef - (self.max_coef - self.min_coef) * progress
        return entropy_coef


class GradientMonitor:
    """Monitor gradient health during training."""
    def __init__(self):
        self.grad_norms = []
        self.nan_count = 0
    
    def check_gradients(self, model, name=""):
        """Check for NaN or extremely large gradients."""
        total_norm = 0.0
        has_nan = False
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                if torch.isnan(param_norm):
                    has_nan = True
        
        total_norm = total_norm ** 0.5
        self.grad_norms.append(total_norm)
        
        if has_nan:
            self.nan_count += 1
            print(f"[WARNING] NaN gradient detected in {name}")
            return False
        
        if total_norm > 100:
            print(f"[WARNING] Large gradient norm in {name}: {total_norm:.4f}")
        
        return True
    
    def get_avg_gradient_norm(self, window=10):
        """Get recent average gradient norm."""
        if len(self.grad_norms) == 0:
            return 0.0
        return np.mean(self.grad_norms[-window:])


def adaptive_value_epochs(
    episode,
    total_episodes,
    base_epochs=25,
    warmup_episodes=0,
    warmup_multiplier=1.0,
    min_epochs=None,
):
    """
    Reduce value network training epochs during training to prevent overfitting.
    Start with more training, reduce as training progresses.

    warmup_episodes: run extra value updates early so critic can get ahead.
    """
    progress = episode / total_episodes

    # Optional warmup boost for early episodes.
    warmup_factor = warmup_multiplier if episode < warmup_episodes else 1.0

    # Linear schedule: base_epochs -> base_epochs//4, with optional warmup boost.
    default_min = max(1, base_epochs // 4)
    effective_min = default_min if min_epochs is None else max(1, int(min_epochs))
    epochs = max(effective_min, int(base_epochs * warmup_factor * (1 - 0.75 * progress)))
    return epochs


def compute_return_statistics(returns):
    """Compute useful statistics for training monitoring."""
    returns_array = np.array(returns)
    return {
        'mean': np.mean(returns_array),
        'std': np.std(returns_array),
        'min': np.min(returns_array),
        'max': np.max(returns_array),
        'median': np.median(returns_array)
    }
