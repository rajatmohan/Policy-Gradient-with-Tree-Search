# Training Stabilization Guide for PGTS vs PG Comparison

## Overview
Your RL project was experiencing training instability due to several factors. This guide explains the improvements implemented and how to use them effectively.

## Key Issues Identified & Fixes

### 1. **Fixed Learning Rates Problem**
   - **Issue**: Policy LR (5e-5) and Value LR (1e-3) were fixed. Static LRs can't adapt to different phases of training.
   - **Fix**: Implemented `DynamicLearningRateScheduler` that:
     - Increases LR when training stalls (plateaus)
     - Decreases LR when rewards are volatile (unstable)
     - Uses warmup phase (first 50 episodes) for gentle start
     - Bounds LRs to safe ranges

### 2. **Reward Scaling Issue**
   - **Issue**: Static 0.01 scaling might cause gradient vanishing or unexpected behavior
   - **Fix**: Implemented `AdaptiveRewardScaler` that:
     - Dynamically adjusts scaling to keep returns at target std (~1.0)
     - Scales smoothly to avoid sudden changes
     - Keeps rewards in numerically stable range

### 3. **Entropy Coefficient**
   - **Issue**: Fixed entropy (0.1) doesn't change during training
   - **Fix**: Implemented `AdaptiveEntropyScheduler` that:
     - Starts with high entropy for exploration (0.3)
     - Linearly decays to low entropy for convergence (0.01)
     - Balances exploration-exploitation automatically

### 4. **Value Network Overfitting**
   - **Issue**: Training value network for 25 epochs per episode causes overfitting
   - **Fix**: Implemented `adaptive_value_epochs()` that:
     - Starts with base epochs (25) in early training
     - Gradually reduces to 6 epochs to prevent overfitting
     - Adapts based on training progress

### 5. **Gradient Safety**
   - **Fix**: Added NaN detection and gradient monitoring:
     - Skips updates if NaN gradients detected
     - Better error handling for numerical issues

## Implementation Details

### New Classes in `training_utils.py`

1. **DynamicLearningRateScheduler**
   ```python
   scheduler = DynamicLearningRateScheduler(
       initial_lr_policy=5e-5,
       initial_lr_value=1e-3,
       min_lr_policy=1e-6,
       max_lr_policy=1e-3
   )
   
   # In training loop:
   lr_policy, lr_value = scheduler.get_learning_rates(episode, avg_reward)
   scheduler.update_optimizer_lr(optimizer_p, optimizer_v, lr_policy, lr_value)
   ```

2. **AdaptiveRewardScaler**
   ```python
   scaler = AdaptiveRewardScaler(init_scale=0.01)
   
   # In training loop:
   scaled_rewards = [r * scaler.get_scale() for r in rewards]
   scaler.update_scale(scaled_rewards)
   ```

3. **AdaptiveEntropyScheduler**
   ```python
   entropy_scheduler = AdaptiveEntropyScheduler(
       initial_entropy_coef=0.1,
       total_episodes=500
   )
   
   # In training loop:
   entropy_coef = entropy_scheduler.get_entropy_coef(episode)
   ```

## Updated Files

### 1. `continuous_experiments.py`
- No changes needed; will automatically use new adaptive components

### 2. `continuous_pgts.py`
- Updated `run_pgts()`: Adds adaptive LR, entropy, reward scaling, and value epochs
- Updated `run_pgts_online()`: Same improvements for online training
- Enhanced `train_value_network()`: NaN detection

### 3. `continuous_pg.py`
- Updated `run_pg()`: Adds entropy bonus and adaptive components

## Usage & Hyperparameters to Tune

### Start with these settings:
```python
# In continuous_experiments.py, modify run_single function:

# For PGTS training:
rewards = run_pgts_online(
    env,
    policy,
    value_net,
    optimizer_p,
    optimizer_v,
    episodes=EPISODES,
    adaptive=(m == -1),
    max_m=m if m != -1 else 20,
    m=m,
    entropy_coef=0.1,  # Auto-decayed by scheduler
    use_lagging=True,  # Recommended for stability
    clip_epsilon=0.2,
    tau=0.01,
    v_epochs=25,  # Will be auto-reduced by scheduler
    K=3  # Tree search branching factor
)
```

### Key Tuning Parameters

1. **Tree Search Budget (m and K)**
   - Increase `m` for better lookahead (more stable but slower)
   - Increase `K` for more action samples (better exploration)
   - Start with `m=2-4`, `K=3`

2. **Learning Rates**
   - `initial_lr_policy=5e-5`: Base policy LR (scheduler multiplies from this)
   - `initial_lr_value=1e-3`: Base value LR (scheduler multiplies from this)
   - If training diverges: Reduce initial LRs
   - If training stalls: Increase initial LRs

3. **Reward Scaling**
   - Auto-adjusted, but monitor the printouts
   - If rewards stay ~0.0, check your environment

4. **Entropy**
   - Will decay from 0.3 → 0.01 over training
   - Adjust `max_coef` in `AdaptiveEntropyScheduler` if needed

5. **Lagging Policy**
   - `use_lagging=True`: Keeps old policy copy, updates slowly
   - `tau=0.01`: EMA update rate (lower = slower updates, more stable)
   - Helpful for PGTS to prevent tree search target chasing

## What to Monitor During Training

Open the console output while training. Look for:

```
[PGTS-Online] Ep    0 | Reward: X.XXXXXX | m: M | LR_P: Y.YYe-0Z | LR_V: A.AAe-00 | Entropy: 0.XXX | Scale: 0.0100
```

**Good signs:**
- ✅ Reward trend increasing (even if noisy)
- ✅ LR stable (not jumping drastically)
- ✅ Scale stays ~0.01
- ✅ Entropy decays slowly

**Red flags:**
- ❌ Rewards stuck at same value (plateau)
- ❌ Rewards oscillating wildly
- ❌ NaN warnings in console
- ❌ Large jumps in LR every episode

## Advanced Tuning

### If training is too slow:
1. Increase initial learning rates slightly
2. Reduce `v_epochs` base value
3. Reduce `m` (less tree search)
4. Increase batch size if using batch PGTS

### If training is unstable (rewards jumping):
1. Decrease initial learning rates
2. Increase `tau` for lagging policy (slower updates)
3. Reduce `K` (fewer action samples per node)
4. Increase smoothing in `AdaptiveRewardScaler`

### If stuck in local minima:
1. Increase entropy coefficient base
2. Increase `m` for better lookahead
3. Use `adaptive=True` to auto-tune `m`
4. Reduce `clip_epsilon` for less conservative updates

## Comparison: Before vs After

### Before (Old Code):
- Fixed LR everywhere
- No entropy scheduling
- Overtrains value network
- Uses static reward scaling

### After (New Code):
- Dynamic LR based on training progress
- Entropy decays for convergence
- Adaptive value training epochs
- Auto-adjusting reward scaling

## Recommended Experiment Settings

For Lunar MDP:
```python
SEEDS = [60, 61, 62]  # Multiple seeds for stability
EPISODES = 500

# Try these configurations:
# 1. PGTS with m=4, K=3, lagging=True
# 2. PGTS with m=10, K=3, lagging=True (more search)
# 3. PG baseline for comparison
# 4. PGTS with adaptive m
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Training converges to local minima | Increase m, use adaptive=True |
| Training crashes (NaN) | Decrease learning rates |
| Training stalls (no improvement) | Check if in warmup phase; increase entropy_coef |
| Rewards very small/large | Scale factor wrong; check reward function |
| PGTS much slower than baseline | Reduce K or m, use strided search |

## Next Steps

1. Run experiment with default settings
2. Monitor the metrics
3. If unstable: decrease LRs by 50%
4. If too slow: increase LRs by 50%
5. Record which configuration works best
6. Report both average reward AND variance
