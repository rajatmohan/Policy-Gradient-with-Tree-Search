# Quick Reference: Training Stability Improvements

## What Changed

### New File: `training_utils.py`
Complete training stability utilities with 4 main classes:
- `DynamicLearningRateScheduler`: Adjusts policy/value LRs based on training progress
- `AdaptiveRewardScaler`: Keeps rewards in stable range
- `AdaptiveEntropyScheduler`: Entropy decays from 0.3 → 0.01
- `GradientMonitor`: Detects NaN gradients

### Updated: `continuous_pgts.py`
✅ `run_pgts()` - Full batch training with all adaptive features  
✅ `run_pgts_online()` - Online training with adaptive components  
✅ `train_value_network()` - NaN detection added

### Updated: `continuous_pg.py`
✅ `run_pg()` - Added entropy scheduling & adaptive reward scaling

## Before vs After Comparison

| Aspect | Before | After |
|--------|--------|-------|
| Learning Rates | Fixed 5e-5 (policy), 1e-3 (value) | Dynamic, ranges: 1e-6 to 1e-3 |
| Reward Scaling | Static 0.01 | Auto-adjusts targeting std=1.0 |
| Entropy Coef | Fixed 0.1 | Decays: 0.3 → 0.01 over episodes |
| Value Training | 25 epochs always | 6-25, auto-reduced with progress |
| Gradient Safety | Basic clipping | NaN detection + safe skipping |
| Warmup | None | Soft start (first 50 episodes) |

## Key Parameters to Control

```python
# Learning rate scheduler (in training_utils.py)
initial_lr_policy=5e-5      # Base policy learning rate
initial_lr_value=1e-3       # Base value learning rate  
min_lr_policy=1e-6          # Don't go below this
max_lr_policy=1e-3          # Don't go above this

# Reward scaler (in training_utils.py)
init_scale=0.01             # Starting reward scale factor
target_return_std=1.0       # Target std for returns

# Entropy scheduler (in training_utils.py)
initial_entropy_coef=0.1    # Used to compute bounds
max_coef=0.3                # Maximum entropy (early training)
min_coef=0.01               # Minimum entropy (convergence)

# Value network (in run_pgts calls)
v_epochs=25                 # Base epochs; auto-reduced to 6
```

## What To Monitor

Print output will show:
```
[PGTS-Online] Ep    0 | Reward: -28.123456 | m: 4 | LR_P: 5.00e-05 | LR_V: 1.00e-03 | Entropy: 0.300 | Scale: 0.0100
```

- **Reward**: Should trend upward (typical for Lunar: -50 → -10 to 0)
- **m**: Tree search depth (stays constant unless adaptive=True)
- **LR_P/LR_V**: Learning rates - should be stable
- **Entropy**: Decays gradually
- **Scale**: Adapts to keep returns normalized

## Debugging Checklist

❌ Rewards not improving after 100 episodes?
- Check entropy is decaying (not stuck at 0.3)
- Try increasing m or enable adaptive=True
- Verify env.rollout() is working

❌ Training diverges (NaN)?
- Reduce initial_lr_policy and initial_lr_value
- Check [WARNING] messages in console
- Verify input data (state ranges)

❌ Training too slow?
- Initial LRs might be too conservative
- Reduce v_epochs base value
- Check if stuck in warmup (first 50 episodes)

❌ Rewards oscillating wildly?
- LR might be too high - reduce initial values
- Increase tau (lagging policy update rate)
- Use use_lagging=True for PGTS

## Quick Test Run

To verify everything works:
```bash
cd /Users/rajatmohan/repos/rl_project
source venv/bin/activate
python continuous_experiments.py
```

Expected: Should see dynamically changing LR, entropy, and scale values.

## Files Modified Summary

1. ✅ Created: `training_utils.py` - Central stability utilities (145 lines)
2. ✅ Modified: `continuous_pgts.py` - Added adaptive components to 3 functions
3. ✅ Modified: `continuous_pg.py` - Added entropy scheduling & adaptive scaling
4. ✅ Created: `TRAINING_STABILITY_GUIDE.md` - Comprehensive documentation

All changes are backwards compatible - old code structure unchanged.
