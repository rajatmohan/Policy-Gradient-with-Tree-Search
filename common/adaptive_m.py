import numpy as np

def get_adaptive_m(
    rewards_history,
    ep,
    current_m,
    max_m=10,
    min_m=2,
    window=30,
    std_threshold=15.0,
    improvement_threshold=5.0,
    gamma=0.85
):
    """
    Improved Adaptive m logic to break out of reward plateaus.
    """
    # 1. Warm-up phase: keep m stable until we have enough data
    if ep % window != 0 or len(rewards_history) < window * 2:
        return current_m

    recent_window = rewards_history[-window:]
    prev_window = rewards_history[-2*window:-window]
    
    reward_std = np.std(recent_window)
    avg_recent = np.mean(recent_window)
    avg_prev = np.mean(prev_window)
    
    improvement = avg_recent - avg_prev
    target_m = current_m

    # --- LOGIC ---

    # A. HIGH INSTABILITY (Standard Deviation is high)
    # The agent is confused; increase lookahead to find a consistent path.
    if reward_std > std_threshold:
        target_m += 1
    
    # B. PLATEAU/STAGNATION (Improvement is flat or negative)
    # The current m=2 isn't enough to find the next 'best' thing.
    elif improvement < improvement_threshold:
        target_m += 1
        
    # C. STABILITY & MASTERY (Stable and getting better)
    # Reduce m to save compute and let the policy take over.
    elif reward_std < (std_threshold * 0.5) and improvement > improvement_threshold:
        target_m -= 1

    # Apply Smoothing (EMA) to prevent m from flickering every episode
    m_smooth = gamma * current_m + (1 - gamma) * target_m
    
    # Final cleanup: round, clip, and return as int
    # new_m = int(np.clip(round(m_smooth), min_m, max_m))
    new_m = int(np.clip(target_m, min_m, max_m))
    
    if new_m != current_m:
        print (f"Episode {ep}: Reward Std={reward_std:.2f}, Improvement={improvement:.2f}, Current m={current_m}, Target m={target_m}, Smoothed m={m_smooth:.2f}, New m={new_m}")
    return new_m