import numpy as np


def get_adaptive_m(
    rewards_history,
    ep,
    current_m=None,
    max_m=20,
    min_m=1,
    window=20,
    std_threshold=5.0,
    growth_rate=10,
    gamma = 0.7
):
    if len(rewards_history) < window:
        return min_m

    recent_rewards = rewards_history[-window:]
    reward_std = np.std(recent_rewards)

    if reward_std > std_threshold:
        return max_m

    m_schedule = min_m + ep // growth_rate
    print(f"Episode {ep}: Reward std = {reward_std:.2f}, m_schedule = {m_schedule}")

    if current_m is not None:
        m_smooth = int(gamma * current_m + (1 - gamma) * m_schedule)
    else:
        m_smooth = m_schedule

    return int(np.clip(m_smooth, min_m, max_m))
