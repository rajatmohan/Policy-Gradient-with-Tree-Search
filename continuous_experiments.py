import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
from continuous.continous_policy import Policy
from continuous.continuous_value import Value
from continuous.continuous_pg import run_pg
from continuous.continuous_pgts import run_pgts, run_pgts_online
from continuous.continuous_pgts import run_pgts_td
from continuous.continuous_pgts import run_pg_mstep
from gymnasium.wrappers import RecordVideo
from continuous.env_two_peak import TwoPeakMDP
from continuous.env_three_peak import ThreePeakMDP
from continuous.lunar_mdp import LunarMDP

SEEDS = [60]
EPISODES = 1000

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

def evaluate_policy(env, policy, tag, episodes=5, deterministic=False):
    policy_was_training = policy.training
    policy.eval()

    episode_returns = []

    try:
        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                state_t = torch.FloatTensor(state).unsqueeze(0)

                if deterministic:
                    with torch.no_grad():
                        mean, _ = policy(state_t)
                        action = mean.squeeze(0).cpu().tolist()
                else:
                    action, _ = policy.sample_action(state)

                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated

            episode_returns.append(total_reward)
    finally:
        if policy_was_training:
            policy.train()

    summary = {
        "tag": tag,
        "episodes": episodes,
        "deterministic": deterministic,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "min_return": float(np.min(episode_returns)),
        "max_return": float(np.max(episode_returns)),
        "returns": [float(x) for x in episode_returns],
    }

    summary_path = os.path.join(RESULT_DIR, f"{tag}_eval.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        f"[EVAL] {tag} | mean: {summary['mean_return']:.3f} | std: {summary['std_return']:.3f} | "
        f"min: {summary['min_return']:.3f} | max: {summary['max_return']:.3f}"
    )

    return summary

def get_envs():
    lunar = LunarMDP()
    return [
        # TwoPeakMDP(),
        # ThreePeakMDP(),
        lunar
    ]

def run_single(method, seed, env, m = 10):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(env, "set_seed"):
        env.set_seed(seed)
    else:
        env.seed = seed

    state_dim = 8 if env.name == "LUNAR_MDP" else 1
    action_dim = 2 if env.name == "LUNAR_MDP" else 1

    policy = Policy(state_dim, action_dim)
    value_net = Value(state_dim)
    
    optimizer_p = torch.optim.Adam(policy.parameters(), lr=5e-5)        
    optimizer_v = torch.optim.Adam(value_net.parameters(), lr=1e-3)
    env.reset()

    if method == "PG":
        rewards = run_pg(env, policy, optimizer_p, episodes=EPISODES)

    elif method == "PGTS":        
        rewards = run_pgts_td(
            env,
            policy,
            value_net,
            optimizer_p,
            optimizer_v,
            episodes=EPISODES,
            adaptive=(m == -1),
            max_m = m if m != -1 else 20,
            m=m,
            v_epochs=6,
            critic_warmup_episodes=400,
            critic_warmup_multiplier=2.5,
            early_stop=False,
            early_stop_min_episodes=1200,
            early_stop_check_every=50,
            early_stop_window=100,
            early_stop_delta=10.0,
            early_stop_patience=5,
            eval_interval=100,
            eval_episodes=5,
            eval_deterministic=True,
        )
    
    elif method == "PGTS_WITH_LAGGING":
        rewards = run_pgts_td(
            env,
            policy,
            value_net,
            optimizer_p,
            optimizer_v,
            m=m,
            episodes=EPISODES,
            adaptive=(m == -1),
            max_m = m if m != -1 else 20,
            use_lagging=True,
            clip_epsilon=0.2,
            tau = 0.01,
            v_epochs=6,
            critic_warmup_episodes=400,
            critic_warmup_multiplier=2.5,
            early_stop=False,
            early_stop_min_episodes=1200,
            early_stop_check_every=50,
            early_stop_window=100,
            early_stop_delta=10.0,
            early_stop_patience=5,
            eval_interval=100,
            eval_episodes=5,
            eval_deterministic=True,
        )

    final_state = env.state
    return rewards, final_state, policy

def run_experiment(method, env, m = 10):
    all_rewards = []
    final_states = []
    best_policy = None
    best_score = -float("inf")

    for seed in SEEDS:
        print(f"\nRunning {method} | {env.name} | Seed {seed}")

        rewards, final_state, policy = run_single(method, seed, env, m=m)

        all_rewards.append(rewards)
        final_states.append(final_state)

        score = np.mean(rewards[-10:])

        if score > best_score:
            best_score = score
            best_policy = policy

    return np.array(all_rewards), np.array(final_states), best_policy

def plot_all(env, pg_rewards, pgts_dict):
    plt.figure(figsize=(10, 6))

    pg_mean = pg_rewards.mean(axis=0)
    plt.plot(pg_mean, label="PG", linewidth=3)

    for m, rewards in pgts_dict.items():
        mean = rewards.mean(axis=0)
        plt.plot(mean, label=f"PGTS m={m}" if isinstance(m, int) else f"PGTS {m}", linewidth=3, linestyle="--")

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title(f"{env.name}: PG vs PGTS (multi-m)")
    plt.legend()

    plt.savefig(f"{RESULT_DIR}/{env.name}_comparison.png")
    # plt.show()

def record_agent(env, model_path, tag):
    video_dir = f"{RESULT_DIR}/videos_{tag}"
    os.makedirs(video_dir, exist_ok=True)

    state_dim = env.reset()[0].shape[0]

    if hasattr(env, "env"):
        base_env = env.env
        action_dim = base_env.action_space.shape[0]
    else:
        env.render_mode = "rgb_array"
        base_env = env
        action_dim = env.action_space.shape[0]

    policy = Policy(state_dim, action_dim)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    video_env = RecordVideo(base_env, video_dir, episode_trigger=lambda e: True)

    try:
        state, _ = video_env.reset()
        done = False
        step = 0

        while not done and step < 1000:
            action, _ = policy.sample_action(state)
            state, _, done, _, _ = video_env.step(action)
            step += 1

    finally:
        video_env.close()

if __name__ == "__main__":
    envs = get_envs()
    # M_VALUES = [1, 5, 10]
    M_VALUES = [2, -1]
    for env in envs:
        env.reset()
        pg_eval = None
        pg_rewards = None

        # env.init_state = init_state
        # pg_rewards, pg_states, pg_policy = run_experiment("PG", env)
        # torch.save(pg_policy.state_dict(), f"{RESULT_DIR}/{env.name}_pg.pt")
        # record_agent(env, f"{RESULT_DIR}/{env.name}_pg.pt", f"{env.name}_PG")
        # pg_eval = evaluate_policy(env, pg_policy, f"{env.name}_PG")

        pgts_results = {}
        pgts_policies = {}
        eval_summary = {}
        for m in M_VALUES:
            # PGTS without lagging
            print(f"\nRunning PGTS m={m} | {env.name}")
            rewards, states, policy = run_experiment("PGTS", env, m=m)
            pgts_results[m] = rewards
            pgts_policies[m] = policy
            model_path = f"{RESULT_DIR}/{env.name}_pgts_m{m}.pt"
            torch.save(policy.state_dict(), model_path)
            record_agent(env, model_path, f"{env.name}_PGTS_m{m}")
            eval_summary[f"PGTS_m{m}"] = evaluate_policy(env, policy, f"{env.name}_PGTS_m{m}")

            # PGTS with lagging
            print(f"\nRunning PGTS LAG m={m} | {env.name}")
            rewards, states, policy = run_experiment("PGTS_WITH_LAGGING", env, m=m)
            pgts_results[f"LAG_m={m}"] = rewards
            pgts_policies[f"LAG_m={m}"] = policy
            model_path = f"{RESULT_DIR}/{env.name}_pgts_lag_m{m}.pt"
            torch.save(policy.state_dict(), model_path)
            record_agent(env, model_path, f"{env.name}_PGTS_lag_m{m}")
            eval_summary[f"PGTS_LAG_m{m}"] = evaluate_policy(env, policy, f"{env.name}_PGTS_lag_m{m}")

        with open(os.path.join(RESULT_DIR, f"{env.name}_evaluation_summary.json"), "w", encoding="utf-8") as handle:
            json.dump({"PG": pg_eval, "PGTS": eval_summary}, handle, indent=2)

        if pg_rewards is not None:
            plot_all(env, pg_rewards, pgts_results)
        break