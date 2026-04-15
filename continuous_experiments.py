import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
from continuous.continous_policy import Policy
from continuous.continuous_value import Value
from continuous.continuous_pg import run_pg
from continuous.continuous_pgts import run_pgts_batch
from gymnasium.wrappers import RecordVideo
from continuous.env_two_peak import TwoPeakMDP
from continuous.env_three_peak import ThreePeakMDP
from continuous.lunar_mdp import LunarMDP

SEEDS = [60]
EPISODES = 1000

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


def save_experiment_results(env_name, pg_rewards=None, pgts_results=None, seeds=None, m_values=None):
    payload = {
        "env_name": env_name,
        "seeds": list(seeds) if seeds is not None else None,
        "m_values": list(m_values) if m_values is not None else None,
        "pg_rewards": pg_rewards.tolist() if pg_rewards is not None else None,
        "pgts_results": {
            key: value.tolist() for key, value in (pgts_results or {}).items()
        },
    }

    results_path = f"{RESULT_DIR}/{env_name}_reward_curves.json"
    with open(results_path, "w", encoding="utf-8") as results_file:
        json.dump(payload, results_file, indent=2)

    return results_path


def load_experiment_results(results_path):
    with open(results_path, "r", encoding="utf-8") as results_file:
        return json.load(results_file)


def plot_comparison(env_name, pg_rewards, series_dict, title_suffix, output_path, label_prefix):
    plt.figure(figsize=(10, 6))

    pg_mean = np.array(pg_rewards).mean(axis=0)
    plt.plot(pg_mean, label="PG", linewidth=3)

    for label, rewards in series_dict.items():
        mean = np.array(rewards).mean(axis=0)
        plt.plot(mean, label=f"{label_prefix} {label}", linewidth=3, linestyle="--")

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title(f"{env_name}: PG vs {title_suffix}")
    plt.legend()

    plt.savefig(output_path)
    plt.close()


def plot_saved_results(results_path):
    payload = load_experiment_results(results_path)
    env_name = payload["env_name"]
    pg_rewards = payload["pg_rewards"]
    pgts_results = payload["pgts_results"] or {}

    standard_pgts = {
        key.replace("m=", "m=") : value
        for key, value in pgts_results.items()
        if key.startswith("m=")
    }
    lagging_pgts = {
        key.replace("LAG_m=", "m=") : value
        for key, value in pgts_results.items()
        if key.startswith("LAG_m=")
    }

    plot_comparison(
        env_name,
        pg_rewards,
        standard_pgts,
        "PGTS (multi-m)",
        f"{RESULT_DIR}/{env_name}_pg_vs_pgts.png",
        "PGTS",
    )

    plot_comparison(
        env_name,
        pg_rewards,
        lagging_pgts,
        "PGTS with Lagging (multi-m)",
        f"{RESULT_DIR}/{env_name}_pg_vs_pgts_lagging.png",
        "PGTS Lag",
    )

def get_envs():
    lunar = LunarMDP()
    return [
        TwoPeakMDP(),
        # ThreePeakMDP(),
        # lunar
    ]

def run_single(method, seed, env, m = 10):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed = seed

    state_dim = 8 if env.name == "LUNAR_MDP" else 1
    action_dim = 2 if env.name == "LUNAR_MDP" else 1

    policy = Policy(state_dim, action_dim) # actor
    value_net = Value(state_dim) # critic
    
    optimizer_p = torch.optim.Adam(policy.parameters(), lr=3e-4)        
    optimizer_v = torch.optim.Adam(value_net.parameters(), lr=1e-3)

    env.reset()

    if method == "PG":
        rewards = run_pg(env, policy, optimizer_p, episodes=EPISODES)

    elif method == "PGTS":        
        rewards = run_pgts_batch(
            env,
            policy,
            value_net,
            optimizer_p,
            optimizer_v,
            m=m
        )
    
    elif method == "PGTS_WITH_LAGGING":
        rewards = run_pgts_batch(
            env,
            policy,
            value_net,
            optimizer_p,
            optimizer_v,
            m=m,
            use_lagging=True,
            clip_epsilon=0.2,
            tau = 0.01
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
            state_t = torch.FloatTensor(state).unsqueeze(0)
            # get pure mean from the forward pass
            with torch.no_grad():
                mean_action, _ = policy(state_t)  # ignore std
                
            action_np = mean_action.squeeze(0).numpy()
            state, _, done, _, _ = video_env.step(action_np)
            step += 1

    finally:
        video_env.close()

if __name__ == "__main__":
    envs = get_envs()
    # M_VALUES = [1, 5, 10]
    M_VALUES = [1, 2, 3]
    for env in envs:
        env.reset()
        init_state = env.state

        env.init_state = init_state
        pg_rewards, pg_states, pg_policy = run_experiment("PG", env)
        results_path = save_experiment_results(env.name, pg_rewards=pg_rewards, seeds=SEEDS, m_values=M_VALUES)
        torch.save(pg_policy.state_dict(), f"{RESULT_DIR}/{env.name}_pg.pt")
        record_agent(env, f"{RESULT_DIR}/{env.name}_pg.pt", f"{env.name}_PG")

        pgts_results = {}
        pgts_policies = {}
        for m in M_VALUES:
            m = m - 1
            # PGTS without lagging
            print(f"\nRunning PGTS m={m} | {env.name}")
            env.init_state = init_state
            rewards, states, policy = run_experiment("PGTS", env, m=m)
            pgts_results[f"m={m}"] = rewards
            pgts_policies[f"m={m}"] = policy
            results_path = save_experiment_results(env.name, pg_rewards=pg_rewards, pgts_results=pgts_results, seeds=SEEDS, m_values=M_VALUES)
            model_path = f"{RESULT_DIR}/{env.name}_pgts_m{m}.pt"
            torch.save(policy.state_dict(), model_path)
            record_agent(env, model_path, f"{env.name}_PGTS_m{m}")

            # PGTS with lagging
            # print(f"\nRunning PGTS LAG m={m} | {env.name}")
            # env.init_state = init_state
            # rewards, states, policy = run_experiment("PGTS_WITH_LAGGING", env, m=m)
            # pgts_results[f"LAG_m={m}"] = rewards
            # pgts_policies[f"LAG_m={m}"] = policy
            # results_path = save_experiment_results(env.name, pg_rewards=pg_rewards, pgts_results=pgts_results, seeds=SEEDS, m_values=M_VALUES)
            # model_path = f"{RESULT_DIR}/{env.name}_pgts_lag_m{m}.pt"
            # torch.save(policy.state_dict(), model_path)
            # record_agent(env, model_path, f"{env.name}_PGTS_lag_m{m}")

        # run_experiment will crash if you pass multiple seeds. need to handle it
        results_path = save_experiment_results(env.name, pg_rewards=pg_rewards, pgts_results=pgts_results, seeds=SEEDS, m_values=M_VALUES)
        plot_saved_results(results_path)