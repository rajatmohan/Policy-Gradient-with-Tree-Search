import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from continuous.continous_policy import Policy
from continuous.continuous_value import Value
from continuous.continuous_pg import run_pg
from continuous.continuous_pgts import run_pgts
from gymnasium.wrappers import RecordVideo
from continuous.env_two_peak import TwoPeakMDP
from continuous.env_three_peak import ThreePeakMDP

SEEDS = [56]
EPISODES = 800

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

def get_envs():
    return [
        TwoPeakMDP(),
        ThreePeakMDP(),
    ]

def run_single(method, seed, env, m = 10):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env.reset()
    init_state = env.state

    policy = Policy(1, 1)
    optimizer_p = torch.optim.Adam(policy.parameters(), lr=3e-3)

    env.init_state = init_state

    if method == "PG":
        rewards = run_pg(env, policy, optimizer_p, episodes=EPISODES)

    elif method == "PGTS":
        value_net = Value(1)
        optimizer_v = torch.optim.Adam(value_net.parameters(), lr=3e-3)

        rewards = run_pgts(
            env,
            policy,
            value_net,
            optimizer_p,
            optimizer_v,
            episodes=EPISODES,
            m=m
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
        plt.plot(mean, label=f"PGTS m={m}")

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title(f"{env.name}: PG vs PGTS (multi-m)")
    plt.legend()

    plt.savefig(f"{RESULT_DIR}/{env.name}_comparison.png")
    # plt.show()

def record_agent(env, model_path, tag):
    video_dir = f"{RESULT_DIR}/videos_{tag}"
    os.makedirs(video_dir, exist_ok=True)

    policy = Policy(1, 1)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    env.render_mode = "rgb_array"
    env = RecordVideo(env, video_dir, episode_trigger=lambda e: True)

    state, _ = env.reset()
    done = False
    step = 0

    while not done and step < 200:
        action, _ = policy.sample_action(state)
        state, _, done, _, _ = env.step(action)
        step += 1

    env.close()

if __name__ == "__main__":
    envs = get_envs()
    M_VALUES = [1, 5, 10]
    for env in envs:
        env.reset()
        init_state = env.state

        env.init_state = init_state
        pg_rewards, pg_states, pg_policy = run_experiment("PG", env)
        torch.save(pg_policy.state_dict(), f"{RESULT_DIR}/{env.name}_pg.pt")
        record_agent(env, f"{RESULT_DIR}/{env.name}_pg.pt", f"{env.name}_PG")

        pgts_results = {}
        pgts_policies = {}
        for m in M_VALUES:
            print(f"\nRunning PGTS m={m} | {env.name}")

            env.init_state = init_state

            rewards, states, policy = run_experiment("PGTS", env, m=m)

            pgts_results[m] = rewards
            pgts_policies[m] = policy

            model_path = f"{RESULT_DIR}/{env.name}_pgts_m{m}.pt"
            torch.save(policy.state_dict(), model_path)

            record_agent(env, model_path, f"{env.name}_PGTS_m{m}")

        plot_all(env, pg_rewards, pgts_results)