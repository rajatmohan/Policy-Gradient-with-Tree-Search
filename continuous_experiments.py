import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from continuous.continous_policy import Policy
from continuous.continuous_value import Value
from continuous.continuous_pg import run_pg
from continuous.continuous_pgts import run_pgts_batch
from gymnasium.wrappers import RecordVideo
from continuous.lunar_mdp import LunarMDP
import pickle
import time

SEEDS = [60]
EPISODES = 1000

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

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

        experiment_times = {}

        env.init_state = init_state
        print(f"\n[START] PG Baseline | {env.name}")
        
        start_time = time.time()
        pg_rewards, pg_states, pg_policy = run_experiment("PG", env)
        pg_duration = time.time() - start_time
        experiment_times["PG"] = pg_duration
        
        print(f"[DONE] PG Baseline finished in {pg_duration:.2f} seconds.")
        torch.save(pg_policy.state_dict(), f"{RESULT_DIR}/{env.name}_pg.pt")
        record_agent(env, f"{RESULT_DIR}/{env.name}_pg.pt", f"{env.name}_PG")

        pgts_results = {}
        pgts_policies = {}
        for m in M_VALUES:
            # PGTS without lagging
            print(f"\n[START] PGTS m={m} | {env.name}")
            env.init_state = init_state
            
            start_time = time.time()
            rewards, states, policy = run_experiment("PGTS", env, m=m)
            pgts_duration = time.time() - start_time
            experiment_times[f"PGTS_m{m}"] = pgts_duration
            
            print(f"[DONE] PGTS m={m} finished in {pgts_duration:.2f} seconds.")
            pgts_results[f"m={m}"] = rewards
            pgts_policies[f"m={m}"] = policy
            model_path = f"{RESULT_DIR}/{env.name}_pgts_m{m}.pt"
            torch.save(policy.state_dict(), model_path)
            record_agent(env, model_path, f"{env.name}_PGTS_m{m}")

            # PGTS with lagging
            print(f"\n[START] PGTS LAG m={m} | {env.name}")
            env.init_state = init_state
            
            start_time = time.time()
            rewards, states, policy = run_experiment("PGTS_WITH_LAGGING", env, m=m)
            lag_duration = time.time() - start_time
            experiment_times[f"PGTS_LAG_m{m}"] = lag_duration
            
            print(f"[DONE] PGTS LAG m={m} finished in {lag_duration:.2f} seconds.")
            pgts_results[f"LAG_m={m}"] = rewards
            pgts_policies[f"LAG_m={m}"] = policy
            model_path = f"{RESULT_DIR}/{env.name}_pgts_lag_m{m}.pt"
            torch.save(policy.state_dict(), model_path)
            record_agent(env, model_path, f"{env.name}_PGTS_lag_m{m}")
        
        experiment_data = {
            "PG": pg_rewards,
            "PGTS": pgts_results,
            "TIMES": experiment_times
        }
        
        data_path = f"{RESULT_DIR}/{env.name}_rewards_data.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(experiment_data, f)

        # run_experiment will crash if you pass multiple seeds. need to handle it
        plot_all(env, pg_rewards, pgts_results)