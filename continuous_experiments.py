import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import warnings
import json
import pickle
from time import perf_counter

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from continuous.continous_policy import Policy
from continuous.continuous_value import Value
from continuous.continuous_pg import run_pg
from continuous.continuous_pgts import run_pgts_batch
from gymnasium.wrappers import RecordVideo
from continuous.env_two_peak import TwoPeakMDP
from continuous.env_three_peak import ThreePeakMDP
from continuous.lunar_mdp import LunarMDP
from utils import get_torch_device

SEEDS = [60]
EPISODES = 1500

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = get_torch_device(prefer_gpu=True)


def move_model_with_fallback(model, preferred_device):
    """Move model to preferred device; fall back to CPU on CUDA OOM."""
    try:
        model.to(preferred_device)
        return model, preferred_device
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda error" in msg:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpu_device = torch.device("cpu")
            model.to(cpu_device)
            print(f"[WARN] CUDA OOM, using CPU for this run: {exc}")
            return model, cpu_device
        raise

def get_envs():
    lunar = LunarMDP()
    return [
        # lunar,
        # TwoPeakMDP(),
        ThreePeakMDP(),
    ]

def run_single(method, seed, env, m = 10):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed = seed

    state_dim = 8 if env.name == "LUNAR_MDP" else 1
    action_dim = 2 if env.name == "LUNAR_MDP" else 1

    policy = Policy(state_dim, action_dim) # actor
    value_net = Value(state_dim) # critic
    policy, runtime_device = move_model_with_fallback(policy, DEVICE)
    value_net, _ = move_model_with_fallback(value_net, runtime_device)
    
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
    experiment_start = perf_counter()
    seed_times = {}

    for seed in SEEDS:
        print(f"\nRunning {method} | {env.name} | Seed {seed}")
        seed_start = perf_counter()

        rewards, final_state, policy = run_single(method, seed, env, m=m)
        seed_elapsed = perf_counter() - seed_start
        seed_times[str(seed)] = seed_elapsed
        print(f"[TIME] {method} | {env.name} | Seed {seed} finished in {seed_elapsed:.2f}s")

        all_rewards.append(rewards)
        final_states.append(final_state)

        score = np.mean(rewards[-10:])

        if score > best_score:
            best_score = score
            best_policy = policy

    experiment_elapsed = perf_counter() - experiment_start
    print(f"[TIME] {method} | {env.name} | m={m} total experiment time: {experiment_elapsed:.2f}s")

    timing = {
        "total_seconds": experiment_elapsed,
        "seed_seconds": seed_times,
    }

    return np.array(all_rewards), np.array(final_states), best_policy, timing

def plot_all(env, pg_result, pgts_dict):
    plt.figure(figsize=(10, 6))

    pg_rewards = pg_result["rewards"]
    pg_mean = pg_rewards.mean(axis=0)
    plt.plot(pg_mean, label="PG", linewidth=3)

    for m, result in pgts_dict.items():
        rewards = result["rewards"]
        mean = rewards.mean(axis=0)
        plt.plot(mean, label=f"PGTS m={m}" if isinstance(m, int) else f"PGTS {m}", linewidth=3, linestyle="--")

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title(f"{env.name}: PG vs PGTS (multi-m)")
    plt.legend()

    plt.savefig(f"{RESULT_DIR}/{env.name}_comparison.png")
    # plt.show()


def plot_timing(env, pg_result, pgts_dict):
    labels = ["PG"]
    times = [pg_result["timing"]["total_seconds"]]

    for key, result in pgts_dict.items():
        labels.append(key)
        times.append(result["timing"]["total_seconds"])

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, times, color="tab:orange", alpha=0.85)
    ax.set_xlabel("Experiment Variant")
    ax.set_ylabel("Total Time (seconds)")
    ax.set_title(f"{env.name}: Timing Comparison")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)

    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    timing_plot_path = f"{RESULT_DIR}/{env.name}_timing_comparison.png"
    fig.savefig(timing_plot_path)
    plt.close(fig)
    print(f"[SAVE] Timing plot saved: {timing_plot_path}")

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
    policy, runtime_device = move_model_with_fallback(policy, DEVICE)
    policy.load_state_dict(torch.load(model_path, map_location=runtime_device))
    policy.eval()

    video_env = RecordVideo(base_env, video_dir, episode_trigger=lambda e: True)

    try:
        state, _ = video_env.reset()
        done = False
        step = 0

        while not done and step < 1000:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=runtime_device).unsqueeze(0)
            # get pure mean from the forward pass
            with torch.no_grad():
                mean_action, _ = policy(state_t)  # ignore std
                
            action_np = mean_action.squeeze(0).detach().cpu().numpy()
            state, _, terminated, truncated, _ = video_env.step(action_np)
            done = terminated or truncated
            step += 1

    except NotImplementedError:
        print(f"[WARN] Skipping video for {tag}: environment render() is not implemented.")

    finally:
        video_env.close()

if __name__ == "__main__":
    print(f"Using torch device: {DEVICE}")
    envs = get_envs()
    # M_VALUES = [1, 5, 10]
    M_VALUES = [1, 3, 5, -1]
    for env in envs:
        env.reset()
        init_state = env.state
        pg_result = None

        env.init_state = init_state
        pg_rewards, pg_states, pg_policy, pg_timing = run_experiment("PG", env)
        pg_model_path = f"{RESULT_DIR}/{env.name}_pg.pt"
        torch.save(pg_policy.state_dict(), pg_model_path)
        record_agent(env, pg_model_path, f"{env.name}_PG")
        pg_result = {
            "rewards": pg_rewards,
            "states": pg_states,
            "timing": pg_timing,
            "model_path": pg_model_path,
        }

        pgts_results = {}
        pgts_policies = {}
        for m in M_VALUES:
            # PGTS without lagging
            print(f"\nRunning PGTS m={m} | {env.name}")
            env.init_state = init_state
            rewards, states, policy, timing = run_experiment("PGTS", env, m=m)
            pgts_policies[f"m={m}"] = policy
            model_path = f"{RESULT_DIR}/{env.name}_pgts_m{m}.pt"
            torch.save(policy.state_dict(), model_path)
            record_agent(env, model_path, f"{env.name}_PGTS_m{m}")
            pgts_results[f"m={m}"] = {
                "rewards": rewards,
                "states": states,
                "timing": timing,
                "model_path": model_path,
            }

            # PGTS with lagging
            print(f"\nRunning PGTS LAG m={m} | {env.name}")
            env.init_state = init_state
            rewards, states, policy, timing = run_experiment("PGTS_WITH_LAGGING", env, m=m)
            pgts_policies[f"LAG_m={m}"] = policy
            model_path = f"{RESULT_DIR}/{env.name}_pgts_lag_m{m}.pt"
            torch.save(policy.state_dict(), model_path)
            record_agent(env, model_path, f"{env.name}_PGTS_lag_m{m}")
            pgts_results[f"LAG_m={m}"] = {
                "rewards": rewards,
                "states": states,
                "timing": timing,
                "model_path": model_path,
            }

        # Save detailed experiment artifacts for future analysis/comparison.
        bundle = {
            "env": env.name,
            "seeds": SEEDS,
            "m_values": M_VALUES,
            "pg_result": pg_result,
            "pgts_results": pgts_results,
        }
        bundle_path = f"{RESULT_DIR}/{env.name}_experiment_results.pkl"
        with open(bundle_path, "wb") as f:
            pickle.dump(bundle, f)

        timing_summary = {
            "env": env.name,
            "pg_timing": pg_result["timing"],
            "pgts_timing": {k: v["timing"] for k, v in pgts_results.items()},
        }
        timing_path = f"{RESULT_DIR}/{env.name}_experiment_timing.json"
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(timing_summary, f, indent=2)

        print(f"[SAVE] Detailed results saved: {bundle_path}")
        print(f"[SAVE] Timing summary saved: {timing_path}")

        # run_experiment will crash if you pass multiple seeds. need to handle it
        if pg_result is not None:
            plot_all(env, pg_result, pgts_results)
            plot_timing(env, pg_result, pgts_results)