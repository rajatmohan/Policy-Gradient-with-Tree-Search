import numpy as np
import matplotlib.pyplot as plt
import os

from grid_mdp import create_grid_mdp
from ladder_mdp import create_ladder_mdp
import mdp
from random_mdp import create_random_mdp
from policy import init_policy
from pg import policy_gradient_update
from pgts import pgts_update
from tightrope_mdp import create_tightrope_mdp

import numpy as np
import random

np.random.seed(0)
random.seed(0)

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


# PG RUN
def run_pg(mdp, steps, lr, init_type):
    pi = init_policy(mdp.S, mdp.A, init_type)
    rewards = []

    for _ in range(steps):
        v = mdp.value_function(pi)
        rewards.append(float(np.dot(mdp.mu, v)))
        pi = policy_gradient_update(mdp, pi, lr)

    return rewards


# PGTS RUN
def run_pgts(mdp, steps, lr, m, init_type):
    pi = init_policy(mdp.S, mdp.A, init_type)
    rewards = []
    initial_m = m
    adaptive_m = False
    if initial_m == -1:
        adaptive_m = True
        print("Using adaptive m strategy")
        adaptive_m_config = {
            'max_m': 20,
            'min_m': 1,
            'window': 5,
            'std_threshold': 2.0,
            'growth_rate': 5
        }
        m = adaptive_m_config['min_m']  # Start with minimum m


    for _ in range(steps):
        v = mdp.value_function(pi)
        rewards.append(float(np.dot(mdp.mu, v)))
        pi = pgts_update(mdp, pi, lr, m, rewards, adaptive_m=adaptive_m, episode=_, adaptive_m_config=adaptive_m_config if adaptive_m else None)

    return rewards

# MDP CREATION
def make_mdp(mdp_type, mu_type):
    if mdp_type == 'ladder':
        S = 5
        if mu_type == 'start':
            mu = np.zeros(S)
            mu[0] = 1.0
        elif mu_type == 'uniform':
            mu = np.ones(S) / S
        else:
            raise ValueError(f"Unsupported mu_type: {mu_type}")
        return create_ladder_mdp(mu=mu)
    elif mdp_type == 'random':
        mdp = create_random_mdp()
        if mu_type == 'uniform':
            mdp.mu = np.ones(mdp.S) / mdp.S
        elif mu_type == 'start':
            mu = np.zeros(mdp.S)
            mu[0] = 1.0
            mdp.mu = mu
        return mdp
    elif mdp_type == 'tightrope':
        S = 4
        if mu_type == 'start':
            mu = np.zeros(S)
            mu[0] = 1.0
        elif mu_type == 'uniform':
            mu = np.ones(S) / S
        else:
            raise ValueError(f"Unsupported mu_type: {mu_type}")
        return create_tightrope_mdp(mu=mu)
    elif mdp_type == 'grid':
        S = 100
        if mu_type == 'start':
            mu = np.zeros(S)
            mu[0] = 1.0
        elif mu_type == 'uniform':
            mu = np.ones(S) / S
        else:
            raise ValueError(f"Unsupported mu_type: {mu_type}")
        return create_grid_mdp(mu=mu)
    else:
        raise ValueError(f"Unsupported mdp_type: {mdp_type}")

# PLOTTING FUNCTION
def plot_experiment(mdp, mdp_type, mu_type, init_type, lr, steps, m_values):
    plt.figure(figsize=(10, 6))

    # Compute optimal return for reference
    _, _, optimal_return = mdp.optimal_policy_value_iteration()
    
    # ---- PG (run once) ----
    pg_rewards = run_pg(mdp, steps, lr, init_type)
    plt.plot(pg_rewards, label="PG", linewidth=2)

    print(f"[PG] final return = {pg_rewards[-1]:.4f}")

    # ---- PGTS (multiple depths) ----
    styles = ['--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.']
    markers = ['o', 's', '^', 'd', 'x', 'P', 'v', '*', 'h', '+']
    for m in m_values:
        pgts_rewards = run_pgts(mdp, steps, lr, m, init_type)
        plt.plot(pgts_rewards, label=f"PGTS (m={m})", linestyle=styles[m % len(styles)], marker=markers[m % len(markers)], markevery=20)

        print(f"[PGTS m={m}] final return = {pgts_rewards[-1]:.4f}")

    # Add optimal return as horizontal line
    plt.axhline(y=optimal_return, color='red', linestyle='-', linewidth=2, label=f'Optimal Return = {optimal_return:.4f}')
    print(f"[OPTIMAL] optimal return = {optimal_return:.4f}")

    plt.xlabel("Iterations")
    plt.ylabel("Return")
    plt.title(f"{mdp_type} | mu={mu_type} | init={init_type} | lr={lr}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = (
        f"{mdp_type}_mu-{mu_type}_init-{init_type}_"
        f"lr-{str(lr).replace('.', 'p')}_steps-{steps}.png"
    )
    save_path = os.path.join(RESULT_DIR, filename)
    plt.savefig(save_path, dpi=150)
    print(f"[SAVED] figure saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    mdp_types = ['ladder', 'random', 'tightrope', 'grid']  
    mdp_types = ['tightrope']  
    
    config = {
        'ladder': {
            'start': {
                0.05: {
                    'steps': 200,
                    'm_values': [1, 2, 4, 6],
                    'init_type': ['left']
                },
                2.0: {
                    'steps': 30,
                    'm_values': [1, 2, 4],
                    'init_type': ['left']
                }
            },
            'uniform': {
                0.05: {
                    'steps': 300,
                    'm_values': [1, 2, 4, 6],
                    'init_type': ['left']
                },
                2.0: {
                    'steps': 50,
                    'm_values': [1, 2, 4],
                    'init_type': ['left']
                }
            }
        },
        'random': {
            'start': {
                0.05: {
                    'steps': 300,
                    'm_values': [1, 3, 5],
                    'init_type': ['left']
                },
                2.0: {
                    'steps': 50,
                    'm_values': [1, 3, 5],
                    'init_type': ['left']
                }
            },
            'uniform': {
                0.05: {
                    'steps': 500,
                    'm_values': [1, 3, 5],
                    'init_type': ['left']
                },
                2.0: {
                    'steps': 80,
                    'm_values': [1, 3, 5],
                    'init_type': ['left']
                }
            }
        },
        'tightrope': {
            'start': {
                0.05: {
                    'steps': 25,
                    'm_values': [1, 2],
                    'init_type': ['left']
                },
                2.0: {
                    'steps': 5,
                    'm_values': [1, 2],
                    'init_type': ['left']
                }
            },
            'uniform': {
                0.05: {
                    'steps': 25,
                    'm_values': [1, 2],
                    'init_type': ['left']
                },
                2.0: {
                    'steps': 5,
                    'm_values': [1, 2],
                    'init_type': ['left']
                }
            }
        },
        'grid': {
            'start': {
                0.1: {
                    'steps': 200,
                    'm_values': [1, 2, 3],
                    'init_type': ['uniform']
                },
                2.0: {
                    'steps': 5,
                    'm_values': [1, 2, 3],
                    'init_type': ['uniform']
                }
            },
            'uniform': {
                0.1: {
                    'steps': 25,
                    'm_values': [1, 2],
                    'init_type': ['uniform']
                },
                2.0: {
                    'steps': 5,
                    'm_values': [1, 2],
                    'init_type': ['uniform']
                }
            }
        },
    }

    for mdp_type in mdp_types:
        for mu_type in config[mdp_type].keys():
                for lr, expe in config[mdp_type][mu_type].items():
                    for init_type in expe['init_type']:

                        steps = expe['steps']
                        m_values = expe['m_values']

                        print("\n" + "="*60)
                        print(f"MDP={mdp_type}, mu={mu_type}, init={init_type}, lr={lr}, steps={steps}")
                        print("="*60)

                        mdp = make_mdp(mdp_type, mu_type)

                        plot_experiment(
                            mdp,
                            mdp_type,
                            mu_type,
                            init_type,
                            lr,
                            steps,
                            m_values
                        )

