# RL Project: Policy Gradient vs Policy Gradient with Tree Search

This project compares two policy optimization methods:

- Policy Gradient (PG)
- Policy Gradient with Tree Search (PGTS)

The codebase includes:

- Discrete tabular MDP experiments (ladder, random, tightrope, grid)
- Continuous-control experiments (two-peak and three-peak reward landscapes)
- Saved model checkpoints and rollout videos under the results folder

## 1) What is being compared?

### Policy Gradient (PG)
In the tabular setting, the update is:

pi_new(s) = ProjectSimplex(pi(s) + alpha * d_pi(s) * Q_pi(s, :))

where:

- d_pi(s): discounted occupancy under policy pi
- Q_pi(s, a): action-value function
- ProjectSimplex: projection back to valid action probabilities

In continuous experiments, PG uses REINFORCE-style returns.

### Policy Gradient with Tree Search (PGTS)
PGTS replaces Q_pi with an m-step lookahead transformed value:

Q_m = T^m(Q_pi)

and then updates:

pi_new(s) = ProjectSimplex(pi(s) + alpha * d_pi(s) * Q_m(s, :))

In continuous experiments, PGTS uses m-step bootstrapped returns with a value network.

### Intuition

- PG is simpler and cheaper per update.
- PGTS injects lookahead/planning signal and can improve direction quality, especially when immediate gradients are misleading.
- Larger m usually means stronger planning signal but more computation and possible sensitivity.

## 2) Repository layout

- experiment.py: runs tabular/discrete comparisons and plots returns
- pg.py: PG update for tabular MDPs
- pgts.py: PGTS update for tabular MDPs
- mdp.py: tabular MDP utilities (value, Q, occupancy, optimal return)
- continuous_experiments.py: runs continuous PG vs PGTS with multiple m
- continuous/continuous_pg.py: REINFORCE-style PG training loop
- continuous/continuous_pgts.py: m-step PGTS training with value learning
- results/: saved models, comparison plots, and recorded videos

## 3) Setup

Create and activate a virtual environment, then install dependencies.

Example:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy matplotlib torch gymnasium opencv-python
```

Note: requirements.txt is currently empty, so dependencies are installed explicitly above.

## 4) Running experiments

### Discrete/tabular comparison

```bash
python experiment.py
```

What it does:

- Trains PG once for each experiment config
- Trains PGTS for multiple m values
- Plots learning curves and draws an optimal-return reference line

### Continuous comparison

```bash
python continuous_experiments.py
```

What it does:

- Runs PG and PGTS on TwoPeakMDP and ThreePeakMDP
- Tests PGTS with m in [1, 5, 10]
- Saves policy checkpoints and videos into results/
- Saves comparison plots as results/<env_name>_comparison.png

## 5) How to read the comparison

Use these signals when comparing PG vs PGTS:

- Final return: last-episode (or last-iteration) performance
- Sample efficiency: how quickly return improves
- Stability: variance/oscillation across training
- Sensitivity to m: whether planning depth helps or hurts in each environment

Practical interpretation:

- If PGTS beats PG early and stays stable, lookahead is helping.
- If PGTS only helps for small m, deep lookahead may be too aggressive/noisy.
- If PG catches up late, PG may still be competitive with lower compute.

## 6) Existing outputs

The results folder already contains:

- PG and PGTS checkpoints for two-peak and three-peak environments
- Comparison plots for these environments
- Recorded rollout videos for PG and PGTS variants (different m)

## 7) Quick takeaway

This project is structured to study when planning-augmented policy gradients (PGTS) provide a measurable advantage over standard policy gradients (PG), both in tabular MDPs and in continuous environments with multimodal rewards.
