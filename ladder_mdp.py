import numpy as np
from mdp import MDP

def create_ladder_mdp(n_states=5, gamma=0.9, mu=None):
    S = n_states
    A = 2  # left=0, right=1

    P = np.zeros((S, A, S))
    R = np.zeros((S, A))

    for s in range(S):
        # transitions
        if s == S - 1:
            P[s, :, s] = 1.0  # absorbing
        else:
            P[s, 0, max(0, s - 1)] = 1.0
            P[s, 1, s + 1] = 1.0

    # 🔴 CRITICAL CHANGE
    R[S - 1, 1] = 1.0   # reward ONLY for (s4, right)

    if mu is None:
        mu = np.zeros(S)
        mu[0] = 1.0  # start at s0

    return MDP(P, R, gamma, mu)
