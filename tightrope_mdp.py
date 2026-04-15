import numpy as np
from mdp import MDP

def create_tightrope_mdp(gamma=0.9, mu=None):
    S = 4   # states: 0,1,2,3
    A = 2   # actions: left(0), right(1)

    P = np.zeros((S, A, S))
    R = np.zeros((S, A))

    good_terminal = 2
    bad_terminal = 3

    for s in range(S):
        # Terminal states (absorbing)
        if s == good_terminal:
            P[s, :, s] = 1.0
            R[s, :] = 1.0
            continue

        if s == bad_terminal:
            P[s, :, s] = 1.0
            R[s, :] = -10.0
            continue

        if s == 0:
            P[s, 0, s] = 1.0
        else:
            P[s, 0, bad_terminal] = 1.0

        R[s, 0] = 0.0

        if s == 1:
            P[s, 1, good_terminal] = 1.0
        else:
            P[s, 1, s + 1] = 1.0

        R[s, 1] = 0.0

    # for s in range(S):
    #     # Terminal states (absorbing, but yield ZERO continuous reward)
    #     if s == good_terminal:
    #         P[s, :, s] = 1.0
    #         R[s, :] = 0.0  # Changed from 1.0
    #         continue

    #     if s == bad_terminal:
    #         P[s, :, s] = 1.0
    #         R[s, :] = 0.0  # Changed from -10.0
    #         continue

    #     # Action 0 (Left)
    #     if s == 0:
    #         P[s, 0, s] = 1.0
    #         R[s, 0] = 0.0
    #     elif s == 1:
    #         P[s, 0, bad_terminal] = 1.0
    #         R[s, 0] = -10.0  # <--- Penalty applied ONCE upon falling

    #     # Action 1 (Right)
    #     if s == 0:
    #         P[s, 1, s + 1] = 1.0
    #         R[s, 1] = 0.0
    #     elif s == 1:
    #         P[s, 1, good_terminal] = 1.0
    #         R[s, 1] = 1.0    # <--- Reward applied ONCE upon success

    if mu is None:
        mu = np.zeros(S)
        mu[0] = 1.0

    return MDP(P, R, gamma, mu)
