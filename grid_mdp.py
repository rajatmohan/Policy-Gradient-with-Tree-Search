import numpy as np
from mdp import MDP

def create_grid_mdp(gamma=0.9, mu = None):
    grid_size = 10
    S = grid_size * grid_size   # 100 states
    A = 4  # left, right, up, down

    P = np.zeros((S, A, S))
    R = np.zeros((S, A))

    def to_state(i, j):
        return i * grid_size + j

    def from_state(s):
        return divmod(s, grid_size)

    # Define special states
    goal = to_state(9, 9)      # s99
    traps = [to_state(0, 1), to_state(2, 3)]  # s12, s32

    # Actions: 0=left, 1=right, 2=up, 3=down
    moves = {
        0: (0, -1),
        1: (0, 1),
        2: (-1, 0),
        3: (1, 0)
    }

    for s in range(S):
        i, j = from_state(s)
        # Terminal states (absorbing)
        if s == goal:
            P[s, :, s] = 1.0
            R[s, :] = 10.0
            continue

        if s in traps:
            P[s, :, s] = 1.0
            R[s, :] = -10.0
            continue

        for a in range(A):
            # Intended move (0.99)
            di, dj = moves[a]
            ni, nj = i + di, j + dj

            # Stay in place if out of bounds
            if ni < 0 or ni >= grid_size or nj < 0 or nj >= grid_size:
                ns = s
            else:
                ns = to_state(ni, nj)

            P[s, a, ns] += 0.99

            # Random move (0.01)
            for a_rand in range(A):
                di_r, dj_r = moves[a_rand]
                ni_r, nj_r = i + di_r, j + dj_r

                if ni_r < 0 or ni_r >= grid_size or nj_r < 0 or nj_r >= grid_size:
                    ns_r = s
                else:
                    ns_r = to_state(ni_r, nj_r)

                P[s, a, ns_r] += 0.01 / A

            # Rewards: only when reaching special states
            if ns == goal:
                R[s, a] = 10.0
            elif ns in traps:
                R[s, a] = -10.0
            else:
                R[s, a] = 0.0

    if mu is None:
        mu = np.zeros(S)
        mu[to_state(0, 0)] = 1.0 

    return MDP(P, R, gamma, mu)
