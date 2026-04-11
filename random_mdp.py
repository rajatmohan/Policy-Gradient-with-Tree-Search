import numpy as np
from mdp import MDP

def create_random_mdp(S=10, A=2, gamma=0.9):
    P = np.random.rand(S, A, S)
    P = P / P.sum(axis=2, keepdims=True)

    R = np.random.rand(S, A)

    mu = np.ones(S) / S

    return MDP(P, R, gamma, mu)
