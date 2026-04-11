import numpy as np
from utils import project_simplex, greedy_projection

def policy_gradient_update(mdp, pi, lr):
    Q = mdp.Q_function(pi)
    d = mdp.occupancy(pi)

    new_pi = np.zeros_like(pi)

    for s in range(mdp.S):
        grad = d[s] * Q[s]
        updated = pi[s] + lr * grad
        new_pi[s] = project_simplex(updated)

    return new_pi
