import numpy as np
from utils import project_simplex

def T_operator(mdp, Q):
    V = np.max(Q, axis=1)
    return mdp.R + mdp.gamma * np.einsum('sab,b->sa', mdp.P, V)

def T_m(mdp, Q, m):
    Qm = Q.copy()
    for _ in range(m):
        Qm = T_operator(mdp, Qm)
    return Qm

def pgts_update(mdp, pi, lr, m):
    Q = mdp.Q_function(pi)
    Qm = T_m(mdp, Q, m)

    d = mdp.occupancy(pi)

    new_pi = np.zeros_like(pi)

    for s in range(mdp.S):
        grad = d[s] * Qm[s]
        updated = pi[s] + lr * grad
        new_pi[s] = project_simplex(updated)

    return new_pi
