import numpy as np
from utils import project_simplex
from common.adaptive_m import get_adaptive_m

def T_operator(mdp, Q):
    V = np.max(Q, axis=1)
    return mdp.R + mdp.gamma * np.einsum('sab,b->sa', mdp.P, V)

def T_m(mdp, Q, m):
    Qm = Q.copy()
    for _ in range(m):
        Qm = T_operator(mdp, Qm)
    return Qm

def pgts_update(mdp, pi, lr, m, rewards, adaptive_m=False, episode=None, adaptive_m_config=None):
    if adaptive_m:
        m = get_adaptive_m(rewards, episode, current_m=m, max_m=adaptive_m_config['max_m'], min_m=adaptive_m_config['min_m'], window=adaptive_m_config['window'], std_threshold=adaptive_m_config['std_threshold'], growth_rate=adaptive_m_config['growth_rate'])
        print (f"Episode {episode}: Using m = {m}")
        
    Q = mdp.Q_function(pi)
    Qm = T_m(mdp, Q, m)

    d = mdp.occupancy(pi)

    new_pi = np.zeros_like(pi)

    for s in range(mdp.S):
        grad = d[s] * Qm[s]
        updated = pi[s] + lr * grad
        new_pi[s] = project_simplex(updated)

    return new_pi
