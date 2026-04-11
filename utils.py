import numpy as np

def project_simplex(v):
    """Euclidean projection onto probability simplex"""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u + (1 - cssv) / (np.arange(n) + 1) > 0)[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)

def greedy_projection(q_values):
    pi = np.zeros_like(q_values)
    pi[np.argmax(q_values)] = 1.0
    return pi
