import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def init_policy(S, A, init_type='uniform', epsilon=0.05):
    """
    Initialize policy π(a|s)

    init_type:
        'uniform' → equal probability
        'right'   → always choose right
        'left'    → always choose left
        'biased_right' → mostly right, some exploration
        'biased_left'  → mostly left, some exploration
    """

    if init_type == 'uniform':
        return np.ones((S, A)) / A

    elif init_type == 'right':
        pi = np.zeros((S, A))
        pi[:, 1] = 1.0
        return pi
    elif init_type == 'left':
        pi = np.zeros((S, A))
        pi[:, 0] = 1.0
        return pi
    elif init_type == 'biased_right':
        pi = np.ones((S, A)) * (epsilon / (A - 1))
        pi[:, 1] = 1 - epsilon
        return pi

    elif init_type == 'biased_left':
        pi = np.ones((S, A)) * (epsilon / (A - 1))
        pi[:, 0] = 1 - epsilon
        return pi

    else:
        raise ValueError(f"Unknown init_type: {init_type}")
