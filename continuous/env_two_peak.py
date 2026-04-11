import numpy as np
from continuous.base_env import BaseMDP


class TwoPeakMDP(BaseMDP):

    def __init__(self):
        super().__init__(name="two_peak", state_low=-6, state_high=6, noise=0.02)

    def reward(self, x):
        x = np.array(x)
        return np.where(
            x < 0,
            -0.5 * (x + 3)**2 + 2,
            -1.0 * (x - 3)**2 + 4
        )
