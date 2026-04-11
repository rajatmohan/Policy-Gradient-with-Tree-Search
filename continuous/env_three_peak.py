import numpy as np
from continuous.base_env import BaseMDP


class ThreePeakMDP(BaseMDP):

    def __init__(self):
        super().__init__(name="three_peak", state_low=-8, state_high=8, noise=0.03)

    def reward(self, x):
        return (
            np.exp(-0.3 * (x + 4)**2)
            + 0.8 * np.exp(-0.3 * (x - 1)**2)
            + 1.5 * np.exp(-0.3 * (x - 5)**2)
        )
