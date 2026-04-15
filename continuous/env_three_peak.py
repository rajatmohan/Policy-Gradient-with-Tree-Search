import numpy as np
from continuous.base_env import BaseMDP

import matplotlib.pyplot as plt
import io

class ThreePeakMDP(BaseMDP):

    def __init__(self):
        super().__init__(name="three_peak", state_low=-8, state_high=8, noise=0.03)

    def reward(self, x):
        return (
            np.exp(-0.3 * (x + 4)**2)
            + 0.8 * np.exp(-0.3 * (x - 1)**2)
            + 1.5 * np.exp(-0.3 * (x - 5)**2)
        )

    def render(self):
        fig, ax = plt.subplots(figsize=(5, 3))
        x = np.linspace(-8, 8, 200)
        y = self.reward(x)

        ax.plot(x, y, 'b-', label='Reward Landscape', alpha=0.7)

        agent_x = self.state[0] if isinstance(self.state, (np.ndarray, list)) else self.state
        agent_y = self.reward(agent_x)
        ax.plot(agent_x, agent_y, 'ro', markersize=10, label='Agent')

        ax.set_title(f"ThreePeak - Position: {agent_x:.2f} | Reward: {agent_y:.2f}")
        ax.set_ylim(-0.1, 1.7)
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)

        img = plt.imread(buf)
        plt.close(fig)

        return (img[:, :, :3] * 255).astype(np.uint8)

