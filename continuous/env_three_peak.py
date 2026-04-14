import numpy as np
from continuous.base_env import BaseMDP

import matplotlib.pyplot as plt
import io

def render(self):
    # 1. Create a plot of the peaks
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.linspace(-2, 12, 100) # Range of your environment
    
    # Example for TwoPeak: Replace with your actual reward function
    y = np.exp(-(x - 2)**2) + 0.5 * np.exp(-(x - 8)**2) 
    
    ax.plot(x, y, 'b-', label='Peaks')
    # Draw the agent as a red dot
    ax.plot(self.state, 0, 'ro', markersize=10, label='Agent') 
    
    ax.set_title(f"{self.name} - State: {self.state[0]:.2f}")
    ax.set_ylim(-0.1, 1.2)
    
    # 2. Convert the plot to an RGB array
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    
    # Return as uint8 (0-255) pixels for the video recorder
    return (img[:, :, :3] * 255).astype(np.uint8)

class ThreePeakMDP(BaseMDP):

    def __init__(self):
        super().__init__(name="three_peak", state_low=-8, state_high=8, noise=0.03)

    def reward(self, x):
        return (
            np.exp(-0.3 * (x + 4)**2)
            + 0.8 * np.exp(-0.3 * (x - 1)**2)
            + 1.5 * np.exp(-0.3 * (x - 5)**2)
        )

