import numpy as np
from continuous.base_env import BaseMDP
import matplotlib.pyplot as plt
import io

class TwoPeakMDP(BaseMDP):

    def __init__(self):
        # We'll set the bounds to -6 and 6 as per your init
        super().__init__(name="two_peak", state_low=-6, state_high=6, noise=0.02)

    def reward(self, x):
        """Your mathematical reward function defining the 'terrain'."""
        x = np.array(x)
        return np.where(
            x < 0,
            -0.5 * (x + 3)**2 + 2,
            -1.0 * (x - 3)**2 + 4
        )

    def render(self):
        """Generates the RGB frame for the video recorder."""
        # 1. Create the plot
        fig, ax = plt.subplots(figsize=(5, 3))
        
        # Use the bounds defined in __init__
        x_axis = np.linspace(-6, 6, 200) 
        
        # ### FIX: Use your actual reward function to draw the peaks!
        y_axis = self.reward(x_axis)
        
        ax.plot(x_axis, y_axis, 'b-', label='Reward Landscape', alpha=0.7)
        
        # Draw the agent's current position as a red dot on the curve
        agent_x = self.state[0] if isinstance(self.state, (np.ndarray, list)) else self.state
        agent_y = self.reward(agent_x)
        
        ax.plot(agent_x, agent_y, 'ro', markersize=10, label='Agent') 
        
        ax.set_title(f"TwoPeak - Position: {agent_x:.2f} | Reward: {agent_y:.2f}")
        ax.set_ylim(-1, 5) # Set Y limits to show the peaks clearly
        ax.grid(True, alpha=0.3)
        
        # 2. Convert the Matplotlib figure to an RGB array
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Read the image and strip the Alpha channel (Gymnasium needs RGB, not RGBA)
        img = plt.imread(buf)
        plt.close(fig)
        
        # Return as uint8 (0-255) pixels
        return (img[:, :, :3] * 255).astype(np.uint8)