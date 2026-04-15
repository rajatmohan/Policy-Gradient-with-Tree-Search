import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # final output needs to be squashed b/w left and right
        self.mean = nn.Sequential(
            nn.Linear(64, action_dim),
            # nn.Tanh() # Forces mean action between -1 and 1
        )

        # init -0.5 for all
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Orthogonal init with a specific gain for Tanh
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)
        
        # Make the output layer even smaller to start with "gentle" actions
        nn.init.orthogonal_(self.mean[0].weight, gain=0.01)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        # make sure the std dev > 0 
        std = torch.exp(self.log_std)
        # std = torch.clamp(std, 0.1, 1.0)
        std = torch.clamp(std, 0.01, 1.0)
        return mean, std

    def sample_action(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32).view(1, -1)

        mean, std = self(state_t)

        dist = Normal(mean, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        
        # FIXED: Manually clip the action for the environment here
        action_np = action.detach().numpy()[0]
        action_clipped = np.clip(action_np, -1.0, 1.0)

        return action_np, log_prob
