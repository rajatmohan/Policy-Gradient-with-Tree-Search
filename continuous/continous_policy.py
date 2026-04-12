import torch
import torch.nn as nn
from torch.distributions import Normal


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.mean = self.mean = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Tanh() # Forces mean action between -1 and 1
        )

        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Orthogonal init with a specific gain for Tanh
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.constant_(m.bias, 0)
        
        # Make the output layer even smaller to start with "gentle" actions
        nn.init.orthogonal_(self.mean[0].weight, gain=0.01)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        std = torch.clamp(std, 0.01, 1.0)
        return mean, std

    def sample_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)

        mean, std = self(state_t)

        dist = Normal(mean, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action.detach().numpy()[0], log_prob
