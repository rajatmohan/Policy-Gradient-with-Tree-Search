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

        self.mean = nn.Linear(64, action_dim)

        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        std = torch.clamp(std, 1e-3, 0.5)

        return mean, std

    def sample_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)

        mean, std = self(state_t)

        dist = Normal(mean, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action.detach().numpy()[0], log_prob
