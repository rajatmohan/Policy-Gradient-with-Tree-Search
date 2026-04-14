import torch
import torch.nn as nn
from torch.distributions import Normal

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, log_std_init=0.0):
        super().__init__()

        # 1. Feature Extractor (Matches the Repo)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 2. Mean Head
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.mu_head.weight.data.mul_(0.1)
        self.mu_head.bias.data.mul_(0.0)

        # 3. Learnable Log Std
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std_init)
        
        # 4. Action Range Metadata
        self.max_action = 1.0 

    def forward(self, state):
        x = self.net(state)
        # Sigmoid keeps the math stable [0, 1]
        mu = torch.sigmoid(self.mu_head(x))
        return mu

    def _adapt_action(self, action):
        """Internal helper: Maps [0, 1] to [-max_action, max_action]"""
        return 2 * (action - 0.5) * self.max_action

    def get_dist(self, state):
        mu = self.forward(state)
        action_log_std = self.action_log_std.expand_as(mu)
        action_std = torch.exp(action_log_std)
        return Normal(mu, action_std)

    def sample_action(self, state, device='cpu'):
        """Returns the adapted action [-1, 1] for the environment."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            dist = self.get_dist(state_t)
            raw_action = dist.sample()
            
            # Clamp raw sample to [0, 1] for sigmoid stability
            raw_action = torch.clamp(raw_action, 0.0, 1.0)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)

            # ADAPT HERE: Hand out the -1 to 1 version
            adapted_action = self._adapt_action(raw_action)

        return adapted_action.squeeze(0).cpu().numpy(), log_prob.squeeze(0)

    def deterministic_action(self, state, device='cpu'):
        """Returns the adapted mean [-1, 1] for evaluation."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            mu = self.forward(state_t)
            adapted_mu = self._adapt_action(mu)
        return adapted_mu.squeeze(0).cpu().numpy()