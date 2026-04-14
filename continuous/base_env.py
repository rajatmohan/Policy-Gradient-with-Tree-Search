from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

class BaseMDP(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, name="env", state_low=-10, state_high=10, noise=0.02, seed=0):
        super().__init__()
        self.name = name
        self.state = None
        self.init_state = None
        self.state_low = state_low
        self.state_high = state_high
        self.noise = noise
        self.seed = seed

        self.observation_space = spaces.Box(
            low=state_low, high=state_high, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.max_steps = 200
        self.step_count = 0

    def _to_scalar(self, value):
        arr = np.asarray(value, dtype=np.float32)
        return float(arr.reshape(-1)[0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.init_state is None:
            self.state = float(np.random.uniform(-2, 2))
            self.init_state = self.state
        else:
            self.state = self._to_scalar(self.init_state)

        self.step_count = 0
        return np.array([self.state], dtype=np.float32), {}
    
    def get_checkpoint(self):
        return {
            'state': self._to_scalar(self.state),
            'step_count': int(self.step_count)
        }

    def restore_checkpoint(self, checkpoint):
        self.state = self._to_scalar(checkpoint['state'])
        self.step_count = checkpoint['step_count']

    def reward(self, x):
        raise NotImplementedError

    def step(self, action):
        # Normalize action to a scalar so state stays scalar throughout rollout.
        a = np.clip(self._to_scalar(action), -1.0, 1.0)

        self.state = self._to_scalar(self.state + a + np.random.randn() * self.noise)

        # Boundary logic
        terminated = False
        if self.state < self.state_low:
            self.state = self.state_low + (self.state_low - self.state)
            reward = -5
            # terminated = True # Uncomment if you want boundary hit to end episode
        elif self.state > self.state_high:
            self.state = self.state_high - (self.state - self.state_high)
            reward = -5
            # terminated = True # Uncomment if you want boundary hit to end episode
        else:
            reward = self.reward(self.state)

        # Smooth boundary penalty
        dist_penalty = 0.5 * (max(0, abs(self.state) - (0.8 * self.state_high)))
        reward -= dist_penalty

        self.step_count += 1
        
        # --- THE TRICK: SEPARATE TERMINATION FROM TRUNCATION ---
        # BaseMDP usually doesn't have a 'death' condition unless you add one above.
        # But we must check if we hit max_steps.
        truncated = self.step_count >= self.max_steps

        return np.array([self.state], dtype=np.float32), float(reward), terminated, truncated, {}

    def rollout(self, policy, max_steps=None):
        if max_steps is None:
            max_steps = self.max_steps

        states, actions, rewards, log_probs = [], [], [], []
        dones = [] # To track actual physical terminations
        checkpoints = []

        state, _ = self.reset()

        for _ in range(max_steps):
            checkpoints.append(self.get_checkpoint())

            action, log_prob = policy.sample_action(state)
            
            # Unpack the 5 variables now returned by step
            next_state, reward, terminated, truncated, _ = self.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(terminated) # Tree search cares about physical termination

            state = next_state

            if terminated or truncated:
                break

        return states, actions, rewards, dones, log_probs, checkpoints