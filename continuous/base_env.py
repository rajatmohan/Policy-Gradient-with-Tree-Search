from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

class BaseMDP(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, name="env", state_low=-10, state_high=10, noise=0.02, seed=0, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.init_state is None:
            self.state = np.random.uniform(-2, 2)
            self.init_state = self.state
        else:
            self.state = self.init_state

        self.step_count = 0
        return np.array([self.state], dtype=np.float32), {}

    def get_checkpoint(self):
        return {
            'state': float(self.state),
            'step_count': int(self.step_count)
        }

    def restore_checkpoint(self, checkpoint):
        self.state = checkpoint['state']
        self.step_count = checkpoint['step_count']

    def reward(self, x):
        raise NotImplementedError

    def step(self, action):
        a = action[0] if isinstance(action, (np.ndarray, list)) else action
        a = np.clip(a, -1, 1)

        self.state = self.state + a + np.random.randn() * self.noise

        terminated = False
        if self.state < self.state_low:
            self.state = self.state_low + (self.state_low - self.state)
            reward = -5
        elif self.state > self.state_high:
            self.state = self.state_high - (self.state - self.state_high)
            reward = -5
        else:
            reward = self.reward(self.state)

        dist_penalty = 0.5 * (max(0, abs(self.state) - (0.8 * self.state_high)))
        reward -= dist_penalty

        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        return np.array([self.state], dtype=np.float32), float(reward), terminated, truncated, {}

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        x = np.linspace(self.state_low, self.state_high, 500)
        y = self.reward(x)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y)

        ax.scatter(
            self.state,
            self.reward(self.state),
            color="red",
            s=100
        )

        ax.set_xlim(self.state_low, self.state_high)

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)

        return img

    def rollout(self, policy, max_steps=None):
        if max_steps is None:
            max_steps = self.max_steps

        states, actions, rewards, log_probs = [], [], [], []
        dones = []
        checkpoints = []

        state, _ = self.reset()

        for _ in range(max_steps):
            checkpoints.append(self.get_checkpoint())

            action, log_prob = policy.sample_action(state)

            next_state, reward, terminated, truncated, _ = self.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(terminated)

            state = next_state

            if terminated or truncated:
                break

        return states, actions, rewards, dones, log_probs, checkpoints
