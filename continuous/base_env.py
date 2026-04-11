import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


class BaseMDP(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, name="env", state_low=-10, state_high=10, noise=0.02):
        super().__init__()

        self.name = name
        self.state = None
        self.init_state = None

        self.state_low = state_low
        self.state_high = state_high
        self.noise = noise

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

    def reward(self, x):
        raise NotImplementedError

    def step(self, action):
        action = np.clip(action, -1, 1)[0]

        self.state = self.state + action + np.random.randn() * self.noise

        if self.state < self.state_low or self.state > self.state_high:
            self.state = np.clip(self.state, self.state_low, self.state_high)
            reward = -5
        else:
            reward = self.reward(self.state)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return np.array([self.state], dtype=np.float32), float(reward), done, False, {}

    def render(self):
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


    def rollout(self, policy, max_steps=200):
        states, actions, rewards, log_probs = [], [], [], []

        state, _ = self.reset()

        for _ in range(max_steps):
            action, log_prob = policy.sample_action(state)
            next_state, reward, done, _, _ = self.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state

            if done:
                break

        return states, actions, rewards, log_probs
