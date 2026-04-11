import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import cv2


class MultiModalEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        self.state = None
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(1,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )

        self.max_steps = 200
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.random.uniform(-2, 2)
        self.step_count = 0

        return np.array([self.state], dtype=np.float32), {}
    
    def reward(self, x):
        return np.exp(-(x + 2)**2) + 1.5 * np.exp(-(x - 3)**2)

    def step(self, action):
        action = np.clip(action, -1, 1)[0]

        noise = np.random.randn() * 0.05

        self.state = self.state + action + noise
        self.state = np.clip(self.state, -10, 10)

        reward = self.reward(self.state)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return (
            np.array([self.state], dtype=np.float32),
            float(reward),
            done,
            False,
            {}
        )

    def render(self):
        x = np.linspace(-10, 10, 500)
        y = np.sin(3 * x) + 0.5 * np.cos(5 * x)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(x, y)
        ax.scatter(
            self.state,
            np.sin(3 * self.state) + 0.5 * np.cos(5 * self.state),
            color="red",
            s=100
        )

        ax.set_xlim(-10, 10)
        ax.set_ylim(-2, 2)

        fig.canvas.draw()

        buf = np.asarray(fig.canvas.buffer_rgba())

        img = buf[:, :, :3]

        plt.close(fig)

        return img

    
    def rollout(self, policy, max_steps=200):

        states = []
        actions = []
        rewards = []
        log_probs = []

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


