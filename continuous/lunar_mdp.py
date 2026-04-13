import gymnasium as gym
import numpy as np
import copy

class LunarMDP:
    def __init__(self, render_mode=None, seed=0):
        self.render_mode = render_mode
        self.env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
        self.name = "LUNAR_MDP"

        self.state = None
        self.init_state = None
        self.seed = seed

        self.max_steps = 1000

    def reset(self):
        state, _ = self.env.reset(seed=self.seed)

        if self.init_state is None:
            self.init_state = state.copy()
        else:
            state = self.init_state.copy()

        self.state = state
        return state, {}
    
    def get_checkpoint(self):
        lander = self.env.unwrapped.lander
        legs = self.env.unwrapped.legs
        
        checkpoint = {
            'lander_state': {
                'pos': lander.position.copy(),
                'angle': lander.angle,
                'v_lin': lander.linearVelocity.copy(),
                'v_ang': lander.angularVelocity,
            },
            'legs_state': [
                {'pos': leg.position.copy(), 'angle': leg.angle, 
                'v_lin': leg.linearVelocity.copy(), 'v_ang': leg.angularVelocity}
                for leg in legs
            ],
            'step_count': self.env.unwrapped.step_ctr if hasattr(self.env.unwrapped, 'step_ctr') else 0
        }
        return checkpoint

    def restore_checkpoint(self, checkpoint):
        # Restore the main body
        lander = self.env.unwrapped.lander
        lander.position = checkpoint['lander_state']['pos']
        lander.angle = checkpoint['lander_state']['angle']
        lander.linearVelocity = checkpoint['lander_state']['v_lin']
        lander.angularVelocity = checkpoint['lander_state']['v_ang']
        
        # Restore the legs
        for leg, state in zip(self.env.unwrapped.legs, checkpoint['legs_state']):
            leg.position = state['pos']
            leg.angle = state['angle']
            leg.linearVelocity = state['v_lin']
            leg.angularVelocity = state['v_ang']

    def step(self, action):
        action = np.clip(action, -1, 1)

        next_state, reward, terminated, truncated, info = self.env.step(action)

        self.state = next_state

        return next_state, float(reward), terminated, truncated, info

    def render(self):
        return self.env.render()

    def rollout(self, policy, max_steps=None):
        if max_steps is None:
            max_steps = self.max_steps

        states = []
        actions = []
        rewards = []
        log_probs = []
        dones = []

        checkpoints = []

        state, _ = self.reset()

        for _ in range(max_steps):
            checkpoints.append(self.get_checkpoint())
            action, log_prob = policy.sample_action(state)

            next_state, reward, terminated, truncated, info = self.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            dones.append(terminated)

            state = next_state

            if terminated or truncated:
                break
        
        return states, actions, rewards, dones, log_probs, checkpoints

    def close(self):
        self.env.close()
