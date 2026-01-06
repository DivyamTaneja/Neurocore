# rl_env.py

import gym
import numpy as np

from controller import *

class SNNControllerEnv(gym.Env):
    """
    RL environment for row-timestep scheduling.
    """

    def __init__(self, processor, spike_mem, neuron_mem, Tmax):
        super().__init__()

        self.processor = processor
        self.spike_mem = spike_mem
        self.neuron_mem = neuron_mem
        self.Tmax = Tmax

        self.state = ControllerState(processor, Tmax)
        self.output_layer = processor.num_layers - 1

        self.max_actions = sum(
            processor.shapes[L][1] for L in range(processor.num_layers)
        )

        self.action_space = gym.spaces.Discrete(self.max_actions)

        self.observation_space = gym.spaces.Box(
            low=0, high=1e6,
            shape=(self._obs_dim(),),
            dtype=np.float32
        )

    def _obs_dim(self):
        return len(self.state.t_progress) * 3 + 1

    def _get_obs(self):
        obs = []
        for (L, R), t in self.state.t_progress.items():
            obs.extend([
                L / 10.0,
                t / self.Tmax,
                self.neuron_mem.num_active(L, R)
            ])
        obs.append(self.neuron_mem.total_active())
        return np.array(obs, dtype=np.float32)

    def get_action_mask(self):
        actions = get_action_space(self.state, self.processor, self.neuron_mem)
        mask = np.zeros(self.max_actions, dtype=np.int8)
        mask[:len(actions)] = 1
        return mask

    def step(self, action_idx):
        actions = get_action_space(self.state, self.processor, self.neuron_mem)

        if action_idx >= len(actions):
            return self._get_obs(), -10.0, False, {}

        execute_action(
            actions[action_idx],
            self.state,
            self.processor,
            self.spike_mem,
            self.neuron_mem
        )

        reward = -self.neuron_mem.total_active()
        done = should_terminate(self.spike_mem, self.output_layer)

        if done:
            reward += 100.0

        return self._get_obs(), reward, done, {}

    def reset(self):
        self.state = ControllerState(self.processor, self.Tmax)
        return self._get_obs()
