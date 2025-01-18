import torch
import torch.nn as nn
import torch.optim as optim

class EnvCfg:
    num_envs = 1000
    num_dims = 5
    episode_length = 20
    value_limit = 3.0

class BaseEnv:
    def __init__(self, cfg: EnvCfg, device="cuda"):
        self.device = device

        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.num_actions = cfg.num_dims
        self.num_observations = cfg.num_dims
        self.num_states = cfg.num_dims
        self.episode_length = cfg.episode_length
        self.value_limit = cfg.value_limit

        # Initialize environment states and time steps
        self.states = torch.zeros((self.num_envs, self.num_states), device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_states), device=self.device)
        self.episode_length_buffer = torch.zeros((self.num_envs,), device=self.device)

    def set_state(self, states):
        self.states = states

    def get_state(self):
        return self.states

    def check_termination(self):
        return self._check_termination()

    def compute_observation(self):
        return self._compute_observation()

    def compute_reward(self):
        return self._compute_reward()

    def get_info(self):
        return {"true_rewards": self.compute_reward()}

    def reset(self):
        self._reset_idx(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))

        return self.compute_observation()

    def step(self, action):
        self.actions = action
        self._step(action)
        self.episode_length_buffer += 1

        done = self.check_termination()
        reward = self.compute_reward()
        info = self.get_info()
        value_obs = self.compute_observation()

        self._reset_idx(done)
        actor_obs = self.compute_observation()

        return actor_obs, reward, done, info

    def _check_termination(self):
        return self.episode_length_buffer >= self.episode_length

    def _compute_observation(self):
        return self.states

    def _compute_reward(self):
        raise NotImplementedError

    def _reset_idx(self, reset_idx):
        self.states[reset_idx] = torch.zeros((reset_idx.sum(), self.num_states), device=self.device)
        self.episode_length_buffer[reset_idx] = torch.zeros((reset_idx.sum(),), device=self.device)

    def _step(self, action):
        self.states = self.states + action
        self.states.clamp_(-self.value_limit, self.value_limit)