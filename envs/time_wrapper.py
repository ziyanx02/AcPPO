import numpy as np
import torch
import torch.nn as nn

import genesis as gs
from vecenv import VecEnv

class TimeWrapper:
    def __init__(self, env: VecEnv, period_length: int, reset_each_period: bool):
        self.env = env
        self.period_length = period_length
        self.reset_each_period = reset_each_period

        self.device = env.device
        self.num_envs = env.num_envs
        self.num_states = env.num_states

        self.time_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.state_mean = nn.Parameter(torch.zeros(
            (self.period_length, self.num_states), device=self.device
        ))
        self.state_std = nn.Parameter(torch.zeros(
            (self.period_length, self.num_states), device=self.device
        ))

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.time_buf = torch.remainder(self.time_buf + 1, self.period_length)

        return obs, reward, done, info