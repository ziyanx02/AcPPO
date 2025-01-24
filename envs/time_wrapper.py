import numpy as np
import torch
import gym

class TimeWrapper(gym.Wrapper):
    def __init__(self, env):
        super(TimeWrapper, self).__init__(env)
        self.time = 0

    def reset(self):
        self.time = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.time += 1
        return obs, reward, done, {"time": self.time}

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()