import numpy as np
import torch

from envs.grid_env import *

class TwoGrids(GridEnv):

    def _reward_success(self):
        return self.dis_to_target < self.target_width

    def _reward_distance(self):
        return self.last_dis_to_target - self.dis_to_target

    def _reward_actions(self):
        return torch.square(self.actions).sum(dim=1)