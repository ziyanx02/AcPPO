import numpy as np
import torch

from envs.grid_env import *

class TwoGrids(GridEnv):

    def _reward_success(self):
        dis_to_target = torch.norm(self.state_buf - self.target_pos, dim=1)
        return dis_to_target < self.target_width
