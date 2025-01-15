import torch
import torch.nn as nn
import torch.optim as optim

from maze.envs.base_env import BaseEnv, EnvCfg

class VanillaReachEnvCfg(EnvCfg):
    reward_type = "additive" # "additive" or "multiplicative"

class VanillaReachEnv(BaseEnv):
    def __init__(self, cfg: VanillaReachEnvCfg, device="cuda"):

        super(VanillaReachEnv, self).__init__(cfg, device)

        self.reward_type = cfg.reward_type
        self.target = 2 * torch.ones((self.num_states,), device=self.device)

    def _compute_reward(self):
        distance = torch.abs(self.states - self.target)
        reward = torch.exp(-distance)
        if self.reward_type == "multiplicative":
            reward = reward.prod(dim=1)
        elif self.reward_type == "additive":
            reward = reward.sum(dim=1)
        else:
            raise NotImplementedError
        return reward
