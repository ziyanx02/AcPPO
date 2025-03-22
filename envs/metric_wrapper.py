from envs.locomotion_wrapper import GaitEnv

import torch

class GaitEnvMetric(GaitEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_reward(self):
        super().compute_reward()
        self.compute_metric()
    
    def compute_metric(self):
        metric = {}

        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        metric['lin_vel'] = torch.exp(-lin_vel_error / 0.25)

        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        metric['ang_vel'] = torch.exp(-ang_vel_error / 0.1)

        foot_forces = torch.norm(self.link_contact_forces[:, self.feet_link_indices, :], dim=-1)
        desired_contact = self.desired_contact_states
        metric['contact_force'] = torch.mean((1 - desired_contact) * (1 - torch.exp(-foot_forces ** 2 / 100.)), dim=-1)

        base_height_error = torch.square(self.base_pos[:, 2] - self.gait_base_height)
        metric['base_height'] = torch.exp(-base_height_error / 0.25)

        dof_pos_diff = torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=-1)
        metric['dof_pos_diff'] = torch.exp(-dof_pos_diff)

        metric['terminate'] = self.terminate_buf.float()

        self.extras['metric'] = metric