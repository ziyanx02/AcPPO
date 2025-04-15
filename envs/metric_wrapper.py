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
        # Metrics for tracking robot locomotion performance

        # Measures how well the robot follows the commanded linear velocity (xy-plane)
        # Uses an exponential function to reward smaller velocity errors
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.body_lin_vel[:, :2]), dim=1)
        metric['lin_vel'] = torch.exp(-lin_vel_error / 0.1)

        # Measures how well the robot follows the commanded angular velocity (yaw)
        # Encourages precise yaw control using an exponential decay function
        ang_vel_error = torch.square(self.commands[:, 2] - self.body_ang_vel[:, 2])
        metric['ang_vel'] = torch.exp(-ang_vel_error / 0.1)

        # Measures the deviation of foot contact forces from the desired contact states
        # Encourages proper foot placement and avoids unnecessary ground impact
        foot_forces = torch.norm(self.link_contact_forces[:, self.feet_link_indices, :], dim=-1)
        desired_contact = self.desired_contact_states
        metric['contact_force'] = torch.mean((1 - desired_contact) * (1 - torch.exp(-foot_forces ** 2 / 100.)), dim=-1)

        # Measures the deviation of the robot's base height from the desired gait-defined height
        # Encourages stable locomotion at the appropriate height
        body_height_error = torch.square(self.body_pos[:, 2] - self.gait_body_height)
        metric['base_height'] = torch.exp(-body_height_error / 0.25)

        # Measures the difference between current dof position and the default dof pos
        # Encourage more natural motions
        dof_pos_diff = torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=-1)
        metric['dof_pos_diff'] = torch.exp(-dof_pos_diff)

        # Indicates the number of failures during evaluation (e.g., falling, instability)
        metric['terminate'] = self.terminate_buf.float()

        self.extras['metric'] = metric