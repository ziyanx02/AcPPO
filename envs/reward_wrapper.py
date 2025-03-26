import numpy as np
import torch
import torch.nn as nn

import genesis as gs

class RewardWrapper:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # Command tracking terms

    def _reward_lin_vel(self):
        # Tracking linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.body_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / 0.25)

    def _reward_ang_vel(self):
        # Tracking angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.body_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / 0.25)

    def _reward_base_height(self):
        # Behavior: tracking base height
        return torch.square(self.body_pos[:, 2] - self.gait_body_height)

    def _reward_contact_force(self):
        # Behavior: tracking contact force of feet
        foot_forces = torch.norm(self.link_contact_forces[:, self.feet_link_indices, :], dim=-1)
        desired_contact = self.desired_contact_states
        return torch.mean((1 - desired_contact) * (1 - torch.exp(-foot_forces ** 2 / 100.)), dim=-1)
    
    def _reward_feet_height(self):
        # Behavior: tracking desired feet height
        rew_foot_height = torch.square(self.feet_pos_local[..., 2] - self.desired_feet_pos_local[..., 2]) * (1 - self.desired_contact_states)
        return torch.mean(rew_foot_height, dim=-1)
    
    def _reward_feet_pos(self):
        # Behavior: tracking desired feet position of x,y-axis
        rew_foot_pos = torch.sum(torch.square(self.feet_pos_local[..., 0:2] - self.desired_feet_pos_local[..., 0:2]), dim=-1)
        return torch.mean(rew_foot_pos, dim=-1)

    def _reward_alive(self):
        # Reward for staying alive
        return 1 - self.terminate_buf.float()

    def _reward_terminate(self):
        # Penalize termination
        return self.terminate_buf.float()

    # Regularization terms

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.body_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.body_ang_vel[:, :2]), dim=-1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.body_projected_gravity[:, :2]), dim=-1)

    def _reward_torques(self):
        # Penalize torques
        return torch.mean(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.mean(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.mean(torch.square(self.last_dof_vel - self.dof_vel), dim=-1)

    def _reward_dof_pos_diff(self):
        # Penalize dof positions deviate from default pose
        return torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=-1)

    def _reward_contact_vel(self):
        # Pernalize big vel_z with contact
        foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.num_envs, -1)
        desired_contact = self.desired_contact_states
        return torch.mean(desired_contact * (1 - torch.exp(-foot_velocities ** 2 / 10.)), dim=-1)

    def _reward_action_smoothness_1(self):
        # Penalize 1st-order deviation in actions
        diff = torch.square(self.actions - self.last_actions)
        diff = diff * (self.last_actions != 0)  # ignore first step
        return torch.mean(diff, dim=-1)

    def _reward_action_smoothness_2(self):
        # Penalize 2st-order deviation in actions
        diff = torch.square(self.actions - 2 * self.last_actions + self.last_last_actions)
        diff = diff * (self.last_actions != 0) * (self.last_last_actions != 0)  # ignore first&second step
        return torch.mean(diff, dim=-1)
    
def RewardFactory(base_class):
    return type(f"{base_class.__name__}Reward", (base_class, RewardWrapper), {})