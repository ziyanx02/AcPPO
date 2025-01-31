import numpy as np
import torch

import genesis as gs
from envs.locomotion_env import *

class Walk(LocoEnv):
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(
                self.commands[:, :2] - self.base_lin_vel[:, :2]
            ),
            dim=1,
        )
        return torch.exp(-lin_vel_error / self.reward_cfg['tracking_sigma'])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2]
        )
        return torch.exp(-ang_vel_error / self.reward_cfg['tracking_sigma'])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1
        )

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.base_pos[:, 2]
        base_height_target = self.reward_cfg['base_height_target']
        return torch.square(base_height - base_height_target)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(
            1.0
            * (
                torch.norm(
                    self.link_contact_forces[:, self.penalized_contact_link_indices, :],
                    dim=-1,
                )
                > 0.1
            ),
            dim=1,
        )

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)  # upper limit
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_pos_diff(self):
        # Penalize dof positions deviate from default pose
        return torch.square(self.dof_pos - self.default_dof_pos).sum(dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.link_contact_forces[:, self.feet_link_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_alive(self):
        # Reward for staying alive
        return 1 - self.terminate_buf.float()

    def _reward_terminate(self):
        # Terminal reward / penalty
        return self.terminate_buf.float()

class Jump(Walk):

    def _prepare_obs_noise(self):
        self.obs_noise[:3] = self.obs_cfg['obs_noise']['ang_vel']
        self.obs_noise[3:6] = self.obs_cfg['obs_noise']['gravity']
        self.obs_noise[6:18] = self.obs_cfg['obs_noise']['dof_pos']
        self.obs_noise[18:30] = self.obs_cfg['obs_noise']['dof_vel']

    def compute_observation(self):
        phase = self.episode_length_buf.float().unsqueeze(1) / self.max_episode_length * 2 * np.pi
        obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales['ang_vel'],                     # 3
                self.projected_gravity,                                             # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'], # 12
                self.dof_vel * self.obs_scales['dof_vel'],                          # 12
                self.actions,                                                       # 12
                torch.sin(phase),
                torch.cos(phase),
                torch.sin(phase / 2),
                torch.cos(phase / 2),
                torch.sin(phase / 4),
                torch.cos(phase / 4),
            ],
            axis=-1,
        )
        # add noise
        if not self.eval:
            obs_buf += gs_rand_float(
                -1.0, 1.0, (self.num_obs,), self.device
            )  * self.obs_noise

        clip_obs = 100.0
        self.obs_buf = torch.clip(obs_buf, -clip_obs, clip_obs)

    def compute_critic_observation(self):
        phase = self.episode_length_buf.float().unsqueeze(1) / self.max_episode_length * 2 * np.pi
        privileged_obs_buf = torch.cat(
            [
                self.base_pos[:, 2:3],                                              # 1
                self.base_lin_vel * self.obs_scales['lin_vel'],                     # 3
                self.base_ang_vel * self.obs_scales['ang_vel'],                     # 3
                self.projected_gravity,                                             # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'], # 12
                self.dof_vel * self.obs_scales['dof_vel'],                          # 12
                self.actions,                                                       # 12
                self.last_actions,                                                  # 12
                torch.sin(phase),
                torch.cos(phase),
                torch.sin(phase / 2),
                torch.cos(phase / 2),
                torch.sin(phase / 4),
                torch.cos(phase / 4),
            ],
            axis=-1,
        )
        clip_obs = 100.0
        self.privileged_obs_buf = torch.clip(privileged_obs_buf, -clip_obs, clip_obs)

    def check_termination(self):
        self.terminate_buf = (
            self.episode_length_buf > self.max_episode_length
        )
        self.reset_buf = (
            self.episode_length_buf > self.max_episode_length
        )

    def _prepare_temporal_distribution(self):
        super()._prepare_temporal_distribution()
        for i in range(50, 75):
            self.state_mean[i, 4] = 2
            self.state_mean[i, 6] = 9.8 * (62.5 - i) / 50
            self.state_mean[i, 0] = 0.2 + 9.8 / 2 * (12.5 ** 2 - (62.5 - i) ** 2) / 50 ** 2

class Backflip(Walk):

    def _prepare_obs_noise(self):
        self.obs_noise[:3] = self.obs_cfg['obs_noise']['ang_vel']
        self.obs_noise[3:6] = self.obs_cfg['obs_noise']['gravity']
        self.obs_noise[6:18] = self.obs_cfg['obs_noise']['dof_pos']
        self.obs_noise[18:30] = self.obs_cfg['obs_noise']['dof_vel']

    def compute_observation(self):
        phase = self.episode_length_buf.float().unsqueeze(1) / self.max_episode_length * 2 * np.pi
        obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales['ang_vel'],                     # 3
                self.projected_gravity,                                             # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'], # 12
                self.dof_vel * self.obs_scales['dof_vel'],                          # 12
                self.actions,                                                       # 12
                torch.sin(phase),
                torch.cos(phase),
                torch.sin(phase / 2),
                torch.cos(phase / 2),
                torch.sin(phase / 4),
                torch.cos(phase / 4),
            ],
            axis=-1,
        )
        # add noise
        if not self.eval:
            obs_buf += gs_rand_float(
                -1.0, 1.0, (self.num_obs,), self.device
            )  * self.obs_noise

        clip_obs = 100.0
        self.obs_buf = torch.clip(obs_buf, -clip_obs, clip_obs)

    def compute_critic_observation(self):
        phase = self.episode_length_buf.float().unsqueeze(1) / self.max_episode_length * 2 * np.pi
        privileged_obs_buf = torch.cat(
            [
                self.base_pos[:, 2:3],                                              # 1
                self.base_lin_vel * self.obs_scales['lin_vel'],                     # 3
                self.base_ang_vel * self.obs_scales['ang_vel'],                     # 3
                self.projected_gravity,                                             # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'], # 12
                self.dof_vel * self.obs_scales['dof_vel'],                          # 12
                self.actions,                                                       # 12
                self.last_actions,                                                  # 12
                torch.sin(phase),
                torch.cos(phase),
                torch.sin(phase / 2),
                torch.cos(phase / 2),
                torch.sin(phase / 4),
                torch.cos(phase / 4),
            ],
            axis=-1,
        )
        clip_obs = 100.0
        self.privileged_obs_buf = torch.clip(privileged_obs_buf, -clip_obs, clip_obs)

    def check_termination(self):
        self.terminate_buf = (
            self.episode_length_buf > self.max_episode_length
        )
        self.reset_buf = (
            self.episode_length_buf > self.max_episode_length
        )

    def _prepare_temporal_distribution(self):
        super()._prepare_temporal_distribution()
        for i in range(50, 75):
            flip_angle = 2 * np.pi * (i - 50) / 25
            self.state_mean[i, 1] = -np.sin(flip_angle)
            self.state_mean[i, 3] = -np.cos(flip_angle)
            self.state_mean[i, 8] = -4 * np.pi
            self.state_mean[i, 6] = 9.8 * (62.5 - i) / 50
            self.state_mean[i, 0] = 0.2 + 9.8 / 2 * (12.5 ** 2 - (62.5 - i) ** 2) / 50 ** 2
