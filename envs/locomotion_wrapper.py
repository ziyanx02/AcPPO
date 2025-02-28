import numpy as np
import torch

import genesis as gs
from envs.locomotion_env import *

class Walk(LocoEnv):
    
    def _reward_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(
                self.commands[:, :2] - self.base_lin_vel[:, :2]
            ),
            dim=1,
        )
        return torch.exp(-lin_vel_error / self.reward_cfg['tracking_sigma'])

    def _reward_ang_vel(self):
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

    def _reward_tracking_base_height(self):
        time = self.episode_length_buf.remainder(self.period_length)
        target_state = self.state_mean[time]
        base_state_error = (target_state[:, 0] - self.state_buf[:, 0]).square()
        return base_state_error * (time > 0)

    def _reward_tracking_projected_gravity(self):
        time = self.episode_length_buf.remainder(self.period_length)
        target_state = self.state_mean[time]
        base_state_error = (target_state[:, 1:4] - self.state_buf[:, 1:4]).square().sum(dim=1)
        return base_state_error * (time > 0)

    def _reward_tracking_lin_vel(self):
        time = self.episode_length_buf.remainder(self.period_length)
        target_state = self.state_mean[time]
        base_state_error = (target_state[:, 4:7] - self.state_buf[:, 4:7]).square().sum(dim=1)
        return base_state_error * (time > 0)

    def _reward_tracking_ang_vel(self):
        time = self.episode_length_buf.remainder(self.period_length)
        target_state = self.state_mean[time]
        base_state_error = (target_state[:, 7:10] - self.state_buf[:, 7:10]).square().sum(dim=1)
        return base_state_error * (time > 0)

    def _reward_tracking_dof_pos(self):
        time = self.episode_length_buf.remainder(self.period_length)
        target_state = self.state_mean[time]
        base_state_error = (target_state[:, 10:22] - self.state_buf[:, 10:22]).square().sum(dim=1)
        return base_state_error * (time > 0)

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
        for i in range(0, 25):
            self.state_mean[i, 0] = 0.3 - 0.1 * i / 25
            self.state_mean[i, [11, 14, 17, 20]] += 0.4 * i / 25
            self.state_mean[i, [12, 15, 18, 21]] += -0.8 * i / 25
        for i in range(25, 50):
            self.state_mean[i, 4] = 2
            self.state_mean[i, 6] = 9.8 * (37.5 - i) / 50
            self.state_mean[i, 0] = 0.2 + 9.8 / 2 * (12.5 ** 2 - (37.5 - i) ** 2) / 50 ** 2
        for i in range(25, 28):
            self.state_mean[i, [11, 14, 17, 20]] += 0.4 * (28 - i) / 3
            self.state_mean[i, [12, 15, 18, 21]] += -0.8 * (28 - i) / 3
        for i in range(47, 50):
            self.state_mean[i, [11, 14, 17, 20]] += 0.4 * (i - 47) / 3
            self.state_mean[i, [12, 15, 18, 21]] += -0.8 * (i - 47) / 3
        for i in range(50, 75):
            self.state_mean[i, 0] = 0.3 - 0.1 * (75 - i) / 25
            self.state_mean[i, [11, 14, 17, 20]] += 0.4 * (75 - i) / 25
            self.state_mean[i, [12, 15, 18, 21]] += -0.8 * (75 - i) / 25

class Backflip(Jump):

    def _prepare_temporal_distribution(self):
        super()._prepare_temporal_distribution()
        for i in range(25, 50):
            flip_angle = 2 * np.pi * (i - 25) / 25
            self.state_mean[i, 0] = 0.2 + 9.8 / 2 * (12.5 ** 2 - (37.5 - i) ** 2) / 50 ** 2
            self.state_mean[i, 1] = -np.sin(flip_angle)
            self.state_mean[i, 3] = -np.cos(flip_angle)
            self.state_mean[i, 4] = 0
            self.state_mean[i, 8] = -4 * np.pi
            self.state_mean[i, 6] = 9.8 * (37.5 - i) / 50

class Walk_Gaits(Walk):
    def __init__(self, num_envs, env_cfg, show_viewer, eval, debug, n_rendered_envs=1, device='cuda'):
        super().__init__(num_envs, env_cfg, show_viewer, eval, debug, n_rendered_envs, device)
        self._load_gait(env_cfg['gait'])

    def _init_buffers(self):
        super()._init_buffers()

        # current phase
        self.gait_indices = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )
        # current phase per foot 
        self.foot_indices = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )

        # desired gait
        self.gait_frequency = torch.ones(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )
        self.gait_duration = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )
        self.gait_offset = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )
        self.gait_feet_height = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )
        self.gait_feet_stationary_pos = torch.zeros(
            (self.num_envs, len(self.feet_link_indices), 2), device=self.device, dtype=gs.tc_float,
        )
        self.gait_base_height = torch.zeros(
            self.num_envs, device=self.device, dtype=gs.tc_float,
        )

        # time embed 
        self.clock_inputs = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )
        self.doubletime_clock_inputs = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )
        self.halftime_clock_inputs = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )

        # reference buffer 
        self.desired_contact_states = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )
        self.desired_feet_height = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)), device=self.device, dtype=gs.tc_float,
        )
        self.desired_feet_pos_local = torch.zeros(
            (self.num_envs, len(self.feet_link_indices), 2), device=self.device, dtype=gs.tc_float,
        )

    def _load_gait(self, gait_cfg):
        self.gait_cfg = gait_cfg
        self.gait_base_height[:] = gait_cfg['base_height_target']
        for i in range(len(self.feet_link_indices)):
            self.gait_frequency[:, i] = gait_cfg['frequency'][i]
            self.gait_duration[:, i] = gait_cfg['duration'][i]
            self.gait_offset[:, i] = gait_cfg['offset'][i]
            self.gait_feet_height[:, i] = gait_cfg['feet_height_target'][i]
            self.gait_feet_stationary_pos[:, i, 0] = gait_cfg['stationary_position'][i][0]
            self.gait_feet_stationary_pos[:, i, 1] = gait_cfg['stationary_position'][i][1]

    def _update_buffers(self):
        super()._update_buffers()
        self._update_desired_contact_states()
        self._update_desired_feet_height()
        self._update_desired_feet_pos_local()

    def _update_desired_contact_states(self):
        num_feet = len(self.feet_link_indices)

        self.gait_indices = torch.remainder(self.gait_indices + self.dt * self.gait_frequency, 1.0)
        
        self.foot_indices = torch.remainder(self.gait_indices + self.gait_offset, 1.0)

        for i in range(num_feet):
            idxs = self.foot_indices[:, i]
            duration = self.gait_duration[:, i]
            stance_idxs = torch.remainder(idxs, 1) < duration
            swing_idxs = torch.remainder(idxs, 1) > duration

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / duration[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - duration[swing_idxs]) * (
                        0.5 / (1 - duration[swing_idxs]))

        # if self.cfg.commands.durations_warp_clock_inputs:

        self.clock_inputs = torch.sin(2 * np.pi * self.foot_indices)        
        self.doubletime_clock_inputs= torch.sin(4 * np.pi * self.foot_indices)
        self.halftime_clock_inputs = torch.sin(np.pi * self.foot_indices)

        # von mises distribution
        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        for i in range(num_feet):
            idxs = self.foot_indices[:, i]
            smoothing_multiplier = (smoothing_cdf_start(torch.remainder(idxs, 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(idxs, 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(idxs, 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(idxs, 1.0) - 0.5 - 1)))
            self.desired_contact_states[:, i] = smoothing_multiplier

    def _update_desired_feet_height(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        self.desired_feet_height = self.gait_feet_height * phases + 0.02
    
    def _update_desired_feet_pos_local(self):
        # Raibert heuristic
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5 # (num_envs, num_feet)
        frequency = self.gait_frequency
        x_vel_des = self.commands[:, 0:1]
        y_vel_des = self.commands[:, 1:2]
        yaw_vel_des = self.commands[:, 2:3]
        desired_xs_norm = self.gait_feet_stationary_pos[:, :, 0]
        desired_ys_norm = self.gait_feet_stationary_pos[:, :, 1]

        desired_xs_offset = phases * x_vel_des * (0.5 / frequency)
        desired_ys_offset = phases * y_vel_des * (0.5 / frequency)
        yaw_to_y_vel_des = yaw_vel_des * desired_xs_norm
        desired_yaw_to_ys_offset = phases * yaw_to_y_vel_des * (0.5 / frequency)
        yaw_to_x_vel_des = - yaw_vel_des * desired_ys_norm
        desired_yaw_to_xs_offset = phases * yaw_to_x_vel_des * (0.5 / frequency)

        self.desired_feet_pos_local[:, :, 0] = desired_xs_norm + (desired_xs_offset + desired_yaw_to_xs_offset)
        self.desired_feet_pos_local[:, :, 1] = desired_ys_norm + (desired_ys_offset + desired_yaw_to_ys_offset)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.gait_base_height)

    def _reward_contact_force(self):
        # Tracking contact force: penalize big force without contact
        foot_forces = torch.norm(self.link_contact_forces[:, self.feet_link_indices, :], dim=-1)
        desired_contact = self.desired_contact_states
        reward = torch.sum((1 - desired_contact) * (1 - torch.exp(-foot_forces ** 2 / 100.)), dim=1) / 4
        return reward
    
    def _reward_contact_vel(self):
        # Tracking contact vel: pernalize big vel_z with contact
        foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.num_envs, -1)
        desired_contact = self.desired_contact_states
        reward = torch.sum(desired_contact * (1 - torch.exp(-foot_velocities ** 2 / 10.)), dim=1) / 4
        return reward

    def _reward_feet_height(self):
        # Tracking desired feet height
        reference_heights = self.terrain_heights[:, None]
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1) - reference_heights
        target_height = self.desired_feet_height
        rew_foot_height = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)
        return torch.sum(rew_foot_height, dim=1)


    def _reward_feet_pos(self):
        # Tracking desired feet position of x,y-axis
        num_feet = len(self.feet_link_indices)
        feet_pos_translated = self.foot_positions - self.com.unsqueeze(1)
        feet_pos_local = torch.zeros(self.num_envs, num_feet, 3, device=self.device)
        for i in range(num_feet):
            feet_pos_local[:, i, :] = gs_quat_apply_yaw(gs_quat_conjugate(self.base_quat),
                                                                 feet_pos_translated[:, i, :])

        err_raibert_heuristic = torch.abs(self.desired_feet_pos_local - feet_pos_local[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
    
    def _draw_debug_vis(self):
        super()._draw_debug_vis()

        ## Fix base
        # init_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        # init_pos[:, 2] = 0.35
        # self.robot.set_pos(init_pos)

        num_feet = len(self.feet_link_indices)
        feet_pos_translated = self.foot_positions - self.com.unsqueeze(1)
        feet_pos_local = torch.zeros(self.num_envs, num_feet, 3, device=self.device)
        for i in range(num_feet):
            feet_pos_local[:, i, :] = gs_quat_apply_yaw(gs_quat_conjugate(self.base_quat),
                                                                 feet_pos_translated[:, i, :])

        feet_pos = feet_pos_local
        for i in range(num_feet):
            feet_pos[:, i, :] = gs_quat_apply_yaw(self.base_quat, feet_pos_local[:, i, :])
        feet_pos += self.com.unsqueeze(1)

        desired_feet_pos = torch.cat([self.desired_feet_pos_local, self.desired_feet_height.unsqueeze(-1)], dim=-1)
        for i in range(num_feet):
            desired_feet_pos[:, i, :] = gs_quat_apply_yaw(self.base_quat, desired_feet_pos[:, i, :])
        desired_feet_pos[:, :, :2] += self.com.unsqueeze(1)[:, :, :2]

        for i in range(num_feet):
            self.scene.draw_debug_sphere(pos=feet_pos[0, i, :], radius=0.05, color=(0, 1, 0, 0.7))
            self.scene.draw_debug_sphere(pos=desired_feet_pos[0, i, :], radius=0.05, color=(1, 1 - self.desired_contact_states[0, i], 0, 0.7))

    def compute_observation(self):
        obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales['ang_vel'],                     # 3
                self.projected_gravity,                                             # 3
                self.commands * self.commands_scale,                                # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],
                self.dof_vel * self.obs_scales['dof_vel'],                         
                self.actions,
                self.clock_inputs,
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
        privileged_obs_buf = torch.cat(
            [
                self.base_lin_vel * self.obs_scales['lin_vel'],                     # 3
                self.base_ang_vel * self.obs_scales['ang_vel'],                     # 3
                self.projected_gravity,                                             # 3
                self.commands * self.commands_scale,                                # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],
                self.dof_vel * self.obs_scales['dof_vel'],
                self.actions,
                self.last_actions,
                self.clock_inputs,
            ],
            axis=-1,
        )
        clip_obs = 100.0
        self.privileged_obs_buf = torch.clip(privileged_obs_buf, -clip_obs, clip_obs)