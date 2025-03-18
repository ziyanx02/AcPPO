import numpy as np
import torch

import genesis as gs
from envs.locomotion_env import *

class GaitEnv(LocoEnv):
    def __init__(self, num_envs, env_cfg, show_viewer, eval, debug, n_rendered_envs=1, device='cuda'):
        super().__init__(num_envs, env_cfg, show_viewer, eval, debug, n_rendered_envs, device)
        self._load_gait(env_cfg['gait'])

        if self.debug:
            self.ref_base_pos = self.base_init_pos
            self.ref_base_pos[2] = self.gait_cfg['base_height_target']

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
        self.desired_feet_pos_local = torch.zeros(
            (self.num_envs, len(self.feet_link_indices), 3), device=self.device, dtype=gs.tc_float,
        )
        self.feet_pos_local = torch.zeros(
            (self.num_envs, len(self.feet_link_indices), 3), device=self.device, dtype=gs.tc_float,
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
        self._update_feet_pos_local()

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
    
    def _update_feet_pos_local(self):
        # Linear feet height
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        self.desired_feet_pos_local[:, :, 2] = self.gait_feet_height * phases + 0.02

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

        # Feet positions in local frame
        center = self.base_pos.clone().unsqueeze(1) # self.com.unsqueeze(1)
        center[:, :, 2] = 0.0
        feet_pos_translated = self.foot_positions - center
        for i in range(len(self.feet_link_indices)):
            self.feet_pos_local[:, i, :] = gs_quat_apply_yaw(gs_quat_conjugate(self.base_quat),
                                                                 feet_pos_translated[:, i, :])

    def _draw_debug_vis(self):
        super()._draw_debug_vis()

        ## Fix base
        fix_base = False
        if fix_base :
            self.ref_base_pos[2] = self.gait_base_height[0]
            self.ref_base_pos[:2] += self.commands[0, :2] * self.dt
            self.robot.set_pos(self.ref_base_pos.unsqueeze(0))
        else: 
            self.ref_base_pos = self.base_pos[0].clone()
            self.ref_base_pos[:2] += self.commands[0, :2] 
            self.ref_base_pos[2] = self.gait_base_height[0]

        num_feet = len(self.feet_link_indices)
        feet_pos = self.feet_pos_local.clone()
        for i in range(num_feet):
            feet_pos[:, i, :] = gs_quat_apply_yaw(self.base_quat, feet_pos[:, i, :])
        feet_pos[:, :, :2] += self.base_pos.unsqueeze(1)[:, :, :2]

        desired_feet_pos = self.desired_feet_pos_local.clone()
        for i in range(num_feet):
            desired_feet_pos[:, i, :] = gs_quat_apply_yaw(self.base_quat, desired_feet_pos[:, i, :])
        desired_feet_pos[:, :, :2] += self.base_pos.unsqueeze(1)[:, :, :2]

        self.scene.draw_debug_sphere(pos=self.base_pos[0], radius=0.1, color=(0, 1, 0, 0.7))
        self.scene.draw_debug_sphere(pos=self.ref_base_pos, radius=0.1, color=(0, 0, 1, 0.7))
        for i in range(num_feet):
            self.scene.draw_debug_sphere(pos=feet_pos[0, i, :], radius=0.05, color=(0, 1, 0, 0.7))
            self.scene.draw_debug_sphere(pos=desired_feet_pos[0, i, :], radius=0.05, color=(1, 1 - self.desired_contact_states[0, i].cpu(), 0, 0.7))

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