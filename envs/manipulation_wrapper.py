import numpy as np
import torch

import genesis as gs
from envs.manipulation_env import *
from ddpm import *

class PickCube(ManiEnv):

    def _add_entities(self):
        super()._add_entities()
        self.cube = self.scene.add_entity( 
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=(0.5, 0.0, 0.021),
            ),
        )

    def _init_buffers(self):
        super()._init_buffers()
        self.cube_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.cube_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )
        self.cube_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.cube_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.cube_contact_force = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )
        self.target_pos = torch.tensor(
            [0.4, 0, 0.4], device=self.device, dtype=gs.tc_float
        )

    def _update_buffers(self):
        super()._update_buffers()
        self.cube_pos[:] = self.cube.get_pos()
        self.cube_quat[:] = self.cube.get_quat()
        self.cube_lin_vel[:] = self.cube.get_vel()
        self.cube_ang_vel[:] = self.cube.get_ang()

    def compute_state(self):
        self.state_buf = torch.cat(
            [
                self.dof_pos,
                self.dof_vel,
                self.cube_pos,
                self.cube_quat,
                self.cube_lin_vel,
                self.cube_ang_vel,
            ],
            axis=-1,
        )

    def compute_observation(self):
        obs_buf = torch.cat(
            [
                self.commands * self.commands_scale,                                # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],
                self.dof_vel * self.obs_scales['dof_vel'],
                self.actions,
                self.ee_pos,
                self.ee_quat,
                self.cube_pos,
                self.cube_quat,
            ],
            axis=-1,
        )

        clip_obs = 100.0
        self.obs_buf = torch.clip(obs_buf, -clip_obs, clip_obs)

    def compute_critic_observation(self):
        privileged_obs_buf = torch.cat(
            [
                self.commands * self.commands_scale,                                # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],
                self.dof_vel * self.obs_scales['dof_vel'],
                self.actions,
                self.ee_pos,
                self.ee_quat,
                self.cube_pos,
                self.cube_quat,
            ],
            axis=-1,
        )
        clip_obs = 100.0
        self.privileged_obs_buf = torch.clip(privileged_obs_buf, -clip_obs, clip_obs)

    def _set_state(self, states, envs_idx):
        super()._set_state(states, envs_idx)
        cube_pos = states[:, 18:21]
        cube_quat = normalize(states[:, 21:25])
        cube_lin_vel = states[:, 25:28]
        cube_ang_vel = states[:, 28:31]

        self.cube_pos[envs_idx] = cube_pos
        self.cube_quat[envs_idx] = cube_quat
        self.cube_lin_vel[envs_idx] = cube_lin_vel
        self.cube_ang_vel[envs_idx] = cube_ang_vel
        self.cube_ang_vel[envs_idx] = gs_transform_by_quat(cube_ang_vel, self.cube_quat[envs_idx])

        self.cube.set_pos(
            pos=self.cube_pos[envs_idx],
            envs_idx=envs_idx,
        )
        self.cube.set_quat(
            quat=self.cube_quat[envs_idx],
            envs_idx=envs_idx,
        )

        cube_vel = torch.concat(
            [self.cube_lin_vel[envs_idx], self.cube_ang_vel[envs_idx]], dim=1
        )
        self.cube.set_dofs_velocity(
            velocity=cube_vel,
            dofs_idx_local=[0, 1, 2, 3, 4, 5],
            envs_idx=envs_idx,
        )

    def resample_commands(self, envs_idx):
        # resample commands
        self.commands[envs_idx, 0] = gs_rand_float(
            *self.command_cfg['x_range'], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 1] = gs_rand_float(
            *self.command_cfg['y_range'], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 2] = gs_rand_float(
            *self.command_cfg['z_range'], (len(envs_idx),), self.device
        )

    def _prepare_temporal_distribution(self):
        init_state_mean = torch.cat(
            [
                self.default_dof_pos,
                torch.zeros((9,), device=self.device, dtype=gs.tc_float),
                0.5 * torch.ones((1,), device=self.device, dtype=gs.tc_float),
                torch.zeros((1,), device=self.device, dtype=gs.tc_float),
                0.021 * torch.ones((1,), device=self.device, dtype=gs.tc_float),
                torch.ones((1,), device=self.device, dtype=gs.tc_float),
                torch.zeros((3,), device=self.device, dtype=gs.tc_float),
                torch.zeros((3,), device=self.device, dtype=gs.tc_float),
                torch.zeros((3,), device=self.device, dtype=gs.tc_float),
            ],
            axis=-1,
        )
        init_state_std = torch.cat(
            [
                0.0 * torch.ones((9,), device=self.device, dtype=gs.tc_float),
                0.0 * torch.ones((9,), device=self.device, dtype=gs.tc_float),
                0.2 * torch.ones((1,), device=self.device, dtype=gs.tc_float),
                0.2 * torch.ones((1,), device=self.device, dtype=gs.tc_float),
                0.0 * torch.ones((1,), device=self.device, dtype=gs.tc_float),
                0.0 * torch.ones((4,), device=self.device, dtype=gs.tc_float),
                0.0 * torch.ones((3,), device=self.device, dtype=gs.tc_float),
                0.0 * torch.ones((3,), device=self.device, dtype=gs.tc_float),
            ],
            axis=-1,
        )
        if self.eval:
            init_state_std *= 0.0
        self.state_mean = init_state_mean.unsqueeze(0).repeat(self.period_length, 1)
        self.state_std = init_state_std.unsqueeze(0).repeat(self.period_length, 1)
        self.init_state_min = torch.cat(
            [
                self.dof_pos_limits[:, 0],
                -torch.ones((9,), device=self.device, dtype=gs.tc_float),
                0.3 * torch.ones((1,), device=self.device, dtype=gs.tc_float),
                -0.2 * torch.ones((1,), device=self.device, dtype=gs.tc_float),
                0.021 * torch.ones((1,), device=self.device, dtype=gs.tc_float),
                torch.ones((1,), device=self.device, dtype=gs.tc_float),
                torch.zeros((3,), device=self.device, dtype=gs.tc_float),
                torch.zeros((3,), device=self.device, dtype=gs.tc_float),
                torch.zeros((3,), device=self.device, dtype=gs.tc_float),
            ],
            axis=-1,
        )
        self.init_state_max = torch.cat(
            [
                self.dof_pos_limits[:, 1],
                torch.ones((9,), device=self.device, dtype=gs.tc_float),
                0.7 * torch.ones((1,), device=self.device, dtype=gs.tc_float),
                0.2 * torch.ones((1,), device=self.device, dtype=gs.tc_float),
                0.021 * torch.ones((1,), device=self.device, dtype=gs.tc_float),
                torch.ones((1,), device=self.device, dtype=gs.tc_float),
                torch.zeros((3,), device=self.device, dtype=gs.tc_float),
                torch.zeros((3,), device=self.device, dtype=gs.tc_float),
                torch.zeros((3,), device=self.device, dtype=gs.tc_float),
            ],
            axis=-1,
        )
        self.link_contact_forces_limit = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0,], device=self.device
        )

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

    def _reward_ee_to_object_dis(self):
        # Penalize distance from end effector to object
        return torch.sum(torch.square(self.ee_pos - self.cube_pos), dim=1)

    def _reward_object_to_target_dis(self):
        # Penalize distance from object to target
        return torch.sum(torch.square(self.cube_pos - self.target_pos), dim=1)

    def _reward_ee_dis(self):
        # Penalize distance from end effector to target
        ee_pos = self.robot.get_links_pos(self.end_effector_link_indices)
        ee_dis = torch.norm(ee_pos[:, 0, :] - ee_pos[:, 1, :], dim=1)
        ee_to_object_dis = torch.norm(self.ee_pos - self.cube_pos, dim=1)
        return ee_dis * (ee_to_object_dis > 0.02).float() - ee_dis * (ee_to_object_dis <= 0.02).float()