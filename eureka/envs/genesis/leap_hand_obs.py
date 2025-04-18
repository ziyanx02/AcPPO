def rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class LeapHand:

    def _update_buffers(self):

        self.links_pos = self.robot.get_links_pos()

        self.base_pos[:, :3] = self.robot.get_pos()
        self.base_quat[:, :4] = self.robot.get_quat()
        base_quat_rel = gs_quat_mul(self.base_quat, gs_inv_quat(self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1)))
        self.base_euler = gs_quat2euler(base_quat_rel)

        inv_quat_yaw = gs_quat_from_angle_axis(-self.base_euler[:, 2],
                                               torch.tensor([0, 0, 1], device=self.device, dtype=torch.float))

        self.base_lin_vel[:, :3] = gs_transform_by_quat(self.robot.get_vel(), inv_quat_yaw)
        self.base_ang_vel[:, :3] = gs_transform_by_quat(self.robot.get_ang(), inv_quat_yaw)
        self.projected_gravity = gs_transform_by_quat(
            self.global_gravity, gs_inv_quat(base_quat_rel)
        )

        # Same as base
        self.body_pos[:, :3] = self.links_pos[:, self.body_link_index].squeeze(1)
        self.body_quat[:, :4] = self.robot.get_links_quat()[:, self.body_link_index].squeeze(1)
        body_quat_rel = gs_quat_mul(self.body_quat, gs_inv_quat(self.body_init_quat.reshape(1, -1).repeat(self.num_envs, 1)))
        self.body_euler = gs_quat2euler(body_quat_rel)

        inv_quat_yaw = gs_quat_from_angle_axis(-self.body_euler[:, 2],
                                               torch.tensor([0, 0, 1], device=self.device, dtype=torch.float))

        self.body_lin_vel[:, :3] = gs_transform_by_quat(self.robot.get_links_vel()[:, self.body_link_index].squeeze(1), inv_quat_yaw)
        self.body_ang_vel[:, :3] = gs_transform_by_quat(self.robot.get_links_ang()[:, self.body_link_index].squeeze(1), inv_quat_yaw)
        self.body_projected_gravity = gs_transform_by_quat(
            self.global_gravity, gs_inv_quat(body_quat_rel)
        )
        # print(self.body_projected_gravity)
        # print(gs_transform_by_quat(gs_transform_by_quat(self.global_gravity, gs_inv_quat(self.base_quat)), self.base_init_quat))

        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.link_contact_forces[:] = self.robot.get_links_net_contact_force()
        self.com[:] = self.rigid_solver.get_links_COM([self.base_link_index,]).squeeze(dim=1)

        self.thumb_fingertip_pos = self.links_pos[:, self.thumb_fingertip_link_index].squeeze(1)
        self.index_fingertip_pos = self.links_pos[:, self.index_fingertip_link_index].squeeze(1)
        self.middle_fingertip_pos = self.links_pos[:, self.index_fingertip_link_index].squeeze(1)
        self.ring_fingertip_pos = self.links_pos[:, self.ring_fingertip_link_index].squeeze(1)
        self.thumb_fingertip_contact_force = self.link_contact_forces[:, self.thumb_fingertip_link_index].squeeze(1)
        self.index_fingertip_contact_force = self.link_contact_forces[:, self.index_fingertip_link_index].squeeze(1)
        self.middle_fingertip_contact_force = self.link_contact_forces[:, self.index_fingertip_link_index].squeeze(1)
        self.ring_fingertip_contact_force = self.link_contact_forces[:, self.ring_fingertip_link_index].squeeze(1)

    def compute_observation(self):
        obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales['ang_vel'],                     # 3
                self.projected_gravity,                                             # 3
                self.commands * self.commands_scale,                                # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],
                self.dof_vel * self.obs_scales['dof_vel'],
                self.actions,
            ],
            axis=-1,
        )

    def resample_commands(self, envs_idx):
        # resample commands

        # lin_vel
        self.commands[envs_idx, 0] = rand_float(
            *self.command_cfg['lin_vel_x_range'], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 1] = rand_float(
            *self.command_cfg['lin_vel_y_range'], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, :2] *= (
            torch.norm(self.commands[envs_idx, :2], dim=1) > 0.3
        ).unsqueeze(1)

        # ang_vel
        self.commands[envs_idx, 2] = rand_float(
            *self.command_cfg['ang_vel_range'], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 2] *= torch.abs(self.commands[envs_idx, 2]) > 0.2