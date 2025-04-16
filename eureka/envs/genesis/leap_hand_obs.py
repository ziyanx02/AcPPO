def rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class LeapHand:

    def _init_buffers(self):
        self.base_euler = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.global_gravity = torch.tensor(
            np.array([0.0, 0.0, -1.0]), device=self.device, dtype=gs.tc_float
        )
        self.forward_vec = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.forward_vec[:, 0] = 1.0

        self.state_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=gs.tc_float
        )
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float
        )
        self.obs_noise = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float
        )
        self._prepare_obs_noise()

        self.privileged_obs_buf = (
            None
            if self.num_privileged_obs is None
            else torch.zeros(
                (self.num_envs, self.num_privileged_obs),
                device=self.device,
                dtype=gs.tc_float,
            )
        )
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.rew_buf_pos = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.rew_buf_neg = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.terminate_buf = torch.ones(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.time_out_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.reset_buf = torch.ones(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )

        # commands
        self.commands = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.commands_scale = torch.tensor(
            [
                self.obs_scales['lin_vel'],
                self.obs_scales['lin_vel'],
                self.obs_scales['ang_vel'],
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.stand_still = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )

        # names to indices
        self.motor_dofs = [
            self.robot.get_joint(name).dof_idx_local
            for name in self.env_cfg['dof_names']
        ]

        def find_link_indices(names, accurate=False):
            link_indices = list()
            availible = [True for i in range(len(self.robot.links))]
            for name in names:
                for i, link in enumerate(self.robot.links):
                    if availible[i] and (accurate==False and name in link.name or name == link.name):
                        availible[i] = False
                        link_indices.append(link.idx - self.robot.link_start)
            return link_indices

        self.termination_contact_link_indices = find_link_indices(
            self.env_cfg['termination_contact_link_names']
        )
        self.penalized_contact_link_indices = find_link_indices(
            self.env_cfg['penalized_contact_link_names']
        )
        self.thumb_fingertip_link_index = find_link_indices(
            "thumb_fingertip",
            accurate=True,
        )
        self.index_fingertip_link_index = find_link_indices(
            "fingertip",
            accurate=True,
        )
        self.index_fingertip_link_index = find_link_indices(
            "fingertip_2",
            accurate=True,
        )
        self.ring_fingertip_link_index = find_link_indices(
            "fingertip_3",
            accurate=True,
        )
        assert len(self.termination_contact_link_indices) > 0
        assert len(self.penalized_contact_link_indices) > 0

        # actions
        self.actions = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.last_last_actions = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.dof_pos = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.dof_vel = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.last_dof_vel = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.root_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.last_root_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )

        self.base_reset_pos = torch.tensor(
            self.env_cfg.get('base_reset_pos', self.env_cfg['base_init_pos']), device=self.device
        )
        self.base_reset_quat = torch.tensor(
            self.env_cfg.get('base_reset_quat', self.env_cfg['base_init_quat']), device=self.device
        )
        self.projected_gravity_reset = gs_transform_by_quat(
            self.global_gravity, gs_inv_quat(gs_quat_mul(self.base_reset_quat, gs_inv_quat(self.base_init_quat)))
        )

        # extras
        self.continuous_push = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.env_identities = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_int, 
        )
        self.common_step_counter = 0
        self.extras = {"log": {}}
        self.extras["observations"] = {"critic": self.privileged_obs_buf}

        self.terrain_heights = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_float,
        )

        # PD control
        stiffness = self.env_cfg['PD_stiffness']
        damping = self.env_cfg['PD_damping']

        self.p_gains, self.d_gains = [], []
        for dof_name in self.env_cfg['dof_names']:
            for key in stiffness.keys():
                if key in dof_name:
                    self.p_gains.append(stiffness[key])
                    self.d_gains.append(damping[key])
        self.p_gains = torch.tensor(self.p_gains, device=self.device, dtype=gs.tc_float)
        self.d_gains = torch.tensor(self.d_gains, device=self.device, dtype=gs.tc_float)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)

        self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)

        default_joint_angles = self.env_cfg['default_joint_angles']
        self.default_dof_pos = torch.tensor(
            [default_joint_angles[name] for name in self.env_cfg['dof_names']],
            device=self.device,
        )
        reset_joint_angles = self.env_cfg.get('reset_joint_angles', self.env_cfg['default_joint_angles'])
        self.reset_dof_pos = torch.tensor(
            [reset_joint_angles[name] for name in self.env_cfg['dof_names']],
            device=self.device,
        )

        self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)
        self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1] * pow(self.scale, 5)
        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = (
                m - 0.5 * r * 0.9
            )
            self.dof_pos_limits[i, 1] = (
                m + 0.5 * r * 0.9
            )
        self._prepare_temporal_distribution()

        self.motor_strengths = torch.ones((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.motor_offsets = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)

        self.base_link_index = 1

        # body
        self.body_link_index = find_link_indices(self.env_cfg['base_link_name'], accurate=True)
        if self.body_link_index[0] != 0:
            self.body_init_pos = torch.tensor(
                self.env_cfg['body_init_pos'], device=self.device
            )
            self.body_init_quat = torch.tensor(
                self.env_cfg['body_init_quat'], device=self.device
            )   
        else:
            self.body_init_pos = self.base_init_pos
            self.body_init_quat = self.base_init_quat

        self.body_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.body_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )
        self.body_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.body_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.body_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.body_projected_gravity = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.com = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=gs.tc_float,
        )

    def _update_buffers(self):

        self.links_pos = self.robot.get_links_pos()

        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        base_quat_rel = gs_quat_mul(self.base_quat, gs_inv_quat(self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1)))
        self.base_euler = gs_quat2euler(base_quat_rel)

        inv_quat_yaw = gs_quat_from_angle_axis(-self.base_euler[:, 2],
                                               torch.tensor([0, 0, 1], device=self.device, dtype=torch.float))

        self.base_lin_vel[:] = gs_transform_by_quat(self.robot.get_vel(), inv_quat_yaw)
        self.base_ang_vel[:] = gs_transform_by_quat(self.robot.get_ang(), inv_quat_yaw)
        self.projected_gravity = gs_transform_by_quat(
            self.global_gravity, gs_inv_quat(base_quat_rel)
        )

        # Same as base
        self.body_pos[:] = self.links_pos[:, self.body_link_index].squeeze(1)
        self.body_quat[:] = self.robot.get_links_quat()[:, self.body_link_index].squeeze(1)
        body_quat_rel = gs_quat_mul(self.body_quat, gs_inv_quat(self.body_init_quat.reshape(1, -1).repeat(self.num_envs, 1)))
        self.body_euler = gs_quat2euler(body_quat_rel)

        inv_quat_yaw = gs_quat_from_angle_axis(-self.body_euler[:, 2],
                                               torch.tensor([0, 0, 1], device=self.device, dtype=torch.float))

        self.body_lin_vel[:] = gs_transform_by_quat(self.robot.get_links_vel()[:, self.body_link_index].squeeze(1), inv_quat_yaw)
        self.body_ang_vel[:] = gs_transform_by_quat(self.robot.get_links_ang()[:, self.body_link_index].squeeze(1), inv_quat_yaw)
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