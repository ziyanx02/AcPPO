import numpy as np
import torch
import math

import genesis as gs
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from utils import *

class ManiEnv:
    def __init__(
        self,
        num_envs,
        env_cfg,
        show_viewer,
        eval,
        debug,
        n_render_envs=1,
        device='cuda',
    ) -> None:
        self.num_envs = num_envs
        self.cfg = env_cfg
        self.env_cfg = env_cfg
        self.num_actions = env_cfg['num_actions']
        self.num_states = env_cfg['num_states']
        self.num_dof = env_cfg['num_dofs']
        self.obs_cfg = env_cfg['observation']
        self.delay_action = env_cfg['delay_action']
        self.command_cfg = env_cfg['command']
        self.is_PPO = env_cfg['PPO']
        self.dt = 1 / env_cfg['control_freq']
        self.max_episode_length_s = env_cfg['episode_length_s']
        self.period_length_s = env_cfg['period_length_s']
        self.obs_scales = self.obs_cfg['obs_scales']
        self.num_obs = self.obs_cfg['num_obs']
        self.num_privileged_obs = self.obs_cfg['num_priv_obs']
        self.reward_cfg = env_cfg['reward']
        self.reward_scales = self.reward_cfg['reward_scales']
        self.max_episode_length = int(np.ceil(self.max_episode_length_s / self.dt))
        self.period_length = int(np.ceil(self.period_length_s / self.dt))

        self.headless = not show_viewer
        self.eval = eval
        self.debug = debug
        self.n_rendered_envs = n_render_envs

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            assert device in ['cpu', 'cuda']
            self.device = torch.device(device)

        self._create_scene()
        if gs.platform != 'macOS':
            self._set_camera()
        self.scene.build(n_envs=num_envs)
        self._init_buffers()
        self._prepare_reward_function()
        self._domain_randomization()

    def _create_scene(self):
        sim_dt = self.dt / self.env_cfg['decimation']
        sim_substeps = self.env_cfg['substeps']

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=sim_dt,
                substeps=sim_substeps,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt * self.env_cfg['decimation']),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                n_rendered_envs=self.n_rendered_envs,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=not self.headless,
            show_FPS=False,
        )

        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver
        
        self._add_entities()

    def _add_entities(self):
        self.scene.add_entity(
            gs.morphs.URDF(file='urdf/plane/plane.urdf', fixed=True),
        )

        self.base_init_pos = torch.tensor(
            self.env_cfg['base_init_pos'], device=self.device
        )
        self.base_init_quat = torch.tensor(
            self.env_cfg['base_init_quat'], device=self.device
        )

        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=self.env_cfg['urdf_path'],
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
            visualize_contact=self.debug,
        )

    def _prepare_reward_function(self):
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == 'termination':
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            for name in self.reward_scales.keys()
        }

    def _init_buffers(self):
        self.ee_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.ee_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )
        self.state_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=gs.tc_float
        )
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float
        )
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
                self.obs_scales['command'],
                self.obs_scales['command'],
                self.obs_scales['command'],
            ],
            device=self.device,
            dtype=gs.tc_float,
        )

        # names to indices
        self.motor_dofs = [
            self.robot.get_joint(name).dof_idx_local
            for name in self.env_cfg['dof_names']
        ]

        def find_link_indices(names):
            link_indices = list()
            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices

        self.penalized_contact_link_indices = find_link_indices(
            self.env_cfg['penalized_contact_link_names']
        )
        self.end_effector_link_indices = find_link_indices(
            self.env_cfg['end_effector_link_names']
        )
        assert len(self.penalized_contact_link_indices) > 0

        # actions
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float
        )
        self.last_last_actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float
        )
        self.target_arm_dof_pos = torch.zeros(
            (self.num_envs, 7), device=self.device, dtype=gs.tc_float
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
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )

        # extras
        self.env_identities = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_int, 
        )
        self.common_step_counter = 0
        self.extras = {"log": {}}
        self.extras["observations"] = {"critic": self.privileged_obs_buf}

        # PD control
        stiffness = self.env_cfg['PD_stiffness']
        damping = self.env_cfg['PD_damping']
        action_scale = self.env_cfg['action_scale']

        self.p_gains, self.d_gains, self.action_scale = [], [], []
        for dof_name in self.env_cfg['dof_names']:
            self.p_gains.append(stiffness[dof_name])
            self.d_gains.append(damping[dof_name])
            self.action_scale.append(action_scale[dof_name])
        self.p_gains = torch.tensor(self.p_gains, device=self.device)
        self.d_gains = torch.tensor(self.d_gains, device=self.device)
        self.action_scale = torch.tensor(self.action_scale, device=self.device)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)

        # self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        # self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)

        default_joint_angles = self.env_cfg['default_joint_angles']
        self.default_dof_pos = torch.tensor(
            [default_joint_angles[name] for name in self.env_cfg['dof_names']],
            device=self.device,
        )
        self.target_arm_dof_pos = self.default_dof_pos[None, :7].repeat(self.num_envs, 1)

        self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)
        self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1]
        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = (
                m - 0.5 * r * self.reward_cfg['soft_dof_pos_limit']
            )
            self.dof_pos_limits[i, 1] = (
                m + 0.5 * r * self.reward_cfg['soft_dof_pos_limit']
            )
        self._prepare_temporal_distribution()

        self.motor_strengths = torch.ones((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.motor_offsets = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)

    def _prepare_temporal_distribution(self):
        init_state_mean = torch.cat(
            [
                self.default_dof_pos,
                torch.zeros((9,), device=self.device, dtype=gs.tc_float),
            ],
            axis=-1,
        )
        init_state_std = torch.cat(
            [
                0.0 * torch.ones((9,), device=self.device, dtype=gs.tc_float),
                0.0 * torch.ones((9,), device=self.device, dtype=gs.tc_float),
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
            ],
            axis=-1,
        )
        self.init_state_max = torch.cat(
            [
                self.dof_pos_limits[:, 1],
                torch.ones((9,), device=self.device, dtype=gs.tc_float),
            ],
            axis=-1,
        )
        self.link_contact_forces_limit = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0,], device=self.device
        )

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return None, None

    def step(self, actions):
        clip_actions = self.env_cfg['clip_actions']
        self.actions = torch.clip(actions, -clip_actions, clip_actions)
        exec_actions = self.last_actions if self.delay_action else self.actions
        exec_actions = torch.cat(
            [
                exec_actions,
                exec_actions[:, -1:],
            ],
            axis=-1,
        )
        target_dof_pos = self._compute_target_dof_pos(exec_actions)
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.post_physics_step()
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        init_states = self.state_mean[0] + gs_rand_float(
            -1.0, 1.0, (len(envs_idx), self.num_states), self.device
        ) * self.state_std[0]
        self.set_state(init_states, 0, envs_idx)
        self.resample_commands(envs_idx)

    def _compute_target_dof_pos(self, actions):
        actions_scaled = actions * self.action_scale
        self.target_arm_dof_pos += actions_scaled[:, :7] * self.dt
        target_dof_pos = torch.cat(
            [
                self.target_arm_dof_pos,
                actions_scaled[:, 7:] + self.default_dof_pos[7:],
            ],
            axis=-1,
        )
        target_dof_pos = torch.clip(target_dof_pos, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
        return target_dof_pos

    def post_physics_step(self):
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._update_buffers()
        self.check_termination()
        self.compute_reward()
        self.compute_critic_observation()

        resampling_time_s = self.env_cfg['resampling_time_s']
        envs_idx = (
            (self.episode_length_buf % int(resampling_time_s / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.resample_commands(envs_idx)
        self._randomize_rigids(envs_idx)
        self._randomize_controls(envs_idx)

        envs_idx = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.update_extras(envs_idx)
        if self.is_PPO:
            self.reset_idx(envs_idx)
            self.compute_observation()
        self.compute_state()

        if gs.platform != 'macOS':
            self._render_headless()
        if not self.headless and self.debug:
            self._draw_debug_vis()

        self.last_actions[:] = self.actions[:]
        self.last_last_actions[:] = self.last_actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

    # ------------ update buffers ----------------

    def _update_buffers(self):
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.link_contact_forces[:] = self.robot.get_links_net_contact_force()

        self.ee_pos[:] = self.robot.get_links_pos(self.end_effector_link_indices).mean(dim=1)
        self.ee_quat[:] = self.robot.get_links_quat(self.end_effector_link_indices[0:1]).mean(dim=1)


    def check_termination(self):
        self.terminate_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = torch.logical_or(self.terminate_buf, self.time_out_buf)

    def compute_state(self):
        self.state_buf = torch.cat(
            [
                self.dof_pos,
                self.dof_vel,
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
            ],
            axis=-1,
        )
        clip_obs = 100.0
        self.privileged_obs_buf = torch.clip(privileged_obs_buf, -clip_obs, clip_obs)

    def compute_reward(self):
        self.rew_buf[:] = 0.
        self.extras['rewards'] = {}
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.extras['rewards'][name] = torch.mean(rew).item()

    def update_extras(self, reset_envs_idx):

        # fill extras
        self.extras['episode'] = {}
        for key in self.episode_sums.keys():
            mean_episode_sum = torch.mean(self.episode_sums[key][reset_envs_idx]).item()
            self.extras['episode'][key] = None if math.isnan(mean_episode_sum) else mean_episode_sum
            self.episode_sums[key][reset_envs_idx] = 0.0

        if self.env_cfg['use_timeout']:
            self.extras['time_outs'] = self.time_out_buf

    def get_state(self):
        return self.state_buf, self.episode_length_buf

    def get_observations(self):
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def set_state(self, states, times, envs_idx=None):
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=self.device)
        if len(envs_idx) == 0:
            return

        self.episode_length_buf[envs_idx] = times
        self._set_state(states, envs_idx)

    def _set_state(self, states, envs_idx):
        dof_pos = states[:, :9]
        dof_vel = states[:, 9:18]
        dof_pos = torch.clip(dof_pos, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])

        # reset dofs
        self.dof_pos[envs_idx] = dof_pos
        self.dof_vel[envs_idx] = dof_vel
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=False,
            envs_idx=envs_idx,
        )
        self.robot.set_dofs_velocity(
            velocity=self.dof_vel[envs_idx],
            dofs_idx_local=self.motor_dofs,
            envs_idx=envs_idx,
        )
        self.target_arm_dof_pos[envs_idx] = self.dof_pos[envs_idx, :7]

        # reset buffers
        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.last_last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.reset_buf[envs_idx] = 1

    def resample_commands(self, envs_idx):
        # resample commands
        return

    # ------------ domain randomization ----------------

    def _domain_randomization(self):
        self._randomize_controls()
        self._randomize_rigids()
        self.pos_init_randomization_scale = self.env_cfg['pos_randomization']
        self.rot_init_randomization_scale = self.env_cfg['rot_randomization']
        self.dof_pos_init_randomization_scale = self.env_cfg['dof_pos_randomization']

    def _randomize_rigids(self, env_ids=None):

        if self.eval:
            return

        if env_ids == None:
            env_ids = torch.arange(0, self.num_envs)
        elif len(env_ids) == 0:
            return

        if self.env_cfg['randomize_friction']:
            self._randomize_link_friction(env_ids)

        self.robot.set_dofs_damping([self.env_cfg['dof_damping'],] * self.num_dof, dofs_idx_local=self.motor_dofs)
        self.robot.set_dofs_armature([self.env_cfg['armature'],] * self.num_dof, dofs_idx_local=self.motor_dofs)

    def _randomize_controls(self, env_ids=None):

        if self.eval:
            return

        if env_ids == None:
            env_ids = torch.arange(0, self.num_envs)
        elif len(env_ids) == 0:
            return

        if self.env_cfg['randomize_motor_strength']:
            self._randomize_motor_strength(env_ids)
        if self.env_cfg['randomize_motor_offset']:
            self._randomize_motor_offset(env_ids)
        if self.env_cfg['randomize_kp_scale']:
            self._randomize_kp(env_ids)
        if self.env_cfg['randomize_kd_scale']:
            self._randomize_kd(env_ids)

    def _randomize_link_friction(self, env_ids):

        min_friction, max_friction = self.env_cfg['friction_range']

        solver = self.rigid_solver

        ratios = gs.rand((len(env_ids), 1), dtype=float).repeat(1, solver.n_geoms) \
                 * (max_friction - min_friction) + min_friction
        solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), env_ids)

    def _randomize_base_mass(self, env_ids):

        min_mass, max_mass = self.env_cfg['added_mass_range']
        base_link_id = 1

        added_mass = gs.rand((len(env_ids), 1), dtype=float) \
                        * (max_mass - min_mass) + min_mass

        self.rigid_solver.set_links_mass_shift(added_mass, [base_link_id,], env_ids)

    def _randomize_com_displacement(self, env_ids):

        min_displacement, max_displacement = self.env_cfg['com_displacement_range']
        base_link_id = 1

        com_displacement = torch.rand((len(env_ids), 1, 3), dtype=torch.float, device=self.device) \
                            * (max_displacement - min_displacement) + min_displacement
        # com_displacement[:, :, 0] -= 0.02

        self.rigid_solver.set_links_COM_shift(com_displacement, [base_link_id,], env_ids)

    def _randomize_motor_strength(self, env_ids):

        min_strength, max_strength = self.env_cfg['motor_strength_range']
        self.motor_strengths[env_ids, :] = torch.rand((len(env_ids), 1), dtype=torch.float, device=self.device) \
                                           * (max_strength - min_strength) + min_strength

    def _randomize_motor_offset(self, env_ids):

        min_offset, max_offset = self.env_cfg['motor_offset_range']
        self.motor_offsets[env_ids, :] = torch.rand((len(env_ids), self.num_dof), dtype=torch.float, device=self.device) \
                                         * (max_offset - min_offset) + min_offset

    def _randomize_kp(self, env_ids):

        min_scale, max_scale = self.env_cfg['kp_scale_range']
        kp_scales = torch.rand((len(env_ids), self.num_dof), dtype=torch.float, device=self.device) \
                    * (max_scale - min_scale) + min_scale
        self.batched_p_gains[env_ids, :] = kp_scales * self.p_gains[None, :]

    def _randomize_kd(self, env_ids):

        min_scale, max_scale = self.env_cfg['kd_scale_range']
        kd_scales = torch.rand((len(env_ids), self.num_dof), dtype=torch.float, device=self.device) \
                    * (max_scale - min_scale) + min_scale
        self.batched_d_gains[env_ids, :] = kd_scales * self.d_gains[None, :]

    # ------------ visualization ----------------

    def _draw_debug_vis(self):
        ''' Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        '''
        self.scene.clear_debug_objects()

    def _set_camera(self):
        ''' Set camera position and direction
        '''
        self._floating_camera = self.scene.add_camera(
            pos=np.array([2, 0, 1]),
            lookat=np.array([0, 0, 0]),
            # res=(720, 480),
            fov=40,
            GUI=False,
        )

        self._recording = False
        self._recorded_frames = []

        self.camera_x = self.scene.add_camera(
            pos=np.array([3, 0, 0.5]),
            lookat=np.array([0, 0, 0.5]),
            # res=(720, 480),
            fov=40,
            GUI=False,
        )
        self.camera_y = self.scene.add_camera(
            pos=np.array([0, 3, 0.5]),
            lookat=np.array([0, 0, 0.5]),
            # res=(720, 480),
            fov=40,
            GUI=False,
        )
        self.camera_z = self.scene.add_camera(
            pos=np.array([0, 0, 3]),
            lookat=np.array([0, 0, 0]),
            # res=(720, 480),
            fov=40,
            GUI=False,
        )

    def render_headless(self):
        return self.camera_x.render()[0], self.camera_y.render()[0], self.camera_z.render()[0]

    def _render_headless(self):
        if self._recording and len(self._recorded_frames) < 150:
            # import time
            # start = time.time()
            frame = self._floating_camera.render()[0]
            # end = time.time()
            # print(end-start)
            self._recorded_frames.append(frame)
            # from PIL import Image
            # img = Image.fromarray(np.uint8(frame))
            # img.save('./test.png')
            # print('save')

    def get_recorded_frames(self):
        if len(self._recorded_frames) == 150:
            frames = self._recorded_frames
            self._recorded_frames = []
            self._recording = False
            return frames
        else:
            return None

    def start_recording(self, record_internal=True):
        self._recorded_frames = []
        self._recording = True
        if record_internal:
            self._record_frames = True
        else:
            self._floating_camera.start_recording()

    def stop_recording(self, save_path=None):
        self._recorded_frames = []
        self._recording = False
        if save_path is not None:
            print("fps", int(1 / self.dt))
            self._floating_camera.stop_recording(save_path, fps = int(1 / self.dt))
