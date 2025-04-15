ENVIRONMENT_TEMPLATE = '''
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
self.terminate_buf = torch.ones(
    (self.num_envs,), device=self.device, dtype=gs.tc_int
)
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

self.penalized_contact_link_indices = find_link_indices(
    self.env_cfg['penalized_contact_link_names']
)
self.feet_link_indices = find_link_indices(
    self.env_cfg['feet_link_names'],
    accurate=True,
)

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
# gait control
self.foot_positions = torch.ones(
    self.num_envs, len(self.feet_link_indices), 3, device=self.device, dtype=gs.tc_float,
)
self.foot_quaternions = torch.ones(
    self.num_envs, len(self.feet_link_indices), 4, device=self.device, dtype=gs.tc_float,
)
self.foot_velocities = torch.ones(
    self.num_envs, len(self.feet_link_indices), 3, device=self.device, dtype=gs.tc_float,
)
self.com = torch.zeros(
    self.num_envs, 3, device=self.device, dtype=gs.tc_float,
)

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
'''


REWARD_CLASS_TEMPLATE = '''
class RewardWrapper:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # Command tracking terms
    # These rewards encourage the robot to follow desired commands (linear and angular velocity) and maintain desired gait behavior.

    def _reward_lin_vel(self):
        """
        Reward for tracking linear velocity commands (x, y axes).
        - Computes the squared error between the desired linear velocity (commands[:, :2]) and the actual body linear velocity (body_lin_vel[:, :2]).
        - Applies an exponential decay to the error to encourage smooth tracking.
        - Higher reward for smaller errors.
        """
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.body_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / 0.25)

    def _reward_ang_vel(self):
        """
        Reward for tracking angular velocity commands (yaw axis).
        - Computes the squared error between the desired angular velocity (commands[:, 2]) and the actual body angular velocity (body_ang_vel[:, 2]).
        - Applies an exponential decay to the error to encourage smooth tracking.
        - Higher reward for smaller errors.
        """
        ang_vel_error = torch.square(self.commands[:, 2] - self.body_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / 0.25)

    def _reward_base_height(self):
        """
        Reward for maintaining the desired body height.
        - Computes the squared error between the current body height (body_pos[:, 2]) and the desired gait body height (gait_body_height).
        - Encourages the robot to maintain a stable body height during locomotion.
        """
        return torch.square(self.body_pos[:, 2] - self.gait_body_height)

    def _reward_contact_force(self):
        """
        Reward for tracking desired contact forces on the feet.
        - Computes the norm of contact forces on the feet (link_contact_forces) and compares them to the desired contact states (desired_contact_states).
        - Encourages feet to exert force when in contact and minimize force when not in contact.
        - Uses an exponential decay to penalize large forces when contact is not desired.
        """
        foot_forces = torch.norm(self.link_contact_forces[:, self.feet_link_indices, :], dim=-1)
        desired_contact = self.desired_contact_states
        return torch.mean((1 - desired_contact) * (1 - torch.exp(-foot_forces ** 2 / 100.)), dim=-1)
    
    def _reward_feet_height(self):
        """
        Reward for tracking desired feet height.
        - Computes the squared error between the current feet height (feet_pos_local[..., 2]) and the desired feet height (desired_feet_pos_local[..., 2]).
        - Only applies the reward when the feet are not in contact (1 - desired_contact_states).
        - Encourages the robot to lift its feet to the correct height during swing phases.
        """
        rew_foot_height = torch.square(self.feet_pos_local[..., 2] - self.desired_feet_pos_local[..., 2]) * (1 - self.desired_contact_states)
        return torch.mean(rew_foot_height, dim=-1)
    
    def _reward_feet_pos(self):
        """
        Reward for tracking desired feet position (x, y axes).
        - Computes the squared error between the current feet position (feet_pos_local[..., 0:2]) and the desired feet position (desired_feet_pos_local[..., 0:2]).
        - Encourages the robot to place its feet accurately in the x-y plane.
        """
        rew_foot_pos = torch.sum(torch.square(self.feet_pos_local[..., 0:2] - self.desired_feet_pos_local[..., 0:2]), dim=-1)
        return torch.mean(rew_foot_pos, dim=-1)

    def _reward_alive(self):
        """
        Reward for staying alive (not terminating).
        - Returns 1 if the robot is alive (terminate_buf is False), otherwise 0.
        - Encourages the robot to avoid conditions that lead to termination (e.g., falling).
        """
        return 1 - self.terminate_buf.float()

    def _reward_terminate(self):
        """
        Penalty for termination.
        - Returns 1 if the robot has terminated (terminate_buf is True), otherwise 0.
        - Discourages behaviors that lead to termination.
        """
        return self.terminate_buf.float()

    # Regularization terms
    # These rewards penalize undesirable behaviors to encourage smooth, stable, and efficient locomotion.

    def _reward_collision(self):
        """
        Penalty for undesired collisions on specific body links.
        - Penalizes contact forces on links that shouldn't touch the ground in stable poses.
        - Especially important when intermediate poses might involve brief contact—this reward discourages such transient but undesirable behaviors.
        """
        return torch.sum(torch.norm(self.link_contact_forces[:, self.penalized_contact_link_indices, :], dim=-1,) > 0.1, dim=1)

    def _reward_lin_vel_z(self):
        """
        Penalty for body linear velocity in the z-axis.
        - Computes the squared z-axis linear velocity (body_lin_vel[:, 2]).
        - Discourages unnecessary vertical motion of the body.
        """
        return torch.square(self.body_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        """
        Penalty for body angular velocity in the x and y axes.
        - Computes the squared angular velocity in the x and y axes (body_ang_vel[:, :2]).
        - Discourages unnecessary tilting or rolling of the body.
        """
        return torch.sum(torch.square(self.body_ang_vel[:, :2]), dim=-1)

    def _reward_orientation(self):
        """
        Penalty for non-flat body orientation.
        - Computes the squared deviation of the projected gravity vector from the vertical axis (body_projected_gravity[:, :2]).
        - Encourages the robot to maintain a flat body orientation.
        """
        return torch.sum(torch.square(self.body_projected_gravity[:, :2]), dim=-1)

    def _reward_torques(self):
        """
        Penalty for high joint torques.
        - Computes the squared torques applied to the joints (torques).
        - Encourages energy-efficient locomotion by minimizing joint torques.
        """
        return torch.mean(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        """
        Penalty for high joint velocities.
        - Computes the squared velocities of the degrees of freedom (dof_vel).
        - Encourages smooth and controlled joint movements.
        """
        return torch.mean(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_acc(self):
        """
        Penalty for high joint accelerations.
        - Computes the squared accelerations of the degrees of freedom (dof_vel changes over time).
        - Encourages smooth transitions in joint movements.
        """
        return torch.mean(torch.square((self.last_dof_vel - self.dof_vel)), dim=-1)

    def _reward_dof_pos_diff(self):
        """
        Penalty for deviations from the default joint positions.
        - Computes the squared difference between current joint positions (dof_pos) and default joint positions (default_dof_pos).
        - Encourages the robot to maintain a natural pose.
        """
        return torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=-1)

    def _reward_contact_vel(self):
        """
        Penalty for high vertical velocities during contact.
        - Computes the norm of foot velocities (foot_velocities) and penalizes large vertical velocities when feet are in contact (desired_contact_states).
        - Encourages stable foot placement during contact phases.
        """
        foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.num_envs, -1)
        desired_contact = self.desired_contact_states
        return torch.mean(desired_contact * (1 - torch.exp(-foot_velocities ** 2 / 10.)), dim=-1)

    def _reward_action_smoothness_1(self):
        """
        Penalty for 1st-order action deviations.
        - Computes the squared difference between current actions (actions) and previous actions (last_actions).
        - Encourages smooth transitions between actions.
        """
        diff = torch.square(self.actions - self.last_actions)
        diff = diff * (self.last_actions != 0)  # ignore first step
        return torch.mean(diff, dim=-1)

    def _reward_action_smoothness_2(self):
        """
        Penalty for 2nd-order action deviations.
        - Computes the squared difference between current actions (actions), previous actions (last_actions), and actions before that (last_last_actions).
        - Encourages even smoother transitions between actions.
        """
        diff = torch.square(self.actions - 2 * self.last_actions + self.last_last_actions)
        diff = diff * (self.last_actions != 0) * (self.last_last_actions != 0)  # ignore first & second steps
        return torch.mean(diff, dim=-1)

    def _reward_dof_pos_limits(self):
        """
        Penalty for dof positions too close to the limit 
        - Compute the difference between current dof positions and dof pos limits.
        - Encourage proper dof position to avoid extreme pose.
        - Critical for deploying in real.
        """
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)  # upper limit
        return torch.sum(out_of_limits, dim=1) # >=0
'''

REWARD_CONFIG_TEMPLATE = '''
reward_scales:
    lin_vel: 1.0
    ang_vel: 0.5
    alive: 10.0
    ang_vel_xy: -0.05
    base_height: -30.0
    dof_acc: -0.05
    dof_vel: -0.01
    dof_pos_diff: -0.1
    lin_vel_z: -2.0
    orientation: -5.0
    contact_force: -5.
    contact_vel: -5.
    feet_height: -400.
    feet_pos: -30.0
    action_smoothness_1: -0.1
    action_smoothness_2: -0.1
    torques: -0.0001
    collision: -1.0
    dof_pos_limits: -10.0
'''

METRIC_FUNCTION_TEMPLATE = '''
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
'''

TASK_NOTE_TEMPLATE = '''
A quadruped robot dog with 12 degrees of freedom.
'''