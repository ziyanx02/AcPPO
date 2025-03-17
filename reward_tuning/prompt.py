ENVIRONMENT = '''
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

    def _reward_lin_vel(self):
        # Tracking linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / 0.25)

    def _reward_ang_vel(self):
        # Tracking angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / 0.25)

    def _reward_base_height(self):
        # Behavior: tracking base height
        return torch.square(self.base_pos[:, 2] - self.gait_base_height)

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
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)

    def _reward_torques(self):
        # Penalize torques
        return torch.mean(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.mean(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.mean(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=-1)

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
'''

REWARD_CONFIG_TEMPLATE = '''
reward_scales:
    lin_vel: 1.0
    ang_vel: 1.0
    ang_vel_xy: -0.05
    base_height: -5.0
    lin_vel_z: -1.0
    orientation: -5.0
    alive: 10.0

    contact_force: -5.
    feet_height: -100.
    feet_pos: -10.0
    action_smoothness_1: -0.1
    action_smoothness_2: -0.1
'''


CODE_OUTPUT_TIP = f'''
The output should include two components:  
1. **Reward Function Class**  
2. **Reward Scale Configuration**  

**Formatting Requirements:**  
- The reward function class should be provided as a Python code string:
  ```python  
  # Your reward function class here  
  ```  
- The reward scale configuration should be provided as a YAML code string:
  ```yaml  
  # Your reward scale configuration here  
  ```  

**Guidelines:**  
- Use only the environment's predefined variables; any new tensor or variable must be on the same device as the input tensors.  
- Refer to the given `reward_function` and `reward_scale` templates for guidance.  
- The reward function consists of two types of terms:  
  1. **Command Tracking Terms:** Crucial terms directly evaluated by the user to ensure task success.  
  2. **Regularization Terms:** Penalization terms to suppress undesired behaviors, ensuring stable learning.  
- The provided reward scale configuration has been tested in locomotion tasks and can serve as a reference.  

**Best Practices for Writing the Reward Function:**  
1. **Normalization:** Apply transformations (e.g., `torch.exp`) to keep reward values within a fixed range.  
2. **Temperature Parameters:** If applying transformations, introduce a named temperature variable inside the function. Each transformed reward component should have its own temperature variable.  
3. **Type Consistency:** Ensure input variables have correct types (e.g., `float` values should not be defined as `torch.Tensor`).  

**Most Importantly: In no case you can create new variables for your reward function. You can only use the follwing variables of environment:**

{ENVIRONMENT}
'''

INITIAL_SYSTEM = f'''
You are a **reward engineer** designing reward functions for **locomotion reinforcement learning tasks**. Your goal is to create a reward function that optimally guides the agent in learning the locomotion task.  

**Structure of the Reward Function:**  
- **Python Class:** Defines the reward function.  
- **YAML Configuration:** Specifies the scaling factors for different reward terms.  

#### **Example: Reward Function Class**  
```python
{REWARD_CLASS_TEMPLATE}
```

#### **Example: Reward Scale Configuration**  
```yaml
{REWARD_CONFIG_TEMPLATE}
```

**Key Notes:**  
- The reward configuration defines **scaling factors** for different reward terms.  
- The return of each reward function is positive. If you want to penalize it, you should only change its scale to negative.
- Reward scale names must match their corresponding reward function terms.  
- Final reward = **Sum of all reward terms after applying the scale factors**.  

Refer to the following guidelines while writing your reward function:  
{CODE_OUTPUT_TIP}  
'''

ROBOT_DESCRIPTION = '''
A quadruped robot dog with 12 degrees of freedom.
'''

INITIAL_USER = f'''
**Task:** Write a reward function that enables the quadruped robot to walk in a **desired gait**.  
- The **desired gait** is computed automatically within the environment.  
- The **robot description** is: {ROBOT_DESCRIPTION}  
'''

JUDGE_SYSTEM = f'''
You are an evaluator assigned to select the optimal reward parameter set for a robot locomotion task using reinforcement learning. You will be provided with evaluation data containing the follwing metrics:
1. Termination Rate: The count of failures observed during evaluation, where lower values indicate better performance.
2. Other Metrics: Numerical values ranging from 0 to 1, where higher values represent better outcomes.

Your task is to carefully analyze both metrics and determine the reward parameter set that demonstrates the best overall performance. Provide the identification number corresponding to the highest-performing set based on the evaluation data.
'''

JUDGE_USER = f'''
The task is to train a robot to walk while following a desired command (comprising linear and angular velocities) and the desired gait (including the intended foot contact). 
The robot description is {ROBOT_DESCRIPTION}.

Below are the evaluation results for different reward parameters. Please identify and output the index of the best reward parameter set in the following format:
```best
index
```
Important: The index **must** be an integer which matches the index of reward parameters. 
'''

POLICY_FEEDBACK = '''
We trained a RL policy using the provided reward function code and tracked the values of the individual components in the reward function as well as global policy metrics such as episode reward and episode lengths after every {epoch_freq} epochs and the maximum, mean, minimum values encountered:
'''
CODE_FEEDBACK = '''
Please carefully analyze the policy feedback and provide a new, improved reward function that can better solve the task. Some helpful tips for analyzing the policy feedback:
    (1) If the episode length are always much lower than the max episode length {max_episode_length}, which shows the policy can't survive, then you must rewrite the entire reward function
    (2) If the values for a certain reward component are near identical throughout, then this means RL is not able to optimize this component as it is written. You may consider
        (a) Changing its scale or the value of its temperature parameter
        (b) Re-writing the reward component 
        (c) Discarding the reward component
    (3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range
Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.
'''