RESPONSE_SAMPLE_REWARD = '''\
Here is the reward function class and the YAML configuration for enabling the quadruped robot to walk in a desired gait.

### Reward Function Class

```python
class RewardWrapper:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _reward_desired_gait(self):
        """
        Reward for following a pre-defined desired gait pattern.
        - This compares the current foot contact state and position with the desired contact states and foot positions in the gait.
        """
        contact_reward = torch.mean(
            self.desired_contact_states * (1 - torch.exp(-torch.sum(self.link_contact_forces[:, self.feet_link_indices, :]**2, dim=-1) / 50.)), dim=-1
        )
        
        foot_position_reward = torch.mean(
            torch.sum(torch.square(self.feet_pos_local - self.desired_feet_pos_local), dim=-1), dim=-1
        )
        
        total_reward = contact_reward + torch.exp(-foot_position_reward/100.)
        
        return total_reward

    def _reward_gait_smoothness(self):
        """
        Reward for smooth gait transitions.
        - This penalizes abrupt changes in the desired contact states.
        """
        gait_smoothness_penalty = torch.mean(
            torch.square(self.desired_contact_states - self.gait_indices), dim=-1
        )
        return torch.exp(-gait_smoothness_penalty / 0.01)
    
    def _reward_base_height(self):
        """
        Penalize deviation of the base height from the desired gait base height.
        """
        base_height_penalty = torch.square(self.base_pos[:, 2] - self.gait_base_height)
        return torch.exp(-base_height_penalty / 0.05)

    def _reward_alive(self):
        """
        Reward for staying alive (not being terminated).
        """
        return 1 - self.terminate_buf.float()

    def _reward_terminate(self):
        """
        Penalize termination.
        """
        return self.terminate_buf.float()

    def _reward_orientation(self):
        """
        Penalize non-flat orientations of the base.
        """
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)

    # Regularization terms
    def _reward_torques(self):
        """
        Penalize excessive torques applied to the joints.
        """
        return torch.mean(torch.square(self.torques), dim=-1)

    def _reward_action_smoothness_1(self):
        """
        Penalize first-order deviations in actions (smooth action transitions).
        """
        diff = torch.square(self.actions - self.last_actions)
        diff = diff * (self.last_actions != 0)  # ignore first step
        return torch.mean(diff, dim=-1)

    def _reward_action_smoothness_2(self):
        """
        Penalize second-order deviations in actions (smooth action transitions).
        """
        diff = torch.square(self.actions - 2 * self.last_actions + self.last_last_actions)
        diff = diff * (self.last_actions != 0) * (self.last_last_actions != 0)  # ignore first & second step
        return torch.mean(diff, dim=-1)
```

### YAML Configuration for Reward Scales

```yaml
reward_scales:
    desired_gait: 10.0
    gait_smoothness: 1.0
    base_height: -5.0
    alive: 5.0
    terminate: -1.0
    orientation: -3.0
    torques: -0.1
    action_smoothness_1: -0.05
    action_smoothness_2: -0.05
```

### Explanation of Reward Components:

1. **_reward_desired_gait**:  
   - Provides a large reward when the quadruped closely follows the desired gait pattern.  
   - Combines foot contact state and foot position tracking with desired states to create a total gait reward.

2. **_reward_gait_smoothness**:  
   - Smooth gait transitions are crucial for stable locomotion. This term penalizes abrupt changes in contact patterns, promoting smoother transitions.

3. **_reward_base_height**:  
   - Penalizes deviation from the desired base height computed for the current gait. Helps maintain consistent locomotion dynamics.

4. **_reward_alive**:  
   - Encourages staying alive and avoids termination.

5. **_reward_orientation**:  
   - Penalizes deviations from a flat body orientation.

6. **Regularization terms**:  
   - Smooth action transitions (`_reward_action_smoothness_1`, `_reward_action_smoothness_2`).  
   - Penalizes excessive torques applied to the joints (`_reward_torques`).  

This reward function and configuration aim to balance locomotion performance (desired gait tracking) with smoothness, stability, and control efficiency.
'''