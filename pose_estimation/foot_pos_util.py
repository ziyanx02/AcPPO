import torch

def foot_pos_by_gait(
        time,               # (n_envs, )
        feet_swing_height,  # (n_envs, )
        vel,                # (n_envs, 3)
        foot_pos,           # (n_feet, 3)
        phase,              # (n_feet)
        duration,           # (n_feet)
    ):
    feet_gait_time = torch.remainder(time + phase, 1.0)
    feet_swing = (feet_gait_time > duration).float()
    feet_scaled_time = 0.5 * (feet_gait_time / duration) * (1 - feet_swing) + (0.5 + 0.5 * (feet_gait_time - duration) / (1 - duration)) * feet_swing

time = 0
feet_swing_height = 0
vel = 0
foot_pos = 0
phase = torch.tensor([0.1, 0.3, 0.6])
duration = torch.tensor([0.2, 0.2, 0.5])

foot_pos_by_gait(time, feet_swing_height, vel, foot_pos, phase, duration)