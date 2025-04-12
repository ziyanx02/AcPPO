import numpy as np
import time
import torch
import yaml

from utils.low_state_controller import LowStateCmdHandler
from transforms3d import quaternions

cfg_path = "g1-walk.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

base_init_quat = torch.tensor(cfg["environment"]["base_init_quat"])
frequency = torch.tensor(cfg["environment"]["gait"]["frequency"])
offset = torch.tensor(cfg["environment"]["gait"]["offset"])

def get_clock_input(t):
    gait_indices = torch.remainder(t * frequency, 1.0)
    foot_indices = torch.remainder(gait_indices + offset, 1.0)
    clock_inputs = torch.sin(2 * np.pi * foot_indices) 
    return clock_inputs.numpy()

def gs_transform_by_quat(pos, quat):
    qw, qx, qy, qz = quat.unbind(-1)

    rot_matrix = torch.stack(
        [
            1.0 - 2 * qy**2 - 2 * qz**2,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx**2 - 2 * qz**2,
            2 * qy * qz - 2 * qx * qw,
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx**2 - 2 * qy**2,
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)
    rotated_pos = torch.matmul(rot_matrix, pos.unsqueeze(-1)).squeeze(-1)

    return rotated_pos

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg["robot_name"] = "g1"

    handler = LowStateCmdHandler(cfg)
    handler.init()
    handler.start()

    policy = torch.jit.load("./ckpts/g1.pt")
    policy.to(device)
    policy.eval()

    default_dof_pos = handler.default_pos
    reset_dof_pos = handler.reset_pos.copy()
    commands = np.array([0., 0., 0.,])
    last_action = np.array([0.0] * 12)

    try:
        while not handler.Start:
            time.sleep(0.1)

        print("Start runing policy")
        last_update_time = time.time()

        step_id = 0
        while not handler.emergency_stop:
            if time.time() - last_update_time < 0.02:
                time.sleep(0.001)
                continue
            last_update_time = time.time()
            projected_gravity = quaternions.rotate_vector(
                v=np.array([0, 0, -1]),
                q=quaternions.qinverse(handler.quat),
            )
            commands[0] = handler.Ly
            commands[1] = -handler.Lx
            commands[2] = -handler.Rx
            clock_input = get_clock_input(step_id * 0.02)
            obs = np.concatenate(
                [
                    np.array(handler.ang_vel) * cfg["environment"]["observation"]["obs_scales"]["ang_vel"],
                    projected_gravity,
                    commands[:3] * 1.0,
                    (np.array(handler.joint_pos) - default_dof_pos) * cfg["environment"]["observation"]["obs_scales"]["dof_pos"],
                    np.array(handler.joint_vel) * cfg["environment"]["observation"]["obs_scales"]["dof_vel"],
                    last_action,
                    clock_input,
                ]
            )
            action = policy(torch.tensor(obs).to(device).float()).cpu().detach().numpy()
            last_action = action
            # action[[5, 11]] = 0
            handler.target_pos = reset_dof_pos + (default_dof_pos + action * cfg["environment"]["action_scale"] - reset_dof_pos) * 1.0
            step_id += 1
            print(time.time() - last_update_time)
    except KeyboardInterrupt:
        pass

    handler.recover()