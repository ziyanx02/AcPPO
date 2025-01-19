import numpy as np
import time
import torch
import yaml

from utils.low_state_controller import LowStateCmdHandler
from transforms3d import quaternions

def get_clock_inputs(t, commands):
    t = t - int(t / 1000) * 1000
    frequencies = commands[4]
    phases = commands[5]
    offsets = commands[6]
    bounds = commands[7]

    gait_indices = t * frequencies - int(t * frequencies)
    foot_indices = torch.tensor([phases + offsets + bounds, offsets, bounds, phases]) + gait_indices

    clock_inputs = torch.sin(2 * np.pi * foot_indices)

    return clock_inputs.numpy()

if __name__ == '__main__':

    cfg = yaml.safe_load(open(f"go2.yaml"))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    handler = LowStateCmdHandler(cfg)
    handler.init()
    handler.start()

    while not handler.Start:
        time.sleep(0.1)

    default_dof_pos = np.array([0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5])
    commands = np.array([0., 0., 0., 0., 2., 0.5, 0., 0., 0., 0.3, 0.05, 0., 0., 0.3, 0.4])
    last_action = np.array([0.0] * 12)

    last_update_time = time.time()

    policy = torch.jit.load("./ckpt4000.pt")
    policy.to(device)
    policy.eval()

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
        clock_inputs = get_clock_inputs(step_id * 0.02, commands)
        commands[0] = handler.Ly
        commands[1] = -handler.Lx
        commands[2] = -handler.Rx
        obs = np.concatenate(
            [
                np.array(handler.ang_vel) * 0.25,
                projected_gravity,
                commands[:2] * 2.0 * 0.3,
                commands[2:3] * 0.25,
                np.array(handler.joint_pos) - default_dof_pos,
                np.array(handler.joint_vel) * 0.05,
                last_action,
                commands[4:],
                clock_inputs,
            ]
        )
        action = policy(torch.tensor(obs).to(device).float()).cpu().detach().numpy()
        last_action = action
        handler.target_pos = default_dof_pos + action * 0.25
        step_id += 1
