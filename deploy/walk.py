import numpy as np
import time
import torch
import yaml

from utils.low_state_controller import LowStateCmdHandler
from transforms3d import quaternions

if __name__ == '__main__':

    cfg = yaml.safe_load(open(f"./ckpts/walk.yaml"))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    handler = LowStateCmdHandler(cfg)
    handler.init()
    handler.start()

    policy = torch.jit.load("./ckpts/walk.pt")
    policy.to(device)
    policy.eval()

    default_dof_pos = np.array([0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5])
    commands = np.array([0., 0., 0., 0., 2., 0.5, 0., 0., 0., 0.3, 0.05, 0., 0., 0.3, 0.4])
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
            obs = np.concatenate(
                [
                    np.array(handler.ang_vel) * 0.25,
                    projected_gravity,
                    commands[:3] * 2.0 * 0.3,
                    np.array(handler.joint_pos) - default_dof_pos,
                    np.array(handler.joint_vel) * 0.05,
                    last_action,
                ]
            )
            action = policy(torch.tensor(obs).to(device).float()).cpu().detach().numpy()
            last_action = action
            handler.target_pos = default_dof_pos + action * 0.25 * 1.0
            step_id += 1
    except:
        pass
    
    handler.recover()
