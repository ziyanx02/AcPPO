import os
import yaml
import argparse
import threading

from robot_display.gui_display import GUIDisplay

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='go2')
parser.add_argument('-n', '--name', type=str, default='default')
parser.add_argument('-c', '--cfg', type=str, default=None)
args = parser.parse_args()

cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.name}_body_name.yaml"))
if args.cfg is not None:
    cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.cfg}.yaml"))

def save(display):
    cfg["robot"]["foot_pos"] = {}
    for i in range(len(display.robot.foot_links)):
        link = display.robot.foot_links[i]
        pos = display.values[display.value_foot_pos_idx_start + 3 * i:display.value_foot_pos_idx_start + 3 * (i + 1)]
        pos = [round(val, 2) for val in pos]
        cfg["robot"]["foot_pos"][link.name] = pos
    cfg["robot"]["default_body_pos"] = [round(val, 2) for val in display.robot.target_body_pos.tolist()]
    cfg["robot"]["default_body_quat"] = [round(val, 4) for val in display.robot.target_body_quat.tolist()]
    cfg["control"]["default_dof_pos"] = {}
    default_dof_pos = [round(val, 2) for val in display.robot.dof_pos.numpy().tolist()]
    for i in range(display.robot.num_dofs):
        cfg["control"]["default_dof_pos"][display.robot.dof_name[i]] = default_dof_pos[i]
    yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_body_pos.yaml", "w"))
    print("Save to", f"./cfgs/{args.robot}/{args.name}_body_pos.yaml")

display = GUIDisplay(
    cfg=cfg,
    body_pos=True,
    body_pose=True,
    dofs_pos=False,
    foot_pos=True,
    save_callable=save,
)
display.run()