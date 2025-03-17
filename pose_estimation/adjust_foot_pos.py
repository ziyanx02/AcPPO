import os
import yaml
import argparse
import threading
import time

import cv2
from robot_display.display import Display

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
    cfg["control"]["default_dof_pos"] = {}
    default_dof_pos = [round(val, 2) for val in display.robot.dof_pos.numpy().tolist()]
    for i in range(display.robot.num_dofs):
        cfg["control"]["default_dof_pos"][display.robot.dof_name[i]] = default_dof_pos[i]
    yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_foot_pos.yaml", "w"))
    print("Save to", f"./cfgs/{args.robot}/{args.name}_foot_pos.yaml")

class VisOptions:
    def __init__(self):
        self.visualize_skeleton = True
        self.visualize_target_foot_pos = False
        self.merge_fixed_links = True
        self.show_world_frame = False
        self.shadow = False
        self.background_color = (0.8, 0.8, 0.8)
        self.show_viewer = False

display = Display(
    cfg=cfg,
    vis_options=VisOptions(),
)

log_dir = f"./cfgs/{args.robot}/foot_pos"
os.makedirs(log_dir, exist_ok=True)

solved = False
while not solved:
    display.step_target()
    for i in range(len(display.foot_links)):
        time.sleep(0.1)
        link = display.foot_links[i]
        display.step_vis()
        display.visualize_frame(link.get_pos())
        robot, _, _, skeleton = display.render(rgb=True, depth=False, segmentation=True)
        axes = ["x", "y", "z", "-x", "-y", "-z"]
        for j in range(len(robot)):
            image = robot[j].copy()[:, :, ::-1]
            cv2.imwrite(os.path.join(log_dir, f"robot_{i}_{axes[j]}.png"), image)
            image = skeleton[j].copy()[:, :, ::-1]
            cv2.imwrite(os.path.join(log_dir, f"skeleton_{i}_{axes[j]}.png"), image)
    raise NotImplementedError