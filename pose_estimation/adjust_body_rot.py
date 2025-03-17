import os
import yaml
import argparse
import threading

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
    cfg["robot"]["default_body_quat"] = [round(val, 4) for val in display.robot.target_body_quat.tolist()]
    yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_body_rot.yaml", "w"))
    print("Save to", f"./cfgs/{args.robot}/{args.name}_body_rot.yaml")

class VisOptions:
    def __init__(self):
        self.visualize_skeleton = True
        self.visualize_robot_frame = True
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

log_dir = f"./cfgs/{args.robot}/body_rot"
os.makedirs(log_dir, exist_ok=True)

solved = False
while not solved:
    display.step_target()
    display.step_vis()
    robot, _, _, skeleton = display.render(rgb=True, depth=False, segmentation=True)
    axes = ["x", "y", "z", "-x", "-y", "-z"]
    for i in range(len(robot)):
        image = robot[i].copy()[:, :, ::-1]
        cv2.imwrite(os.path.join(log_dir, f"robot_{axes[i]}.png"), image)
        image = skeleton[i].copy()[:, :, ::-1]
        cv2.imwrite(os.path.join(log_dir, f"skeleton_{axes[i]}.png"), image)
    raise NotImplementedError