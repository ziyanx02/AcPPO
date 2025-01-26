import argparse
import yaml
import time

from robot_display.display import Display
from robot_display.utils.gui import start_gui

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='go2')
parser.add_argument('-n', '--name', type=str, default='default')
parser.add_argument('-c', '--cfg', type=str, default=None)
args = parser.parse_args()

cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.name}_body_pos.yaml"))
if args.cfg is not None:
    cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.cfg}.yaml"))

robot = Display(cfg)

labels = []
ranges = {}
labels.append("lin_vel_x")
ranges["lin_vel_x"] = [- robot.diameter * 2, robot.diameter * 2]
labels.append("lin_vel_y")
ranges["lin_vel_y"] = [- robot.diameter * 2, robot.diameter * 2]
labels.append("ang_vel_z")
ranges["ang_vel_z"] = [- 2.0, 2.0]
labels.append("feet swing height")
ranges["feet swing height"] = [0.0, robot.diameter / 2]
for foot in cfg["robot"]["foot_names"]:
    labels.append(f"{foot} phase")
    ranges[f"{foot} phase"] = [0.0, 1.0]
    labels.append(f"{foot} duration")
    ranges[f"{foot} duration"] = [0.0, 1.0]
gui_cfg = {"label": labels, "range": ranges}
values = [0.0] * (4 + 2 * len(cfg["robot"]["foot_names"]))
start_gui(cfg=gui_cfg, values=values)

while True:
    time.sleep(1)
    print(values)
    pass