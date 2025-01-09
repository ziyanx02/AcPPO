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

cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/basic.yaml"))
if args.cfg is not None:
    cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.cfg}.yaml"))
body_name = "base"
foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
links_to_keep = foot_names + [body_name]
cfg["robot"]["body_name"] = body_name
cfg["robot"]["foot_names"] = foot_names
cfg["robot"]["links_to_keep"] = links_to_keep
yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_body_name.yaml", "w"))