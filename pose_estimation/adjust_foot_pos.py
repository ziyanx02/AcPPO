import os
import yaml
import argparse
import threading

from robot_display.gui_display import GUIDisplay

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, default='go2')
parser.add_argument('-b', '--body', type=str, default='base')
args = parser.parse_args()

cfg = yaml.safe_load(open(f"./cfgs/{args.name}/basic.yaml"))
display = GUIDisplay(
    cfg=cfg,
    body_pos=True,
    body_pose=True,
    dofs_pos=False,
    foot_pos=True,
)
display.run()