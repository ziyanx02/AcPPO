import os
import yaml
import argparse

from robot_display.gui_display import GUIDisplay

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, default='go2')
args = parser.parse_args()

cfg = yaml.safe_load(open(f"./cfgs/{args.name}.yaml"))
interact = GUIDisplay(cfg)
interact.run()