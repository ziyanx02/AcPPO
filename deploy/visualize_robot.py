import argparse
import yaml
import time

from robot_display.display import Display
from utils.low_state_handler import LowStateMsgHandler

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='go2')
parser.add_argument('-c', '--cfg', type=str, default=None)
args = parser.parse_args()

cfg = yaml.safe_load(open(f"../pose_estimation/cfgs/{args.robot}/basic.yaml"))
if args.cfg is not None:
    cfg = yaml.safe_load(open(f"../pose_estimation/cfgs/{args.robot}/{args.cfg}.yaml"))

robot = Display(cfg)

cfg_path = "go2-handstand-walk-llm-ground.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

low_state_handler = LowStateMsgHandler(cfg)
low_state_handler.init()

while True:
    time.sleep(0.1)
    robot.set_body_quat(low_state_handler.quat)
    robot.set_dofs_position(low_state_handler.joint_pos)
    robot.update()