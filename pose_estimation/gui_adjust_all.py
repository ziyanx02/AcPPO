import os
import yaml
import argparse
import threading

from robot_display.gui_display import GUIDisplay

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='go2')
parser.add_argument('-c', '--cfg', type=str, default=None)
parser.add_argument('-s', '--save', type=str, default=None)
args = parser.parse_args()

cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/basic.yaml"))
if args.cfg is not None:
    cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.cfg}.yaml"))

def save(display):
    cfg["control"]["base_init_pos"] = [round(val, 4) for val in display.robot.target_body_pos.tolist()]
    cfg["control"]["base_init_quat"] = [round(val, 4) for val in display.robot.target_body_quat.tolist()]
    dof_pos = display.robot.target_dof_pos.tolist()
    cfg["control"]["default_joint_angles"] = {display.robot.dof_name[i] : round(dof, 2) for i, dof in enumerate(dof_pos)}
    yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.save}.yaml", "w"))
    print("Save to", f"./cfgs/{args.robot}/{args.save}.yaml")

class VisOptions:
    def __init__(self):
        self.visualize_skeleton = False
        self.visualize_robot_frame = True
        self.visualize_target_foot_pos = False
        self.merge_fixed_links = True
        self.show_world_frame = True
        self.shadow = False
        self.background_color = (0.8, 0.8, 0.8)

'''
For hand-designed pose
'''
display = GUIDisplay(
    cfg=cfg,
    body_pos=True,
    body_pose=True,
    dofs_pos=True,
    pd_control=False,
    save_callable=save,
    vis_options=VisOptions(),
)

def run():
    display.run()
display_thread = threading.Thread(target=run)
display_thread.start()