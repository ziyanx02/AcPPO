import os
import yaml
import argparse
import threading

import numpy as np
import torch
import cv2
from PIL import Image
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
import pickle

from robot_display.display import Display
from api.azure_openai import complete, local_image_to_data_url

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='leap_hand')
parser.add_argument('-n', '--name', type=str, default='default')
args = parser.parse_args()

cfg_path = f"./cfgs/{args.robot}/{args.name}_dof_pos.yaml"
cfg = yaml.safe_load(open(cfg_path))

task = cfg["task"]

from agent import Agent

agent = Agent(cfg_path)

body_link_id = agent.display.get_link_by_name(cfg["base_link_name"]).idx_local
agent.set_body_link(body_link_id)
agent.set_body_quat(cfg["body_init_quat"])
extremities = cfg["extremity ids"]
dof_pos = agent.display.dof_pos
joint_name_to_dof_order = agent.display.joint_name_to_dof_order
for joint_name in cfg["dof_names"]:
    dof_pos[joint_name_to_dof_order[joint_name]] = cfg["default_joint_angles"][joint_name]
agent.display.set_dofs_position(dof_pos)
agent.update()

agent.render_from_xyz(agent.get_body_pos())

prompt = f"""
Your task is find a gait suitable for the robot to complete the task: {task}
First you should pick foot links from {extremities}. Typically foot links are the lowest along z axis.
Then for each foot, you should design three parameters: duration, phase and frequency.
duration is in range [0, 1], describe in a cycle the proportion of the time that the feet touch the ground. This is typically 0.5 for regular gaits.
phase is in range [0, 1], describe in a cycle when the feet touch the ground. A foot with phase 0.7 means this foot touch the at 0.7 of the cycle.
frequency is how many times the foot should touch the gound in a second. Typically different foot should have the same frequency.

At the end of your reply, you should contain the ids you think are reasonable in this format (remamber to add \\n after Answer:):
Example1:
Answer:
feet:
18, 20, 22, 24
duration:
0.5, 0.5, 0.5, 0.5
phase:
0.5, 0.0, 0.5, 0.0
frequency:
3, 3, 3, 3

Example2:
Answer:
feet:
18, 20
duration:
0.5, 0.5
phase:
0.5, 0.0
frequency:
4, 4

The blue arrow is z axis (up). The red arrow is x axis (front, forward). The gree arrow is y axis (left).
Here are the images:
"""

messages = [
    {"role": "system", "content": "You are an expert in robot kinematics and motion planning."},
    {"role": "user", "content": prompt},
]

for orientation in ["y", "z"]:
    image_path = f"./label_{orientation}.png"
    if orientation == "x":
        prompt = f"The picture is taken from front of the robot. In this picture, left is robot's right."
    if orientation == "y":
        prompt = f"The picture is taken from left side of the link. In this picture, left is robot's front."
    if orientation == "z":
        prompt = f"The picture is taken from up of the link. In this picture, up is robot's front."
    messages.append(
        {
            "role": "user",
            "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(image_path)},
                        },
                    ],
        }
    )
response = complete(messages)
print(response)

lines = response.split("\n")
for line_id in range(len(lines)):
    if "Answer:" in lines[line_id]:
        lines = lines[line_id + 1:]
        break
for line_id in range(len(lines)):
        if "feet:" in lines[line_id]:
            feet = [int(num.strip()) for num in lines[line_id + 1].split(", ")]
        if "duration:" in lines[line_id]:
            duration = [float(num.strip()) for num in lines[line_id + 1].split(", ")]
        if "phase:" in lines[line_id]:
            phase = [float(num.strip()) for num in lines[line_id + 1].split(", ")]
        if "frequency:" in lines[line_id]:
            frequency = [float(num.strip()) for num in lines[line_id + 1].split(", ")]


feet_pos = [agent.get_link_pos(link_id) for link_id in feet]
body_pos = agent.get_body_pos()
base_height_target = (body_pos[2] - torch.mean(torch.cat([foot_pos.unsqueeze(0) for foot_pos in feet_pos])[:, 2])).item()

feet_link_names = [agent.display.links[foot_id].name for foot_id in feet]
stationary_position = [foot_pos[:2].numpy().tolist() for foot_pos in feet_pos]
feet_height_target = [0.2 * base_height_target for _ in feet_link_names]

gait_cfg = {}
gait_cfg["base_height_target"] = base_height_target
gait_cfg["frequency"] = frequency
gait_cfg["duration"] = duration
gait_cfg["offset"] = phase
gait_cfg["stationary_position"] = stationary_position
cfg["gait"] = gait_cfg
cfg["feet_link_names"] = feet_link_names
cfg["base_init_pos"] = [0.0, 0.0, 1.2 * (agent.get_link_pos(0)[2].item() - agent.display.entity.get_AABB()[0, 2].item())]
yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_gait.yaml", "w"))
print(f"./cfgs/{args.robot}/{args.name}_gait.yaml", "w")