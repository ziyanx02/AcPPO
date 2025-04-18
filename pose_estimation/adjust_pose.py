import os
import yaml
import argparse
import threading

import numpy as np
import cv2
import pickle

from robot_display.display import Display
from prompts.prompts import *
from api.azure_openai import complete, local_image_to_data_url

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='leap_hand')
parser.add_argument('-n', '--name', type=str, default='default')
parser.add_argument('--task', type=str, default="walk with index and middle fingers, stand like a human")
args = parser.parse_args()

cfg_path = f"./cfgs/{args.robot}/basic.yaml"
cfg = yaml.safe_load(open(cfg_path))

from agent import Agent

agent = Agent(cfg_path)

visible_links_id, camera_transforms = agent.render_from_xyz(agent.get_body_pos())

task = args.task

prompt = f"""
The robot in the images is divided into multiple parts (called links). Each link is highlighted with a unique color and labeled with an ID.

Visible link IDs in the images:
{visible_links_id}

The robot's target walking behavior is described as:
{task}

Your task:
- Identify the base link. The base is the part of the robot that should stay stable while the robot moves according to velocity commands.

Instructions:
- Respond with **only one number** — the ID of the base link.

You are given images of the robot from different views.
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": prompt},
]

for axis in ["x", "y", "z"]:
    image_path = f"./label_{axis}.png"
    messages.append(
        {
            "role": "user",
            "content": [
                        {"type": "text", "text": ORIENTATION_PROMPT[axis]},
                        {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(image_path)},
                        },
                    ],
        }
    )

response = complete(messages)
print(response)
body_link_id = int(response)

def rotate_robot(agent, axis, angle):
    if axis == "x":
        agent.rotate_along_x(angle)
    if axis == "y":
        agent.rotate_along_y(angle)
    if axis == "z":
        agent.rotate_along_z(angle)

agent.set_body_link(body_link_id)
agent.update()

agent.render()
agent.render_from_xyz(agent.get_body_pos())

rotated_axis = []
for _ in range(3):

    prompt = f"""
    Your task is to help rotate the robot so it can complete the following task: {task}

    All robot links are movable in the following steps **except** for link {body_link_id}, which should be treated as the fixed base. So, your job is to consider how link {body_link_id} should be rotated to align the robot for the task.
    You should first think about which links you want to move by rotating, and how will they change their pose after rotating along different axis,

    Please determine:
    - Whether any further rotation is needed.
    - If yes, specify which axis the robot should rotate around, and by how many degrees.

    Format your answer like one of the following examples:

    Example 1:
    (Your analysis)
    Answer:
    no

    Example 2:
    (Your analysis)
    Answer:
    yes
    y
    90

    Example 3:
    (Your analysis)
    Answer:
    yes
    x
    45
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for axis in ["x", "y", "z"]:
        image_path = f"./label_{axis}.png"
        messages.append(
            {
                "role": "user",
                "content": [
                            {"type": "text", "text": ORIENTATION_PROMPT[axis]},
                            {
                                "type": "image_url",
                                "image_url": {"url": local_image_to_data_url(image_path)},
                            },
                        ],
            }
        )

    response = complete(messages)
    print(response)
    response = response.split("Answer:")[1]
    lines = response.split("\n")
    no_further_rotation = False
    axis = "y"
    for line in lines:
        if "yes" in line:
            continue
        elif "x" in line:
            axis = "x"
        elif "y" in line:
            axis = "y"
        elif "z" in line:
            axis = "z"
        elif "no" in line:
            no_further_rotation = True
    if no_further_rotation:
        break
    if axis in rotated_axis:
        continue
    rotated_axis.append(axis)

    prompt = f"""
    Your task now is to judge which rotation is the best for the robot to complete the task: {task}

    Determine:
    - which degree is the best for rotation

    At the end of your reply, you should contain your answer as:
    Example 1:
    Answer:
    0

    Example 2:
    Answer:
    90

    Example 3:
    Answer:
    180

    Example 4:
    Answer:
    270

    Here are images for different rotations.
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    total_rotate = 0
    agent.render_from_xyz(agent.get_body_pos())
    messages.append(
        {
            "role": "user",
            "content": [
                        {"type": "text", "text": ORIENTATION_PROMPT[axis] + " The robot has been rotated for 0 degrees."},
                        {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(f"./label_{axis}.png")},
                        },
                    ],
        }
    )

    for _ in range(3):
        rotate_robot(agent, axis, 90)
        total_rotate += 90
        agent.update()
        agent.render_from_xyz(agent.get_body_pos())
        messages.append(
            {
                "role": "user",
                "content": [
                            {"type": "text", "text": ORIENTATION_PROMPT[axis] + f" The robot has been rotated for {total_rotate} degrees."},
                            {
                                "type": "image_url",
                                "image_url": {"url": local_image_to_data_url(f"./label_{axis}.png")},
                            },
                        ],
            }
        )

    rotate_robot(agent, axis, 90)
    total_rotate += 90
    agent.update()

    response = complete(messages)
    print(response)
    lines = response.split("\n")
    response = "0"
    for i in range(len(lines)):
        if "Answer:" in lines[i]:
            response = lines[i+1]

    rotate_robot(agent, axis, int(response))
    agent.update()

    prompt = f"""
    Your task now is to judge which rotation is the best for the robot to complete the task: {task}

    Determine:
    - which degree is the best for rotation

    At the end of your reply, you should contain your answer as:
    Example 1:
    Answer:
    -45

    Example 2:
    Answer:
    0

    Example 3:
    Answer:
    45

    Here are images for different rotations.
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    rotate_robot(agent, axis, -45)
    total_rotate = -45
    agent.render_from_xyz(agent.get_body_pos())
    messages.append(
        {
            "role": "user",
            "content": [
                        {"type": "text", "text": ORIENTATION_PROMPT[axis] + f" The robot has been rotated for {total_rotate} degrees."},
                        {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(f"./label_{axis}.png")},
                        },
                    ],
        }
    )

    for _ in range(2):
        rotate_robot(agent, axis, 45)
        total_rotate += 45
        agent.render_from_xyz(agent.get_body_pos())
        messages.append(
            {
                "role": "user",
                "content": [
                            {"type": "text", "text": ORIENTATION_PROMPT[axis] + f" The robot has been rotated for {total_rotate} degrees."},
                            {
                                "type": "image_url",
                                "image_url": {"url": local_image_to_data_url(f"./label_{axis}.png")},
                            },
                        ],
            }
        )

    rotate_robot(agent, axis, -45)
    agent.update()

    response = complete(messages)
    print(response)
    lines = response.split("\n")
    response = "0"
    for i in range(len(lines)):
        if "Answer:" in lines[i]:
            response = lines[i+1]

    rotate_robot(agent, axis, int(response))
    agent.update()

links = agent.display.links

cfg["base_link_name"] = links[body_link_id].name
cfg["body_init_quat"] = agent.get_body_quat().numpy().tolist()
cfg["base_init_quat"] = agent.get_link_quat(0).numpy().tolist()
cfg["task"] = args.task
yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_pose.yaml", "w"))
input("finished")