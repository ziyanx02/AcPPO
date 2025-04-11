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
parser.add_argument('--task', type=str, default="walk with palm parallel to the ground")
args = parser.parse_args()

cfg_path = f"./cfgs/{args.robot}/basic.yaml"
cfg = yaml.safe_load(open(cfg_path))

from agent import Agent

agent = Agent(cfg_path)

visible_links_id, camera_transforms = agent.render_from_xyz(agent.get_body_pos())
visible_links_id, camera_transforms = agent.render()

task = args.task

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": BODY_SELECTION_PROMPT.format(visible_links_id=visible_links_id, task=task)},
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

max_attempts = 5  # avoid infinite loops
rotation_history = []

for attempt in range(max_attempts):
    # Save pre-rotation image
    agent.render_from_xyz(agent.get_body_pos())
    before_images = {}
    for axis in ["x", "y", "z"]:
        before_images[axis] = local_image_to_data_url(f"./label_{axis}.png")

    # Ask VLM to propose a rotation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ROTATION_PROPOSE_PROMPT.format(task=task, body_link_id=body_link_id)},
    ]
    for axis in ["x", "y", "z"]:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": ORIENTATION_PROMPT[axis]},
                {"type": "image_url", "image_url": {"url": before_images[axis]}}
            ]
        })

    response = complete(messages)
    print(response)

    lines = response.split("Answer:")[-1].strip().split("\n")
    if lines[0].strip().lower() == "no":
        break

    axis = lines[1].strip().lower()
    angle = float(lines[2].strip())
    rotation_history.append((axis, angle))

    # Perform rotation
    rotate_robot(agent, axis, angle)
    agent.update()

    # Render new image and get feedback
    agent.render_from_xyz(agent.get_body_pos())
    after_images = {}
    for ax in ["x", "y", "z"]:
        after_images[ax] = local_image_to_data_url(f"./label_{ax}.png")

    # Ask VLM to evaluate the rotation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ROTATION_EVALUATE_PROMPT.format(task=task, axis=axis, angle=angle)},
    ]
    for ax in ["x", "y", "z"]:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Before rotation. " + ORIENTATION_PROMPT[ax]},
                {"type": "image_url", "image_url": {"url": before_images[ax]}},
                {"type": "text", "text": "After rotation. " + ORIENTATION_PROMPT[ax]},
                {"type": "image_url", "image_url": {"url": after_images[ax]}}
            ]
        })

    response = complete(messages)
    print(response)
    answer = response.split("Answer:")[-1].strip().lower()

    if "cancel" in answer:
        # Undo last rotation
        print(f"Cancelling last rotation: -{angle} along {axis}")
        rotate_robot(agent, axis, -angle)
        agent.update()
    elif "done" in answer:
        break
    else:
        continue

links = agent.display.links

cfg["base_link_name"] = links[body_link_id].name
cfg["body_init_quat"] = agent.get_body_quat().numpy().tolist()
cfg["base_init_quat"] = agent.get_link_quat(0).numpy().tolist()
cfg["task"] = args.task
yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_pose.yaml", "w"))
input("finished")
