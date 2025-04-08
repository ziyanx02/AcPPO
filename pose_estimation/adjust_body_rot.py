import os
import yaml
import argparse
import threading

import numpy as np
import cv2
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
import pickle

from robot_display.display import Display
from api.azure_openai import complete, local_image_to_data_url

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='leap_hand')
parser.add_argument('-n', '--name', type=str, default='default')
parser.add_argument('-c', '--cfg', type=str, default=None)
parser.add_argument('--txt', type=str, default="walk palm parallel to xy-plane")
args = parser.parse_args()

cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/basic.yaml"))
if args.cfg is not None:
    cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.cfg}.yaml"))

from agent import Agent

agent = Agent(f"./cfgs/{args.robot}/basic.yaml")

body_link_id = 1

task = args.txt

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
agent.render_from_xyz()

for _ in range(3):

    # prompt = f"""
    # Your task now is to rotate the robot to complete the task: {task}

    # All robot links are movable except link {body_link_id}. So your task is actually consider the rotation of link {body_link_id}.

    # Determine:
    # - whether further rotation is needed
    # - if yes, which axis should the robot rotate along now and the degree

    # The blue arrow is z axis (upward). The red arrow is x axis (forwward). The gree arrow is y axis (leftward).

    # Here are images from multiple perspectives.
    # """

    prompt = f"""
    Your task now is to rotate the robot to complete the task: {task}

    All robot links are movable except link {body_link_id}. So your task is actually consider the rotation of link {body_link_id}.

    Determine:
    - whether further rotation is needed
    - if yes, which axis should the robot rotate along now and the degree

    At the end of your reply, you should contain your answer as:
    Example1:
    Answer:
    no

    Example2:
    Answer:
    yes
    y

    The blue arrow is z axis (upward). The red arrow is x axis (forwward). The gree arrow is y axis (leftward).

    """

    messages = [
        {"role": "system", "content": "You are an expert in robot kinematics and motion planning."},
        {"role": "user", "content": prompt},
    ]

    for axis in ["x", "y", "z"]:
        image_path = f"./label_{axis}.png"
        messages.append(
            {
                "role": "user",
                "content": [
                            {"type": "text", "text": f"The picture is taken from {axis} axis of the robot, along -{axis} axis."},
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

    prompt = f"""
    Your task now is to judge which rotation is the best for the robot to complete the task: {task}

    Determine:
    - which degree is the best for rotation

    The blue arrow is z axis (upward). The red arrow is x axis (forwward). The gree arrow is y axis (leftward).

    At the end of your reply, you should contain your answer as:
    Example1:
    Answer:
    90

    Example2:
    Answer:
    270

    Here are images for different rotations.
    """

    messages = [
        {"role": "system", "content": "You are an expert in robot kinematics and motion planning."},
        {"role": "user", "content": prompt},
    ]

    total_rotate = 0
    agent.render_from_xyz()
    messages.append(
        {
            "role": "user",
            "content": [
                        {"type": "text", "text": f"The picture is taken from {axis} axis of the robot with 0 degrees rotated."},
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
        agent.render_from_xyz()
        messages.append(
            {
                "role": "user",
                "content": [
                            {"type": "text", "text": f"The picture is taken from {axis} axis of the robot with {total_rotate} degrees rotated."},
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

    messages = [
        {"role": "system", "content": "You are an expert in robot kinematics and motion planning."},
        {"role": "user", "content": prompt},
    ]

    rotate_robot(agent, axis, -45)
    total_rotate = -45
    agent.render_from_xyz()
    messages.append(
        {
            "role": "user",
            "content": [
                        {"type": "text", "text": f"The picture is taken from {axis} axis of the robot with 0 degrees rotated."},
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
        agent.render_from_xyz()
        messages.append(
            {
                "role": "user",
                "content": [
                            {"type": "text", "text": f"The picture is taken from {axis} axis of the robot with {total_rotate} degrees rotated."},
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

input()