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
parser.add_argument('--txt', type=str, default="walk with index and middle fingers, palm vertical to xy-plane")
args = parser.parse_args()

cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/basic.yaml"))
if args.cfg is not None:
    cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.cfg}.yaml"))

from agent import Agent

agent = Agent(f"./cfgs/{args.robot}/basic.yaml")

visible_links_id = agent.render_from_xyz()

task = args.txt

prompt = f"""
The robot's structure is segmented into multiple links, each labeled with a unique ID in the given images.
The following link IDs are present in the segmentation data:
{visible_links_id}

The description of the target way of walking is:
{task}

Determine:
- The base is the link that should remain stable while following velocity commands.

Your reply should only contain one number that is the id of the body link

Here are images from multiple perspectives.
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
                        {"type": "text", "text": f"The picture is taken from {axis} axis of the robot."},
                        {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(image_path)},
                        },
                    ],
        }
    )

response = complete(messages)
print(response)
id = int(response)

exit()

# Split the string into lines
lines = response.strip().split('\n')

for line in lines:
    if "base" in line:
        base_id = int(line.split(':')[1].strip())
    # if "feet" in line:
    #     feet_ids = [int(id.strip()) for id in line.split(':')[1].split(',')]

# Print the results
print("Base ID:", base_id)