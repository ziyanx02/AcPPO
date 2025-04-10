import os
import yaml
import argparse
import threading
import ast

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

cfg_path = f"./cfgs/{args.robot}/{args.name}_pose.yaml"
cfg = yaml.safe_load(open(cfg_path))

task = cfg["task"]

from agent import Agent

agent = Agent(cfg_path)

body_link_id = agent.display.get_link_by_name(cfg["base_link_name"]).idx_local
agent.set_body_link(body_link_id)
agent.set_body_quat(cfg["body_init_quat"])
agent.update()

def world_to_pixel(point_world, extrinsics, intrinsics):
    # Convert 3D point to homogeneous
    point_homog = np.append(point_world, 1.0)

    # Transform to camera coordinates
    point_camera = extrinsics @ point_homog

    # Drop points behind the camera
    if point_camera[2] <= 0:
        return None  # or np.nan, or mark as invalid

    # Project to normalized image plane
    point_norm = point_camera[:3] / point_camera[2]

    # Apply intrinsic matrix to get pixel coordinates
    pixel = intrinsics @ point_norm

    # Return x, y pixel coordinates
    return pixel[:2]

agent.set_body_link(body_link_id)
agent.update()

visible_links, _ = agent.render_from_xyz(agent.get_body_pos())

extremities = []
for link in agent.display.links:
    if link.is_leaf:
        extremities.append(link.idx_local)
extremities = list(set(extremities) & set(visible_links))

agent.render()

diameter = agent.display.get_diameter()

def min_projected_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    assert p1.shape == (3,) and p2.shape == (3,), "Inputs must be shape (3,)"

    # Project to xy, xz, yz planes and compute distances
    d_xy = np.linalg.norm(p1[[0, 1]] - p2[[0, 1]])
    d_xz = np.linalg.norm(p1[[0, 2]] - p2[[0, 2]])
    d_yz = np.linalg.norm(p1[[1, 2]] - p2[[1, 2]])

    return min(d_xy, d_xz, d_yz)

num_iters = 2

for link_id in extremities:
    agent.render()
    _, camera_transforms = agent.render_from_xyz(agent.get_link_pos(link_id))
    mean = agent.get_link_pos(link_id)
    sigma = torch.ones(3, dtype=torch.float) * diameter / 8
    axis = ["x", "y", "z"]
    circle_radius = 20

    for iter in range(num_iters):
        sampled_points = []
        for sample in range(1000):
            if sample == 0:
                point = (mean + 0 * sigma).numpy()
            else:
                point = (mean + torch.randn(3) * sigma).numpy()
            skip_this_point = False
            if min_projected_distance(point, agent.get_link_pos(link_id).numpy()) < diameter / 15:
                skip_this_point = True
            for sampled_point in sampled_points:
                if min_projected_distance(point, sampled_point) < diameter / 15:
                    skip_this_point = True
            if skip_this_point:
                continue
            if len(sampled_points) == 15:
                break
            if agent.try_set_link_pose(link_id, point):
                sampled_points.append(point)
            else:
                continue

        for i in range(3):
            img = cv2.imread(f"rgb_{axis[i]}.png")
            extrinsics = camera_transforms[i]["extrinsics"]
            intrinsics = camera_transforms[i]["intrinsics"]
            center = world_to_pixel(agent.get_link_pos(link_id).numpy(), extrinsics, intrinsics).astype(int)
            for j in range(len(sampled_points)):
                point = sampled_points[j]
                pixel_coord = world_to_pixel(point, extrinsics, intrinsics).astype(int)

                # Vector from arrow root to circle center
                direction = np.array(pixel_coord) - np.array(center)
                length = np.linalg.norm(direction)

                if length == 0:
                    raise ValueError("Arrow root and circle center cannot be the same")

                # Normalize and compute arrow tip (on the edge of the circle)
                direction = direction / length
                arrow_tip = (np.array(pixel_coord) - direction * circle_radius).astype(int)

                # Draw arrow
                cv2.arrowedLine(
                    img,
                    tuple(center.tolist()),
                    tuple(arrow_tip.tolist()),
                    (255.0, 128.0, 128.0),
                    2,
                    tipLength=0.1
                )

            for j in range(len(sampled_points)):
                point = sampled_points[j]
                pixel_coord = world_to_pixel(point, extrinsics, intrinsics).astype(int)

                # Vector from arrow root to circle center
                direction = np.array(pixel_coord) - np.array(center)
                length = np.linalg.norm(direction)

                if length == 0:
                    raise ValueError("Arrow root and circle center cannot be the same")

                # Normalize and compute arrow tip (on the edge of the circle)
                direction = direction / length
                arrow_tip = (np.array(pixel_coord) - direction * circle_radius).astype(int)

                # --- Semi-transparent white background circle ---
                overlay = img.copy()
                bg_radius = circle_radius
                cv2.circle(overlay, tuple(pixel_coord.tolist()), bg_radius, (255, 255, 255), -1)
                # Blend the overlay onto the original image
                alpha = 0.5
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

                # Draw circle
                cv2.circle(img, tuple(pixel_coord.tolist()), circle_radius, (255.0, 128.0, 128.0), 2)

                # Draw number at center
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                text_size, _ = cv2.getTextSize(str(j), font, font_scale, 2)
                text_x = pixel_coord[0] - text_size[0] // 2
                text_y = pixel_coord[1] + text_size[1] // 2
                cv2.putText(img, str(j), (text_x, text_y), font, font_scale, (0.0, 0.0, 0.0), 2)

            # Save or display image
            output_path = f"./annotated_{axis[i]}.png"
            cv2.imwrite(output_path, img)

        selected_ids = []
        prompt = f"""
Your task is find a standing pose of the robot to complete the task: {task}
Now the link {body_link_id} is adjusted and fixed. So the rotation of the robot is not your task now.
You are moving link {link_id} by inverse kinametics (as link {body_link_id} is fixed so only joint between link {body_link_id} and link {link_id} will change during ik).
The potential positions are labeled by ids in the annotated images, including: {[i for i in range(len(sampled_points))]}

First analysis the front / back, left / right in each image.
Then analysis which way you should move the link. And for the image taken from each axis, find out reasonable movements. Do not consider the connections between different views.

At the end of your reply, you should contain the ids you think are reasonable in this format (remamber to add \\n after Answer:):
Example1:
Answer:
x:
1, 3, 9, 14
y:
2, 3
z:
3, 5, 7

Example2:
Answer:
x:
3, 7
y:
2, 7, 9, 15
z:
5, 8

The blue arrow is z axis (upward). The red arrow is x axis (forwward). The gree arrow is y axis (leftward).
Here are the images:
"""

        messages = [
            {"role": "system", "content": "You are an expert in robot kinematics and motion planning."},
            {"role": "user", "content": prompt},
        ]

        messages.append(
            {
                "role": "user",
                "content": [
                            {"type": "text", "text": f"Links are labeled in this image."},
                            {
                                "type": "image_url",
                                "image_url": {"url": local_image_to_data_url("./label.png")},
                            },
                        ],
            }
        )

        for orientation in ["x", "y", "z"]:
            image_path = f"./annotated_{orientation}.png"
            messages.append(
                {
                    "role": "user",
                    "content": [
                                {"type": "text", "text": f"The picture is taken from {orientation} axis of the link, along -{orientation} axis."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": local_image_to_data_url(image_path)},
                                },
                            ],
                }
            )
        # if iter == num_iters - 1:
        #     messages.append({"role": "user", "content": "This time you should only choose one id, which means you have to consider both views. The reply format should not be changed."})
        response = complete(messages)
        print(response)
        lines = response.split("\n")
        selected_x = []
        selected_y = []
        selected_z = []
        for line_id in range(len(lines)):
            if "Answer:" in lines[line_id]:
                lines = lines[line_id + 1:]
                break
        for line_id in range(len(lines)):
                if "x:" in lines[line_id]:
                    selected_x = [int(num) for num in lines[line_id + 1].split(", ")]
                if "y:" in lines[line_id]:
                    selected_y = [int(num) for num in lines[line_id + 1].split(", ")]
                if "z:" in lines[line_id]:
                    selected_z = [int(num) for num in lines[line_id + 1].split(", ")]
        selected_for_x = selected_y + selected_z
        selected_for_y = selected_x + selected_z
        selected_for_z = selected_x + selected_y
        x_dist = np.array([sampled_points[i][0] for i in selected_for_x])
        y_dist = np.array([sampled_points[i][1] for i in selected_for_y])
        z_dist = np.array([sampled_points[i][2] for i in selected_for_z])
        mean[0] = np.mean(x_dist).item()
        mean[1] = np.mean(y_dist).item()
        mean[2] = np.mean(z_dist).item()
        sigma[0] = np.std(x_dist).item()
        sigma[1] = np.std(x_dist).item()
        sigma[2] = np.std(x_dist).item()

    agent.set_link_pose(link_id, mean)
    agent.update()

extremities_pos = []
for extremity in extremities:
    extremities_pos.append(agent.get_link_pos(extremity).numpy().tolist())

extremities_pos_as_prompt = ""
for row in extremities_pos:
    formatted_row = [f"{x:.3f}" for x in row]
    extremities_pos_as_prompt += ("[" + ", ".join(formatted_row) + "]\n")

agent.render_from_xyz(agent.get_body_pos())

com = agent.display.entity.solver.get_links_COM([0,]).squeeze().numpy().tolist()
com = [f"{x:.3f}" for x in com]

prompt = f"""
Your task is to adjust the robot pose to better complete the task: {task}

The robot link {body_link_id} is fixed. You should move exrtemities {extremities} to fit the task and comply symmetry and some commensense (i.e. the height of feet should be the same, when there are only two feet then the line should across center of mass). The current positions of the extremities are (the order is same as previous extremites' order):
{extremities_pos_as_prompt}

For those foot links (not all extremities are feet, decide which are), try to keep the mean height unchanged. (which is the average of z axis of the selected points).

18, 20 are the front legs

x axis is forward, y axis is leftward, z axis is upward.

The current center of mass is {com}.

First you have to analysis what rule you should comply. Then from the current positions, find out reasonable target positions.

At the end of your reply, you should contain the target position in this format (same order as the input) after "Answer:":
Example1:
Answer:
[0.15, 0.2, -0.3]
[0.15, -0.2, -0.3]
[-0.15, 0.2, -0.3]
[-0.15, -0.2, -0.3]

Example2:
Answer:
[0.0, -0.2, -0.4]
[0.0, 0.2, -0.4]
[0.2, 0.2, -0.0]
[-0.15, -0.2, -0.0]

Here are the images:
"""

messages = [
    {"role": "system", "content": "You are an expert in robot kinematics and motion planning."},
    {"role": "user", "content": prompt},
]

messages.append(
    {
        "role": "user",
        "content": [
                    {"type": "text", "text": f"Links are labeled in this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": local_image_to_data_url("./label.png")},
                    },
                ],
    }
)

for orientation in ["x", "y", "z"]:
    image_path = f"./label_{orientation}.png"
    messages.append(
        {
            "role": "user",
            "content": [
                        {"type": "text", "text": f"The picture is taken from {orientation} axis of the link, along -{orientation} axis."},
                        {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(image_path)},
                        },
                    ],
        }
    )

response = complete(messages)
print(response)
while True:
    lines = response.strip().splitlines()
    for i in range(len(lines)):
        if "Answer:" in lines[i]:
            response = lines[i+1:]

    # Split into lines, strip whitespace, and convert each line from str to list
    data = [ast.literal_eval(line.strip()) for line in response]

    for i in range(len(extremities)):
        agent.set_link_pos(extremities[i], torch.tensor(data[i]))
    agent.update()
    agent.render_from_xyz(agent.get_body_pos())

    messages.append({"role": "user", "content": "This is the robot pose after applying your last reply."})
    for orientation in ["x", "y", "z"]:
        image_path = f"./label_{orientation}.png"
        messages.append(
            {
                "role": "user",
                "content": [
                            {"type": "text", "text": f"The picture is taken from {orientation} axis of the link, along -{orientation} axis."},
                            {
                                "type": "image_url",
                                "image_url": {"url": local_image_to_data_url(image_path)},
                            },
                        ],
            }
        )
    
    # Ask if we're done
    check_prompt = f"""
Do you think further adjustment is needed? Please respond with one of: yes / no
If yes, please provide the new target positions in the same format as before.
Example 1:
no

Example 2:
yes
Reason why the previous answer is not good.
Answer:
[0.0, -0.2, -0.4]
[0.0, 0.2, -0.4]
[0.2, 0.2, -0.0]
[-0.15, -0.2, -0.0]
"""
    messages.append({"role": "user", "content": check_prompt})
    response = complete(messages)
    lines = response.strip().splitlines()
    if "no" in lines[0].strip().lower():
        break

cfg["extremity ids"] = extremities
cfg["diameter"] = agent.display.get_diameter().item()
cfg["diameter"] = float(agent.display.entity.get_mass())
cfg["base_init_quat"] = agent.get_link_quat(0).numpy().tolist()
if "dof_names" not in cfg:
    cfg["dof_names"] = agent.display.dof_name
dof_pos = agent.display.dof_pos
joint_name_to_dof_order = agent.display.joint_name_to_dof_order
cfg["default_joint_angles"] = {}
for joint_name in cfg["dof_names"]:
    cfg["default_joint_angles"][joint_name] = dof_pos[joint_name_to_dof_order[joint_name]].item()
yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_dof_pos.yaml", "w"))
input("finished")