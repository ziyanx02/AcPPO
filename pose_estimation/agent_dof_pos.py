import os
import yaml
import argparse
import ast

import numpy as np
import torch
import cv2
from PIL import Image

from robot_display.display import Display
from api.azure_openai import complete, local_image_to_data_url
from agent import Agent

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='leap_hand')
parser.add_argument('-n', '--name', type=str, default='default')
args = parser.parse_args()

cfg_path = f"./cfgs/{args.robot}/{args.name}_pose.yaml"
cfg = yaml.safe_load(open(cfg_path))

agent = Agent(cfg_path)
task = cfg["task"]

body_link_id = agent.display.get_link_by_name(cfg["base_link_name"]).idx_local
agent.set_body_link(body_link_id)
agent.set_body_quat(cfg["body_init_quat"])
agent.update()

visible_links, _ = agent.render_from_xyz(agent.get_body_pos())

extremities = [link.idx_local for link in agent.display.links if link.is_leaf]
extremities = list(set(extremities) & set(visible_links))

agent.render()
diameter = agent.display.get_diameter()


def min_projected_distance(p1, p2):
    d_xy = np.linalg.norm(p1[[0, 1]] - p2[[0, 1]])
    d_xz = np.linalg.norm(p1[[0, 2]] - p2[[0, 2]])
    d_yz = np.linalg.norm(p1[[1, 2]] - p2[[1, 2]])
    return min(d_xy, d_xz, d_yz)

for link_id in extremities:
    agent.render()
    # _, camera_transforms = agent.render_from_xyz(agent.get_link_pos(link_id))
    camera_transforms, axis = agent.render_link(link_id)
    camera_transforms = [camera_transforms[axis[i]] for i in range(3)]
    mean = agent.get_link_pos(link_id)
    sigma = torch.ones(3, dtype=torch.float) * diameter / 8
    circle_radius = 20

    sampled_points = []
    for _ in range(300):
        point = (mean + torch.randn(3) * sigma).numpy()
        if any(min_projected_distance(point, sp) < diameter / 20 for sp in sampled_points):
            continue
        if min_projected_distance(point, agent.get_link_pos(link_id).numpy()) < diameter / 15:
            continue
        if len(sampled_points) == 15:
            break
        if agent.try_set_link_pose(link_id, point):
            sampled_points.append(point)

    for i in range(3):
        img = cv2.imread(f"rgb_{axis[i]}.png")
        extrinsics = camera_transforms[i]["extrinsics"]
        intrinsics = camera_transforms[i]["intrinsics"]
        center = agent.get_link_pos(link_id).numpy()
        center = (intrinsics @ ((extrinsics @ np.append(center, 1.0))[:3] / (extrinsics @ np.append(center, 1.0))[2]))[:2].astype(int)

        for j, point in enumerate(sampled_points):
            pixel = (intrinsics @ ((extrinsics @ np.append(point, 1.0))[:3] / (extrinsics @ np.append(point, 1.0))[2]))[:2].astype(int)
            direction = pixel - center
            length = np.linalg.norm(direction)
            if length == 0: continue
            direction = direction / length
            arrow_tip = (pixel - direction * circle_radius).astype(int)
            cv2.arrowedLine(img, tuple(center), tuple(arrow_tip), (255, 0, 0), 2, tipLength=0.1)

        for j, point in enumerate(sampled_points):
            pixel = (intrinsics @ ((extrinsics @ np.append(point, 1.0))[:3] / (extrinsics @ np.append(point, 1.0))[2]))[:2].astype(int)
            overlay = img.copy()
            cv2.circle(overlay, tuple(pixel), circle_radius, (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            cv2.circle(img, tuple(pixel), circle_radius, (255, 0, 0), 2)
            text_size, _ = cv2.getTextSize(str(j), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = pixel[0] - text_size[0] // 2
            text_y = pixel[1] + text_size[1] // 2
            cv2.putText(img, str(j), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imwrite(f"./annotated_{axis[i]}.png", img)

    prompt = f"""
    You are selecting the best target position for link {link_id} to complete the task: {task}.
    Evaluate all three images taken from different axis views (x, y, z). Select **one** ID from 0 to {len(sampled_points) - 1} that appears best in all views.

    Use this format (add \n after Answer:):
    Answer:
    chosen_id
    """

    messages = [
        {"role": "system", "content": "You are an expert in robot kinematics and motion planning."},
        {"role": "user", "content": prompt},
    ]

    for orientation in axis:
        image_path = f"./annotated_{orientation}.png"
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"View from -{orientation} axis."},
                {"type": "image_url", "image_url": {"url": local_image_to_data_url(image_path)}},
            ],
        })

    response = complete(messages)
    chosen_id = int(response.strip().split("Answer:")[-1].strip())

    agent.set_link_pose(link_id, torch.tensor(sampled_points[chosen_id]))
    agent.update()
    agent.render_from_xyz(agent.get_body_pos())

    # Ask if we're done
    check_prompt = f"""
    The robot has updated pose with link {link_id} set to the selected position for task: {task}.
    Please respond with one of: continue / cancel / done
    """

    messages = [
        {"role": "system", "content": "You are an expert in robot kinematics and motion planning."},
        {"role": "user", "content": check_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Updated pose."},
                {"type": "image_url", "image_url": {"url": local_image_to_data_url("./updated_render.png")}},
            ]
        }
    ]
    decision = complete(messages).strip().lower()
    if decision == "done":
        break
    elif decision == "cancel":
        continue

# Finish and save cfg
cfg["extremity ids"] = extremities
cfg["diameter"] = agent.display.get_diameter().item()
cfg["mass"] = float(agent.display.entity.get_mass())
cfg["base_init_quat"] = agent.get_link_quat(0).numpy().tolist()

if "dof_names" not in cfg:
    cfg["dof_names"] = agent.display.dof_name

dof_pos = agent.display.dof_pos
joint_name_to_dof_order = agent.display.joint_name_to_dof_order
cfg["default_joint_angles"] = {
    joint_name: dof_pos[joint_name_to_dof_order[joint_name]].item()
    for joint_name in cfg["dof_names"]
}

yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_dof_pos.yaml", "w"))
input("Finished.")
