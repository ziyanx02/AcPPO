import os
import yaml
import argparse
import threading

import torch
import cv2
from robot_display.display import Display
from api.azure_openai import complete, local_image_to_data_url

def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))

def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='go2')
parser.add_argument('-n', '--name', type=str, default='default')
parser.add_argument('-c', '--cfg', type=str, default=None)
args = parser.parse_args()

cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.name}_body_name.yaml"))
if args.cfg is not None:
    cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.cfg}.yaml"))

def save(display):
    cfg["robot"]["default_body_quat"] = [round(val, 4) for val in display.robot.target_body_quat.tolist()]
    yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_body_rot.yaml", "w"))
    print("Save to", f"./cfgs/{args.robot}/{args.name}_body_rot.yaml")

class VisOptions:
    def __init__(self):
        self.visualize_skeleton = True
        self.visualize_robot_frame = True
        self.visualize_target_foot_pos = False
        self.merge_fixed_links = True
        self.show_world_frame = False
        self.shadow = False
        self.background_color = (0.8, 0.8, 0.8)
        self.show_viewer = False

display = Display(
    cfg=cfg,
    vis_options=VisOptions(),
)

log_dir = f"./cfgs/{args.robot}/body_rot"
os.makedirs(log_dir, exist_ok=True)

use_robot = True
use_skeleton = True
solved = False

axis_to_index = {
    "x": torch.tensor([1.0, 0.0, 0.0]),
    "y": torch.tensor([0.0, 1.0, 0.0]),
    "z": torch.tensor([0.0, 0.0, 1.0]),
}

while not solved:

    display.step_target()
    display.step_vis()
    axes = ["x", "y", "z", "-x", "-y", "-z"]
    views = {
        "x": "front",
        "y": "left",
        "z": "top",
        "-x": "back",
        "-y": "right",
        "-z": "bottom",
    }
    robot, _, _, skeleton = display.render(rgb=True, depth=False, segmentation=True)
    for i in range(len(robot)):
        image = robot[i].copy()[:, :, ::-1]
        cv2.imwrite(os.path.join(log_dir, f"robot_{axes[i]}_prev.png"), image)
        image = skeleton[i].copy()[:, :, ::-1]
        cv2.imwrite(os.path.join(log_dir, f"skeleton_{axes[i]}_prev.png"), image)

    requirement = "walk with index and middle fingers like a human."

    prompt = f"""
    The robot is displayed with world frame rendered.
    The red arrow is x-axis or forward.
    The green arrow is y-axis or leftward.
    The blue arrow is z-axis or upward.

    If images of skeleton are provided, cyan represents the legs and orange represents the base (which link should stay stable while walking) of the robot.

    The description of the target way of walking is:
    {requirement}

    You are going to adjust robot's orientation to make the it suitable for a pose that could start walking after adjusting the joint positions.

    Determine:
    - Whether the current orientation is good enough.
    - If not, how to rotate the robot.

    In your reply, you should contain 1. whether the rotation is good enough and 2. if not, the first step of rotate the robot to a better pose.
    The reply should be in the following format:
    Example 1:
    Yes

    Example 2:
    No
    y 90

    Example 3:
    No
    x -180

    Here are images from multiple perspectives.
    """

    messages = [
        {"role": "system", "content": "You are an expert in robot kinematics and motion planning."},
        {"role": "user", "content": prompt},
    ]

    if use_robot:
        for axis in axes:
            image_path = os.path.join(log_dir, f"robot_{axis}_prev.png")
            messages.append(
                {
                    "role": "user",
                    "content": [
                                {"type": "text", "text": f"The picture is taken from {views[axis]} of the robot."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": local_image_to_data_url(image_path)},
                                },
                            ],
                }
            )

    if use_skeleton:
        for axis in axes:
            image_path = os.path.join(log_dir, f"skeleton_{axis}_prev.png")
            messages.append(
                {
                    "role": "user",
                    "content": [
                                {"type": "text", "text": f"The picture is taken from {views[axis]} of the skeleton."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": local_image_to_data_url(image_path)},
                                },
                            ],
                }
            )

    response = complete(messages)
    print(response)

    # Split the string into lines
    lines = response.strip().split('\n')

    if "yes" in lines[0]:
        break

    axis, degree = lines[1].strip().split(' ')
    axis = axis_to_index[axis]
    degree = torch.tensor([torch.pi * int(degree) / 180,])

    curr_quat = display.target_body_quat
    rot = quat_from_angle_axis(degree, axis).squeeze()

    post_quat = quat_mul(rot, curr_quat)
    display.set_body_quat(post_quat)

    display.step_target()
    display.step_vis()
    robot, _, _, skeleton = display.render(rgb=True, depth=False, segmentation=True)
    for i in range(len(robot)):
        image = robot[i].copy()[:, :, ::-1]
        cv2.imwrite(os.path.join(log_dir, f"robot_{axes[i]}_post.png"), image)
        image = skeleton[i].copy()[:, :, ::-1]
        cv2.imwrite(os.path.join(log_dir, f"skeleton_{axes[i]}_post.png"), image)

    input()