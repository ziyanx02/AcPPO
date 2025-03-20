import os
import yaml
import argparse
import threading

import cv2
from robot_display.display import Display
from api.azure_openai import complete, local_image_to_data_url

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

display.step_target()
display.step_vis()
robot, _, _, skeleton = display.render(rgb=True, depth=False, segmentation=True)
axes = ["x", "y", "z", "-x", "-y", "-z"]
views = {
    "x": "front",
    "y": "left",
    "z": "top",
    "-x": "back",
    "-y": "right",
    "-z": "bottom",
}
for i in range(len(robot)):
    image = robot[i].copy()[:, :, ::-1]
    cv2.imwrite(os.path.join(log_dir, f"robot_{axes[i]}.png"), image)
    image = skeleton[i].copy()[:, :, ::-1]
    cv2.imwrite(os.path.join(log_dir, f"skeleton_{axes[i]}.png"), image)

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

Here are images from multiple perspectives.
"""

messages = [
    {"role": "system", "content": "You are an expert in robot kinematics and motion planning."},
    {"role": "user", "content": prompt},
]

axes = ["x", "y", "z",]

# for axis in axes:
#     image_path = os.path.join(log_dir, f"robot_{axis}.png")
#     messages.append(
#         {
#             "role": "user",
#             "content": [
#                         {"type": "text", "text": f"The picture is taken from {views[axis]} of the robot."},
#                         {
#                             "type": "image_url",
#                             "image_url": {"url": local_image_to_data_url(image_path)},
#                         },
#                     ],
#         }
#     )

for axis in axes:
    image_path = os.path.join(log_dir, f"skeleton_{axis}.png")
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
