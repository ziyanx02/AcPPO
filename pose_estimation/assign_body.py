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
parser.add_argument('-r', '--robot', type=str, default='go2')
parser.add_argument('-n', '--name', type=str, default='default')
parser.add_argument('-c', '--cfg', type=str, default=None)
args = parser.parse_args()

cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/basic.yaml"))
if args.cfg is not None:
    cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.cfg}.yaml"))

class VisOptions:
    def __init__(self):
        self.visualize_skeleton = False
        self.visualize_target_foot_pos = False
        self.merge_fixed_links = False
        self.show_world_frame = False
        self.shadow = False
        self.background_color = (0.8, 0.8, 0.8)
        self.show_viewer = False

display = Display(
    cfg=cfg,
    vis_options=VisOptions(),
)

display.step_target()
display.step_vis()
rgbs, _, segs, _ = display.render(rgb=True, depth=False, segmentation=True)  # Random color

log_dir = f"./cfgs/{args.robot}/segmentation"
os.makedirs(log_dir, exist_ok=True)

alpha = 0.5  # Transparency factor for highlighting segments
def darken_color(color, factor=0.5):
    """Darken the given color by a factor."""
    return tuple(int(c * factor) for c in color)

colors = []
for _ in range(display.entity.n_links):
    colors.append(np.random.randint(0, 256, 3).tolist())
visible_links_id = np.array([-1])

axes = ["x", "y", "z", "-x", "-y", "-z"]
views = {
    "x": "front",
    "y": "left",
    "z": "top",
    "-x": "back",
    "-y": "right",
    "-z": "bottom",
}
for i in range(len(rgbs)):
    image = rgbs[i].copy()[:, :, ::-1]
    segmentation_label = segs[i].copy()

    # Create a color map for each segment ID
    unique_labels = np.unique(segmentation_label)
    visible_links_id = np.unique(np.concatenate([visible_links_id, unique_labels]))
    color_map = {}
    for label_id in unique_labels:
        if label_id == -1:
            color_map[label_id] = (0, 0, 0)  # Black for background
        else:
            color_map[label_id] = tuple(colors[label_id])

    # Create an output image with highlighted segments
    highlighted_image = np.zeros_like(image, dtype=np.float32)
    for label_id in unique_labels:
        if label_id == -1:
            mask = (segmentation_label == label_id)
            highlighted_image[mask] = image[mask]
            continue
        mask = (segmentation_label == label_id)
        highlight_color = np.array(color_map[label_id], dtype=np.float32)
        # Blend the highlight color with the original image
        highlighted_image[mask] = alpha * highlight_color + (1 - alpha) * image[mask]

    # Convert the highlighted image back to uint8
    highlighted_image = np.clip(highlighted_image, 0, 255).astype(np.uint8)

    # Draw borders between segments
    for label_id in unique_labels:
        if label_id == -1:
            continue  # Skip background
        mask = (segmentation_label == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        border_color = darken_color(color_map[label_id])  # Darken the highlight color for the border
        cv2.drawContours(highlighted_image, contours, -1, border_color, 1)  # Draw borders

    # Place segment IDs on the image
    for label_id in unique_labels:
        if label_id == -1:
            continue  # Skip background
        mask = (segmentation_label == label_id)
        if np.any(mask):
            # Calculate the centroid of the segment
            centroid = center_of_mass(mask)
            y, x = map(int, centroid)

            # Ensure the centroid lies within the segment
            if not mask[y, x]:
                # Find the closest pixel in the segment to the centroid
                y, x = np.argwhere(mask)[np.linalg.norm(np.argwhere(mask) - centroid, axis=1).argmin()]

            # Add a black square under the number for better visibility
            text = str(label_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_w, text_h = text_size

            # Coordinates for the black square
            square_top_left = (x - text_w // 2 - 2, y - text_h // 2 - 2)
            square_bottom_right = (x + text_w // 2 + 2, y + text_h // 2 + 2)

            # Draw the black square
            cv2.rectangle(highlighted_image, square_top_left, square_bottom_right, (0, 0, 0), -1)  # -1 fills the rectangle

            # Put the segment ID as text on the image
            cv2.putText(highlighted_image, text, (x - text_w // 2, y + text_h // 2), font, font_scale, (255, 255, 255), thickness)

    cv2.imwrite(os.path.join(log_dir, f"{axes[i]}.png"), highlighted_image)

visible_links_id = visible_links_id[visible_links_id != -1]
links_name = []
for i in range(len(display.links)):
    links_name.append(display.links[i].name)

requirement = "crawl like a dog."

prompt = f"""
The robot's structure is segmented into multiple links, each labeled with a unique ID in the given images.
The following link IDs are present in the segmentation data:
{visible_links_id.tolist()}

These links are extremeties:
[14, 15, 16, 17]

The description of the target way of walking is:
{requirement}

The robot's orientation is adjustable, so the position of each links in the picture and the view direction of each picture do not matter.
Analyze the robot morphology (i.e. which links are extremities) and their utility when the robot is walking.
Determine:
- The **base** is the link that should remain stable while following velocity commands.
- The **feet** are the links that make contact with the ground, typically at the robot's extremities.
- The **legs** are the sequence of links from the base to each foot (base link excluded).

Here are images from multiple perspectives.
"""

messages = [
    {"role": "system", "content": "You are an expert in robot kinematics and motion planning."},
    {"role": "user", "content": prompt},
]

for axis in axes:
    image_path = os.path.join(log_dir, f"{axis}.png")
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

response = complete(messages)
print(response)

# body_name = "base"
# foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
# links_to_keep = foot_names + [body_name]
# cfg["robot"]["body_name"] = body_name
# cfg["robot"]["foot_names"] = foot_names
# cfg["robot"]["links_to_keep"] = links_to_keep
# yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_body_name.yaml", "w"))