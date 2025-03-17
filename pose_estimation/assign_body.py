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

axes = ["x", "y", "z", "-x", "-y", "-z"]
for i in range(len(rgbs)):
    image = rgbs[i].copy()[:, :, ::-1]
    segmentation_label = segs[i].copy()

    # Create a color map for each segment ID
    unique_labels = np.unique(segmentation_label)
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

# padding = 2
# width = 3

# axes = ["x", "y", "z"]
# labels = {}
# for i in range(display.entity.n_links):
#     link_name = display.entity.links[i].name
#     labels[link_name] = {}
#     for j in range(3):
#         axis = axes[j]
#         link_seg = segs[j] == i
#         rgb = rgbs[j].copy()
#         if link_seg.sum() > 0:
#             labels[link_name][axis] = True
#             link_seg = np.where(link_seg)
#             min_x = link_seg[0].min()
#             max_x = link_seg[0].max()
#             min_y = link_seg[1].min()
#             max_y = link_seg[1].max()

#             rgb[min_x - padding - width:min_x - padding, min_y - padding - width:max_y + padding + width + 1] = [255, 0, 0]
#             rgb[max_x + padding + 1:max_x + padding + width + 1, min_y - padding - width:max_y + padding + width + 1] = [255, 0, 0]
#             rgb[min_x - padding - width:max_x + padding + width + 1, min_y - padding - width:min_y - padding] = [255, 0, 0]
#             rgb[min_x - padding - width:max_x + padding + width + 1, max_y + padding +1:max_y + padding + width + 1] = [255, 0, 0]
#         else:
#             labels[link_name][axis] = False

#         plt.imshow(rgb)
#         plt.axis('off')
#         plt.margins(0, 0)
#         plt.savefig(os.path.join(log_dir, f"{link_name}_{axis}.png"), bbox_inches='tight', pad_inches=0)
#         plt.close()

#     labels[link_name]["any"] = False
#     for axis in axes:
#         if labels[link_name][axis]:
#             labels[link_name]["any"] = True

# with open(os.path.join(log_dir, f"labels.pkl"), "wb") as f:
#     pickle.dump(labels, f)

# with open(os.path.join(log_dir, f"labels.pkl"), "rb") as f:
#     labels = pickle.load(f)

# body_name = "base"
# foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
# links_to_keep = foot_names + [body_name]
# cfg["robot"]["body_name"] = body_name
# cfg["robot"]["foot_names"] = foot_names
# cfg["robot"]["links_to_keep"] = links_to_keep
# yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_body_name.yaml", "w"))