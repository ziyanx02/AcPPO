import os
import yaml
import argparse
import threading

import numpy as np
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

log_dir = f"./cfgs/{args.robot}/segmentation"
render = True
try:
    os.makedirs(log_dir)
except:
    if_render = input("Log directory already exists. Override? (y/n)\n")
    if if_render != "y":
        render = False

class VisOptions:
    def __init__(self):
        self.visualize_skeleton = False
        self.visualize_target_foot_pos = True
        self.merge_fixed_links = False
        self.show_world_frame = False
        self.shadow = False
        self.background_color = (0.8, 0.8, 0.8)

display = Display(
    cfg=cfg,
    vis_options=VisOptions(),
)

padding = 2
width = 3

if render:
    rgbs, _, segs = display.render(rgb=True, depth=False, segmentation=True)

    axes = ["x", "y", "z"]
    labels = {}
    for i in range(display.entity.n_links):
        link_name = display.entity.links[i].name
        labels[link_name] = {}
        for j in range(3):
            axis = axes[j]
            link_seg = segs[j] == i
            rgb = rgbs[j].copy()
            if link_seg.sum() > 0:
                labels[link_name][axis] = True
                link_seg = np.where(link_seg)
                min_x = link_seg[0].min()
                max_x = link_seg[0].max()
                min_y = link_seg[1].min()
                max_y = link_seg[1].max()

                rgb[min_x - padding - width:min_x - padding, min_y - padding - width:max_y + padding + width + 1] = [255, 0, 0]
                rgb[max_x + padding + 1:max_x + padding + width + 1, min_y - padding - width:max_y + padding + width + 1] = [255, 0, 0]
                rgb[min_x - padding - width:max_x + padding + width + 1, min_y - padding - width:min_y - padding] = [255, 0, 0]
                rgb[min_x - padding - width:max_x + padding + width + 1, max_y + padding +1:max_y + padding + width + 1] = [255, 0, 0]
            else:
                labels[link_name][axis] = False

            plt.imshow(rgb)
            plt.axis('off')
            plt.margins(0, 0)
            plt.savefig(os.path.join(log_dir, f"{link_name}_{axis}.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

        labels[link_name]["any"] = False
        for axis in axes:
            if labels[link_name][axis]:
                labels[link_name]["any"] = True

    with open(os.path.join(log_dir, f"labels.pkl"), "wb") as f:
        pickle.dump(labels, f)

with open(os.path.join(log_dir, f"labels.pkl"), "rb") as f:
    labels = pickle.load(f)

# body_name = "base"
# foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
# links_to_keep = foot_names + [body_name]
# cfg["robot"]["body_name"] = body_name
# cfg["robot"]["foot_names"] = foot_names
# cfg["robot"]["links_to_keep"] = links_to_keep
# yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_body_name.yaml", "w"))