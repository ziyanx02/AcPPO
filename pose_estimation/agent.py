import os
import yaml
import argparse
import threading
import multiprocessing
import time

import numpy as np
import math
import torch
import cv2
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
import pickle

from robot_display.display import Display
from api.azure_openai import complete, local_image_to_data_url
from prompts.api_example import API_EXAMPLES

class VisOptions:
    visualize_skeleton = False
    visualize_target_foot_pos = False
    merge_fixed_links = False
    show_world_frame = False
    shadow = False
    background_color = (1., 1., 1.)
    show_viewer = True

def rotate_quat_from_rpy(quat, roll, pitch, yaw):
    """
    Rotate a quaternion by given roll, pitch, and yaw angles (in degrees).
    
    Args:
        quat: Input quaternion as a tensor in shape (4,) with order (w, x, y, z)
        roll: Rotation around x-axis in degrees
        pitch: Rotation around y-axis in degrees
        yaw: Rotation around z-axis in degrees

    Returns:
        Rotated quaternion as a tensor in shape (4,)
    """
    # Convert angles from degrees to radians
    roll_rad = math.radians(roll)
    pitch_rad = -math.radians(pitch)
    yaw_rad = math.radians(yaw)
    
    # Compute half angles
    half_roll = roll_rad / 2.0
    half_pitch = pitch_rad / 2.0
    half_yaw = yaw_rad / 2.0
    
    # Create rotation quaternions for each axis
    # Roll (x-axis rotation)
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    q_roll = torch.tensor([cr, sr, 0.0, 0.0], dtype=quat.dtype, device=quat.device)
    
    # Pitch (y-axis rotation)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    q_pitch = torch.tensor([cp, 0.0, sp, 0.0], dtype=quat.dtype, device=quat.device)
    
    # Yaw (z-axis rotation)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    q_yaw = torch.tensor([cy, 0.0, 0.0, sy], dtype=quat.dtype, device=quat.device)
    
    # Combine the rotations (order: yaw -> pitch -> roll)
    q_rpy = quaternion_multiply(q_roll, quaternion_multiply(q_pitch, q_yaw))
    
    # Rotate the input quaternion by the combined rotation
    rotated_quat = quaternion_multiply(q_rpy, quat)
    
    return rotated_quat

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    
    Args:
        q1: First quaternion (w, x, y, z)
        q2: Second quaternion (w, x, y, z)
        
    Returns:
        Product of q1 and q2
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.tensor([w, x, y, z], dtype=q1.dtype, device=q1.device)

class Agent:
    def __init__(self, cfg_path):
        self.cfg = yaml.safe_load(open(cfg_path))
        self.vis_options = VisOptions()
        self.display = Display(
            cfg=self.cfg,
            vis_options=self.vis_options,
        )

    def rotate_along_x(self, degree):
        body_quat = self.get_body_quat()
        body_quat = rotate_quat_from_rpy(body_quat, degree, 0, 0)
        self.set_body_quat(body_quat)

    def rotate_along_y(self, degree):
        body_quat = self.get_body_quat()
        body_quat = rotate_quat_from_rpy(body_quat, 0, -degree, 0)
        self.set_body_quat(body_quat)

    def rotate_along_z(self, degree):
        body_quat = self.get_body_quat()
        body_quat = rotate_quat_from_rpy(body_quat, 0, 0, degree)
        self.set_body_quat(body_quat)

    def get_body_link(self):
        """
        Get the body link.
        Return:
            link_id (int): The id of the link
        
        Body link is the link regarded as the base of the robot while simulating.
        Setting robot's position and quaternion is same as setting base link's position and quaternion.
        """
        link_id = self.display.body_link.idx_local
        return link_id

    def set_body_link(self, link_id):
        """
        Set the body link.
        Args:
            link_id (int): The id of the link

        Body link is the link regarded as the base of the robot while simulating.
        Setting body link's position and quaternion is same as setting robot's position and quaternion.
        """
        self.display.set_body_link_by_id(link_id)

    def get_body_pos(self):
        """
        Get the position of the body link.
        Return:
            body_pos (torch.tensor): The position of the body link
        """
        body_pos = self.display.body_pos
        return body_pos

    def set_body_pos(self, body_pos):
        """
        Set the position of the body link.
        Args:
            body_pos (torch.tensor): The target position of the body link
        """
        self.display.set_body_pos(body_pos)

    def get_body_quat(self):
        """
        Get the quaternion of the body link.
        Return:
            body_quat (torch.tensor): The quaternion of the body link
        """
        body_quat = self.display.body_quat
        return body_quat

    def set_body_quat(self, body_quat):
        """
        Set the quaternion of the body link.
        Args:
            body_quat (torch.tensor): The target quaternion of the body link
        """
        self.display.set_body_quat(body_quat)

    def get_link_pos(self, link_id):
        """
        Get the position of the link.
        Args:
            link_id (int): The id of the link
        Return:
            pos (torch.tensor): The position of the link
        """
        pos = self.display.links_pos[link_id]
        return pos

    def get_link_quat(self, link_id):
        """
        Get the quaternion of the link.
        Args:
            link_id (int): The id of the link
        Return:
            quat (torch.tensor): The quaternion of the link
        """
        quat = self.display.links_quat[link_id]
        return quat

    def set_link_pose(self, link_id, pos, quat=None):
        """
        Use inverse kinematics to set the position (and quaternion) of the link. If the quaternion is not provided, only the position is required.
        If the IK solver succeed, robot will be set to the pose; if not, the robot pose will not be changed.
        Args:
            link_id (int): The id of the link
            pos (torch.tensor): The target position of the link
            quat (torch.tensor): The target quaternion of the link
        Return:
            success (bool): Whether the IK solver succeeded
        """
        success = self.display.set_link_pose(link_id, pos, quat)
        return success

    def set_link_pos(self, link_id, pos):
        """
        Use inverse kinematics to set the position of the link.
        Args:
            link_id (int): The id of the link
            pos (torch.tensor): The target position of the link
        """
        success = self.display.set_link_pos(link_id, pos)
        return success

    def try_set_link_pose(self, link_id, pos, quat=None):
        """
        Use inverse kinematics to set the position (and quaternion) of the link. If the quaternion is not provided, only the position is required.
        If the IK solver succeed, robot will be set to the pose; if not, the robot pose will not be changed.
        Args:
            link_id (int): The id of the link
            pos (torch.tensor): The target position of the link
            quat (torch.tensor): The target quaternion of the link
        Return:
            success (bool): Whether the IK solver succeeded
        """
        success = self.display.try_set_link_pose(link_id, pos, quat)
        return success

    def get_joints_between_links(self, link_id1, link_id2):
        """
        Get the joints between two links.
        Args:
            link_id1 (int): The id of the first link
            link_id2 (int): The id of the second link
        Return:
            joint_ids (list): The joints between the two links
        """
        joint_ids = self.display.get_dofs_between_links(link_id1, link_id2)
        return joint_ids

    # def get_joint_pos(self, joint_id):
    #     """
    #     Get the joint position.
    #     Args:
    #         joint_id (int): The id of the joint
    #     Return:
    #         joint_pos (float): The joint position
    #         joint_limit (tuple): The lower and upper bound of the joint position
    #     """
    #     joint_pos = self.display.dof_pos[joint_id].item()
    #     joint_limit = (self.display.dof_limit[0][joint_id].item(), self.display.dof_limit[1][joint_id].item())
    #     return joint_pos, joint_limit

    def get_joint_pos(self, joint_id):
        """
        Get the joint position.
        Args:
            joint_id (int): The id of the joint
        Return:
            joint_pos (torch.tensor): The joint position
        """
        joint_name = self.display.dof_name[joint_id]
        joint = self.display.get_joint(joint_name)
        return joint.get_pos()

    def set_joint_pos(self, joint_pos, joint_id):
        """
        Set the joint position.
        Args:
            joint_pos (float): The target joint position
            joint_id (int): The id of the joint
        """
        self.display.set_dofs_position(joint_pos, joint_id)

    def get_camera_pose(self):
        """
        Get the current camera pose.
        Return:
            camera_pose (dict): The current camera pose
            camera_pose["azimuth"] (degree): The azimuth angle of the camera
            camera_pose["elevation"] (degree): The elevation angle of the camera
            camera_pose["lookat"] (torch.tensor): The lookat point of the camera
            camera_pose["distance"] (float): The distance of the camera from the lookat point
        The camera's position is defined by the azimuth, elevation, lookat point, and distance by
        camera_pos = lookat + distance * torch.tensor([
            np.cos(azimuth / 180 * np.pi) * np.cos(elevation / 180 * np.pi),
            np.sin(azimuth / 180 * np.pi) * np.cos(elevation / 180 * np.pi),
            np.sin(elevation / 180 * np.pi),
        ])
        """
        camera_pose = {
            "azimuth": self.display.camera_azimuth,
            "elevation": self.display.camera_elevation,
            "distance": self.display.camera_distance,
            "lookat": self.display.camera_lookat,
        }
        return camera_pose

    def set_camera_pose(self, camera_pose):
        """
        Set camera pose.
        Args:
            camera_pose (dict): The target camera pose
            camera_pose["azimuth"] (degree): The azimuth angle of the camera
            camera_pose["elevation"] (degree): The elevation angle of the camera
            camera_pose["lookat"] (torch.tensor): The lookat point of the camera
            camera_pose["distance"] (float): The distance of the camera from the lookat point
        The camera's position is defined by the azimuth, elevation, lookat point, and distance by
        camera_pos = lookat + distance * torch.tensor([
            np.cos(azimuth / 180 * np.pi) * np.cos(elevation / 180 * np.pi),
            np.sin(azimuth / 180 * np.pi) * np.cos(elevation / 180 * np.pi),
            np.sin(elevation / 180 * np.pi),
        ])
        """
        self.display.set_camera_pose(
            azimuth=camera_pose["azimuth"],
            elevation=camera_pose["elevation"],
            distance=camera_pose["distance"],
            lookat=camera_pose["lookat"],
        )

    def pack_camera_transform(self):
        camera_transform = {}
        camera_transform["extrinsics"] = self.display.camera.extrinsics
        camera_transform["intrinsics"] = self.display.camera.intrinsics
        return camera_transform

    def render(self, link_ids=None):
        """
        Render current camera view
        Return (list):
            All visible links ids
        """
        self.update()
        image, _, labelled_image, visible_links = self.display.render(link_ids)
        cv2.imwrite(os.path.join("./rgb.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join("./label.png"), labelled_image[:, :, ::-1])
        return visible_links.tolist()

    def render_from_x(self, camera_lookat, link_ids=None):
        """
        Get an image from x (front)
        Return (list):
            All visible links ids
        """
        self.update()
        initial_camera_pose = self.get_camera_pose()
        camera_pose = initial_camera_pose.copy()
        camera_pose["azimuth"] = 0
        camera_pose["elevation"] = 0
        camera_pose["lookat"] = camera_lookat
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(camera_lookat, x=False)
        image, _, labelled_image, visible_links = self.display.render(link_ids)
        camera_transform = self.pack_camera_transform()
        cv2.imwrite(os.path.join("./rgb_x.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join("./label_x.png"), labelled_image[:, :, ::-1])
        self.set_camera_pose(initial_camera_pose)
        self.display.clear_debug_objects()
        return visible_links.tolist(), camera_transform

    def render_from_y(self, camera_lookat, link_ids=None):
        """
        Get an image from y (left)
        Return (list):
            All visible links ids
        """
        self.update()
        initial_camera_pose = self.get_camera_pose()
        camera_pose = initial_camera_pose.copy()
        camera_pose["azimuth"] = 90
        camera_pose["elevation"] = 0
        camera_pose["lookat"] = camera_lookat
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(camera_lookat, y=False)
        image, _, labelled_image, visible_links = self.display.render(link_ids)
        camera_transform = self.pack_camera_transform()
        cv2.imwrite(os.path.join("./rgb_y.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join("./label_y.png"), labelled_image[:, :, ::-1])
        self.set_camera_pose(initial_camera_pose)
        self.display.clear_debug_objects()
        return visible_links.tolist(), camera_transform

    def render_from_z(self, camera_lookat, link_ids=None):
        """
        Get an image from z (up)
        Return (list):
            All visible links ids
        """
        self.update()
        initial_camera_pose = self.get_camera_pose()
        camera_pose = initial_camera_pose.copy()
        camera_pose["azimuth"] = 180
        camera_pose["elevation"] = 89
        camera_pose["lookat"] = camera_lookat
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(camera_lookat, z=False)
        image, _, labelled_image, visible_links = self.display.render(link_ids)
        camera_transform = self.pack_camera_transform()
        cv2.imwrite(os.path.join("./rgb_z.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join("./label_z.png"), labelled_image[:, :, ::-1])
        self.set_camera_pose(initial_camera_pose)
        self.display.clear_debug_objects()
        return visible_links.tolist(), camera_transform

    def render_from_xyz(self, camera_lookat):
        """
        Render three different views from x (front), y (left) and z (up)
        Return (list):
            All visible links ids
        """
        all_visible_links = np.array([-1])
        camera_transforms = []
        visible_links, camera_transform = self.render_from_x(camera_lookat)
        all_visible_links = np.concatenate([all_visible_links, visible_links])
        camera_transforms.append(camera_transform)
        visible_links, camera_transform = self.render_from_y(camera_lookat)
        all_visible_links = np.concatenate([all_visible_links, visible_links])
        camera_transforms.append(camera_transform)
        visible_links, camera_transform = self.render_from_z(camera_lookat)
        all_visible_links = np.concatenate([all_visible_links, visible_links])
        camera_transforms.append(camera_transform)
        all_visible_links = np.unique(all_visible_links)
        return all_visible_links[all_visible_links != -1].tolist(), camera_transforms

    def render_link(self, link_id):
        self.update()
        initial_camera_pose = self.get_camera_pose()
        camera_pose = initial_camera_pose.copy()
        camera_pose["lookat"] = self.get_link_pos(link_id)
        camera_transforms = {}
        axis = []

        camera_pose["azimuth"] = 0
        camera_pose["elevation"] = 0
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(self.get_link_pos(link_id), x=False)
        image, segmentation_x, labelled_image, _ = self.display.render()
        camera_transforms["x"] = self.pack_camera_transform()
        cv2.imwrite(os.path.join("./rgb_x.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join("./label_x.png"), labelled_image[:, :, ::-1])
        camera_pose["azimuth"] = 180
        camera_pose["elevation"] = 0
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(self.get_link_pos(link_id), x=False)
        image, segmentation_nx, labelled_image, _ = self.display.render()
        camera_transforms["-x"] = self.pack_camera_transform()
        cv2.imwrite(os.path.join("./rgb_-x.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join("./label_-x.png"), labelled_image[:, :, ::-1])
        if np.sum(segmentation_x == link_id) > np.sum(segmentation_nx == link_id):
            axis.append("x")
        else:
            axis.append("-x")
        self.display.clear_debug_objects()

        camera_pose["azimuth"] = 90
        camera_pose["elevation"] = 0
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(self.get_link_pos(link_id), y=False)
        image, segmentation_y, labelled_image, _ = self.display.render()
        camera_transforms["y"] = self.pack_camera_transform()
        cv2.imwrite(os.path.join("./rgb_y.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join("./label_y.png"), labelled_image[:, :, ::-1])
        camera_pose["azimuth"] = 270
        camera_pose["elevation"] = 0
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(self.get_link_pos(link_id), y=False)
        image, segmentation_ny, labelled_image, _ = self.display.render()
        camera_transforms["-y"] = self.pack_camera_transform()
        cv2.imwrite(os.path.join("./rgb_-y.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join("./label_-y.png"), labelled_image[:, :, ::-1])
        if np.sum(segmentation_y == link_id) > np.sum(segmentation_ny == link_id):
            axis.append("y")
        else:
            axis.append("-y")
        self.display.clear_debug_objects()

        camera_pose["azimuth"] = 180
        camera_pose["elevation"] = 89
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(self.get_link_pos(link_id), z=False)
        image, segmentation_z, labelled_image, _ = self.display.render()
        camera_transforms["z"] = self.pack_camera_transform()
        cv2.imwrite(os.path.join("./rgb_z.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join("./label_z.png"), labelled_image[:, :, ::-1])
        camera_pose["azimuth"] = 0
        camera_pose["elevation"] = -89
        self.set_camera_pose(camera_pose)
        self.display.visualize_link_frame(self.get_link_pos(link_id), z=False)
        image, segmentation_nz, labelled_image, _ = self.display.render()
        camera_transforms["-z"] = self.pack_camera_transform()
        cv2.imwrite(os.path.join("./rgb_-z.png"), image[:, :, ::-1])
        cv2.imwrite(os.path.join("./label_-z.png"), labelled_image[:, :, ::-1])
        if np.sum(segmentation_z == link_id) > np.sum(segmentation_nz == link_id):
            axis.append("z")
        else:
            axis.append("-z")
        self.display.clear_debug_objects()

        return camera_transforms, axis

    def update(self):
        "Update the robot display"
        self.display.update()

    def run(self, task):
        self.display.update()
        self.render()

        hist_messages = [
            {"role": "system", "content": "You are an expert in robot kinematics and motion planning."},
            {"role": "user", "content": API_EXAMPLES},
        ]

        from prompts.example1.prompt import prompt as example1

        # hist_messages += example1

        hist_messages.append(
                {
                    "role": "user",
                    "content": f"Now your task is to {task}",
                }
            )
        hist_messages.append(
                {
                    "role": "user",
                    "content": "Your answer should contain only executable python codes. Your thoughts should be in the comments.",
                }
            )

        hist_messages.append(
            {
                "role": "user",
                "content": [
                            {"type": "text", "text": f"This is the current RGB image taken by the camera."},
                            {
                                "type": "image_url",
                                "image_url": {"url": local_image_to_data_url("./rgb.png")},
                            },
                        ],
            }
        )
        hist_messages.append(
            {
                "role": "user",
                "content": [
                            {"type": "text", "text": f"This is the image with different links highlighted and labelled with their ids."},
                            {
                                "type": "image_url",
                                "image_url": {"url": local_image_to_data_url("./label.png")},
                            },
                        ],
            }
        )

        while True:
            hist_messages.append(
                {
                    "role": "user",
                    "content": "Write executable code (not a function) to finish next step. Don't try to solve the task once for all.",
                }
            )
            print("Waiting for Completion...")
            response = complete(hist_messages)
            hist_messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )
            lines = response.split('\n')
            filtered_lines = []
            for line in lines:
                if not line.startswith('```'):
                    filtered_lines.append(line)
            response = '\n'.join(filtered_lines)
            print(response)
            try:
                messages = []
                exec(response)
                for message in messages:
                    if message[0] == "image":
                        hist_messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": message[2]},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": local_image_to_data_url(f"./{message[1]}.png")},
                                    },
                                ],
                            }
                        )
                    else:
                        text = f"{message[2]} {locals()[message[1]]}"
                        print(text)
                        hist_messages.append(
                            {
                                "role": "user",
                                "content": text,
                            }
                        )
                print("Execute successfully!")
            except Exception as e:
                hist_messages.append(
                    {
                        "role": "system",
                        "content": f"Error: {e}",
                    }
                )
                print(f"Error: {e}")
            input()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--robot', type=str, default='go2')
    parser.add_argument('-n', '--name', type=str, default='default')
    args = parser.parse_args()

    cfg_path = f"./cfgs/{args.robot}/basic.yaml"
    agent = Agent(cfg_path)
    # print(agent.display.joint_name_to_dof_order)
    # print(agent.get_joints_between_links(20, 22))
    # time.sleep(0.5)
    # agent.set_body_link(20)
    # agent.set_link_pose(0, torch.tensor([0., 0., 0.5]))
    # print(agent.get_link_pos(20))
    # print(agent.get_link_pos(0))
    # agent.display.update()
    # agent.render()
    # agent.render_from_xyz()
    # exit()

    # agent.run()

    task = "generate a pose that the hand walks with like a dog"
    agent.run(task)