import os
import time

import numpy as np
import torch
from taichi._lib import core as _ti_core
from taichi.lang import impl

import genesis as gs
from robot_display.utils.gs_math import *

def clean():
    gs.utils.misc.clean_cache_files()
    _ti_core.clean_offline_cache_files(os.path.abspath(impl.default_cfg().offline_cache_file_path))
    print("Cleaned up all genesis and taichi cache files.")

class Robot:
    def __init__(self, asset_file, foot_names, links_to_keep=[], scale=1.0, fps=60, vis_options=None):

        gs.init(backend=gs.cpu)

        self.vis_options = vis_options
        self.visualize_interval = 0.5
        self.last_visualize_time = time.time()

        # Create scene
        self.dt = 1 / fps
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=fps,
            ),
            sim_options=gs.options.SimOptions(
                gravity=(0, 0, 0),
            ),
        )

        # Load entity
        if asset_file.endswith(".urdf"):
            morph = gs.morphs.URDF(file=asset_file, collision=True, scale=scale, links_to_keep=links_to_keep)
        elif asset_file.endswith(".xml"):
            morph = gs.morphs.MJCF(file=asset_file, collision=True, scale=scale)
        else:
            raise ValueError(f"Unsupported file format: {asset_file}")
        self.entity = self.scene.add_entity(
            morph,
            surface=gs.surfaces.Default(
                vis_mode="visual",
            ),
        )
        self.body_link = self.links[0]
        self.body_name = self.body_link.name
        self.foot_names = []
        self.foot_links = []
        self.foot_joints = []
        for link in self.links:
            is_foot = False
            for name in foot_names:
                if link.name == name:
                    is_foot = True
                    break
            if is_foot:
                self.foot_names.append(link.name)
                self.foot_links.append(link)
                self.foot_joints.append(link.joint)

        # Build scene
        self.scene.build(compile_kernels=False)
        self.last_step_time = time.time()

        self.diameter = 0
        for pos1 in self.links_pos:
            for pos2 in self.links_pos:
                diameter = torch.norm(pos1 - pos2).item()
                if diameter > self.diameter:
                    self.diameter = diameter
        
        self.scene.viewer.set_camera_pose(
            pos=(self.diameter * 2, self.diameter * 2, self.diameter),
            lookat=(0., 0., 0.),
        )

        self._init_buffers()
        self.sim_step()

    def _init_buffers(self):

        self.link_name = [link.name for link in self.links]
        self.links_by_joint = {}
        self.joint_name = []
        self.dof_name = []
        self.dof_idx = []
        self.dof_idx_local = []
        idx = 6 # Skip the base dofs
        for joint in self.joints:
            if joint.type == gs.JOINT_TYPE.FREE:
                continue
            else:
                self.joint_name.append(joint.name)
                if joint.type == gs.JOINT_TYPE.FIXED:
                    continue
                self.dof_name.append(joint.name)
                self.dof_idx.append(idx)
                self.dof_idx_local.append(joint.dof_idx_local)
                idx += 1
        self.num_links = len(self.link_name)
        self.num_dofs = len(self.dof_name)
        self.num_joints = len(self.joint_name)

        self.update_skeleton()

        print("--------- Link Names ----------")
        print(self.link_name)
        print("--------- Joint Names ---------")
        print(self.dof_name)
        print("-------------------------------")

        self.init_body_pos = torch.tensor([0.0, 0.0, 0.0])
        self.init_body_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.init_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float32)

        self.target_body_pos = self.init_body_pos.clone()
        self.target_body_quat = self.init_body_quat.clone()
        self.target_dof_pos = self.init_dof_pos.clone()

        self.target_foot_pos = torch.zeros((len(self.foot_links), 3), dtype=torch.float32)
        self.target_foot_quat = torch.zeros((len(self.foot_links), 4), dtype=torch.float32)
        self.target_foot_quat[:, 0] = 1.0

    def update_skeleton(self):

        self.link_adjacency_map = [[False for _ in range(self.num_links)] for _ in range(self.num_links)]
        self.leg = []
        self.body = []

        body_joint_name = []
        for idx in self.body_link.child_idxs_local:
            body_joint_name.append(self.links[idx].joint.name)
        if self.body_link.idx_local != 0: # If the base link is not the root link
            body_joint_name.append(self.body_link.joint.name)

        # vertices to visualize body
        for joint_name1 in body_joint_name:
            for joint_name2 in body_joint_name:
                if joint_name1 < joint_name2:
                    self.body.append((joint_name1, joint_name2))

        for link in self.links:
            for idx in link.child_idxs_local:
                self.link_adjacency_map[link.idx_local][idx] = self.links[idx].joint.name
            if link.idx_local == 0:
                continue
            self.link_adjacency_map[link.idx_local][link.parent_idx_local] = link.joint.name

        def dfs(curr, target, visited, path):
            visited[curr] = True
            path.append(curr)
            if curr == target:
                return True
            for i in range(self.num_links):
                if self.link_adjacency_map[curr][i] and not visited[i]:
                    if dfs(i, target, visited, path):
                        return True
            path.pop()
            return False

        paths = []
        for link in self.foot_links:
            foot_idx = link.idx_local
            body_idx = self.body_link.idx_local
            path = []
            visited = [False for _ in range(self.num_links)]
            dfs(foot_idx, body_idx, visited, path)
            paths.append(path)
        
        # distill the path
        # remove the links that are used by other paths
        links_used_counter = [0 for _ in range(self.num_links)]
        for path in paths:
            for idx in path:
                links_used_counter[idx] += 1
        distilled_paths = []
        for path in paths:
            joint_path = []
            for i in range(len(path) - 2):
                if links_used_counter[path[i]] == 1:
                    joint_path.append(self.link_adjacency_map[path[i]][path[i + 1]])
            joint_path.append(self.link_adjacency_map[path[-2]][path[-1]])
            distilled_paths.append(joint_path)
        for dist_path in distilled_paths:
            for i in range(len(dist_path) - 1):
                if dist_path[i] < dist_path[i + 1]:
                    if (dist_path[i], dist_path[i + 1]) not in self.leg:
                        self.leg.append((dist_path[i], dist_path[i + 1]))
                else:
                    if (dist_path[i + 1], dist_path[i]) not in self.leg:
                        self.leg.append((dist_path[i + 1], dist_path[i]))

    def reset(self):
        self.target_body_pos = self.init_body_pos.clone()
        self.target_body_quat = self.init_body_quat.clone()
        self.target_dof_pos = self.init_dof_pos.clone()
        self.sim_step()

    def step(self):
        if self.dt - (time.time() - self.last_step_time) > 0.01:
            time.sleep(self.dt - (time.time() - self.last_step_time))
        self.visualize()
        self.scene.visualizer.update(force=True)
        self.last_step_time = time.time()

    def sim_step(self):
        # Set the joint positions
        self.target_dof_pos = torch.max(torch.min(self.target_dof_pos, self.dof_limit[1]), self.dof_limit[0])
        self.entity.set_dofs_position(self.target_dof_pos, self.dof_idx, zero_velocity=True)

        # Set base rotation
        R = gs_quat_mul(self.target_body_quat, gs_quat_conjugate(self.body_quat))
        self.entity.set_quat(gs_quat_mul(R, self.entity.get_quat()))

        # Set base position
        delta_pos = self.target_body_pos - self.body_pos
        self.entity.set_pos(delta_pos + self.entity.get_pos())

    def visualize(self):

        if time.time() - self.last_visualize_time < self.visualize_interval:
            return
        else:
            self.last_visualize_time = time.time()

        joint_pos = {}
        joint_vis = {}
        for joint in self.joints:
            joint_vis[joint.name] = False
            joint_pos[joint.name] = joint.get_pos()
            joint_pos[joint.name][:2] -= self.diameter

        self.scene.clear_debug_objects()

        body_color = (0, 0, 0.8, 1)
        leg_color = (0, 0.8, 0, 1)
        foot_color = (0.8, 0, 0, 1)

        thickness = self.diameter / 30

        for name1 in self.joint_name:
            pos1 = joint_pos[name1]
            for name2 in self.joint_name:
                if (name1, name2) in self.body:
                    color = body_color
                    joint_vis[name1] = "body"
                    joint_vis[name2] = "body"
                elif (name1, name2) in self.leg:
                    color = leg_color
                    joint_vis[name1] = "leg"
                    joint_vis[name2] = "leg"
                else:
                    continue
                pos2 = joint_pos[name2]
                self.scene.draw_debug_line(pos1, pos2, radius=thickness, color=color)

        for name in self.joint_name:
            if joint_vis[name] == "body":
                color = body_color
            elif joint_vis[name] == "leg":
                color = leg_color
            else:
                continue
            pos = joint_pos[name]
            self.scene.draw_debug_sphere(pos, radius=thickness, color=color)

    def get_joint(self, joint_name=None):
        return self.entity.get_joint(joint_name)

    def get_link(self, link_name=None):
        return self.entity.get_link(link_name)

    def set_body_link(self, link):
        self.body_link = link
        self.body_name = link.name
        self.update_skeleton()
        self.sim_step()

    def set_body_link_by_name(self, body_name):
        self.body_name = body_name
        self.body_link = self.get_link(body_name)
        self.sim_step()

    def set_init_state(self, body_pos, body_qaut, dof_pos):
        self.init_body_pos = torch.tensor(body_pos)
        self.init_body_pos[:2] = 0.0
        self.init_body_quat = torch.tensor(body_qaut)
        self.init_dof_pos = torch.tensor(dof_pos)
        self.target_body_pos = self.init_body_pos.copy()
        self.target_body_quat = self.init_body_quat.copy()
        self.target_dof_pos = self.init_dof_pos.copy()
        self.sim_step()

    def set_dof_order(self, dof_names):
        order = []
        dof_order = []
        for name in dof_names:
            for idx, joint in enumerate(self.dof_name):
                if name == joint:
                    order.append(self.dof_idx[idx])
                    dof_order.append(self.dof_idx_local[idx])
                    break
        assert len(order) == len(dof_names), "Some dof names are not found"
        self.dof_idx = order
        self.dof_idx_local = dof_order
        self.dof_name = dof_names
        self.init_dof_pos = torch.zeros(len(order), dtype=torch.float32)

    def set_body_height(self, height):
        self.target_body_pos[2] = height
        self.sim_step()

    def set_body_quat(self, quat):
        self.target_body_quat = normalize(torch.tensor(quat))
        self.sim_step()

    def set_body_pose(self, roll, pitch, yaw):
        roll, pitch, yaw = roll / 180 * np.pi, pitch / 180 * np.pi, yaw / 180 * np.pi
        xyz = torch.tensor([roll, pitch, yaw])
        R = gs_euler2quat(xyz)
        # Compute the rotation quaternion 
        self.target_body_quat = gs_quat_mul(R, self.init_body_quat)
        self.sim_step()

    def set_dofs_position(self, positions):
        self.target_dof_pos = torch.tensor(positions)
        self.sim_step()

    def set_links_pos(self, links, poss, quats):
        q, err = self.entity.inverse_kinematics_multilink(
            links=links,
            poss=poss,
            # quats=quats,
            return_error=True,
            # rot_mask=[False, False, True],
        )
        self.entity.set_qpos(q)
        # target_dof_pos = q[7:]
        # self.set_dofs_position(target_dof_pos)

    @property
    def links(self):
        return self.entity.links

    @property
    def links_pos(self):
        return self.entity.get_links_pos()

    @property
    def links_quat(self):
        return self.entity.get_links_quat()

    @property
    def body_pos(self):
        return self.links_pos[self.body_link.idx_local]

    @property
    def body_quat(self):
        return self.links_quat[self.body_link.idx_local]

    @property
    def body_pose(self):
        return gs_quat2euler(self.body_quat)

    @property
    def base_pos(self):
        return self.entity.get_pos()

    @property
    def base_quat(self):
        return self.entity.get_quat()

    @property
    def joints(self):
        return self.entity.joints

    @property
    def dof_pos(self):
        return self.entity.get_dofs_position(dofs_idx_local=self.dof_idx)

    @ property
    def dof_limit(self):
        return self.entity.get_dofs_limit(self.dof_idx_local)

    @property
    def foot_pos(self):
        return self.links_pos[[link.idx_local for link in self.foot_links],]

    @property
    def foot_quat(self):
        return self.links_quat[[link.idx_local for link in self.foot_links],]
