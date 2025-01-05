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
    def __init__(self, asset_file, scale=1.0, fps=60, init_plane=False):

        gs.init(backend=gs.cpu)

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
            morph = gs.morphs.URDF(file=asset_file, collision=True, scale=scale, merge_fixed_links=False)
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
        self.body_link = self.entity.links[0]
        self.foot_links = [link for link in self.entity.links if "foot" in link.name]

        # Build scene
        self.scene.build(compile_kernels=False)
        self.last_step_time = time.time()

        self._init_buffers()
        self.sim_step()

    def _init_buffers(self):

        self.link_name = [link.name for link in self.entity.links]
        self.joint_name = []
        self.joint_idx = []
        self.dof_idx = []
        idx = 6 # Skip the base dofs
        for joint in self.entity.joints:
            if joint.type == gs.JOINT_TYPE.FREE:
                continue
            elif joint.type == gs.JOINT_TYPE.FIXED:
                continue
            self.joint_name.append(joint.name)
            self.joint_idx.append(idx)
            self.dof_idx.append(joint.dof_idx_local)
            idx += 1
        self.num_dofs = len(self.joint_name)

        self.init_body_pos = torch.tensor([0.0, 0.0, 0.0])
        self.init_body_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.init_joint_pos = torch.zeros(self.num_dofs, dtype=torch.float32)

        self.target_body_pos = self.init_body_pos.clone()
        self.target_body_quat = self.init_body_quat.clone()
        self.target_joint_pos = self.init_joint_pos.clone()

        self.target_foot_pos = torch.zeros((len(self.foot_links), 3), dtype=torch.float32)
        self.target_foot_quat = torch.zeros((len(self.foot_links), 4), dtype=torch.float32)
        self.target_foot_quat[:, 0] = 1.0

    def reset(self):
        self.target_body_pos = self.init_body_pos.clone()
        self.target_body_quat = self.init_body_quat.clone()
        self.target_joint_pos = self.init_joint_pos.clone()
        self.sim_step()

    def step(self):
        if time.time() - self.last_step_time < self.dt:
            time.sleep(self.dt - (time.time() - self.last_step_time))
        self.scene.visualizer.update(force=True)
        self.last_step_time = time.time()

    def sim_step(self):
        # Set the joint positions
        self.target_joint_pos = torch.max(torch.min(self.target_joint_pos, self.joint_limit[1]), self.joint_limit[0])
        self.entity.set_dofs_position(self.target_joint_pos, self.joint_idx, zero_velocity=True)

        # Set base rotation
        R = gs_quat_mul(self.target_body_quat, gs_quat_conjugate(self.body_quat))
        self.entity.set_quat(gs_quat_mul(R, self.entity.get_quat()))

        # Set base position
        delta_pos = self.target_body_pos - self.body_pos
        self.entity.set_pos(delta_pos + self.entity.get_pos())

    def get_link(self, link_name):
        return self.entity.get_link(link_name)

    def set_body_link(self, link):
        self.body_link = link
        self.sim_step()

    def set_init_state(self, body_pos, body_qaut, joint_pos):
        self.init_body_pos = torch.tensor(body_pos)
        self.init_body_pos[:2] = 0.0
        self.init_body_quat = torch.tensor(body_qaut)
        self.init_joint_pos = torch.tensor(joint_pos)
        self.target_body_pos = self.init_body_pos.copy()
        self.target_body_quat = self.init_body_quat.copy()
        self.target_joint_pos = self.init_joint_pos.copy()
        self.sim_step()

    def set_joint_order(self, joint_names):
        order = []
        dof_order = []
        for name in joint_names:
            for idx, joint in enumerate(self.joint_name):
                if name == joint:
                    order.append(self.joint_idx[idx])
                    dof_order.append(self.dof_idx[idx])
                    break
        assert len(order) == len(joint_names), "Some dof names are not found"
        self.joint_idx = order
        self.dof_idx = dof_order
        self.joint_name = joint_names
        self.init_joint_pos = torch.zeros(len(order), dtype=torch.float32)

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
        self.target_joint_pos = torch.tensor(positions)
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
    def joint_pos(self):
        return self.entity.get_dofs_position(dofs_idx_local=self.joint_idx)

    @ property
    def joint_limit(self):
        return self.entity.get_dofs_limit(self.dof_idx)

    @property
    def foot_pos(self):
        return self.links_pos[[link.idx_local for link in self.foot_links],]

    @property
    def foot_quat(self):
        return self.links_quat[[link.idx_local for link in self.foot_links],]
