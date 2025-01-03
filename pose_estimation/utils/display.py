import os
import time

import numpy as np
import torch
from taichi._lib import core as _ti_core
from taichi.lang import impl

import genesis as gs
from utils import *

def clean():
    gs.utils.misc.clean_cache_files()
    _ti_core.clean_offline_cache_files(os.path.abspath(impl.default_cfg().offline_cache_file_path))
    print("Cleaned up all genesis and taichi cache files.")

class Display:
    def __init__(self, asset_file, scale=1.0, fps=50, init_plane=False):

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
        self.base_link = self.entity.links[0]
        self.foot_links = [link for link in self.entity.links if "foot" in link.name]

        # Build scene
        self.scene.build(compile_kernels=False)
        self.last_step_time = time.time()

        self._init_buffers()

    def _init_buffers(self):

        self.link_name = [link.name for link in self.entity.links]
        self.joint_name = []
        self.joint_idx = []
        for joint in self.entity.joints:
            if joint.type == gs.JOINT_TYPE.FREE:
                continue
            elif joint.type == gs.JOINT_TYPE.FIXED:
                continue
            self.joint_name.append(joint.name)
            self.joint_idx.append(joint.idx_local)

        self.init_base_pos = torch.tensor([0.0, 0.0, 0.0])
        self.init_base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.init_joint_pos = torch.tensor([0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5])

        self.init_entity_pos = torch.tensor([0.0, 0.0, 0.0])
        self.init_entity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        self.target_entity_pos = self.init_entity_pos.copy()
        self.target_entity_quat = self.init_entity_quat.copy()
        self.target_joint_pos = self.init_joint_pos.copy()

        self.target_base_pos = self.init_base_pos
        self.target_base_quat = self.init_base_quat
        self.target_foot_pos = torch.zeros((len(self.foot_links), 3), dtype=torch.float32)
        self.target_foot_quat = torch.zeros((len(self.foot_links), 4), dtype=torch.float32)
        self.target_foot_quat[:, 0] = 1.0

    def step(self):
        if time.time() - self.last_step_time < self.dt:
            time.sleep(self.dt - (time.time() - self.last_step_time))
        self.scene.visualizer.update(force=True)
        self.last_step_time = time.time()

    def get_link(self, link_name):
        return self.entity.get_link(link_name)

    def set_base_pos(self, pos):
        self.target_base_pos = torch.tensor(pos)
        delta_pos = self.target_base_pos - self.base_pos
        self.entity.set_pos(delta_pos + self.entity.get_pos())

    def set_base_quat(self, quat):
        self.target_base_quat = normalize(torch.tensor(quat))
        # Compute the rotation quaternion 
        R = gs_quat_mul(self.target_base_quat, gs_quat_conjugate(self.base_quat))
        target_quat = gs_quat_mul(R, self.entity.get_quat())
        self.entity.set_quat(target_quat)

    def set_base_roll_pitch(self, roll, pitch):
        xyz = torch.tensor([roll, pitch, 0.0])
        gs_euler2quat(xyz)
        # self.target_base_quat = normalize(torch.tensor(quat))
        # Compute the rotation quaternion 
        R = gs_quat_mul(self.target_base_quat, gs_quat_conjugate(self.base_quat))
        target_quat = gs_quat_mul(R, self.entity.get_quat())
        self.entity.set_quat(target_quat)

    def set_links_pos(self, links, poss=None, quats=None):
        if type(links) is not list:
            links = [links,]
            if poss is not None:
                poss = [poss,]
            if quats is not None:
                quats = [quats,]
        links.append(self.base_link)
        if poss is None:
            poss = [link.get_pos() for link in links]
        else:
            poss.append(self.target_base_pos)
        if quats is None:
            quats = [link.get_quat() for link in links]
        else:
            quats.append(self.target_foot_quat)
        poss = [torch.tensor(pos) for pos in poss]
        quats = [normalize(torch.tensor(quat)) for quat in quats]
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

    def set_dofs_position(self, positions):
        self.entity
        self.entity.set_dofs_position(position=torch.tensor(positions), dofs_idx_local=self.joint_idx, zero_velocity=True)

    @property
    def links_pos(self):
        return self.entity.get_links_pos()

    @property
    def links_quat(self):
        return self.entity.get_links_quat()

    @property
    def base_pos(self):
        return self.links_pos[self.base_link.idx_local]

    @property
    def base_quat(self):
        return self.links_quat[self.base_link.idx_local]

    @property
    def foot_pos(self):
        return self.links_pos[[link.idx_local for link in self.foot_links],]

    @property
    def foot_quat(self):
        return self.links_quat[[link.idx_local for link in self.foot_links],]

if __name__ == "__main__":
    display = Display("urdf/go2/urdf/go2.urdf")
    init_time = time.time()
    display.set_links_pos(display.foot_links, poss=[[0.0, 0.0, -0.3] for _ in range(4)])
    while True:
        if time.time() - init_time > 1:
            pass
            # display.set_base_pos([0.0, 0.0, 0.5])
            # display.set_base_quat([0.7, 0.0, 0.0, 0.7])
        display.step()