from robot_display.utils.gui import start_gui
from robot_display.utils.display import Robot

class GUIDisplay:
    def __init__(self, cfg: dict, body_pos=True, body_pose=True, dofs_pos=True, foot_pos=False, links_pos=False):
        assert body_pos or body_pose or dofs_pos or foot_pos or links_pos, "At least one of the interaction modes should be enabled"
        assert not (dofs_pos and foot_pos), "Dofs pos and foot position cannot be enabled at the same time"
        self.cfg = cfg
        self.control_body_height = body_pos
        self.control_body_pose = body_pose
        self.control_dofs_pos = dofs_pos
        self.control_foot_pos = foot_pos
        self.control_links_pos = links_pos
        self.setup_robot()
        self.setup_gui()

    def setup_robot(self):
        if "control" not in self.cfg.keys():
            self.cfg["control"] = {"control_freq": 60}
        if "links_to_keep" not in self.cfg["robot"].keys():
            self.cfg["robot"]["links_to_keep"] = []
        self.robot = Robot(
            asset_file=self.cfg["robot"]["asset_path"],
            foot_names=self.cfg["robot"]["foot_names"],
            links_to_keep=self.cfg["robot"]["links_to_keep"],
            scale=self.cfg["robot"]["scale"],
            fps=self.cfg["control"]["control_freq"],
        )
        if "body_name" in self.cfg["robot"].keys():
            self.robot.set_body_link(self.robot.get_link(self.cfg["robot"]["body_name"]))
        if "dof_names" in self.cfg["control"].keys():
            assert len(self.cfg["control"]["dof_names"]) == self.robot.num_dofs, "Number of dof names should match the number of dofs"
            self.robot.set_dof_order(self.cfg["control"]["dof_names"])

    def setup_gui(self):
        self.labels = []
        self.limits = {}
        self.values = []
        idx = 0
        if self.control_body_height:
            self.labels.append("Body Height")
            self.limits["Body Height"] = [0.0, 1.0]
            self.values.append(self.robot.body_pos[2].item())
            self.value_body_height_idx = idx
            idx += 1
        if self.control_body_pose:
            self.labels.extend(["Body Roll", "Body Pitch" , "Body Yaw"])
            self.limits["Body Roll"] = [-180, 180]
            self.limits["Body Pitch"] = [-180, 180]
            self.limits["Body Yaw"] = [-180, 180]
            self.values.extend(self.robot.body_pose.numpy().tolist())
            self.value_body_pose_idx = idx
            idx += 3
        if self.control_dofs_pos:
            self.labels.extend(self.robot.dof_name)
            dof_limits = self.robot.dof_limit
            for i in range(len(self.robot.dof_name)):
                self.limits[self.robot.dof_name[i]] = [dof_limits[0][i].item(), dof_limits[1][i].item()]
            self.values.extend(self.robot.dof_pos.numpy().tolist())
            self.value_dofs_pos_idx_start = idx
            self.value_dofs_pos_idx_end = idx + self.robot.num_dofs
            idx += self.robot.num_dofs
        if self.control_foot_pos:
            raise NotImplementedError
        if self.control_links_pos:
            raise NotImplementedError
        cfg = {
            "label": self.labels,
            "range": self.limits,
        }
        self.gui = start_gui(
            cfg=cfg,
            values=self.values,
            save_callback=self.save_callback,
            reset_callback=self.reset_callback,
        )

    def save_callback(self):
        pass

    def reset_callback(self):
        self.robot.reset()

    def update(self):
        if self.control_body_height:
            self.robot.set_body_height(self.values[self.value_body_height_idx])
        if self.control_body_pose:
            self.robot.set_body_pose(*self.values[self.value_body_pose_idx:self.value_body_pose_idx+3])
        if self.control_dofs_pos:
            self.robot.set_dofs_position(self.values[self.value_dofs_pos_idx_start:self.value_dofs_pos_idx_end])
        if self.control_foot_pos:
            raise NotImplementedError
        if self.control_links_pos:
            raise NotImplementedError
        self.robot.step()

    def run(self):
        while True:
            self.update()