import time
import sys
import yaml
import argparse

import numpy as np
import torch
from transforms3d import quaternions

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmd_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmd_hg
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from .low_state_handler import LowStateMsgHandler

JointID = {
    "go2": {
        "FR_hip_joint": 0,
        "FR_thigh_joint": 1,
        "FR_calf_joint": 2,
        "FL_hip_joint": 3,
        "FL_thigh_joint": 4,
        "FL_calf_joint": 5,
        "RR_hip_joint": 6,
        "RR_thigh_joint": 7,
        "RR_calf_joint": 8,
        "RL_hip_joint": 9,
        "RL_thigh_joint": 10,
        "RL_calf_joint": 11,
    }
}

HIGHLEVEL = 0xEE
LOWLEVEL = 0xFF
TRIGERLEVEL = 0xF0
PosStopF = 2.146e9
VelStopF = 16000.0

class LowStateCmdHandler(LowStateMsgHandler):
    def __init__(self, cfg, freq=1000):
        super().__init__(cfg, freq)

        if type(self.cfg["control"]["kp"]) is not dict:
            self.kp = [self.cfg["control"]["kp"]] * self.num_dof
        else:
            self.kp = [self.cfg["control"]["kp"][name] for name in cfg["control"]["dof_names"]]
        if type(self.cfg["control"]["kd"]) is not dict:
            self.kd = [self.cfg["control"]["kd"]] * self.num_dof
        else:
            self.kd = [self.cfg["control"]["kd"][name] for name in cfg["control"]["dof_names"]]

        self.default_pos = np.array([self.cfg["control"]["default_dof_pos"][name] for name in cfg["control"]["dof_names"]])
        self.target_pos = self.default_pos.copy()

        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.emergency_stop = False

        # thread handling
        self.lowCmdWriteThreadPtr = None

        self.crc = CRC()

    # Public methods
    def init(self):
        super().init()

        self.init_low_cmd()

        # create publisher #
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_go)
        self.lowcmd_publisher.Init()

        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

    def start(self):

        self.msc.ReleaseMode()

        self.initial_pos = self.joint_pos.copy()
        self.initial_stage = 0.0

        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.update_interval, target=self.LowCmdWrite, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    def recover(self):
        status, result = self.msc.CheckMode()
        while result['name'] != 'normal':
            self.msc.SelectMode("normal")
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def init_low_cmd(self):
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        self.set_stop_cmd()

    def set_stop_cmd(self):
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def set_cmd(self):
        for i in range(self.num_dof):
            self.low_cmd.motor_cmd[self.dof_index[i]].q = self.target_pos[i]
            self.low_cmd.motor_cmd[self.dof_index[i]].dq = 0
            self.low_cmd.motor_cmd[self.dof_index[i]].kp = self.kp[i]
            self.low_cmd.motor_cmd[self.dof_index[i]].kd = self.kd[i]
            self.low_cmd.motor_cmd[self.dof_index[i]].tau = 0

    def LowCmdWrite(self):

        if self.L2 and self.R2:
            self.emrgence_stop()

        if self.initial_stage < 1.0:
            self.target_pos = self.initial_pos + (self.default_pos - self.initial_pos) * self.initial_stage
            self.initial_stage += 0.001

        if self.emergency_stop:
            self.set_stop_cmd()
        else:
            self.set_cmd()

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def emrgence_stop(self):
        self.emergency_stop = True

def get_clock_inputs(t, commands):
    t = t - int(t / 1000) * 1000
    frequencies = commands[4]
    phases = commands[5]
    offsets = commands[6]
    bounds = commands[7]

    gait_indices = t * frequencies - int(t * frequencies)
    foot_indices = torch.tensor([phases + offsets + bounds, offsets, bounds, phases]) + gait_indices

    clock_inputs = torch.sin(2 * np.pi * foot_indices)

    return clock_inputs.numpy()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--robot', type=str, default='go2')
    parser.add_argument('-n', '--name', type=str, default='default')
    parser.add_argument('-c', '--cfg', type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/basic.yaml"))
    if args.cfg is not None:
        cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.cfg}.yaml"))

    # Run steta publisher
    low_state_handler = LowStateCmdHandler(cfg)
    low_state_handler.init()
    low_state_handler.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        low_state_handler.recover()
        sys.exit()
