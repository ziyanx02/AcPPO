import numpy as np
import time
import threading
import yaml
import struct

import argparse

from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

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
    },
    "g1": {
        "left_hip_pitch_joint": 0,
        "left_hip_roll_joint": 1,
        "left_hip_yaw_joint": 2,
        "left_knee_joint": 3,
        "left_ankle_pitch_joint": 4,
        "left_ankle_roll_joint": 5,
        "right_hip_pitch_joint": 6,
        "right_hip_roll_joint": 7,
        "right_hip_yaw_joint": 8,
        "right_knee_joint": 9,
        "right_ankle_pitch_joint": 10,
        "right_ankle_roll_joint": 11,
    }
}

class LowStateMsgHandler:
    def __init__(self, cfg, freq=1000):

        self.cfg = cfg
        self.update_interval = 1.0 / freq
        self.robot_name = cfg["robot_name"]
        self.dof_names = cfg["environment"]["dof_names"]
        self.num_dof = len(self.dof_names)
        self.dof_index = [JointID[self.robot_name][name] for name in self.dof_names]

        self.msg = None
        self.msg_received = False

        # robot
        self.quat = np.zeros(4)
        self.ang_vel = np.zeros(3)
        self.joint_pos = np.zeros(self.num_dof)
        self.joint_vel = np.zeros(self.num_dof)
        self.torque = np.zeros(self.num_dof)
        self.temperature = np.zeros(self.num_dof)

        # button
        self.L1 = 0
        self.L2 = 0
        self.R1 = 0
        self.R2 = 0
        self.A = 0
        self.B = 0
        self.X = 0
        self.Y = 0
        self.Up = 0
        self.Down = 0
        self.Left = 0
        self.Right = 0
        self.Select = 0
        self.F1 = 0
        self.F3 = 0
        self.Start = 0
       
        # Create a thread for the main loop
        self.main_thread = threading.Thread(target=self.main_loop, daemon=True)

    def init(self):

        try:
            ChannelFactoryInitialize(0)
        except:
            pass

        if self.robot_name == "go2":
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_go)
            self.robot_lowstate_subscriber.Init(self.LowStateHandler_go, 10)
        elif self.robot_name == "g1":
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_hg)
            self.robot_lowstate_subscriber.Init(self.LowStateHandler_hg, 10)
        while not self.msg_received:
            print("Waiting for Low State Message...")
            time.sleep(0.1)
        print("Low State Message Received!!!")

        self.main_thread.start()

    def LowStateHandler_go(self, msg: LowState_go):
        self.msg = msg
        self.msg_received = True
    
    def LowStateHandler_hg(self, msg: LowState_hg):
        self.msg = msg
        self.msg_received = True

    def main_loop(self):
        total_publish_cnt = 0
        start_time = time.time()
        while True:
            update_start_time = time.time()

            # Process raw message
            self.parse_imu(self.msg.imu_state)
            self.parse_motor_state(self.msg.motor_state)
            self.parse_wireless_remote(self.msg.wireless_remote)

            if time.time() - update_start_time < self.update_interval:
                time.sleep(self.update_interval - (time.time() - update_start_time))

            # Print publishing rate
            # total_publish_cnt += 1
            # if total_publish_cnt == 1000:
            #     end_time = time.time()
            #     print("-" * 50)
            #     print(f"LowStateMsg Receiving Rate: {total_publish_cnt / (end_time - start_time)}")
            #     start_time = end_time
            #     total_publish_cnt = 0

    def parse_imu(self, imu_state):
        self.quat = np.array(imu_state.quaternion) # w, x, y, z
        self.ang_vel = np.array(imu_state.gyroscope)

    def parse_motor_state(self, motor_state):
        for i in range(self.num_dof):
            self.joint_pos[i] = motor_state[self.dof_index[i]].q
            self.joint_vel[i] = motor_state[self.dof_index[i]].dq
            self.torque[i] = motor_state[self.dof_index[i]].tau_est
            # self.temperature[i] = motor_state[self.dof_index[i]].temperature
            # error_code = motor_state[self.dof_index[i]].reserve[0]
            # if error_code != 0:
            #     print(f"Joint {self.dof_index[i]} Error Code: {error_code}")
        # print(self.joint_pos)
        # print("low_state_big_flag", self.robot_low_state.bit_flag)

    def parse_botton(self, data1, data2):
        self.R1 = (data1 >> 0) & 1
        self.L1 = (data1 >> 1) & 1
        self.Start = (data1 >> 2) & 1
        self.Select = (data1 >> 3) & 1
        self.R2 = (data1 >> 4) & 1
        self.L2 = (data1 >> 5) & 1
        self.F1 = (data1 >> 6) & 1
        self.F3 = (data1 >> 7) & 1
        self.A = (data2 >> 0) & 1
        self.B = (data2 >> 1) & 1
        self.X = (data2 >> 2) & 1
        self.Y = (data2 >> 3) & 1
        self.Up = (data2 >> 4) & 1
        self.Right = (data2 >> 5) & 1
        self.Down = (data2 >> 6) & 1
        self.Left = (data2 >> 7) & 1

    def parse_key(self, data):
        lx_offset = 4
        self.Lx = struct.unpack('<f', data[lx_offset:lx_offset + 4])[0]
        rx_offset = 8
        self.Rx = struct.unpack('<f', data[rx_offset:rx_offset + 4])[0]
        ry_offset = 12
        self.Ry = struct.unpack('<f', data[ry_offset:ry_offset + 4])[0]
        L2_offset = 16
        L2 = struct.unpack('<f', data[L2_offset:L2_offset + 4])[0] # Placeholder，unused
        ly_offset = 20
        self.Ly = struct.unpack('<f', data[ly_offset:ly_offset + 4])[0]

    def parse_wireless_remote(self, remoteData):
        self.parse_key(remoteData)
        self.parse_botton(remoteData[2], remoteData[3])

        # print("Lx:", self.Lx)
        # print("Rx:", self.Rx)
        # print("Ry:", self.Ry)
        # print("Ly:", self.Ly)

        # print("L1:", self.L1)
        # print("L2:", self.L2)
        # print("R1:", self.R1)
        # print("R2:", self.R2)
        # print("A:", self.A)
        # print("B:", self.B)
        # print("X:", self.X)
        # print("Y:", self.Y)
        # print("Up:", self.Up)
        # print("Down:", self.Down)
        # print("Left:", self.Left)
        # print("Right:", self.Right)
        # print("Select:", self.Select)
        # print("F1:", self.F1)
        # print("F3:", self.F3)
        # print("Start:", self.Start)

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
    low_state_handler = LowStateMsgHandler(cfg)
    low_state_handler.init()
    while True:
        time.sleep(1)
        pass
