import numpy as np
import time
import threading
import yaml

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
    }

}

class LowStateMsgHandler:
    def __init__(self, cfg):

        self.cfg = cfg

        self.msg = None
        self.msg_received = False

        self.num_dof = 12

        self.quat = np.zeros(4)
        self.ang_vel = np.zeros(3)
        self.joint_pos = np.zeros(self.num_dof)
        self.joint_vel = np.zeros(self.num_dof)
        self.torque = np.zeros(self.num_dof)
        self.temperature = np.zeros(self.num_dof)

        # Create a thread for the main loop
        self.main_thread = threading.Thread(target=self.main_loop, daemon=True)

    def Init(self):

        ChannelFactoryInitialize(0)

        self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_go)
        self.robot_lowstate_subscriber.Init(self.LowStateHandler_go, 10)
        if "go2" in self.cfg["robot"]["asset_path"]:
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_go)
            self.robot_lowstate_subscriber.Init(self.LowStateHandler_go, 10)
            try: 
                dof_names = self.cfg["control"]["dof_names"]
                self.dof_index = [JointID["go2"][name] for name in dof_names]
            except:
                self.dof_index = list(range(self.num_dof))
        elif "g1" in self.cfg["robot"]["asset_path"]:
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_hg)
            self.robot_lowstate_subscriber.Init(self.LowStateHandler_hg, 10)
            raise NotImplementedError
        while not self.msg_received:
            print("Waiting for Low State Message...")
            time.sleep(0.1)
        print("Low State Message Received!!!")

    def Start(self):
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
            time.sleep(0.001)

            # Process raw message
            imu_state = self.msg.imu_state
            self.quat = np.array(imu_state.quaternion) # w, x, y, z
            self.ang_vel = np.array(imu_state.gyroscope)

            motor_state = self.msg.motor_state
            for i in range(self.num_dof):
                # import ipdb; ipdb.set_trace()
                self.joint_pos[i] = motor_state[self.dof_index[i]].q
                self.joint_vel[i] = motor_state[self.dof_index[i]].dq
                self.torque[i] = motor_state[self.dof_index[i]].tau_est
                self.temperature[i] = motor_state[self.dof_index[i]].temperature
                error_code = motor_state[self.dof_index[i]].reserve[0]
                if error_code != 0:
                    print(f"Joint {self.dof_index[i]} Error Code: {error_code}")
            # print("low_state_big_flag", self.robot_low_state.bit_flag)

            # Print publishing rate
            total_publish_cnt += 1
            if total_publish_cnt == 1000:
                end_time = time.time()
                print("-" * 50)
                print(f"LowStateMsg Receiving Rate: {total_publish_cnt / (end_time - start_time)}")
                start_time = end_time
                total_publish_cnt = 0

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
    low_state_handler.Init()
    low_state_handler.Start()
    while True:
        time.sleep(1)
        pass
