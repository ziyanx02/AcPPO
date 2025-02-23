import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__HeightMap_
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

import numpy as np
import matplotlib.pyplot as plt

class HeightMapHandler:
    def __init__(self):

        self.width = 128
        self.height = 128
        self.resolution = 0.06
        self.origin = [0, 0]
        self.stamp = 0
        self.data = 1e9 * np.ones(self.width * self.height)

        self.crc = CRC()

    # Public methods
    def init(self):

        try:
            ChannelFactoryInitialize(0)
        except:
            pass

        self.lidar_switch_publisher = ChannelPublisher("rt/utlidar/switch", String_)
        self.lidar_switch_publisher.Init()
        self.lidar_switch = std_msgs_msg_dds__String_()
        self.lidar_switch.data = "ON"
        self.lidar_switch_publisher.Write(self.lidar_switch)

        # create subscriber # 
        self.heightmap_subscriber = ChannelSubscriber("rt/utlidar/height_map_array", HeightMap_)
        self.heightmap_subscriber.Init(self.LowStateMessageHandler, 10)

    def LowStateMessageHandler(self, msg: HeightMap_):
        if msg.stamp > self.stamp + 0.1:
            self.msg = msg
            self.width = msg.width
            self.height = msg.height
            self.resolution = msg.resolution
            self.origin = msg.origin
            self.stamp = msg.stamp
            self.data = msg.data

    @property
    def height_map(self):
        height_map = np.reshape(np.array(self.data), (self.width, self.height)).T
        return height_map

if __name__ == '__main__':

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    height_map_handler = HeightMapHandler()
    height_map_handler.init()

    time.sleep(0.5)
    width = height_map_handler.width
    height = height_map_handler.height

    while True:
        height_map = height_map_handler.height_map
        height_map[height_map == 1e9] = 1
        plt.imshow(1 - height_map, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')  # Hide axis for a cleaner look
        print("origin", height_map_handler.origin)
        plt.show()