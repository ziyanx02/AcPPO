import time
import sys

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient as Go2SportClient
import numpy as np
import matplotlib.pyplot as plt

class SportClient:
    def __init__(self):
        try:
            ChannelFactoryInitialize(0)
        except:
            pass

        self.sport_client = Go2SportClient()
        self.sport_client.SetTimeout(3.0)
        self.sport_client.Init()

    # Public methods
    def init(self):
        pass

    def move(self, vx, vy, vz):
        self.sport_client.Move(vx, vy, vz)

if __name__ == '__main__':

    import cv2
    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    client = SportClient()

    time.sleep(0.5)

    client.move(1.0, 0, 0)