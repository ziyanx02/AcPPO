import time
import sys

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient as Go2VideoClient

import numpy as np
import matplotlib.pyplot as plt

class VideoClient:
    def __init__(self):
        try:
            ChannelFactoryInitialize(0)
        except:
            pass

        self.video_client = Go2VideoClient()
        self.video_client.SetTimeout(3.0)
        self.video_client.Init()

    # Public methods
    def init(self):
        pass

    def get_image(self):
        code, data = self.video_client.GetImageSample()

        if code != 0:
            print("get image sample error. code:", code)
        else:
            image_data = np.frombuffer(bytes(data), dtype=np.uint8)
            return image_data

if __name__ == '__main__':

    import cv2
    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    client = VideoClient()

    time.sleep(0.5)

    image_id = 0
    while True:
        input("Press Enter to save image...")
        image_data = client.get_image()
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        # Display image
        cv2.imshow("front_camera", image)
        # Press ESC to stop
        if cv2.waitKey(20) == 27:
            break

        cv2.imwrite(f"image_{image_id}.jpg", image)
        image_id += 1