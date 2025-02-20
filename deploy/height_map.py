import time
import threading
import pickle

from utils.height_map_handler import HeightMapHandler

import numpy as np
import matplotlib.pyplot as plt

class HeightMapThread(threading.Thread):
    def __init__(self, handler: HeightMapHandler):
        super().__init__()
        self.handler = handler
        self.height_map = self.handler.height_map
        self.origin = (
            int(self.handler.origin[0] / self.handler.resolution),
            int(self.handler.origin[1] / self.handler.resolution),
        )
        self.original_origin = (
            int(self.handler.origin[0] / self.handler.resolution),
            int(self.handler.origin[1] / self.handler.resolution),
        )
        self.width = self.handler.width
        self.height = self.handler.height
        self.running = True

    def run(self):
        while self.running:
            height_map = self.handler.height_map
            origin = (
                int(self.handler.origin[0] / self.handler.resolution),
                int(self.handler.origin[1] / self.handler.resolution),
            )
            new_width = max(self.origin[0] + self.width, origin[0] + self.handler.width) - min(self.origin[0], origin[0])
            new_height = max(self.origin[1] + self.height, origin[1] + self.handler.height) - min(self.origin[1], origin[1])
            if new_width != self.width or new_height != self.height:
                new_origin = (
                    min(self.origin[0], origin[0]),
                    min(self.origin[1], origin[1]),
                )
                new_height_map = 1e9 * np.ones((new_width, new_height))
                new_height_map[
                    self.origin[0] - new_origin[0]:self.origin[0] - new_origin[0] + self.width,
                    self.origin[1] - new_origin[1]:self.origin[1] - new_origin[1] + self.height,
                ] = self.height_map
                self.height_map = new_height_map
                self.origin = new_origin
                self.width = new_width
                self.height = new_height
            self.height_map[
                origin[0] - self.origin[0]:origin[0] - self.origin[0] + self.handler.width,
                origin[1] - self.origin[1]:origin[1] - self.origin[1] + self.handler.height,
            ] = height_map * (height_map < 1) + self.height_map[
                origin[0] - self.origin[0]:origin[0] - self.origin[0] + self.handler.width,
                origin[1] - self.origin[1]:origin[1] - self.origin[1] + self.handler.height,
            ] * (height_map >= 1)

            time.sleep(0.5)

    def stop(self):
        self.running = False

if __name__ == '__main__':

    handler = HeightMapHandler()
    handler.init()

    time.sleep(0.3)

    height_map_thread = HeightMapThread(handler)
    height_map_thread.start()

    try:
        while True:
            input("Press ENTER to plot full height map...")
            height_map = height_map_thread.height_map
            height_map[height_map > 1] = 1
            print(height_map_thread.width, height_map_thread.height)
            plt.imshow(1 - height_map, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')  # Hide axis for a cleaner look
            plt.show()
    except KeyboardInterrupt:
        print("Stopping thread...")
        height_map_thread.stop()
        height_map_thread.join()