import argparse
import multiprocessing
import os
import time
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import torch

import genesis as gs


class GUI:
    def __init__(self, root: tk.Tk, cfg: dict, values, save_callback: callable = None, reset_callback: callable = None):

        self.root = root
        self.root.title("Joint Controller")

        self.save_callback = save_callback
        self.reset_callback = reset_callback

        self.cfg = cfg
        self.labels = self.cfg["label"]
        self.values = values
        self.sliders = []
        self.value_labels = []
        self.create_widgets()

    def create_widgets(self):
        for i, name in enumerate(self.labels):

            # Get the min and max limits for the slider
            min_limit, max_limit = self.cfg["range"][name][:2]
            frame = tk.Frame(self.root)
            frame.pack(pady=5, padx=10, fill=tk.X)

            tk.Label(frame, text=f"{name}", font=("Arial", 12), width=20).pack(side=tk.LEFT)

            slider = ttk.Scale(
                frame,
                from_=float(min_limit),
                to=float(max_limit),
                orient=tk.HORIZONTAL,
                length=300,
                command=lambda val, idx=i: self.update_values(idx, val),
            )
            slider.pack(side=tk.LEFT, padx=5)
            slider.set(self.values[i])
            self.sliders.append(slider)

            value_label = tk.Label(frame, text=f"{slider.get():.2f}", font=("Arial", 12))
            value_label.pack(side=tk.LEFT, padx=5)
            self.value_labels.append(value_label)

            # Update label dynamically
            def update_label(s=slider, l=value_label):
                def callback(event):
                    l.config(text=f"{s.get():.2f}")

                return callback

            slider.bind("<Motion>", update_label())

        if self.save_callback is not None:
            tk.Button(self.root, text="Save", font=("Arial", 12), command=self.save).pack(pady=10)

        if self.reset_callback is not None:
            tk.Button(self.root, text="Reset", font=("Arial", 12), command=self.reset).pack(pady=10)

    def update_values(self, idx, val):
        self.values[idx] = float(val)

    def save(self):
        self.save_callback(self.values)

    def reset(self):
        self.reset_callback()

def start_gui(cfg, values, save_callback=None, reset_callback=None):

    # Start GUI in a separate thread
    stop_event = threading.Event()
    is_gui_closed = [False,]

    def on_close():
        is_gui_closed[0] = True
        stop_event.set()

    def start_gui():
        root = tk.Tk()
        app = GUI(root, cfg, values, save_callback=save_callback, reset_callback=reset_callback)
        root.protocol("WM_DELETE_WINDOW", on_close)
        root.mainloop()

    gui_thread = threading.Thread(target=start_gui, daemon=True)
    gui_thread.start()

if __name__ == "__main__":
    values = [0, 0, 0]
    start_gui(
        cfg={"label": ["joint1", "joint2", "joint3"], "range": {"joint1": [-1, 1], "joint2": [-1, 1], "joint3": [-1, 1]}},
        values=values,
        save_callback=lambda x: print(x),
        reset_callback=lambda: print("reset")
    )
    while True:
        time.sleep(0.5)
