
import torch
from panda_butler.envs.butler_tasks import ButlerStackEnv
from butler.utils import show_video
from pl_bolts.models.detection.faster_rcnn.faster_rcnn_module import FasterRCNN
import numpy as np
import time
import cv2

import gymnasium as gym
import panda_gym
import matplotlib.pyplot as plt

render = False
#env = gym.make("PandaStack-v3", render=True)
env = ButlerStackEnv(render=render)

observation, info = env.reset()
pos = env.sim.get_base_position(env.robot.body_name)
pos += np.array([0.5, 0.0, 0.0])

roll = 0.0
pitch = -30
yaw = 0.0
distance = 0.5

width = 224
height = 224

frames = []
segs = []
T = 100
t = time.time()
for _ in range(T):
    action = env.action_space.sample()  # random action
    #action = np.zeros(env.action_space.shape)
    observation, reward, terminated, truncated, info = env.step(action)
    ret = env.render(mode='rgb_array',
                     width=width,
                     height=height,
                     target_position=pos,
                     pitch=pitch,
                     roll=roll,
                     yaw=yaw,
                     distance=distance)
    frames.append(ret["rgb"])
    segs.append(ret["seg"])
    print("FPS: ", 1/(time.time()-t))
    t = time.time()


def annotate_img(img, seg):
    """
    Draw the bounding box around the segmentation
    mask of the two stacks
    """
    # god-given ids for the two stacks
    obj = 5
    obj2 = 3
    mask = np.logical_or(seg == obj, seg == obj2)
    # get the bounding box of the mask
    y, x = np.where(mask)
    y1, y2 = y.min(), y.max()
    x1, x2 = x.min(), x.max()
    im_arr_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.rectangle(im_arr_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return im_arr_bgr


annotated_frames = [annotate_img(f, s) for f, s in zip(frames, segs)]
show_video(annotated_frames)
# show_video(frames)
