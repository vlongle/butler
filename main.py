
from panda_butler.envs.butler_tasks import ButlerStackEnv
from butler.utils import show_video
from pl_bolts.models.detection.faster_rcnn.faster_rcnn_module import FasterRCNN
import numpy as np
import time

import gymnasium as gym
import panda_gym

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

width = 640
height = 480

frames = []

t = time.time()
for _ in range(50):
    #action = env.action_space.sample() # random action
    action = np.zeros(env.action_space.shape)
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render(mode='rgb_array', 
                     width = width,
                     height = height,
                     target_position=pos,
                     pitch=pitch,
                     roll=roll,
                     yaw=yaw,
                     distance=distance))
    print("FPS: ", 1/(time.time()-t))
    t = time.time()

model = FasterRCNN(pretrained=True)
print(model)