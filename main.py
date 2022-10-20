
from panda_butler.envs.butler_tasks import ButlerStackEnv
from butler.utils import show_video
import numpy as np
import time


env = ButlerStackEnv(render=False)

observation, info = env.reset()
pos = env.sim.get_base_position(env.robot.body_name)
pos += np.array([0.5, 0.0, 0.0])

roll = 0.0
pitch = -30
yaw = 0.0
distance = 0.5

frames = []

t = time.time()
for _ in range(100):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render(mode='rgb_array', 
                     width = 80,
                     height = 60,
                     target_position=pos,
                     pitch=pitch,
                     roll=roll,
                     yaw=yaw,
                     distance=distance))
    print("FPS: ", 1/(time.time()-t))
    t = time.time()
    

show_video(frames)