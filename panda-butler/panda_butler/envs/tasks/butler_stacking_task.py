from typing import Any, Dict, Tuple

import numpy as np

from panda_gym.envs.tasks.stack import Stack
from panda_gym.utils import distance


class ButlerStackingTask(Stack):
    
    white = np.array([1.0, 1.0, 1.0, 1.0])
    red = np.array([1.0, 0.0, 0.0, 1.0])
    blue = np.array([0.0, 0.0, 1.0, 1.0])
    transparent = np.array([0.0, 0.0, 0.0, 0.0])
    
    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            ghost=False,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=ButlerStackingTask.red,
        )
        self.sim.create_box(
            body_name="target1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=ButlerStackingTask.transparent,
        )
        self.sim.create_box(
            body_name="object2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            ghost=False,
            position=np.array([0.5, 0.0, 0.05]),
            rgba_color=ButlerStackingTask.red,
        )
        self.sim.create_box(
            body_name="target2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.5, 0.0, 0.05]),
            rgba_color=ButlerStackingTask.transparent,
        )
        
    def reset(self) -> None:
        self.goal = self._sample_goal()
        object1_position = np.array([0.0, 0.0, self.object_size / 2])
        object2_position = np.array([0.0, 0.0, self.object_size / 2 + self.object_size]) 
        self.sim.set_base_pose("target1", np.array([0.0,0.0,-1e3]), np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target2", np.array([0.0,0.0,-1e3]), np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object1", object1_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object2", object2_position, np.array([0.0, 0.0, 0.0, 1.0]))