import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_butler.envs.tasks.butler_stacking_task import ButlerStackingTask
from panda_gym.pybullet import PyBullet
from typing import Dict, Any, Tuple, List, Optional
import time
import pybullet as p


class ButlerPyBullet(PyBullet):
    def __init__(self, render: bool = False, n_substeps: int = 20, background_color: Optional[np.ndarray] = None) -> None:
        super().__init__(render, n_substeps, background_color)

    def render(
        self,
        mode: str = "human",
        width: int = 720,
        height: int = 480,
        target_position: Optional[np.ndarray] = None,
        distance: float = 1.4,
        yaw: float = 45,
        pitch: float = -30,
        roll: float = 0,
    ) -> Optional[np.ndarray]:
        """Render.
        If mode is "human", make the rendering real-time. All other arguments are
        unused. If mode is "rgb_array", return an RGB array of the scene.
        Args:
            mode (str): "human" of "rgb_array". If "human", this method waits for the time necessary to have
                a realistic temporal rendering and all other args are ignored. Else, return an RGB array.
            width (int, optional): Image width. Defaults to 720.
            height (int, optional): Image height. Defaults to 480.
            target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
                Defaults to [0., 0., 0.].
            distance (float, optional): Distance of the camera. Defaults to 1.4.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -30.
            roll (int, optional): Rool of the camera. Defaults to 0.
        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        target_position = target_position if target_position is not None else np.zeros(
            3)
        if mode == "human":
            self.physics_client.configureDebugVisualizer(
                self.physics_client.COV_ENABLE_SINGLE_STEP_RENDERING)
            time.sleep(self.dt)  # wait to seems like real speed
        if mode == "rgb_array":
            # if self.connection_mode == p.DIRECT:
            #     # warnings.warn(
            #     #     "The use of the render method is not recommended when the environment "
            #     #     "has not been created with render=True. The rendering will probably be weird. "
            #     #     "Prefer making the environment with option `render=True`. For example: "
            #     #     "`env = gym.make('PandaReach-v3', render=True)`.",
            #     #     UserWarning,
            #     # )
            view_matrix = self.physics_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target_position,
                distance=distance,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                upAxisIndex=2,
            )
            proj_matrix = self.physics_client.computeProjectionMatrixFOV(
                fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0
            )
            (_, _, px, depth, seg) = self.physics_client.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )

            return {"rgb": px, "depth": depth, "seg": seg}


class ButlerStackEnv(RobotTaskEnv):
    """Custom Stack task wih Panda robot.
    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = ButlerPyBullet(render=render)
        robot = Panda(sim, block_gripper=False, base_position=np.array(
            [-0.6, 0.0, 0.0]), control_type=control_type)
        task = ButlerStackingTask(sim, reward_type=reward_type)
        super().__init__(robot, task)
