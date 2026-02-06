import copy
import time
from franka_env.utils.rotations import euler_2_quat_scipy
from scipy.spatial.transform import Rotation as R
import numpy as np
import requests
from airbot_env.envs.airbot_env import AirbotJointEnv

class TimingBeltEnv(AirbotJointEnv):
    pass
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)


    # def reset(self, joint_reset=False, **kwargs):
    #     self.last_gripper_act = time.time()
    #     if self.save_video:
    #         self.save_video_recording()

    #     self.go_to_reset(joint_reset=False)
    #     self.curr_path_length = 0

    #     self._update_currpos()
    #     obs = self._get_obs()
    #     self.terminate = False
    #     return obs, {}
