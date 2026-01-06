import copy
import time
from franka_env.utils.rotations import euler_2_quat_scipy
from scipy.spatial.transform import Rotation as R
import numpy as np
import requests
try:
    from pynput import keyboard
except:
    pass

from airbot_env.envs.airbot_env import AirbotEnv

class InsertVerticalEnv(AirbotEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_regrasp = False

        def on_press(key):
            if str(key) == "Key.f1":
                self.should_regrasp = True

        try:
            listener = keyboard.Listener(
                on_press=on_press)
            listener.start()
        except:
            pass

    # def go_to_reset(self, joint_reset=False):
    #     """
    #     Move to the rest position defined in base class.
    #     Add a small z offset before going to rest to avoid collision with object.
    #     """        
    #     # use compliance mode for coupled reset
    #     self._update_currpos()
    #     self._send_pos_command(self.currpos)
    #     time.sleep(0.3)

    #     # pull up
    #     self._update_currpos()
    #     reset_pose = copy.deepcopy(self.currpos)
    #     reset_pose[2] = self.reset_pose[2] + 0.04
    #     self.interpolate_move(reset_pose, timeout=1)

    #     super().go_to_reset(joint_reset=joint_reset)


    def regrasp(self):
        input("Press enter to release gripper...")
        self._send_gripper_command(1.0)
        input("Place stick in holder and press enter to grasp...")
        self._send_gripper_command(0.0)
        time.sleep(1)


    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        # if True:
        if self.should_regrasp:
            self.regrasp()
            self.should_regrasp = False

        self.go_to_reset(joint_reset=False)
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {}
