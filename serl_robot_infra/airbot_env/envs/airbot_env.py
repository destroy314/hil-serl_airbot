import os
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import queue
import threading
from datetime import datetime
from typing import Callable, Dict, Literal
try:
    from pynput import keyboard
except:
    pass

from franka_env.camera.video_capture import VideoCapture
from franka_env.utils.rotations import euler_2_quat_scipy, quat_2_euler
from airbot_env.camera.v4l2_capture import V4L2Capture

from airbot_py.arm import AIRBOTArm, RobotMode, SpeedProfile


class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name
        self._window_initialized = False

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            # Enlarge per-camera view for better visibility
            frame = np.concatenate(
                [cv2.resize(v, (512, 512)) for k, v in img_array.items() if "full" in k], axis=1
            )

            if not self._window_initialized:
                # Allow manual window resizing with the mouse
                cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.name, frame.shape[1], frame.shape[0])
                self._window_initialized = True

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


##############################################################################


class DefaultEnvConfig:
    """Default configuration for AirbotEnv. Fill in the values below."""

    NAME: str = "main"
    SERVER_PORT: int = 50051
    V4L2_CAMERAS: Dict = {
        "wrist": {"index": 0},
    }
    IMAGE_CROP: dict[str, Callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((6,))
    RESET_POSE: np.ndarray = np.zeros((6,))
    RESET_JOINT_POS: np.ndarray = np.zeros((7,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,)) # move, rotate, gripper
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    JOINT_POS_LIMIT_LOW = np.zeros((7,))
    JOINT_POS_LIMIT_HIGH = np.zeros((7,))
    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 0.6
    MAX_EPISODE_LENGTH: int = 100


##############################################################################


class AirbotCartesianEnv(gym.Env):
    """For cartesian speed (twist) control"""
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
    ):
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.name = config.NAME
        self.port = config.SERVER_PORT
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP
        self.gripper_max_length = 0.07

        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # x,y,z,qx,qy,qz,w
                        # "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),  # we need jacobi matrix to calculate it
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        # "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        # "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                                for key in config.V4L2_CAMERAS}
                ),
            }
        )

        if fake_env:
            return

        self.cap = None
        self.init_cameras(config.V4L2_CAMERAS)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.name)
            self.displayer.start()

        self.terminate = False
        def on_press(key):
            if key == keyboard.Key.esc:
                self.terminate = True
        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()

        self.robot = AIRBOTArm(port=self.port)
        self.robot.connect()
        self.robot.switch_mode(RobotMode.SERVO_CART_POSE)
        # self.robot.set_speed_profile(SpeedProfile.SLOW)
        self.robot.set_speed_profile(SpeedProfile.DEFAULT)

        self._update_currpos()

        print(f"Initialized {self.name} arm.")

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        action[:3]*=self.action_scale[0]
        action[3:6]*=self.action_scale[1]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + action[:3]

        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (
            Rotation.from_rotvec(action[3:6])
            # Rotation.from_rotvec(action[3:6] * self.action_scale[1])
            # Rotation.from_euler('xyz',action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()
        # print(Rotation.from_quat(self.currpos[3:]).as_euler('xyz'),Rotation.from_quat(self.nextpos[3:]).as_euler('xyz'))

        gripper_action = action[6] * self.action_scale[2]

        self._send_gripper_command(gripper_action)
        # self._send_pos_command(self.clip_safety_box(self.nextpos))
        self._send_pos_command(self.nextpos)

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        return ob, int(reward), done, False, {"succeed": reward}

    def compute_reward(self, obs) -> bool:
        current_pose = obs["state"]["tcp_pose"]
        # convert from quat to euler first
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T  @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))
        # print(f"Delta: {delta}")
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {self._REWARD_THRESHOLD}')
            return False

    def get_im(self, allow_empty=True) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
            except queue.Empty:
                if allow_empty:
                    h, w = self.observation_space["images"][key].shape[:2]
                    resized = np.zeros((h, w, 3), dtype=np.uint8)
                    images[key] = resized[..., ::-1]
                    display_images[key + "_full"] = resized
                    continue
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.V4L2_CAMERAS)
                return self.get_im()

        # Store full resolution cropped images separately
        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image:
            self.img_queue.put(display_images)
        return images

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat_scipy(goal[3:])])
        steps = int(timeout * self.hz)
        self._update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self.nextpos = p
        self._update_currpos()

    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # halt
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self._RESET_POSE.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_random
        else:
            reset_pose = self._RESET_POSE.copy()
        
        self.planning_move(reset_pose)
    
    def planning_move(self, target_pose):
        self.robot.switch_mode(RobotMode.PLANNING_POS)
        target_pos = target_pose[:3]
        target_quat = euler_2_quat_scipy(target_pose[3:])
        self.robot.move_to_cart_pose(cart_pose=[target_pos.tolist(), target_quat.tolist()], blocking=True)
        self.robot.switch_mode(RobotMode.SERVO_CART_POSE)

    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        self.go_to_reset(joint_reset=joint_reset)
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.recording_frames[0].keys():
                    video_path = f'./videos/{self.name}_{camera_key}_{timestamp}.mp4'
                    
                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    
                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")
                
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_index_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = {}
        for cam_name, kwargs in name_index_dict.items():
            cap = VideoCapture(
                V4L2Capture(name=cam_name, **kwargs)
            )
            self.cap[cam_name] = cap

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _send_pos_command(self, pos: np.ndarray):
        """Internal function to send position command to the robot."""
        arr = np.array(pos).astype(np.float32).tolist()
        self.robot.servo_cart_pose([arr[:3], arr[3:]])

    def _send_gripper_command(self, pos: float, mode="binary"):
        """Internal function to send gripper command to the robot."""
        # print(pos, self.curr_gripper_pos)
        if mode == "binary":
            if (pos <= 0.5) and (self.curr_gripper_pos > 0.5) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # close gripper
                self.robot.servo_eef_pos(0.0)
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            elif (pos >= 0.5) and (self.curr_gripper_pos < 0.5) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # open gripper
                self.robot.servo_eef_pos(1.0)
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            else: 
                return
        elif mode == "continuous":
            self.robot.servo_eef_pos(pos * self.gripper_max_length)

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        pose = self.robot.get_end_pose()
        assert pose is not None
        self.currpos = np.array(pose[0] + pose[1])

        # 笛卡尔只提了供位置接口
        # self.currvel = np.array(ps["vel"])

        # self.currforce = np.array(ps["force"])
        # self.currtorque = np.array(ps["torque"])
        # self.currjacobian = np.reshape(np.array(ps["jacobian"]), (6, 7))

        # self.q = np.array(ps["q"])
        # self.dq = np.array(ps["dq"])

        self.curr_gripper_pos = np.array(self.robot.get_eef_pos()) / self.gripper_max_length

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            # "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            # "tcp_force": self.currforce,
            # "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.robot.disconnect()
        self.close_cameras()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()


class AirbotJointEnv(AirbotCartesianEnv):
    """For joint absolute position control"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.RESET_JOINT_POS = kwargs["config"].RESET_JOINT_POS

        self.action_space = gym.spaces.Box(
            kwargs["config"].JOINT_POS_LIMIT_LOW,
            kwargs["config"].JOINT_POS_LIMIT_HIGH,
            dtype=np.float32,
        )
        
        self.observation_space["state"] = gym.spaces.Dict(
            {
                "joint_pos": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "joint_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                "joint_torque": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
            }
        )

        if kwargs["fake_env"]:
            return

        self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)
    
    def step(self, action):
        start_time = time.time()

        self._send_gripper_command(float(action[6]), mode="continuous")
        self._send_joint_command(action[:6])

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        return ob, int(reward), done, False, {"succeed": reward}

    def _send_joint_command(self, pos: np.ndarray):
        self.robot.servo_joint_pos(pos.tolist())

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        self.curr_joint_pos = np.array(self.robot.get_joint_pos())
        self.curr_joint_vel = np.array(self.robot.get_joint_vel())
        self.curr_joint_eff = np.array(self.robot.get_joint_eff())

        pose = self.robot.get_end_pose()
        self.currpos = np.array(pose[0] + pose[1])

        self.curr_gripper_pos = np.array(self.robot.get_eef_pos()) / self.gripper_max_length

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "joint_pos": self.curr_joint_pos,
            "joint_vel": self.curr_joint_vel,
            "gripper_pose": self.curr_gripper_pos,
            "joint_torque": self.curr_joint_eff,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def reset(self, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        self.robot.switch_mode(RobotMode.PLANNING_POS)
        self.robot.move_to_joint_pos(joint_pos=self.RESET_JOINT_POS.tolist()[:6], blocking=True)
        self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)
        self._send_gripper_command(float(self.RESET_JOINT_POS[6]), mode="continuous")
        
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        print("reset used time:", time.time() - self.last_gripper_act)
        return obs, {"succeed": False}