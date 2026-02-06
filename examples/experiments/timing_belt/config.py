import os
import jax
import jax.numpy as jnp
import numpy as np

from airbot_env.envs.dual_airbot_env import DualAirbotEnv
from airbot_env.envs.wrappers import DualAirbotIntervention
from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.timing_belt.wrapper import TimingBeltEnv

_l = 100
_c = 150

class EnvConfigLeft(DefaultEnvConfig):
    NAME = "left"
    SERVER_PORT = 50051
    V4L2_CAMERAS = {
        "wrist": {
            "index": 6,
            "dim": (640, 480),
            "fps": 30,
            # "fake": True,
        },
        "env": {
            "index": 0,
            "dim": (640, 480),
            "fps": 30,
            # "fake": True,
        },
    }
    # IMAGE_CROP = {
    #     "wrist": lambda img: img[240-_l:240+_l, 320-_l:320+_l],
    #     "env": lambda img: img[240-_c:240+_c, 320-_c:320+_c],
    # }
    IMAGE_CROP = {
        "wrist": lambda img: img[:, 80:560],
        "env": lambda img: img[:, 80:560],
    }
    TARGET_POSE = np.array([0.0] * 6) # TODO
    RESET_JOINT_POS = np.array([-0.05, -0.26, 0.56, 1.48, -1.19, -1.34, 0.0])
    JOINT_POS_LIMIT_LOW = RESET_JOINT_POS - np.array([0.2] * 7) #TODO
    JOINT_POS_LIMIT_HIGH = RESET_JOINT_POS + np.array([0.2] * 7)
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.01, 1, 1)
    DISPLAY_IMAGE = False
    # MAX_EPISODE_LENGTH = 100
    MAX_EPISODE_LENGTH = int(1e9)

class EnvConfigRight(EnvConfigLeft):
    NAME: str = "right"
    SERVER_PORT = 50053
    V4L2_CAMERAS = {
        "wrist": {
            "index": 12,
            "dim": (640, 480),
            "fps": 30,
            # "fake": True,
        },
    }
    RESET_JOINT_POS = np.array([-0.05, -0.26, 0.56, -1.48, 1.19, 1.34, 0.0])
    JOINT_POS_LIMIT_LOW = RESET_JOINT_POS - np.array([0.2] * 7) #TODO
    JOINT_POS_LIMIT_HIGH = RESET_JOINT_POS + np.array([0.2] * 7)

class TrainConfig(DefaultTrainingConfig):
    image_keys = ["left/wrist", "right/wrist", "left/env"]
    classifier_keys = ["left/wrist", "right/wrist", "left/env"]
    proprio_keys = [
        "left/joint_pos", "left/joint_vel", "left/joint_torque", "left/gripper_pose",
        "right/joint_pos", "right/joint_vel", "right/joint_torque", "right/gripper_pose"
    ]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "dual-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        left_env = TimingBeltEnv(
            hz=10,
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfigLeft(),
        )
        right_env = TimingBeltEnv(
            hz=10,
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfigRight(),
        )
        env = DualAirbotEnv(left_env, right_env, display_images=not fake_env)
        if not fake_env:
            env = DualAirbotIntervention(
                env,
                left_leader_port=50050,
                left_follower_port=50051,
                right_leader_port=50052,
                right_follower_port=50053,
            )
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        # env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=15, infer_delay=0.11)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                s = sigmoid(classifier(obs))[0]
                print(s)
                return int(s > 0.85)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env