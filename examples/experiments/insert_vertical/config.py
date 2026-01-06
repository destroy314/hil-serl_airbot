import os
import jax
import jax.numpy as jnp
import numpy as np

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
from experiments.insert_vertical.wrapper import InsertVerticalEnv

_l = 100

class EnvConfig(DefaultEnvConfig):
    NAME: str = "main"
    SERVER_PORT = 50053
    V4L2_CAMERAS = {
        "wrist": {
            "index": 2,
            "dim": (640, 480),
            "fps": 30,
        },
        "env": {
            "index": 0,
            "dim": (640, 480),
            "fps": 30,
        },
    }
    IMAGE_CROP = {
        "wrist": lambda img: img[240-_l:240+_l, 320-_l:320+_l],
        "env": lambda img: img[240-_l:240+_l, 320-_l:320+_l],
    }
    TARGET_POSE = np.array([0.26, 0.275, 0.112, 0, np.pi/2, 0])
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.08, 0, 0, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.02, 0.09, 0.01, 0.1, 0.4])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.01, 1, 1)
    DISPLAY_IMAGE = True
    # MAX_EPISODE_LENGTH = 100
    MAX_EPISODE_LENGTH = int(1e9)

class TrainConfig(DefaultTrainingConfig):
    leader_port = 50052
    image_keys = ["wrist", "env"]
    classifier_keys = ["wrist", "env"]
    # proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    proprio_keys = ["tcp_pose", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = InsertVerticalEnv(
            hz=10,
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        env = GripperCloseEnv(env)
        if not fake_env:
            # env = AirbotIntervention(env, port=self.leader_port)
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
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
                # added check for z position to further robustify classifier, but should work without as well
                # return int(sigmoid(classifier(obs)[0]) > 0.85 and obs['state'][0, 6] > 0.04)
                return int(sigmoid(classifier(obs)[0]) > 0.85)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env