import os
import jax
import jax.numpy as jnp
import numpy as np

from airbot_env.envs.airbot_env import DefaultEnvConfig
from airbot_env.envs.dual_airbot_env import DualAirbotEnv
from airbot_env.envs.wrappers import DualAirbotCartesianIntervention
from franka_env.envs.wrappers import MultiCameraBinaryRewardClassifierWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.airbot_cart.wrapper import AirbotCartEnvSingle, GripperPenaltyWrapper

_l = 100


class EnvConfigLeft(DefaultEnvConfig):
    NAME = "left"
    SERVER_PORT = 50051
    V4L2_CAMERAS = {
        "wrist": {"index": 2, "dim": (640, 480), "fps": 30},
        "env":   {"index": 0, "dim": (640, 480), "fps": 30},
    }
    IMAGE_CROP = {
        "wrist": lambda img: img[:, 80:560],
        "env":   lambda img: img[:, 80:560],
    }
    # Cartesian reset/target poses: [x, y, z, roll, pitch, yaw]  (meters / radians)
    # !! Calibrate these values for your physical setup !!
    RESET_POSE  = np.array([0.21, 0.0, 0.17, 0.1, 0.89, -0.18])
    TARGET_POSE = np.array([10] * 6) # far from obs space so the reward keeps 0
    REWARD_THRESHOLD = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
    # fill this by compute_pose_limit.py
    ABS_POSE_LIMIT_LOW  = np.array([0.1877, -0.0260, 0.1490, 0.0021, 0.8386, -0.2711])
    ABS_POSE_LIMIT_HIGH = np.array([0.2301, 0.0201, 0.1902, 0.1512, 0.9464, -0.1291])
    # action_scale: (position_m, rotation_rad, gripper)
    # With scale=(1,1,1) the raw delta from the leader arm is applied directly.
    # ACTION_SCALE = (1.6, 1.6, 1.0) # for human
    # ACTION_SCALE = (0.02, 0.02, 1.0) # for not argmax
    ACTION_SCALE = (1.0, 1.0, 1.0) # for slower argmax
    DISPLAY_IMAGE = False
    MAX_EPISODE_LENGTH = int(1e9)
    GRIPPER_MODE = "binary"
    SPEED_FAST = True


class EnvConfigRight(EnvConfigLeft):
    NAME = "right"
    SERVER_PORT = 50053
    V4L2_CAMERAS = {
        "wrist": {"index": 4, "dim": (640, 480), "fps": 30},
    }
    # !! Calibrate for your right arm !!
    RESET_POSE  = np.array([0.21, 0.0, 0.17, -0.1, 0.89, 0.08])
    TARGET_POSE = np.array([10] * 6)
    ABS_POSE_LIMIT_LOW  = np.array([0.1901, -0.0251, 0.0037, -0.2112, 0.8201, 0.0079])
    ABS_POSE_LIMIT_HIGH = np.array([0.3624, 0.3500, 0.2129, 0.6772, 1.4853, 1.4016])


class TrainConfig(DefaultTrainingConfig):
    # Images from both wrists + left global view
    image_keys      = ["left/wrist", "right/wrist", "left/env"]
    classifier_keys = ["left/wrist", "right/wrist", "left/env"]

    # Proprio: tcp_pose (xyz+quat = 7D) + gripper (1D) per arm → 16D total
    proprio_keys = [
        "left/tcp_pose",  "left/gripper_pose",
        "right/tcp_pose", "right/gripper_pose",
    ]

    buffer_period     = 1000
    checkpoint_period = 5000
    steps_per_update  = 50
    training_starts   = 25 * 10
    encoder_type      = "resnet-pretrained"

    # SACAgentHybridDualArm: 12D continuous + 2×3-way discrete gripper (indices 6, 13)
    setup_mode = "dual-arm-learned-gripper"

    pretrain_steps = 1000
    pretrain_loss  = "bc"

    # Enable action normalisation after recording demos and computing stats:
    #   python examples/compute_action_derivative_stats.py \
    #       --demo_path=demo_data/<your_demo>.pkl \
    #       --output_path=demo_data/airbot_cart_stats.json
    # Then set:
    #   action_norm_scale = 0.9
    #   action_stats_path = "demo_data/airbot_cart_stats.json"
    action_norm_scale      = 1.0   # disabled until demo stats are computed
    action_stats_path      = "demo_data/1.json"
    action_derivative_scale = 0.0  # Cartesian deltas don't require derivative constraints

    # actor_kwargs = {
    #     "std_max": 0.15,
    # }
    actor_kwargs = {}

    # Leader arm ports (must differ from follower ports)
    left_leader_port  = 50050
    right_leader_port = 50052
    rel_ctrl = True

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        left_env = AirbotCartEnvSingle(
            hz=10,
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfigLeft(),
        )
        right_env = AirbotCartEnvSingle(
            hz=10,
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfigRight(),
        )

        env = DualAirbotEnv(left_env, right_env, display_images=not fake_env)

        if not fake_env:
            env = DualAirbotCartesianIntervention(
                env,
                left_leader_port=self.left_leader_port,
                right_leader_port=self.right_leader_port,
            )

        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        if classifier:
            classifier_fn = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return int(sigmoid(classifier_fn(obs))[0] > 0.85)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)

        env = GripperPenaltyWrapper(env, penalty=-0.05)
        return env
