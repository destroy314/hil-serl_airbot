import numpy as np
import gymnasium as gym
from airbot_env.envs.airbot_env import AirbotCartesianEnv


class AirbotCartEnvSingle(AirbotCartesianEnv):
    """Single-arm Cartesian env for airbot_cart experiment.

    Uses SERVO_CART_POSE mode. Action: [dx, dy, dz, drx, dry, drz, gripper] (7D).
    Two of these are combined by DualAirbotEnv into a 14D dual-arm env.
    """
    pass


class GripperPenaltyWrapper(gym.Wrapper):
    """Penalises gripper state transitions (open↔close) for Cartesian arm control.

    Adapted from dual_airbot.wrapper.GripperPenaltyWrapper for the Cartesian
    observation layout and raw-meter gripper_pose values.

    Observation state layout (after SERLObsWrapper with proprio_keys =
    ["left/tcp_pose", "left/gripper_pose", "right/tcp_pose", "right/gripper_pose"]):
        indices  0-6  : left/tcp_pose  (7D, xyz+quat)
        index    7    : left/gripper_pose  (raw metres, 0–0.07 m)
        indices  8-14 : right/tcp_pose (7D)
        index   15    : right/gripper_pose

    Action space (14D, indices 6 and 13 are gripper):
        Binary mode actions ∈ {≈-1, 0, ≈+1}
          action <= action_close_threshold  → close command
          action >= action_open_threshold   → open command
          in between                        → neutral, no penalty

    Gripper-pose thresholds (raw metres, gripper_max_length = 0.07 m):
        obs >= obs_open_threshold   → gripper considered open  (default 0.035 m, half-open)
        obs <= obs_close_threshold  → gripper considered closed (default 0.010 m)
    """

    def __init__(
        self,
        env,
        penalty=-0.05,
        action_close_threshold=-0.5,
        action_open_threshold=0.5,
        obs_open_threshold=0.035,    # raw metres: > half of 0.07 m → open
        obs_close_threshold=0.010,   # raw metres: < 10 mm → closed
        left_gripper_index=7,
        right_gripper_index=15,
    ):
        super().__init__(env)
        self.penalty = penalty
        self.action_close_threshold = action_close_threshold
        self.action_open_threshold = action_open_threshold
        self.obs_open_threshold = obs_open_threshold
        self.obs_close_threshold = obs_close_threshold
        self.left_gripper_index = left_gripper_index
        self.right_gripper_index = right_gripper_index
        self.last_left_gripper_pos = None
        self.last_right_gripper_pos = None

    def _extract_state_pos(self, state, index):
        state_array = np.asarray(state)
        if state_array.ndim == 2:
            state_array = state_array[0]
        return float(state_array[index])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_left_gripper_pos = self._extract_state_pos(
            obs["state"], self.left_gripper_index
        )
        self.last_right_gripper_pos = self._extract_state_pos(
            obs["state"], self.right_gripper_index
        )
        return obs, info

    def _compute_penalty(self, action_value, last_pos):
        """Penalise when action requests closing while gripper was open, or vice versa."""
        if action_value <= self.action_close_threshold and last_pos >= self.obs_open_threshold:
            return self.penalty
        if action_value >= self.action_open_threshold and last_pos <= self.obs_close_threshold:
            return self.penalty
        return 0.0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        action = np.asarray(action).reshape(-1)
        info["grasp_penalty"] = 0.0
        info["grasp_penalty"] += self._compute_penalty(
            action[6], self.last_left_gripper_pos
        )
        info["grasp_penalty"] += self._compute_penalty(
            action[13], self.last_right_gripper_pos
        )

        self.last_left_gripper_pos = self._extract_state_pos(
            observation["state"], self.left_gripper_index
        )
        self.last_right_gripper_pos = self._extract_state_pos(
            observation["state"], self.right_gripper_index
        )
        return observation, reward, terminated, truncated, info
