import numpy as np
import gymnasium as gym
from airbot_env.envs.airbot_env import AirbotJointEnv

class DualAirbotEnvSingle(AirbotJointEnv):
    pass


class GripperPenaltyWrapper(gym.Wrapper):
    """Penalises gripper state transitions (open↔close).

    action_value thresholds (apply to the gripper action dimension):
      action <= action_close_threshold  → close command
      action >= action_open_threshold   → open command
      in between                        → neutral, no penalty

    obs_pos thresholds (apply to normalised gripper_pose in [0, 1] from observation):
      obs >= obs_open_threshold   → gripper considered open
      obs <= obs_close_threshold  → gripper considered closed

    Default values are set for binary mode where actions ∈ {-1, 0, +1}:
      action_close_threshold = -0.5  (catches -1, excludes 0)
      action_open_threshold  =  0.5  (catches +1, excludes 0)
      obs_open_threshold     =  0.06 (gripper ≥ ~4 mm open)
      obs_close_threshold    =  0.01 (gripper ≤ ~0.7 mm open)
    """

    def __init__(
        self,
        env,
        penalty=-0.05,
        action_close_threshold=-0.5,
        action_open_threshold=0.5,
        obs_open_threshold=0.06,
        obs_close_threshold=0.01,
        left_gripper_index=18,
        right_gripper_index=37,
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
