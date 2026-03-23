import gymnasium as gym
import numpy as np
import json
from collections import deque
from airbot_env.envs.action_normalizer import ActionNormalizer
from airbot_env.airbot.airbot_expert import AirbotJointExpert, AirbotCartesianExpert


class DeltaJointActionWrapper(gym.Wrapper):
    """Converts delta joint actions to absolute joint actions.

    The policy outputs small delta joint angles, which are added to the current
    joint positions to produce absolute joint commands sent to the robot.
    Gripper actions remain absolute (pass-through).

    Should wrap *outside* the intervention wrapper so that both policy and
    expert actions are consistently represented in delta space.
    """

    def __init__(self, env, max_delta=0.05):
        super().__init__(env)
        self.max_delta = max_delta
        self._current_joint_pos = None

        # Store original absolute action space bounds for clipping
        self._abs_action_low = self.env.action_space.low.copy()
        self._abs_action_high = self.env.action_space.high.copy()

        n = len(self._abs_action_low)
        half = n // 2  # 7 per arm (6 joints + 1 gripper)

        # Joint indices (non-gripper): 0..5 for left, 7..12 for right
        self._joint_indices = list(range(0, half - 1)) + list(range(half, n - 1))

        # Build delta action space: joints use [-max_delta, max_delta],
        # gripper dimensions keep original bounds
        delta_low = self._abs_action_low.copy()
        delta_high = self._abs_action_high.copy()
        for i in self._joint_indices:
            delta_low[i] = -max_delta
            delta_high[i] = max_delta

        self.action_space = gym.spaces.Box(delta_low, delta_high, dtype=np.float32)

    def _extract_joint_pos(self, obs):
        """Extract current joint positions + gripper from observation."""
        left_jp = obs["state"]["left/joint_pos"]    # (6,)
        right_jp = obs["state"]["right/joint_pos"]  # (6,)
        left_g = obs["state"]["left/gripper_pose"]  # (1,)
        right_g = obs["state"]["right/gripper_pose"]  # (1,)
        return np.concatenate([left_jp, left_g, right_jp, right_g])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_joint_pos = self._extract_joint_pos(obs)
        return obs, info

    def step(self, action):
        # Convert delta joints to absolute; gripper passes through
        absolute_action = action.copy()
        for i in self._joint_indices:
            absolute_action[i] = self._current_joint_pos[i] + action[i]
        # Clip to original absolute joint position limits
        # absolute_action = np.clip(
        #     absolute_action, self._abs_action_low, self._abs_action_high
        # )

        obs, rew, done, truncated, info = self.env.step(absolute_action)

        # Convert expert intervention action from absolute to delta
        if "intervene_action" in info:
            intervene_abs = info["intervene_action"]
            intervene_delta = intervene_abs.copy()
            for i in self._joint_indices:
                intervene_delta[i] = intervene_abs[i] - self._current_joint_pos[i]
                intervene_delta[i] = np.clip(
                    intervene_delta[i], -self.max_delta, self.max_delta
                )
            info["intervene_action"] = intervene_delta

        # Update current joint positions from new observation
        self._current_joint_pos = self._extract_joint_pos(obs)

        return obs, rew, done, truncated, info

class ActionDerivativeConstraintWrapper(gym.Wrapper):
    """Constrains action derivatives (velocity, acceleration, jerk) to stay within specified limits.

    This wrapper maintains a history of past actions and their derivatives, then clips
    the current action to ensure its derivatives don't exceed the configured limits.
    Useful for ensuring smooth, physically plausible robot motions.

    Usage:
        1. First compute derivative statistics from human demonstrations:
           python examples/compute_action_derivative_stats.py \
               --demo_path=demo1.pkl --demo_path=demo2.pkl \
               --output_path=action_derivative_stats.json

        2. Then wrap your environment:
           env = ActionDerivativeConstraintWrapper(
               env,
               stats_path="action_derivative_stats.json",
               scale=1.0  # Use 1.0 for exact demo limits, >1.0 to be more permissive
           )

    Args:
        env: The environment to wrap
        stats_path: Path to JSON file with derivative statistics from demonstrations
        scale: Scale factor for the limits (1.0 = exact demo limits, >1.0 = more permissive)
        max_velocity: Optional manual override for max velocity (action change per step)
        max_acceleration: Optional manual override for max acceleration
        max_jerk: Optional manual override for max jerk
    """

    def __init__(
        self,
        env: gym.Env,
        stats_path: str = "",
        scale: float = 1.0,
    ):
        super().__init__(env)
        self.scale = scale

        # Load limits from file or use manual overrides
        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.max_velocity = np.array(stats["max_velocity"], dtype=np.float32) * scale
        self.max_acceleration = np.array(stats["max_acceleration"], dtype=np.float32) * scale
        self.max_jerk = np.array(stats["max_jerk"], dtype=np.float32) * scale
        print(f"[ActionDerivativeConstraintWrapper] Loaded limits from {stats_path} (scale={scale})")

        # History: store last 3 actions to compute up to jerk
        # [t-3, t-2, t-1] where t-1 is the most recent
        self.action_history = deque(maxlen=3)

        # Store last derivatives
        self.last_velocity = None  # a(t-1) - a(t-2)
        self.last_acceleration = None  # v(t-1) - v(t-2)

    def reset(self, **kwargs):
        """Reset the wrapper state."""
        obs, info = self.env.reset(**kwargs)
        self.action_history.clear()
        # self.action_history.append(obs["state"]["joint_pos"])
        self.last_velocity = None
        self.last_acceleration = None
        return obs, info

    def step(self, action: np.ndarray):
        """Execute action after constraining its derivatives.

        Args:
            action: Desired action from the policy

        Returns:
            obs, reward, done, truncated, info
        """
        action = np.asarray(action, dtype=np.float32).copy()

        # If we don't have history yet, just record and execute
        if len(self.action_history) == 0:
            constrained_action = action
            print("NOT CONSTRAINED")
        else:
            # Constrain based on available history
            constrained_action = self._constrain_action(action)

        # Execute the constrained action
        obs, reward, done, truncated, info = self.env.step(constrained_action)

        # Update history
        if "intervene_action" in info:
            # If intervention happened, the actual executed action is not the constrained action
            actual_action = np.asarray(info["intervene_action"], dtype=np.float32)
            self.action_history.append(actual_action.copy())
        else:
            self.action_history.append(constrained_action.copy())
            info["executed_action"] = constrained_action.copy()

        # # Store constraint info for debugging
        # if not np.allclose(action, constrained_action):
        #     info["action_constrained"] = True
        #     info["action_constraint_diff"] = np.linalg.norm(action - constrained_action)
        #     print("action_constrained", info["action_constraint_diff"])

        return obs, reward, done, truncated, info

    def _constrain_action(self, action: np.ndarray) -> np.ndarray:
        """Constrain action to satisfy derivative limits.

        Args:
            action: Desired action

        Returns:
            Constrained action
        """
        last_action = self.action_history[-1]

        # Compute desired velocity (1st derivative)
        desired_velocity = action - last_action

        # Constrain velocity
        velocity = np.clip(desired_velocity, -self.max_velocity, self.max_velocity)
        action = last_action + velocity

        # # Constrain acceleration (2nd derivative) if we have enough history
        # if len(self.action_history) >= 2 and self.last_velocity is not None:
        #     desired_acceleration = velocity - self.last_velocity
        #     acceleration = np.clip(
        #         desired_acceleration, -self.max_acceleration, self.max_acceleration
        #     )
        #     velocity = self.last_velocity + acceleration
        #     action = last_action + velocity

        # # Constrain jerk (3rd derivative) if we have enough history
        # if len(self.action_history) >= 3 and self.last_acceleration is not None:
        #     desired_jerk = acceleration - self.last_acceleration
        #     jerk = np.clip(desired_jerk, -self.max_jerk, self.max_jerk)
        #     acceleration = self.last_acceleration + jerk
        #     velocity = self.last_velocity + acceleration
        #     action = last_action + velocity

        # # Update stored derivatives for next step
        # self.last_velocity = velocity.copy()
        # if len(self.action_history) >= 2:
        #     self.last_acceleration = acceleration.copy()

        return action


class AirbotIntervention(gym.ActionWrapper):
    """WIP, use DualAirbotIntervention instead. Single-arm intervention using SpaceMouse cartesian commands."""
    def __init__(self, env, leader_port, follower_port, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = AirbotJointExpert(leader_port, follower_port)
        self.left, self.right = False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray):
        expert_a, intervened, gripper = self.expert.get_action()
        g_close, g_open = gripper
        
        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if self.gripper_enabled:
            if g_close:  # close gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif g_open:  # open gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if intervened:
            return expert_a, True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info


class DualAirbotIntervention(gym.ActionWrapper):
    def __init__(self, env, left_leader_port, left_follower_port, right_leader_port, right_follower_port, gripper_mode="continuous"):
        super().__init__(env)

        self.gripper_mode = gripper_mode
        self.left_expert = AirbotJointExpert(left_leader_port, left_follower_port)
        self.right_expert = AirbotJointExpert(right_leader_port, right_follower_port)

        # Previous closed-state for edge detection (binary mode). None = uninitialized.
        self._left_prev_closed: bool | None = None
        self._right_prev_closed: bool | None = None

    def _gripper_to_binary(self, raw_pos: float, prev_closed: bool | None):
        """Convert raw eef position (metres) to a {≈-1, 0, ≈+1} edge-triggered action.

        Mimics the franka SpaceMouse button convention:
          - Gripper just closed (open→close):  uniform(-1, -0.9)  → rounds to -1
          - Gripper just opened (close→open):  uniform(0.9,  1)   → rounds to +1
          - No state change:                   0.0                → rounds to  0

        Threshold at half of gripper_max_length (0.035 m).
        """
        HALF = 0.035
        is_closed = raw_pos < HALF

        if prev_closed is None:
            # First call after reset – treat as no transition
            return 0.0, is_closed

        if is_closed and not prev_closed:
            return float(np.random.uniform(-1, -0.9)), is_closed   # open → close
        elif not is_closed and prev_closed:
            return float(np.random.uniform(0.9, 1)), is_closed     # close → open
        else:
            return 0.0, is_closed                                   # no change

    def action(self, action: np.ndarray):
        expert_a, gripper_a, intervened = self.left_expert.get_action()
        expert_b, gripper_b, _ = self.right_expert.get_action()

        if self.gripper_mode == "binary":
            left_gripper, self._left_prev_closed = self._gripper_to_binary(
                gripper_a, self._left_prev_closed
            )
            right_gripper, self._right_prev_closed = self._gripper_to_binary(
                gripper_b, self._right_prev_closed
            )
        else:
            left_gripper = gripper_a
            right_gripper = gripper_b

        expert = np.concatenate((expert_a, [left_gripper], expert_b, [right_gripper]), axis=0)

        if intervened:
            return expert, True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        self.left_expert.intervened = False
        self.right_expert.intervened = False
        self._left_prev_closed = None
        self._right_prev_closed = None
        return self.env.reset(**kwargs)

    def close(self):
        self.left_expert.close()
        self.right_expert.close()
        self.env.close()


class DualAirbotCartesianIntervention(gym.ActionWrapper):
    """Dual-arm Cartesian expert intervention using leader-arm pose deltas.

    Each leader arm's end-effector displacement (from the moment 's' is pressed)
    is used as the relative Cartesian action for the corresponding follower arm.
    Human holds 's' to intervene; releasing 's' returns control to the policy.

    Action format (14D):
        [dx_l, dy_l, dz_l, drx_l, dry_l, drz_l, gripper_l,
         dx_r, dy_r, dz_r, drx_r, dry_r, drz_r, gripper_r]

    Gripper encoding (binary, edge-triggered, same convention as DualAirbotIntervention):
        open→close transition:  uniform(-1, -0.9)  → close command
        close→open transition:  uniform(0.9,  1)   → open command
        no state change:        0.0                → neutral
    """

    def __init__(self, env, left_leader_port, right_leader_port):
        super().__init__(env)
        self.left_expert = AirbotCartesianExpert(port=left_leader_port)
        self.right_expert = AirbotCartesianExpert(port=right_leader_port)
        # Previous gripper closed-state for edge detection. None = uninitialized.
        self._left_prev_closed: bool | None = None
        self._right_prev_closed: bool | None = None

    def _gripper_to_binary(self, raw_pos: float, prev_closed: "bool | None"):
        """Convert raw eef position (metres) to a {≈-1, 0, ≈+1} edge-triggered action.

        Identical convention to DualAirbotIntervention._gripper_to_binary.
        Threshold at half of gripper_max_length (0.035 m).
        """
        HALF = 0.035
        is_closed = raw_pos < HALF

        if prev_closed is None:
            return 0.0, is_closed

        if is_closed and not prev_closed:
            return float(np.random.uniform(-1, -0.9)), is_closed   # open → close
        elif not is_closed and prev_closed:
            return float(np.random.uniform(0.9, 1)), is_closed     # close → open
        else:
            return 0.0, is_closed                                   # no change

    def action(self, action: np.ndarray):
        left_a, left_g_raw, left_intervened = self.left_expert.get_action()
        right_a, right_g_raw, right_intervened = self.right_expert.get_action()

        left_g, self._left_prev_closed = self._gripper_to_binary(
            left_g_raw, self._left_prev_closed
        )
        right_g, self._right_prev_closed = self._gripper_to_binary(
            right_g_raw, self._right_prev_closed
        )

        expert = np.concatenate([left_a, [left_g], right_a, [right_g]])

        if left_intervened or right_intervened:
            return expert, True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        self.left_expert.intervened = False
        self.right_expert.intervened = False
        self._left_prev_closed = None
        self._right_prev_closed = None
        return self.env.reset(**kwargs)

    def close(self):
        self.left_expert.close()
        self.right_expert.close()
        self.env.close()
