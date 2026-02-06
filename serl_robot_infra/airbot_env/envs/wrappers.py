import gymnasium as gym
import numpy as np
from airbot_env.airbot.airbot_expert import AirbotJointExpert

class AirbotIntervention(gym.ActionWrapper):
    """WIP"""
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
    def __init__(self, env, left_leader_port, left_follower_port, right_leader_port, right_follower_port):
        super().__init__(env)

        self.left_expert = AirbotJointExpert(left_leader_port, left_follower_port)
        self.right_expert = AirbotJointExpert(right_leader_port, right_follower_port)

    def action(self, action: np.ndarray):
        expert_a, gripper_a, intervened = self.left_expert.get_action()
        expert_b, gripper_b, _ = self.right_expert.get_action()
        gripper_a /= self.env.env_left.gripper_max_length
        gripper_b /= self.env.env_right.gripper_max_length
        expert = np.concatenate((expert_a, [gripper_a], expert_b, [gripper_b]), axis=0)
        
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
        return self.env.reset(**kwargs)