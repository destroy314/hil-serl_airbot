import multiprocessing
import threading
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from airbot_py.arm import AIRBOTArm
try:
    from pynput import keyboard
except:
    pass
from franka_env.utils.rotations import quat_2_euler


class ActionDisplayer:
    """Show actions/gripper/intervention for one or more experts in a single window.
    Runs in a separate process to avoid blocking the main thread."""

    def __init__(self, name="ActionDisplayer"):
        self.name = name
        self.manager = multiprocessing.Manager()
        self.entries = self.manager.dict()  # label -> latest_data dict
        self._stop_event = multiprocessing.Event()
        self._process = None

    def register(self, label: str, latest_data):
        """Register a data source (expert) to be displayed."""
        self.entries[label] = latest_data

    def unregister(self, label: str):
        """Unregister a data source."""
        if label in self.entries:
            del self.entries[label]

    def start(self):
        """Start the display process."""
        if self._process is None or not self._process.is_alive():
            self._process = multiprocessing.Process(
                target=self._display_loop,
                args=(self.name, self.entries, self._stop_event),
                daemon=True
            )
            self._process.start()

    def stop(self):
        """Stop the display process."""
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()

    @staticmethod
    def _display_loop(name, entries, stop_event):
        """Display loop running in a separate process."""
        window_initialized = False
        
        while not stop_event.is_set():
            entries_list = list(entries.items())

            if not entries_list:
                time.sleep(0.1)
                continue

            # Layout: stacked groups
            header_h = 40
            row_height = 32
            group_margin = 30
            bar_max_width = 250
            origin_x = 360

            # Compute total height
            total_height = group_margin
            for _, data in entries_list:
                action = np.array(data.get("action", [0.0] * 6))
                total_height += header_h + len(action) * row_height + group_margin

            canvas_h = max(200, total_height)
            canvas_w = 720
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            y_cursor = group_margin
            for label, data in entries_list:
                action = np.array(data.get("action", [0.0] * 6))
                intervened = data.get("s_pressed", False) or data.get("intervened", False)
                gripper = data.get("gripper_control", [False, False])

                header = f"[{label}] Intervening: {'YES' if intervened else 'NO '}"
                grip_txt = f"Gripper Close/Open: {gripper}"
                cv2.putText(canvas, header, (20, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(canvas, grip_txt, (20, y_cursor + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 0), 2)

                max_abs = max(np.max(np.abs(action)), 1e-6)
                start_y = y_cursor + header_h

                for i, v in enumerate(action):
                    y = start_y + i * row_height
                    length = int(min(bar_max_width, abs(v) / max_abs * bar_max_width))
                    color = (0, 200, 0) if v >= 0 else (0, 0, 200)

                    cv2.putText(canvas, f"a{i}: {v:+.3f}", (20, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.line(canvas, (origin_x - bar_max_width, y), (origin_x + bar_max_width, y), (80, 80, 80), 1)

                    if v >= 0:
                        cv2.rectangle(canvas, (origin_x, y - 10), (origin_x + length, y + 10), color, -1)
                    else:
                        cv2.rectangle(canvas, (origin_x - length, y - 10), (origin_x, y + 10), color, -1)

                y_cursor = start_y + len(action) * row_height + group_margin

            if not window_initialized:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(name, canvas.shape[1], canvas.shape[0])
                window_initialized = True

            cv2.imshow(name, canvas)
            cv2.waitKey(1)
            time.sleep(0.05)


# Shared singleton displayer to allow multiple experts in one window
_global_displayer = None
_displayer_lock = threading.Lock()


def get_action_displayer():
    global _global_displayer
    with _displayer_lock:
        if _global_displayer is None:
            _global_displayer = ActionDisplayer()
            _global_displayer.start()
        return _global_displayer


class AirbotCartesianExpert:
    """WIP, SpaceMouseExpert performs better."""
    def __init__(self, port):
        self.robot = AIRBOTArm(port=port)
        self.robot.connect()

        self.intervened = False
        self.prev_pos = None
        self.prev_euler = None

        def on_press(key):
            if hasattr(key, "char") and key.char == "s":
                self.intervened = not self.intervened
                print("Intervened:", self.intervened)
                if self.intervened:
                    pose = self.robot.get_end_pose()
                    if pose is not None:
                        self.prev_pos = np.array(pose[0])
                        self.prev_euler = quat_2_euler(pose[1])
                else:
                    self.prev_pos = None
                    self.prev_euler = None

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()

    def get_action(self):
        action = np.zeros(6)
        gripper = 0.0

        if self.intervened:
            pose = self.robot.get_end_pose()  # [[x,y,z], [qx, qy, qz, w]]
            if pose is not None:
                current_pos = np.array(pose[0])
                current_euler = quat_2_euler(pose[1])
                if self.prev_pos is not None and self.prev_euler is not None:
                    action[:3] = current_pos - self.prev_pos
                    # action[3:] = (current_euler - self.prev_euler + np.pi) % (2 * np.pi) - np.pi
                    # 在 SO(3) 旋转群上做正确的旋转差值再转回欧拉角
                    prev_rotmax = R.from_euler("xyz", self.prev_euler, degrees=False)
                    curr_rotmax = R.from_euler("xyz", current_euler, degrees=False)
                    delta_rot = curr_rotmax * prev_rotmax.inv()
                    action[3:] = delta_rot.as_euler("xyz", degrees=False)
                self.prev_pos = current_pos
                self.prev_euler = current_euler

            gripper_pos = self.robot.get_eef_pos()
            if gripper_pos is not None:
                gripper = gripper_pos[0]

        return np.array(action), gripper, self.intervened

    def close(self):
        self.listener.stop()
        self.robot.disconnect()

class AirbotJointExpert():
    def __init__(self, leader_port, follower_port, display_action=False):
        self.leader_port = leader_port
        self.leader = AIRBOTArm(port=leader_port)
        self.leader.connect()
        self.follower = AIRBOTArm(port=follower_port)
        self.follower.connect()
        self.intervened = False
        self.leader_start_pos = np.zeros(6)
        self.follower_start_pos = np.zeros(6)

        def on_press(key):
            if hasattr(key, "char") and key.char == "s":
                self.intervened = not self.intervened
                print("Intervened:", self.intervened)
                if self.intervened:
                    self.leader_start_pos = np.array(self.leader.get_joint_pos())
                    self.follower_start_pos = np.array(self.follower.get_joint_pos())
                    # print(leader_port, "Leader Start Pos:", self.leader_start_pos, "Follower Start Pos:", self.follower_start_pos)

        self.key_listener = keyboard.Listener(on_press=on_press)
        self.key_listener.start()

    def get_action(self):
        if self.intervened:
            leader_pos = np.array(self.leader.get_joint_pos())
            action = leader_pos - self.leader_start_pos + self.follower_start_pos
            # print(self.leader_port, "Leader Pos:", leader_pos, "Action:", action, gripper)
            gripper = self.leader.get_eef_pos()[0]
        else:
            # will be ignored
            action = np.zeros(6)
            gripper = 0.0
        return np.array(action), gripper, self.intervened

    def close(self):
        self.key_listener.stop()
        self.leader.disconnect()
        self.follower.disconnect()
