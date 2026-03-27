#!/usr/bin/env python3
"""
Compute ABS_POSE_LIMIT_LOW / ABS_POSE_LIMIT_HIGH from Cartesian demo data.

Reads tcp_pose observations (xyz + quaternion, 7D per arm) stored by the
airbot_cart wrapper stack (ChunkingWrapper obs_horizon=1 → state shape (1, 16)):
state[0:7]  = left/tcp_pose
state[7]    = left/gripper_pose
state[8:15] = right/tcp_pose
state[15]   = right/gripper_pose

Outputs ready-to-paste Python for EnvConfigLeft / EnvConfigRight.

Usage:
    cd examples
    python compute_pose_limits.py \\
        --demo_path=experiments/airbot_cart/demo_data/1.pkl \\
        [--margin_xyz=0.05] [--margin_rpy=0.1]
"""

import json
import pickle as pkl
from typing import List

import numpy as np
from absl import app, flags
from scipy.spatial.transform import Rotation

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("demo_path", None, "Path(s) to demo pkl file(s).")
flags.DEFINE_float("margin_xyz", 0.02, "Safety margin to add beyond observed xyz range(metres).")
flags.DEFINE_float("margin_rpy", 0.05, "Safety margin to add beyond observed rpy range(radians).")
flags.DEFINE_string(
    "output_json", None,
    "If set, also write limits to this JSON file (for reference).",
)

# Indices within the flattened state vector (airbot_cart proprio layout)
_LEFT_POSE_SLICE  = slice(0, 7)   # xyz + quat
_RIGHT_POSE_SLICE = slice(8, 15)  # xyz + quat


def _load_tcp_poses(demo_paths: List[str]):
    """Return (left_poses, right_poses) each as (N, 6) arrays of xyz+euler_xyz."""
    left_all, right_all = [], []
    total = 0
    for path in demo_paths:
        with open(path, "rb") as f:
            transitions = pkl.load(f)
        for t in transitions:
            state = np.asarray(t["observations"]["state"]).flatten()
            left_all.append(state[_LEFT_POSE_SLICE])
            right_all.append(state[_RIGHT_POSE_SLICE])
            total += 1
        print(f"  {path}: {len(transitions)} transitions")

    print(f"Total transitions: {total}")

    def to_xyz_euler(poses_7d):
        """(N, 7) xyz+quat  →  (N, 6) xyz+euler_xyz."""
        poses_7d = np.array(poses_7d)
        xyz   = poses_7d[:, :3]
        euler = Rotation.from_quat(poses_7d[:, 3:]).as_euler("xyz")
        return np.concatenate([xyz, euler], axis=1)

    return to_xyz_euler(left_all), to_xyz_euler(right_all)


def _compute_limits(poses_6d: np.ndarray, margin_xyz: float, margin_rpy: float):
    """Return (low, high) each 6D, with safety margin applied."""
    margin = np.array([margin_xyz, margin_xyz, margin_xyz,
                    margin_rpy,  margin_rpy,  margin_rpy])
    low  = poses_6d.min(axis=0) - margin
    high = poses_6d.max(axis=0) + margin
    return low, high


def _print_report(name: str, poses_6d: np.ndarray, low: np.ndarray, high: np.ndarray):
    labels = ["x  ", "y  ", "z  ", "roll ", "pitch", "yaw  "]
    units  = ["m", "m", "m", "rad", "rad", "rad"]
    print(f"\n{'='*62}")
    print(f"  {name} arm  (N={len(poses_6d)} samples)")
    print(f"{'='*62}")
    print(f"  {'dim':<7}  {'min_obs':>10}  {'max_obs':>10}  {'low_limit':>10}  {'high_limit':>10}unit")
    print(f"  {'-'*60}")
    for i, (lbl, unit) in enumerate(zip(labels, units)):
        print(
            f"  {lbl:<7}  {poses_6d[:, i].min():10.4f}  {poses_6d[:, i].max():10.4f}"
            f"  {low[i]:10.4f}  {high[i]:10.4f}  {unit}"
        )
    print()
    low_s  = ", ".join(f"{v:.4f}" for v in low)
    high_s = ", ".join(f"{v:.4f}" for v in high)
    print(f"  # Paste into EnvConfig{name.capitalize()}:")
    print(f"  ABS_POSE_LIMIT_LOW  = np.array([{low_s}])")
    print(f"  ABS_POSE_LIMIT_HIGH = np.array([{high_s}])")


def _check_clip_safety_box():
    """Verify clip_safety_box euler[0] logic and report any issues."""
    print(f"\n{'='*62}")
    print("  clip_safety_box sanity check")
    print(f"{'='*62}")

    # Replicate the current implementation
    def clip_current(euler0, low0, high0):
        sign = np.sign(euler0)
        return sign * np.clip(np.abs(euler0), low0, high0)

    def clip_correct(euler0, low0, high0):
        return np.clip(euler0, low0, high0)

    # Asymmetric limits matching default config:
    # RESET_POSE[3]=0.1, margin=0.5 → low=-0.4, high=0.6
    low0, high0 = -0.4, 0.6
    test_cases = [
        ("positive, in range",  0.3),
        ("positive, too high",  0.8),
        ("negative, in range", -0.2),
        ("negative, too low",  -0.6),  # should clip to -0.4 but doesn't
        ("exactly zero",        0.0),
    ]

    has_bug = False
    print(f"  limits: low={low0}, high={high0}  (e.g. RESET_POSE roll=0.1 ± 0.5)")
    print(f"  {'case':<25}  {'input':>8}  {'current':>10}  {'correct':>10}  {'ok?':>4}")
    print(f"  {'-'*58}")
    for desc, val in test_cases:
        cur = clip_current(val, low0, high0)
        cor = clip_correct(val, low0, high0)
        ok = np.isclose(cur, cor)
        if not ok:
            has_bug = True
        print(f"  {desc:<25}  {val:8.3f}  {cur:10.4f}  {cor:10.4f}  {'OK' if ok else 'BUG':>4}")

    if has_bug:
        print()
        print("  BUG DETECTED: sign * clip(|x|, low, high) != clip(x, low, high)")
        print("  for asymmetric limits (low != -high).")
        print()
        print("  Fix in airbot_env/envs/airbot_env.py, clip_safety_box():")
        print("    Replace:")
        print("      sign = np.sign(euler[0])")
        print("      euler[0] = sign * np.clip(np.abs(euler[0]), low[0], high[0])")
        print("    With:")
        print("      euler[0] = np.clip(euler[0], low[0], high[0])")
        print()
        print("  The 'discontinuity' comment is misleading: for any reasonable")
        print("  workspace limit (< pi rad), plain clip is correct and safe.")
    else:
        print("\n  No issues found.")


def main(_):
    assert FLAGS.demo_path, "Must provide at least one --demo_path"

    print(f"Loading {len(FLAGS.demo_path)} demo file(s)...")
    left_poses, right_poses = _load_tcp_poses(FLAGS.demo_path)

    left_low,  left_high  = _compute_limits(left_poses,  FLAGS.margin_xyz, FLAGS.margin_rpy)
    right_low, right_high = _compute_limits(right_poses, FLAGS.margin_xyz, FLAGS.margin_rpy)

    _print_report("left",  left_poses,  left_low,  left_high)
    _print_report("right", right_poses, right_low, right_high)

    _check_clip_safety_box()

    if FLAGS.output_json:
        out = {
            "margin_xyz": FLAGS.margin_xyz,
            "margin_rpy": FLAGS.margin_rpy,
            "left":  {"low": left_low.tolist(),  "high": left_high.tolist()},
            "right": {"low": right_low.tolist(), "high": right_high.tolist()},
        }
        with open(FLAGS.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {FLAGS.output_json}")


if __name__ == "__main__":
    app.run(main)