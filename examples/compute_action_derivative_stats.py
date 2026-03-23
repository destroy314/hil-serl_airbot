#!/usr/bin/env python3
"""
Compute action derivative statistics (velocity, acceleration, jerk) from demonstration data.

This script analyzes human demonstrations to extract the maximum derivatives of actions,
which can then be used to constrain policy actions for smoother, more human-like behavior.
Actions are considered absolute (positions).

Derivatives computed:
- Velocity (1st derivative): action[t] - action[t-1]
- Acceleration (2nd derivative): velocity[t] - velocity[t-1]
- Jerk (3rd derivative): acceleration[t] - acceleration[t-1]

Usage:
    python compute_action_derivative_stats.py \
        --demo_path=path/to/demo1.pkl \
        --demo_path=path/to/demo2.pkl \
        --output_path=action_derivative_stats.json

The output JSON file can be passed to ActionDerivativeConstraintWrapper via stats_path.
"""

import json
import os
import pickle as pkl
from typing import List, Tuple

import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("demo_path", None, "Path to demo pkl file(s).")
flags.DEFINE_string("output_path", None, "Output JSON path. Defaults to <first_demo_path>.json")
flags.DEFINE_bool("verbose", True, "Print detailed statistics.")
flags.DEFINE_bool("rel", False, "Relative (delta) mode: compute min/max of per-step action deltas "
                  "for joint dims, and absolute min/max for gripper dims.")
flags.DEFINE_string("gripper_indices", "6,13",
                    "Comma-separated gripper dimension indices (used in --rel mode). "
                    "These dims use absolute action bounds; joint dims use delta bounds.")


def compute_trajectory_derivatives(actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute velocity, acceleration, and jerk for a single trajectory.

    Args:
        actions: Array of shape (T, action_dim) where T is trajectory length

    Returns:
        velocity: (T-1, action_dim)
        acceleration: (T-2, action_dim)
        jerk: (T-3, action_dim)
    """
    # Velocity: first difference
    velocity = np.diff(actions, axis=0)

    # Acceleration: second difference
    acceleration = np.diff(velocity, axis=0)

    # Jerk: third difference
    jerk = np.diff(acceleration, axis=0)

    return velocity, acceleration, jerk


def _load_trajectories(demo_paths):
    """Load demo transitions and group into per-episode trajectory arrays."""
    trajectories = []
    total_transitions = 0
    for path in demo_paths:
        with open(path, "rb") as f:
            transitions = pkl.load(f)
        current_traj = []
        for t in transitions:
            action = np.asarray(t["actions"], dtype=np.float32).reshape(-1)
            current_traj.append(action)
            total_transitions += 1
            if t.get("dones", False) or t.get("masks", 1.0) == 0.0:
                if current_traj:
                    trajectories.append(np.array(current_traj))
                    current_traj = []
        if current_traj:
            trajectories.append(np.array(current_traj))
        print(f"  {path}: {len(transitions)} transitions")
    return trajectories, total_transitions


def main(_):
    assert FLAGS.demo_path is not None, "Must provide at least one --demo_path"

    output_path = FLAGS.output_path
    if output_path is None:
        base = os.path.splitext(FLAGS.demo_path[0])[0]
        num_demos = len(FLAGS.demo_path)
        if num_demos > 1:
            base = f"{base}_{num_demos}"
        output_path = base + ".json"

    print(f"Loading demonstrations from {len(FLAGS.demo_path)} file(s)...")
    trajectories, total_transitions = _load_trajectories(FLAGS.demo_path)

    if FLAGS.rel:
        # --rel mode: compute min/max of per-step deltas for joint dims,
        # and absolute min/max for gripper dims.
        gripper_indices = set(int(i) for i in FLAGS.gripper_indices.split(",") if i.strip())
        print(f"\nRelative (delta) mode. Gripper indices (absolute bounds): {sorted(gripper_indices)}")

        all_abs = np.concatenate(trajectories, axis=0)  # (N, action_dim)
        all_deltas = np.concatenate(
            [np.diff(traj, axis=0) for traj in trajectories if len(traj) >= 2], axis=0
        )  # (M, action_dim)

        action_dim = all_abs.shape[1]
        min_action = np.zeros(action_dim, dtype=np.float32)
        max_action = np.zeros(action_dim, dtype=np.float32)
        for i in range(action_dim):
            if i in gripper_indices:
                min_action[i] = all_abs[:, i].min()
                max_action[i] = all_abs[:, i].max()
            else:
                min_action[i] = all_deltas[:, i].min()
                max_action[i] = all_deltas[:, i].max()

        if FLAGS.verbose:
            print(f"\n{'Dim':>4s}  {'Type':>11s}  {'min_action':>12s}  {'max_action':>12s}")
            print("-" * 46)
            for i in range(action_dim):
                kind = "gripper(abs)" if i in gripper_indices else "joint(delta)"
                print(f"  {i:2d}  {kind:>11s}  {min_action[i]:12.6f}  {max_action[i]:12.6f}")

        stats = {
            "action_dim": int(action_dim),
            "num_samples": int(total_transitions),
            "rel_ctrl": True,
            "gripper_indices": sorted(gripper_indices),
            "min_action": min_action.tolist(),
            "max_action": max_action.tolist(),
        }

    else:
        # Default absolute mode: compute full derivative stats.
        all_actions_list = []
        all_velocities = []
        all_accelerations = []
        all_jerks = []

        for traj_actions in trajectories:
            if len(traj_actions) < 4:
                continue
            all_actions_list.append(traj_actions)
            velocity, acceleration, jerk = compute_trajectory_derivatives(traj_actions)
            all_velocities.append(velocity)
            all_accelerations.append(acceleration)
            all_jerks.append(jerk)

        if not all_velocities:
            raise ValueError("No valid trajectories found (need at least 4 steps per trajectory)")

        all_actions = np.concatenate(all_actions_list, axis=0)
        all_velocities = np.concatenate(all_velocities, axis=0)
        all_accelerations = np.concatenate(all_accelerations, axis=0)
        all_jerks = np.concatenate(all_jerks, axis=0)

        action_dim = all_velocities.shape[1]
        print(f"\nTotal: {total_transitions} transitions, action dim: {action_dim}")

        max_action = all_actions.max(axis=0)
        min_action = all_actions.min(axis=0)
        max_velocity = np.abs(all_velocities).max(axis=0)
        max_acceleration = np.abs(all_accelerations).max(axis=0)
        max_jerk = np.abs(all_jerks).max(axis=0)

        velocity_p95 = np.percentile(np.abs(all_velocities), 95, axis=0)
        velocity_p99 = np.percentile(np.abs(all_velocities), 99, axis=0)
        acceleration_p95 = np.percentile(np.abs(all_accelerations), 95, axis=0)
        acceleration_p99 = np.percentile(np.abs(all_accelerations), 99, axis=0)
        jerk_p95 = np.percentile(np.abs(all_jerks), 95, axis=0)
        jerk_p99 = np.percentile(np.abs(all_jerks), 99, axis=0)

        if FLAGS.verbose:
            print("\n" + "=" * 80)
            print("VELOCITY (1st derivative: action[t] - action[t-1])")
            print("=" * 80)
            print(f"  {'Dim':>4s}  {'Max':>10s}  {'95th %ile':>10s}  {'99th %ile':>10s}")
            for i in range(action_dim):
                print(f"  {i:4d}  {max_velocity[i]:10.6f}  {velocity_p95[i]:10.6f}  {velocity_p99[i]:10.6f}")

            print("\n" + "=" * 80)
            print("ACCELERATION (2nd derivative)")
            print("=" * 80)
            print(f"  {'Dim':>4s}  {'Max':>10s}  {'95th %ile':>10s}  {'99th %ile':>10s}")
            for i in range(action_dim):
                print(f"  {i:4d}  {max_acceleration[i]:10.6f}  {acceleration_p95[i]:10.6f}  {acceleration_p99[i]:10.6f}")

            print("\n" + "=" * 80)
            print("JERK (3rd derivative)")
            print("=" * 80)
            print(f"  {'Dim':>4s}  {'Max':>10s}  {'95th %ile':>10s}  {'99th %ile':>10s}")
            for i in range(action_dim):
                print(f"  {i:4d}  {max_jerk[i]:10.6f}  {jerk_p95[i]:10.6f}  {jerk_p99[i]:10.6f}")

        stats = {
            "action_dim": int(action_dim),
            "num_samples": int(total_transitions),
            "rel_ctrl": False,
            "min_action": min_action.tolist(),
            "max_action": max_action.tolist(),
            "max_velocity": max_velocity.tolist(),
            "max_acceleration": max_acceleration.tolist(),
            "max_jerk": max_jerk.tolist(),
            "velocity_p95": velocity_p95.tolist(),
            "velocity_p99": velocity_p99.tolist(),
            "acceleration_p95": acceleration_p95.tolist(),
            "acceleration_p99": acceleration_p99.tolist(),
            "jerk_p95": jerk_p95.tolist(),
            "jerk_p99": jerk_p99.tolist(),
        }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Saved statistics to: {output_path}")
    print(f"{'=' * 80}")
    if not FLAGS.rel:
        print("\nTo use derivative limits:")
        print(f'  env = ActionDerivativeConstraintWrapper(env, stats_path="{output_path}", scale=1.0)')
    print("\nTo use action normalization bounds:")
    print(f'  action_stats_path = "{output_path}"')
    print(f'  action_norm_scale = 0.9')


if __name__ == "__main__":
    app.run(main)
