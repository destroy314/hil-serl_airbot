#!/usr/bin/env python3
"""
Offline evaluation script: loads a checkpoint and demo data,
runs inference on each demo observation, and plots GT vs predicted action curves.

Usage:
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    python eval_offline.py \
        --exp_name=timing_belt \
        --checkpoint_path=<path_to_checkpoint_dir> \
        --demo_path=<path_to_demo_pkl> \
        [--checkpoint_step=0]  # 0 means latest
"""

import os
import json
import importlib.util
import jax
import jax.numpy as jnp
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from absl import app, flags

from flax.training import checkpoints

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
)

from experiments.mappings import CONFIG_MAPPING
from airbot_env.envs.wrappers import ActionNormalizer

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("checkpoint_path", None, "Path to the checkpoint directory.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo pkl file(s).")
flags.DEFINE_integer("checkpoint_step", 0, "Checkpoint step to load (0 = latest).")
flags.DEFINE_integer("traj_idx", -1, "Trajectory index to plot (-1 = all).")
flags.DEFINE_integer("n_trajs", 0, "Max number of trajectories to evaluate (0 = all).")
flags.DEFINE_string("output_dir", None, "Directory to save plots.")
flags.DEFINE_boolean("argmax", True, "Whether to use argmax actions during evaluation.")
flags.DEFINE_boolean("no_state", False, "If True, skip plotting the state curve.")

devices = jax.local_devices()
sharding = jax.sharding.PositionalSharding(devices)


ACTION_LABELS_DUAL = [
    "L joint 1", "L joint 2", "L joint 3",
    "L joint 4", "L joint 5", "L joint 6",
    "L gripper",
    "R joint 1", "R joint 2", "R joint 3",
    "R joint 4", "R joint 5", "R joint 6",
    "R gripper",
]

ACTION_LABELS_SINGLE = [
    "joint 1", "joint 2", "joint 3",
    "joint 4", "joint 5", "joint 6",
    "gripper",
]


def split_trajectories(transitions):
    """Split a flat list of transitions into trajectories by 'dones' boundaries."""
    trajectories = []
    current = []
    for t in transitions:
        current.append(t)
        if t.get("dones", False):
            trajectories.append(current)
            current = []
    if current:
        trajectories.append(current)
    return trajectories


def extract_state_vector(obs, target_dim):
    """Extract joint-position-like state aligned with action layout."""
    state = obs.get("state", obs) if isinstance(obs, dict) else obs

    state_vec = np.asarray(state, dtype=np.float32)
    if state_vec.ndim >= 2:
        state_vec = state_vec[-1]
    state_vec = state_vec.reshape(-1)

    # SERLObsWrapper flattened timing_belt order:
    # left/joint_pos(6), left/joint_vel(6), left/joint_torque(6), left/gripper(1),
    # right/joint_pos(6), right/joint_vel(6), right/joint_torque(6), right/gripper(1)
    if target_dim == 14 and state_vec.size >= 38:
        state_vec = np.concatenate([state_vec[0:6], state_vec[18:19], state_vec[19:25], state_vec[37:38]])
        # old order for now
        # state_vec = np.concatenate([state_vec[0:6], state_vec[18:19], state_vec[20:26], state_vec[19:20]])
    elif target_dim == 7 and state_vec.size >= 19:
        state_vec = np.concatenate([state_vec[0:6], state_vec[18:19]])

    if state_vec.size >= target_dim:
        return state_vec[:target_dim]

    padded = np.zeros((target_dim,), dtype=np.float32)
    padded[:state_vec.size] = state_vec
    return padded


def plot_trajectory(state_values, gt_actions, pred_actions, traj_idx, action_labels, output_dir, plot_state=True):
    """Plot state, GT action and predicted actions for a single trajectory."""
    n_steps, n_dims = gt_actions.shape
    n_cols = 2
    n_rows = (n_dims + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 2.5 * n_rows), sharex=True)
    axes = axes.flatten()

    timesteps = np.arange(n_steps)
    for dim in range(n_dims):
        ax = axes[dim]
        if plot_state:
            ax.plot(timesteps, state_values[:, dim], label="State", color="tab:blue", linewidth=1.5)
        ax.plot(timesteps, gt_actions[:, dim], label="GT action", color="tab:green", linewidth=1.5)
        ax.scatter(
            timesteps,
            pred_actions[:, dim],
            label="Pred action",
            color="tab:red",
            s=5,
            alpha=0.85,
        )
        ax.set_ylabel(action_labels[dim] if dim < len(action_labels) else f"dim {dim}",
                       fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for dim in range(n_dims, len(axes)):
        axes[dim].set_visible(False)

    axes[-2].set_xlabel("Timestep", fontsize=10)
    if n_dims % 2 == 0:
        axes[-1].set_xlabel("Timestep", fontsize=10)

    fig.suptitle(f"Trajectory {traj_idx}  ({n_steps} steps)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"traj_{traj_idx}.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def load_config():
    if FLAGS.exp_name:
        return CONFIG_MAPPING[FLAGS.exp_name]()

    config_path = os.path.join(os.getcwd(), "config.py")
    if not os.path.exists(config_path):
        raise ValueError(
            "--exp_name not provided, and no config.py found in current working directory."
        )

    spec = importlib.util.spec_from_file_location("eval_local_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Failed to load config from {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "TrainConfig"):
        raise ValueError(
            f"{config_path} does not define TrainConfig. Please pass --exp_name explicitly."
        )

    return module.TrainConfig()


def main(_):
    config = load_config()

    # Create fake env for observation/action space
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    # Create agent
    if config.setup_mode in ('single-arm-fixed-gripper', 'dual-arm-fixed-gripper'):
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            **config.actor_kwargs,
        )
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    action_labels = ACTION_LABELS_DUAL if 'dual' in config.setup_mode else ACTION_LABELS_SINGLE

    # Replicate agent across devices
    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )

    # Create action normalizer if enabled
    action_normalizer = None
    if config.action_norm_scale > 0:
        with open(config.action_stats_path, "r") as f:
            stats = json.load(f)
        gripper_skip = {
            "single-arm-learned-gripper": [6],
            "dual-arm-learned-gripper": [6, 13],
        }.get(config.setup_mode, [])
        action_normalizer = ActionNormalizer(
            np.asarray(stats["min_action"], dtype=np.float32),
            np.asarray(stats["max_action"], dtype=np.float32),
            scale=config.action_norm_scale,
            skip_indices=gripper_skip,
        )
        print(
            f"Action normalizer enabled from stats {config.action_stats_path} "
            f"(scale={config.action_norm_scale})"
        )

    # Load checkpoint
    assert FLAGS.checkpoint_path is not None, "Must provide --checkpoint_path"
    ckpt_path = os.path.abspath(FLAGS.checkpoint_path)
    default_output = f"eval_plots{'_argmax' if FLAGS.argmax else ''}"
    output_dir = FLAGS.output_dir or os.path.join(ckpt_path, default_output)
    if FLAGS.checkpoint_step > 0:
        ckpt = checkpoints.restore_checkpoint(ckpt_path, agent.state, step=FLAGS.checkpoint_step)
    else:
        ckpt = checkpoints.restore_checkpoint(ckpt_path, agent.state)
    agent = agent.replace(state=ckpt)
    latest = checkpoints.latest_checkpoint(ckpt_path)
    print(f"Loaded checkpoint: {latest}")

    # Load demo data
    assert FLAGS.demo_path is not None, "Must provide --demo_path"
    all_transitions = []
    for path in FLAGS.demo_path:
        with open(path, "rb") as f:
            transitions = pkl.load(f)
            all_transitions.extend(transitions)
    print(f"Loaded {len(all_transitions)} transitions from {len(FLAGS.demo_path)} file(s)")

    # Split into trajectories
    trajectories = split_trajectories(all_transitions)
    print(f"Found {len(trajectories)} trajectories")

    # Determine which trajectories to evaluate
    if FLAGS.traj_idx >= 0:
        indices = [FLAGS.traj_idx]
    else:
        indices = list(range(len(trajectories)))
    if FLAGS.n_trajs > 0:
        indices = indices[:FLAGS.n_trajs]

    rng = jax.random.PRNGKey(FLAGS.seed)

    for traj_idx in indices:
        traj = trajectories[traj_idx]
        n_steps = len(traj)
        state_list = []
        gt_actions_list = []
        pred_actions_list = []

        for t in traj:
            obs = t["observations"]
            gt_action = np.array(t["actions"])

            rng, key = jax.random.split(rng)
            pred_action = agent.sample_actions(
                observations=jax.device_put(obs),
                seed=key,
                argmax=FLAGS.argmax,
            )
            pred_action = np.asarray(jax.device_get(pred_action))
            # Denormalize prediction to raw space for comparison with GT
            if action_normalizer is not None:
                pred_action = action_normalizer.denormalize(pred_action)

            state_vec = extract_state_vector(obs, gt_action.shape[0])
            state_list.append(state_vec)
            gt_actions_list.append(gt_action)
            pred_actions_list.append(pred_action)

        state_values = np.stack(state_list)         # (T, action_dim)
        gt_actions = np.stack(gt_actions_list)       # (T, action_dim)
        pred_actions = np.stack(pred_actions_list)    # (T, action_dim)

        # Print per-dim MAE
        mae = np.abs(gt_actions - pred_actions).mean(axis=0)
        print(f"\nTrajectory {traj_idx} ({n_steps} steps) — MAE per dim:")
        for d, m in enumerate(mae):
            label = action_labels[d] if d < len(action_labels) else f"dim {d}"
            print(f"  {label:12s}: {m:.4f}")
        print(f"  {'TOTAL':12s}: {mae.mean():.4f}")

        plot_trajectory(state_values, gt_actions, pred_actions, traj_idx, action_labels, output_dir,
                       plot_state=not FLAGS.no_state)

        # Save per-trajectory stats to npz
        os.makedirs(output_dir, exist_ok=True)
        npz_path = os.path.join(output_dir, f"traj_{traj_idx}_stats.npz")
        np.savez(
            npz_path,
            gt_actions=gt_actions,
            pred_actions=pred_actions,
            diff=gt_actions - pred_actions,
            mae_per_dim=mae,
            mae_total=np.array(mae.mean()),
            action_labels=np.array(action_labels),
        )
        print(f"  Saved stats: {npz_path}")

    # Aggregate summary across all evaluated trajectories: mean MAE per dim
    summary_path = os.path.join(output_dir, "summary.json")
    all_npz = sorted(
        f for f in os.listdir(output_dir) if f.endswith("_stats.npz")
    )
    if all_npz:
        all_mae = np.stack([
            np.load(os.path.join(output_dir, f))["mae_per_dim"] for f in all_npz
        ])  # (n_trajs, action_dim)
        mean_mae = all_mae.mean(axis=0)
        summary = {
            "n_trajs": len(all_npz),
            "mae_per_dim": {
                (action_labels[d] if d < len(action_labels) else f"dim_{d}"): float(mean_mae[d])
                for d in range(len(mean_mae))
            },
            "mae_total": float(mean_mae.mean()),
        }
        import json as _json
        with open(summary_path, "w") as f:
            _json.dump(summary, f, indent=2)
        print(f"\nSummary saved: {summary_path}  (mae_total={summary['mae_total']:.4f})")

    print(f"\nDone. All plots saved to {output_dir}/")


if __name__ == "__main__":
    app.run(main)
