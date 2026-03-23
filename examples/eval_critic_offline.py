#!/usr/bin/env python3
"""
Offline critic evaluation script: loads a checkpoint and demo data,
computes Q-values for each transition, and plots Q-value curves.

For each transition the script evaluates:
  - Q(s, a_demo)  — Q-value of the ground-truth demo action
  - Q(s, a_policy) — Q-value of the policy's predicted (mode) action
  - TD target      — r + γ * min_Q(s', a') (using target critic)

Usage:
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    python eval_critic_offline.py \
        --exp_name=timing_belt \
        --checkpoint_path=<path_to_checkpoint_dir> \
        --demo_path=<path_to_demo_pkl> \
        [--checkpoint_step=0]   # 0 = latest
        [--traj_idx=-1]          # -1 = all trajectories
        [--output_dir=<dir>]
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
flags.DEFINE_string("output_dir", None, "Directory to save plots.")
flags.DEFINE_float("discount", -1.0, "Discount factor override (-1 = use config value).")

devices = jax.local_devices()
sharding = jax.sharding.PositionalSharding(devices)


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


def batch_from_transition(t, action_normalizer):
    """Build a batch dict of size 1 from a single transition dict."""
    obs = t["observations"]
    next_obs = t["next_observations"]
    action = np.asarray(t["actions"], dtype=np.float32)

    # Normalize action to [-scale, scale] if normalizer is set
    if action_normalizer is not None:
        action = action_normalizer.normalize(action)

    reward = np.asarray([t.get("rewards", 0.0)], dtype=np.float32)
    done = float(t.get("dones", False))
    mask = np.asarray([1.0 - done], dtype=np.float32)

    def add_batch_dim(x):
        if isinstance(x, dict):
            return {k: add_batch_dim(v) for k, v in x.items()}
        x = np.asarray(x, dtype=np.float32)
        return x[None]  # (1, ...)

    return {
        "observations": add_batch_dim(obs),
        "next_observations": add_batch_dim(next_obs),
        "actions": action[None],      # (1, action_dim)
        "rewards": reward,            # (1,)
        "masks": mask,                # (1,)
    }


def eval_critic_on_batch(agent, batch, rng, discount):
    """Return Q-values for demo action and policy action, plus TD target."""
    obs = jax.device_put(batch["observations"])
    next_obs = jax.device_put(batch["next_observations"])
    demo_action = jax.device_put(batch["actions"])
    rewards = jax.device_put(batch["rewards"])
    masks = jax.device_put(batch["masks"])

    rng, key1, key2, key3, key4 = jax.random.split(rng, 5)

    # Q(s, a_demo) — ensemble shape (n_critics, 1)
    q_demo = agent.forward_critic(obs, demo_action, rng=key1, train=False)
    q_demo_min = float(jax.device_get(q_demo.min(axis=0)[0]))
    q_demo_mean = float(jax.device_get(q_demo.mean(axis=0)[0]))
    q_demo_ensemble = jax.device_get(q_demo[:, 0])  # (n_critics,)

    # Policy action (mode)
    policy_action = agent.sample_actions(obs, seed=key2, argmax=True)
    q_policy = agent.forward_critic(obs, policy_action, rng=key3, train=False)
    q_policy_min = float(jax.device_get(q_policy.min(axis=0)[0]))
    q_policy_mean = float(jax.device_get(q_policy.mean(axis=0)[0]))

    # TD target: r + γ * min Q_target(s', a')
    next_policy_action = agent.sample_actions(next_obs, seed=key4, argmax=True)
    q_next_target = agent.forward_target_critic(next_obs, next_policy_action, rng=key4)
    q_next_min = q_next_target.min(axis=0)  # (1,)
    td_target = float(jax.device_get(
        rewards[0] + discount * masks[0] * q_next_min[0]
    ))

    td_error = q_demo_min - td_target

    return {
        "q_demo_min": q_demo_min,
        "q_demo_mean": q_demo_mean,
        "q_demo_ensemble": q_demo_ensemble,
        "q_policy_min": q_policy_min,
        "q_policy_mean": q_policy_mean,
        "td_target": td_target,
        "td_error": td_error,
    }


def plot_trajectory_critic(results, traj_idx, rewards, output_dir):
    """
    Plot Q-value curves for a single trajectory.

    Panels:
      1. Q(s, a_demo) min/mean over time
      2. Q(s, a_policy) min over time
      3. TD target over time
      4. TD error = Q(s, a_demo)_min - TD_target
      5. Reward signal
      6. Q ensemble spread = Q_mean - Q_min (measure of critic uncertainty)
    """
    n = len(results)
    t = np.arange(n)

    q_demo_min = np.array([r["q_demo_min"] for r in results])
    q_demo_mean = np.array([r["q_demo_mean"] for r in results])
    q_policy_min = np.array([r["q_policy_min"] for r in results])
    td_target = np.array([r["td_target"] for r in results])
    td_error = np.array([r["td_error"] for r in results])
    rewards_arr = np.array(rewards, dtype=np.float32)
    ensemble_spread = q_demo_mean - q_demo_min

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    axes = axes.flatten()

    def _plot(ax, y_vals, title, ylabel, color, extra=None):
        ax.plot(t, y_vals, color=color, linewidth=1.5, label=ylabel)
        if extra is not None:
            for label, vals, clr in extra:
                ax.plot(t, vals, color=clr, linewidth=1.2, linestyle="--", label=label)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if extra is not None:
            ax.legend(fontsize=8)

    _plot(axes[0], q_demo_min, "Q(s, a_demo) — min over ensemble", "Q value", "tab:blue",
          extra=[("mean", q_demo_mean, "tab:cyan")])
    _plot(axes[1], q_policy_min, "Q(s, a_policy) — min over ensemble", "Q value", "tab:orange")
    _plot(axes[2], td_target, "TD target  r + γ·min Q_target(s', a')", "TD target", "tab:green")
    _plot(axes[3], td_error, "TD error  Q(s,a_demo)_min − TD_target", "TD error", "tab:red")
    axes[3].axhline(0, color="black", linewidth=0.8, linestyle=":")
    _plot(axes[4], rewards_arr, "Reward signal", "reward", "tab:purple")
    _plot(axes[5], ensemble_spread, "Critic uncertainty  Q_mean − Q_min", "spread", "tab:brown")

    for ax in axes[-2:]:
        ax.set_xlabel("Timestep", fontsize=10)

    # Summary stats in suptitle
    rmse = float(np.sqrt(np.mean(td_error ** 2)))
    fig.suptitle(
        f"Trajectory {traj_idx}  ({n} steps)   "
        f"mean Q(demo)_min={q_demo_min.mean():.3f}   "
        f"TD-RMSE={rmse:.3f}   "
        f"total_reward={rewards_arr.sum():.2f}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"critic_traj_{traj_idx}.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main(_):
    config = load_config()
    discount = FLAGS.discount if FLAGS.discount > 0 else config.discount

    # Fake env for observation/action space
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    # Create agent
    if config.setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=discount,
            **config.actor_kwargs,
        )
    elif config.setup_mode == "single-arm-learned-gripper":
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=discount,
        )
    elif config.setup_mode == "dual-arm-learned-gripper":
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=discount,
        )
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # Replicate across devices
    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )

    # Action normalizer
    action_normalizer = None
    if config.action_norm_scale > 0:
        with open(config.action_stats_path, "r") as f:
            stats = json.load(f)
        action_normalizer = ActionNormalizer(
            np.asarray(stats["min_position"], dtype=np.float32),
            np.asarray(stats["max_position"], dtype=np.float32),
            scale=config.action_norm_scale,
        )
        print(
            f"Action normalizer enabled from {config.action_stats_path} "
            f"(scale={config.action_norm_scale})"
        )

    # Load checkpoint
    assert FLAGS.checkpoint_path is not None, "Must provide --checkpoint_path"
    ckpt_path = os.path.abspath(FLAGS.checkpoint_path)
    output_dir = FLAGS.output_dir or os.path.join(ckpt_path, "eval_critic_plots")
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

    trajectories = split_trajectories(all_transitions)
    print(f"Found {len(trajectories)} trajectories")

    indices = [FLAGS.traj_idx] if FLAGS.traj_idx >= 0 else list(range(len(trajectories)))

    rng = jax.random.PRNGKey(FLAGS.seed)

    all_q_demo = []
    all_td_errors = []

    for traj_idx in indices:
        traj = trajectories[traj_idx]
        results = []
        rewards = []

        for t in traj:
            batch = batch_from_transition(t, action_normalizer)
            rng, key = jax.random.split(rng)
            res = eval_critic_on_batch(agent, batch, key, discount)
            results.append(res)
            rewards.append(float(t.get("rewards", 0.0)))

        # Per-trajectory summary
        q_mins = np.array([r["q_demo_min"] for r in results])
        td_errs = np.array([r["td_error"] for r in results])
        rmse = float(np.sqrt(np.mean(td_errs ** 2)))
        total_reward = float(sum(rewards))

        print(
            f"\nTrajectory {traj_idx} ({len(traj)} steps):"
            f"\n  Q(s,a_demo) min : mean={q_mins.mean():.4f}  std={q_mins.std():.4f}"
            f"  min={q_mins.min():.4f}  max={q_mins.max():.4f}"
            f"\n  TD RMSE          : {rmse:.4f}"
            f"\n  Total reward     : {total_reward:.3f}"
        )

        all_q_demo.extend(q_mins.tolist())
        all_td_errors.extend(td_errs.tolist())

        plot_trajectory_critic(results, traj_idx, rewards, output_dir)

    # Global summary
    all_q_demo = np.array(all_q_demo)
    all_td_errors = np.array(all_td_errors)
    print(
        f"\n{'='*50}"
        f"\nGlobal summary over {len(indices)} trajectories:"
        f"\n  Q(s,a_demo) min : mean={all_q_demo.mean():.4f}  std={all_q_demo.std():.4f}"
        f"\n  TD RMSE overall : {float(np.sqrt(np.mean(all_td_errors**2))):.4f}"
    )
    print(f"\nDone. All plots saved to {output_dir}/")


if __name__ == "__main__":
    app.run(main)
