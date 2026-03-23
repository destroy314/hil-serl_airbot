#!/usr/bin/env python3
"""
Offline reproduction of the online training collapse observed in dual_airbot run_2.

Loads the pretrain checkpoint and saved buffers, then runs the learner training loop
without any actor/env/TCP connections. Saves checkpoints frequently so eval_offline.py
can be used to observe the collapse.

Usage (from examples/):
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.3
    unset LD_LIBRARY_PATH
    python repro_online_collapse.py \
        --exp_name=dual_airbot \
        --checkpoint_path=experiments/dual_airbot/run_2 \
        --demo_path=experiments/dual_airbot/demo_data/2.pkl \
        --output_path=experiments/dual_airbot/repro_collapse \
        [--max_steps=500] \
        [--save_every=50]
"""

import glob
import json
import os
import pickle as pkl

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.core import frozen_dict
from flax.training import checkpoints

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_sac_pixel_agent_hybrid_single_arm,
)
from serl_launcher.utils.train_utils import concat_batches
from airbot_env.envs.action_normalizer import ActionNormalizer

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Experiment name (key in CONFIG_MAPPING).", required=True)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("checkpoint_path", None, "Path to run_2 directory (contains checkpoint_1000/, buffer/, demo_buffer/).", required=True)
flags.DEFINE_multi_string("demo_path", None, "Path(s) to demo pkl file(s) for demo_buffer.", required=True)
flags.DEFINE_string("output_path", None, "Directory to save reproduced checkpoints. Defaults to <checkpoint_path>/repro_collapse.")
flags.DEFINE_integer("max_steps", 500, "Number of online training steps to run.")
flags.DEFINE_integer("save_every", 25, "Save a checkpoint every N steps.")
flags.DEFINE_float("demo_filter_motion_threshold", 0.01, "Filter stationary demo transitions (0 = off).")

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    print("\033[92m {}\033[00m".format(x))


def main(_):
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    output_path = FLAGS.output_path or os.path.join(FLAGS.checkpoint_path, "repro_collapse")
    os.makedirs(output_path, exist_ok=True)

    # ------------------------------------------------------------------ env (fake)
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    # ------------------------------------------------------------------ action normalizer
    action_normalizer = None
    if config.action_norm_scale > 0:
        # Resolve action_stats_path: try relative to demo_path dir, then cwd, then as-is
        demo_dir = os.path.dirname(os.path.abspath(FLAGS.demo_path[0]))
        candidates = [
            os.path.join(demo_dir, config.action_stats_path),
            os.path.join(demo_dir, os.path.basename(config.action_stats_path)),
            config.action_stats_path,
        ]
        stats_path = next((p for p in candidates if os.path.exists(p)), None)
        assert stats_path is not None, (
            f"Cannot find action_stats_path={config.action_stats_path!r}. "
            f"Tried: {candidates}"
        )
        with open(stats_path, "r") as f:
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
        print_green(f"Action normalizer: scale={config.action_norm_scale}, skip={gripper_skip}")

    # ------------------------------------------------------------------ agent
    include_grasp_penalty = False
    if config.setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            **config.actor_kwargs,
        )
    elif config.setup_mode == "single-arm-learned-gripper":
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == "dual-arm-learned-gripper":
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            **config.actor_kwargs,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup_mode: {config.setup_mode}")

    agent = jax.device_put(jax.tree_util.tree_map(jnp.array, agent), sharding.replicate())

    # ------------------------------------------------------------------ load pretrain checkpoint
    ckpt_dir = os.path.abspath(FLAGS.checkpoint_path)
    latest = checkpoints.latest_checkpoint(ckpt_dir)
    assert latest is not None, f"No checkpoint found in {ckpt_dir}"
    ckpt = checkpoints.restore_checkpoint(ckpt_dir, agent.state)
    agent = agent.replace(state=ckpt)
    print_green(f"Loaded checkpoint: {latest}")

    # ------------------------------------------------------------------ replay buffer
    replay_buffer = MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=config.replay_buffer_capacity,
        image_keys=config.image_keys,
        include_grasp_penalty=include_grasp_penalty,
    )
    buffer_dir = os.path.join(FLAGS.checkpoint_path, "buffer")
    for fpath in sorted(glob.glob(os.path.join(buffer_dir, "*.pkl"))):
        with open(fpath, "rb") as f:
            for t in pkl.load(f):
                replay_buffer.insert(t)
    print_green(f"Replay buffer size: {len(replay_buffer)}")

    # ------------------------------------------------------------------ demo buffer
    demo_buffer = MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=config.replay_buffer_capacity,
        image_keys=config.image_keys,
        include_grasp_penalty=include_grasp_penalty,
    )

    # Load original demo pkl(s)
    total_loaded = total_filtered = 0
    for path in FLAGS.demo_path:
        with open(path, "rb") as f:
            transitions = pkl.load(f)
        prev_action = None
        for t in transitions:
            total_loaded += 1
            cur_action = np.asarray(t["actions"], dtype=np.float32)
            if FLAGS.demo_filter_motion_threshold > 0 and prev_action is not None:
                if np.linalg.norm(cur_action - prev_action) < FLAGS.demo_filter_motion_threshold:
                    total_filtered += 1
                    prev_action = cur_action
                    continue
            if "infos" in t and "grasp_penalty" in t["infos"]:
                t["grasp_penalty"] = t["infos"]["grasp_penalty"]
            demo_buffer.insert(t)
            prev_action = cur_action

    # Also load any saved demo_buffer pkl from the run
    demo_buffer_dir = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
    if os.path.exists(demo_buffer_dir):
        for fpath in sorted(glob.glob(os.path.join(demo_buffer_dir, "*.pkl"))):
            with open(fpath, "rb") as f:
                for t in pkl.load(f):
                    demo_buffer.insert(t)

    if FLAGS.demo_filter_motion_threshold > 0:
        print_green(f"Filtered {total_filtered}/{total_loaded} stationary demo transitions")
    print_green(f"Demo buffer size: {len(demo_buffer)}")

    # ------------------------------------------------------------------ training setup
    def normalize_batch(batch):
        if action_normalizer is not None:
            norm_actions = action_normalizer.normalize(np.asarray(batch["actions"]))
            batch = batch.unfreeze()
            batch["actions"] = jnp.array(norm_actions)
            batch = frozen_dict.freeze(batch)
        return batch

    if isinstance(agent, SACAgent):
        train_critic_only = frozenset({"critic"})
        train_all = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_only = frozenset({"critic", "grasp_critic"})
        train_all = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    replay_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs_and_next_obs": True},
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs_and_next_obs": True},
        device=sharding.replicate(),
    )

    # Save step-0 (pretrain) checkpoint as baseline
    checkpoints.save_checkpoint(os.path.abspath(output_path), agent.state, step=0, keep=10000)
    print_green(f"Saved step-0 (pretrain) checkpoint to {output_path}")

    # ------------------------------------------------------------------ training loop
    print_green(f"Running {FLAGS.max_steps} online training steps (save every {FLAGS.save_every})...")
    for step in tqdm.tqdm(range(1, FLAGS.max_steps + 1), dynamic_ncols=True):
        # cta_ratio - 1 critic-only updates
        for _ in range(config.cta_ratio - 1):
            batch = normalize_batch(concat_batches(next(replay_iterator), next(demo_iterator), axis=0))
            agent, _ = agent.update(batch, networks_to_update=train_critic_only)

        # 1 full update
        batch = normalize_batch(concat_batches(next(replay_iterator), next(demo_iterator), axis=0))
        agent, update_info = agent.update(batch, networks_to_update=train_all)

        if step % FLAGS.save_every == 0:
            agent = jax.block_until_ready(agent)
            checkpoints.save_checkpoint(
                os.path.abspath(output_path), agent.state, step=step, keep=10000
            )
            actor_loss = float(np.asarray(update_info.get("actor", {}).get("actor_loss", float("nan"))))
            critic_loss = float(np.asarray(update_info.get("critic", {}).get("critic_loss", float("nan"))))
            temperature = float(np.asarray(update_info.get("actor", {}).get("temperature", float("nan"))))
            tqdm.tqdm.write(
                f"  step={step:4d}  actor_loss={actor_loss:.4f}  critic_loss={critic_loss:.4f}  temperature={temperature:.4f}"
            )

    print_green(f"\nDone. Checkpoints saved to {output_path}/")
    print_green(
        f"Evaluate with:\n"
        f"  python eval_offline.py --exp_name={FLAGS.exp_name} "
        f"--checkpoint_path={output_path} "
        f"--demo_path={FLAGS.demo_path[0]}"
    )


if __name__ == "__main__":
    app.run(main)
