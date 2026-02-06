export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export WANDB_API_KEY=543a96cef34b11ce6e61cd19ed577264a0c51f11 && \
unset LD_LIBRARY_PATH && \
python ../../train_rlpd.py "$@" \
    --exp_name=timing_belt \
    --checkpoint_path=run_1 \
    --demo_path=demo_data/timing_belt_20_demos_2026-02-05_17-36-59.pkl \
    --learner \