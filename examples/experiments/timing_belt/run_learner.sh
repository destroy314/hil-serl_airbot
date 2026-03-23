export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
unset LD_LIBRARY_PATH && \
python ../../train_rlpd.py "$@" \
    --exp_name=timing_belt \
    --checkpoint_path=run_6 \
    --demo_path=demo_data/abs_timing_belt_20_demos_2026-02-05_17-36-59_grasp_penalty.pkl \
    --learner \
    --debug
    # --diagnostics \