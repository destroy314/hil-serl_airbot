export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=insert_vertical \
    --checkpoint_path=run_1 \
    --demo_path=demo_data/insert_vertical_20_demos_2025-12-24_18-53-43.pkl \
    --learner \