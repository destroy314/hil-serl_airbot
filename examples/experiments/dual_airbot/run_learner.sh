export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3
unset LD_LIBRARY_PATH
python ../../train_rlpd.py "$@" \
    --exp_name=dual_airbot \
    --checkpoint_path=run_2 \
    --demo_path=demo_data/2.pkl \
    --learner \
    --debug
