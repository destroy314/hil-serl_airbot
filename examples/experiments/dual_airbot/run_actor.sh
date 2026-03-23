export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
unset LD_LIBRARY_PATH
python ../../train_rlpd.py "$@" \
    --exp_name=dual_airbot \
    --checkpoint_path=run_2 \
    --actor \
    --debug
