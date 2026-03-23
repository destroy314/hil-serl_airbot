#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."   # → examples/

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3
unset LD_LIBRARY_PATH

python train_rlpd.py \
    --exp_name=airbot_cart \
    --checkpoint_path=run_1 \
    --demo_path=demo_data/<your_demo>.pkl \
    --learner \
    --debug
