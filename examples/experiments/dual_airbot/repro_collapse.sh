#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3
unset LD_LIBRARY_PATH

BC_CKPT=run_2
BC_EVAL_DIR="$BC_CKPT/eval_plots/bc"
REPRO_CKPT=run_2/repro_collapse_3
MAX_STEPS=100
SAVE_EVERY=50

run_eval() {
    local ckpt_dir="$1"
    local step="$2"
    local out_dir="$3"
    python ../../eval_offline.py \
        --exp_name=dual_airbot \
        --checkpoint_path="$ckpt_dir" \
        --demo_path=demo_data/2.pkl \
        --checkpoint_step="$step" \
        --output_dir="$out_dir" \
        --n_trajs=5 \
        --argmax \
        --no_state
}

# ── Baseline: eval the BC pretrain checkpoint (step 1000) ──────────────────
if [ ! -f "$BC_EVAL_DIR/summary.json" ]; then
    echo "BC eval data not found — running eval on pretrain checkpoint..."
    run_eval "$BC_CKPT" 1000 "$BC_EVAL_DIR"
else
    echo "BC eval already exists ($BC_EVAL_DIR/summary.json), skipping."
fi

# ── Reproduce online collapse ───────────────────────────────────────────────
python ../../repro_online_collapse.py "$@" \
    --exp_name=dual_airbot \
    --checkpoint_path="$BC_CKPT" \
    --demo_path=demo_data/2.pkl \
    --output_path="$REPRO_CKPT" \
    --max_steps=$MAX_STEPS \
    --save_every=$SAVE_EVERY

# ── Eval every saved checkpoint ─────────────────────────────────────────────
echo ""
echo "Running eval_offline.py on all repro checkpoints..."
for ckpt in "$REPRO_CKPT"/checkpoint_*; do
    step="${ckpt##*_}"
    out_dir="$REPRO_CKPT/eval_plots/step_${step}"
    echo "  Evaluating step $step..."
    run_eval "$REPRO_CKPT" "$step" "$out_dir"
done

echo ""
echo "BC baseline : $BC_EVAL_DIR/summary.json"
echo "Repro plots : $REPRO_CKPT/eval_plots/"
