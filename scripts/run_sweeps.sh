#!/usr/bin/env bash
set -euo pipefail

MODULE="experiments.run_mnist_addition_supervised"
OUTDIR="runs"
mkdir -p "$OUTDIR"

# Common training params
EPOCHS=3
STEPS=100
BATCH=16
SEEDS=3

# Schedule params
LAM0=1.0
LAM1=0.2
LAM2=0.0

timestamp() { date +"%Y%m%d_%H%M%S"; }

run_case () {
  local tag="$1"; shift
  local outfile="${OUTDIR}/$(timestamp)_${tag}.log"
  echo "=== RUN: ${tag} ==="
  echo "CMD: python -m ${MODULE} $*" | tee "$outfile"
  python -m "$MODULE" "$@" 2>&1 | tee -a "$outfile"
  echo "=== DONE: ${tag} -> ${outfile} ==="
  echo
}

# -------------------------
# 1) Baselines (no stress)
# -------------------------
run_case "schedule_noise0_none" \
  --lambda_mode schedule --num_seeds "$SEEDS" \
  --epochs "$EPOCHS" --steps_per_epoch "$STEPS" --batch_size "$BATCH" \
  --lam0 "$LAM0" --lam1 "$LAM1" --lam2 "$LAM2" \
  --noise_eval 0.0 --occlude_eval none --occlude_frac 0.0

run_case "fixed0_noise0_none" \
  --lambda_mode fixed0 --num_seeds "$SEEDS" \
  --epochs "$EPOCHS" --steps_per_epoch "$STEPS" --batch_size "$BATCH" \
  --noise_eval 0.0 --occlude_eval none --occlude_frac 0.0

# -------------------------
# 2) Noise-only robustness (eval only)
# -------------------------
for NOISE in 0.2 0.4; do
  run_case "schedule_noise${NOISE}_none" \
    --lambda_mode schedule --num_seeds "$SEEDS" \
    --epochs "$EPOCHS" --steps_per_epoch "$STEPS" --batch_size "$BATCH" \
    --lam0 "$LAM0" --lam1 "$LAM1" --lam2 "$LAM2" \
    --noise_eval "$NOISE" --occlude_eval none --occlude_frac 0.0

  run_case "fixed0_noise${NOISE}_none" \
    --lambda_mode fixed0 --num_seeds "$SEEDS" \
    --epochs "$EPOCHS" --steps_per_epoch "$STEPS" --batch_size "$BATCH" \
    --noise_eval "$NOISE" --occlude_eval none --occlude_frac 0.0
done

# -------------------------
# 3) Occlusion + noise robustness (eval only)
#    (your “hard” stress test)
# -------------------------
NOISE=0.4
FRAC=0.4

for OCC in d1 d2; do
  run_case "schedule_noise${NOISE}_${OCC}_frac${FRAC}" \
    --lambda_mode schedule --num_seeds "$SEEDS" \
    --epochs "$EPOCHS" --steps_per_epoch "$STEPS" --batch_size "$BATCH" \
    --lam0 "$LAM0" --lam1 "$LAM1" --lam2 "$LAM2" \
    --noise_eval "$NOISE" --occlude_eval "$OCC" --occlude_frac "$FRAC"

  run_case "fixed0_noise${NOISE}_${OCC}_frac${FRAC}" \
    --lambda_mode fixed0 --num_seeds "$SEEDS" \
    --epochs "$EPOCHS" --steps_per_epoch "$STEPS" --batch_size "$BATCH" \
    --noise_eval "$NOISE" --occlude_eval "$OCC" --occlude_frac "$FRAC"
done

echo "All runs completed. Logs are in: ${OUTDIR}/"