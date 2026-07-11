#!/usr/bin/env bash
# Fast synthetic plumbing check (CPU-ok, no downloads) for both pilots.
source "$(dirname "$0")/_common.sh"

echo "=== g1_pilot smoke ==="
"$PY" g1_pilot.py --smoke

echo "=== g4_wflow smoke (+ sample grids) ==="
"$PY" g4_wflow.py --dataset synth --H 32 --N 96 --npool 300 --knn 8 \
    --steps 60 --ae_steps 100 --n_steps 8 --n_draws 4 --lams 0 0.5 1 \
    --device cpu --save_samples figs
