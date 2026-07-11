#!/usr/bin/env bash
# G4 hardened table: 3 seeds (mean+/-std) + conditional-MMD -> results/g4_full.json
# plus sample grids -> figs/. This is the paper's G4 table.
source "$(dirname "$0")/_common.sh"

"$PY" g4_wflow.py --dataset cifar100 --data_root "$DATA_ROOT" \
    --H 32 --N 256 --npool 2000 --knn 16 --steps 1500 --ae_steps 3000 \
    --ae_zc 2 --ae_qscale 0.5 --n_steps 20 --n_draws 8 \
    --lams 0 0.25 0.5 0.75 1 --seeds 0 1 2 --embed pool \
    --out results/g4_full.json --save_samples figs "$@"
