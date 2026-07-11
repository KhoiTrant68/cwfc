#!/usr/bin/env bash
# G1 (the decisive negative): aggregate extended-cost OT, weak AE + K=4,
# non-detached potential loss. Single seed, pool embedding -> results/g1_potential.json
source "$(dirname "$0")/_common.sh"

"$PY" g1_pilot.py --dataset cifar100 --data_root "$DATA_ROOT" \
    --H 32 --mode both --N 384 --steps 2000 --ae_steps 3000 \
    --ae_qscale 0.5 --ae_zc 2 --K 4 --eps 0.02 \
    --seeds 0 --etas 0 0.3 3 30 --embeds pool --loss potential \
    --out results/g1_potential.json "$@"
