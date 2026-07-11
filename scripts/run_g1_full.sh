#!/usr/bin/env bash
# G1 robustness lock: 3 embeddings x 3 seeds -> results/g1_potential_full.json
# Shows the eta-flat / embedding-independent pattern that makes G1 definitive.
source "$(dirname "$0")/_common.sh"

"$PY" g1_pilot.py --dataset cifar100 --data_root "$DATA_ROOT" \
    --H 32 --mode train --N 384 --steps 2000 --ae_steps 3000 \
    --ae_qscale 0.5 --ae_zc 2 --K 4 --eps 0.02 \
    --seeds 0 1 2 --etas 0 0.3 3 30 --embeds raw proj8 pool --loss potential \
    --out results/g1_potential_full.json "$@"
