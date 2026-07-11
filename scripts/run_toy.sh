#!/usr/bin/env bash
# Toy 2-D extended-cost Sinkhorn frontier (the core mechanism, no downloads).
# Q1 = coupling geometry vs eta, Q2 = trained-generator D-P frontier.
source "$(dirname "$0")/_common.sh"

"$PY" derisk_cot.py --q both --N 1024 --steps 4000 "$@"
