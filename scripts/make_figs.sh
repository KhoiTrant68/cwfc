#!/usr/bin/env bash
# Render the argument figures from whatever result JSONs exist in results/.
# g1_frontier (flat=negative), g4_frontier (monotone=positive), dp_plane (contrast).
source "$(dirname "$0")/_common.sh"

"$PY" viz.py \
    --g1 results/g1_potential_full.json \
    --g4 results/g4_full.json \
    --out figs "$@"
