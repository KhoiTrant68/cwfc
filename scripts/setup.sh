#!/usr/bin/env bash
# Install dependencies. Core (torch, numpy) + viz (matplotlib) is enough for the
# G1/G4 pilots and the figures; pass --all for the entropy/DP extras too.
source "$(dirname "$0")/_common.sh"

if [ "${1:-}" = "--all" ]; then
    "$PY" -m pip install -r requirements.txt
else
    "$PY" -m pip install torch numpy matplotlib
fi
echo "done. (use --all for compressai/diffusers/lpips extras)"
