#!/usr/bin/env bash
# Shared setup sourced by every script here. Override on the command line, e.g.
#   DATA_ROOT=/path/to/cifar100 PY=python3 bash scripts/run_g4.sh
set -euo pipefail

PY="${PY:-python}"
DATA_ROOT="${DATA_ROOT:-/kaggle/input/datasets/fedesoriano/cifar100}"

# Run from the repo root so the flat scripts (and `import g1_pilot`) resolve.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results figs
