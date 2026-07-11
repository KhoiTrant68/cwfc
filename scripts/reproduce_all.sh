#!/usr/bin/env bash
# End-to-end paper pipeline (Kaggle 2x T4): the G1 robustness lock, the G4
# hardened table, then all figures. Override the dataset path if needed:
#   DATA_ROOT=/path/to/cifar100 bash scripts/reproduce_all.sh
source "$(dirname "$0")/_common.sh"

echo "### [1/3] G1 robustness (3 embeds x 3 seeds) -> the negative"
bash scripts/run_g1_full.sh

echo "### [2/3] G4 hardened (3 seeds + condMMD + sample grids) -> the positive"
bash scripts/run_g4_full.sh

echo "### [3/3] figures"
bash scripts/make_figs.sh

echo "done. metrics in results/, figures + sample grids in figs/"
