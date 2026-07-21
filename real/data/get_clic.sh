#!/usr/bin/env bash
# Download CLIC (professional + mobile tracks, 2020 vintage) into $ROOT.
# Layout produced:
#   $ROOT/train/   -- professional_train_2020 + mobile_train_2020 (pool for
#                      the kNN conditional set + patch training, real/data.py)
#   $ROOT/valid/    -- professional_valid_2020 (held-out eval, matches the
#                      folder facebookresearch/NeuralCompression's own
#                      eval_folder_example.py expects as `clic_path`)
# All URLs verified reachable at the time this script was written (2026-07);
# re-check https://clic2025.compression.cc if any 404.
set -euo pipefail

ROOT="${1:-./data/clic}"
mkdir -p "$ROOT/train" "$ROOT/valid"

BASE="https://data.vision.ee.ethz.ch/cvl/clic"

fetch_and_unzip() {
  local fname="$1" dest="$2"
  local tmp
  tmp="$(mktemp -d)"
  echo "fetching $fname"
  curl -sL --fail -o "$tmp/$fname" "$BASE/$fname"
  unzip -q -o "$tmp/$fname" -d "$dest"
  rm -rf "$tmp"
}

fetch_and_unzip professional_train_2020.zip "$ROOT/train"
fetch_and_unzip mobile_train_2020.zip "$ROOT/train"
fetch_and_unzip professional_valid_2020.zip "$ROOT/valid"

n_train=$(find "$ROOT/train" -type f \( -iname '*.png' -o -iname '*.jpg' \) | wc -l)
n_valid=$(find "$ROOT/valid" -type f \( -iname '*.png' -o -iname '*.jpg' \) | wc -l)
echo "CLIC train: $n_train images -> $ROOT/train"
echo "CLIC valid: $n_valid images -> $ROOT/valid"
