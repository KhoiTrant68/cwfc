#!/usr/bin/env bash
# Download the 24 Kodak PNG test images (768x512 true-color) into $ROOT.
# The canonical host (r0k.us/graphics/kodak) currently sits behind a
# bot-check page for scripted clients. github.com/lemire/kodakimagecollection
# looked like a good mirror but is missing 6 of the 24 images (06-08, 12-14);
# github.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite has the
# full set (verified: all 24 present, correct 768x512 PNGs) under
# PhotoCD_PCD0992/NN.png (two-digit, no "kodim" prefix) -- renamed to
# kodimNN.png on download so downstream globbing matches the usual convention.
set -euo pipefail

ROOT="${1:-./data/kodak}"
mkdir -p "$ROOT"

BASE="https://raw.githubusercontent.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite/master/PhotoCD_PCD0992"
for i in $(seq -w 1 24); do
  out="$ROOT/kodim${i}.png"
  if [ -f "$out" ]; then
    echo "skip $out (exists)"
    continue
  fi
  echo "fetching kodim${i}.png"
  curl -sL --fail -o "$out" "$BASE/${i}.png"
done

echo "Kodak (24 images) -> $ROOT"
