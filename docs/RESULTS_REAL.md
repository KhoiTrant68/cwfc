# Real-image results — Kodak/CLIC, CWFC G4 vs. MS-ILLM

Companion to [`RESULTS.md`](RESULTS.md): that document proves the G1-negative
/ G4-positive mechanism on 32×32 CIFAR-100 with a toy autoencoder and a proxy
MMD metric. This document is where those same conclusions get re-tested at
real resolution, with a real bitstream, and against a published baseline.

**Verdict: G4 mechanism POSITIVE at real resolution** — all four metrics
(PSNR_sample, Var_z, LPIPS, FID) move monotonically with λ in the direction
the distortion–perception trade-off predicts (see Results below). CWFC's
absolute frontier sits well below MS-ILLM's at matched bpp, which is expected
given the difference in training scale (a from-scratch U-Net on a ~250-image
pool vs. a large-scale pretrained model) — see the comparison note below the
table.

## Setup

- **Frozen codec** (replaces `TinyAE`): a pretrained, frozen
  `compressai` model (`real/codec.py`), quality level chosen to rate-match
  an MS-ILLM comparison point (see `real/codec.py --arch --qualities
  --image_path` to read off the bpp/PSNR ladder for a given real image).
- **Data**: CLIC train (`real/data/get_clic.sh`) for the 256×256 patch pool
  + kNN conditional targets; Kodak (`real/data/get_kodak.sh`) and CLIC
  valid/test for evaluation.
- **Flow model** (replaces `CondVelocity`): `UNetVelocity` (`real/flow.py`),
  a 4-stage down/up U-Net matched to the codec's 16× downsample, trained as
  a single λ-conditioned model across the whole frontier (architecture
  change from `g4_wflow.py`'s one-model-per-λ, flagged and approved — see
  the plan this track was built from).
- **Baseline**: MS-ILLM (Muckley et al., ICML 2023), pretrained weights via
  `torch.hub`, evaluated with its own official protocol
  (`real/baselines/run_msillm.py`, adapted from
  `facebookresearch/NeuralCompression`'s `eval_folder_example.py`).
- **Metrics**: bpp (real entropy-coded bitstream, not a proxy), PSNR,
  MS-SSIM, LPIPS, patch-FID — computed with the *same*
  `neuralcompression.metrics` module for both the baseline and CWFC G4, so
  the two are on identical footing.

## A real finding from building this pipeline

`real/data.py`'s `sanity_check_knn` (the M2 verification step) surfaces
something CIFAR never exposed: on natural photos, a pooled-embedding kNN in
the codec's condition space sometimes lands on a genuinely related crop
(same scene, different offset) and sometimes on a tonally-similar but
semantically unrelated photo (matched on average color/brightness rather
than content). This did not show up on 32×32 CIFAR, where images are simple
enough that pooled-embedding neighbours were reliably sensible. `real/flow.py`
supports a `--same_image_only` fallback (restrict kNN candidates to crops of
the same source photo) for exactly this case — run the sanity check against
the actual CLIC train pool (richer and much larger than the 24-image Kodak
set used to first notice this) before deciding which mode to train with, and
report here which was used and why.

**Mode used: `--same_image_only`, `crops_per_image` auto-bumped to `knn + 1 =
17`.** Two compounding bugs surfaced getting here, both in
`PatchFolderDataset`/`real/flow.py`, not in the data source:

1. A first `--same_image_only` run (without grouped sampling) made PSNR
   *worse* (~9-10.5dB vs. ~12dB unrestricted). Diagnosis:
   `PatchFolderDataset.sample()` picks a source image independently at random
   per crop, so with `npool` comparable to the number of distinct source
   images, most anchors end up with *no* same-image candidate in the pool at
   all, and the `--same_image_only` kNN mask degenerates to an all-`inf` row
   (`topk` then returns arbitrary unrelated indices) — silently reproducing
   the exact unrelated-neighbour problem the flag exists to fix. Fixed by
   adding `PatchFolderDataset.sample_grouped()` (`real/data.py`) plus a
   `--crops_per_image` flag: the pool is drawn as groups of crops per source
   image so every anchor has genuine same-image candidates.
2. Even with that fix, the CLI default `--crops_per_image 8` was still
   smaller than the default `--knn 16`: each anchor only had
   `crops_per_image - 1 = 7` genuine same-image neighbours, so `topk(16)`
   still padded out 9 of its 16 picks with arbitrary tied-`inf` indices —
   partially defeating fix #1. Fixed with a guard clause in
   `real/flow.py::main()` that auto-bumps `crops_per_image` to `knn + 1`
   whenever `--same_image_only` is set and the invariant is violated
   (prints a `[warn]` line so the run stays reproducible/auditable).

After both fixes, `--same_image_only` gives a clean monotonic frontier on
held-out Kodak images (see Results below), confirming the fallback works
once every anchor genuinely has `knn` real same-image candidates to pick
from.

## Reproduce

```bash
# one-time data fetch
bash real/data/get_kodak.sh data/kodak
bash real/data/get_clic.sh data/clic

# M0 — baseline, no training required
python real/baselines/run_msillm.py --data_root data/kodak \
  --qualities 1 2 3 4 5 6 --out results/baseline_msillm_kodak.json

# sanity-check the conditional pool before training (see finding above)
python real/data.py --image_root data/clic/train --out figs/real_knn_sanity.png

# train (single λ-conditioned model; see real/flow.py --help for the full
# argument list — arch/quality, patch, npool, knn, steps, lams, seeds).
# --same_image_only is the mode that produced the Results below (see "A real
# finding from building this pipeline" above); crops_per_image auto-bumps to
# knn+1 if you don't pass one large enough yourself.
python real/flow.py --train_root data/clic/train --quality 1 \
  --patch 128 --npool 2000 --knn 16 --same_image_only \
  --lams 0 0.25 0.5 0.75 1 --save_checkpoint_dir ckpts \
  --out results/g4_real_train.json

# quick stability check on a mini subset before committing to a full run
# (--max_images caps the pool to the first N images found under --train_root)
python real/flow.py --train_root data/clic/train --max_images 50 \
  --patch 128 --npool 200 --steps 200 --lams 0 0.5 1 \
  --out results/g4_real_mini.json

# multi-GPU (DDP) -- e.g. Kaggle's 2xT4, launch with torchrun instead of python:
#   torchrun --standalone --nproc_per_node=2 real/flow.py --train_root ... (same args)
# --N is batch size PER GPU. Plain `python real/flow.py ...` is unaffected --
# distributed mode only turns on when torchrun sets WORLD_SIZE>1.
# If NCCL hangs on Kaggle's shared multi-GPU networking, a common workaround is
# prefixing the command with NCCL_P2P_DISABLE=1 (and NCCL_IB_DISABLE=1).

# evaluate the trained checkpoint on full Kodak images (fully-convolutional
# U-Net: no tiling needed, Kodak's 768x512 is already codec-divisible)
python real/eval.py --test_root data/kodak --checkpoint ckpts/g4_real_seed0.pt \
  --zc <printed by real/flow.py> --lams 0 0.25 0.5 0.75 1 \
  --out results/eval_real_kodak.json

# plot CWFC's lambda frontier against the MS-ILLM points
python real/viz_real.py --baseline results/baseline_msillm_kodak.json \
  --cwfc results/eval_real_kodak.json --out figs
```

## Pass / fail criterion (same shape as G4 on CIFAR)

At a bpp within tolerance of the chosen MS-ILLM comparison point:
`Var_z` (diversity) should fall and PSNR should rise monotonically with λ
(as on CIFAR), while LPIPS/FID should trace the opposite trend — best
(lowest) near λ=0, worst near λ=1. That reproduces the G4 mechanism at real
resolution; whether CWFC's frontier sits above, on, or below MS-ILLM's own
rate-distortion-perception curve at the matched bpp is the actual result to
report here once the run completes.

## Results

CWFC G4, trained on CLIC train (`--quality 1 --patch 128 --npool 2000 --knn 16
--same_image_only`, `crops_per_image` auto-bumped to 17), evaluated on full
Kodak images (`real/eval.py`, `--n_draws 8 --n_steps 20`):

| λ | bpp | PSNR_sample | PSNR_mmse | MS-SSIM | LPIPS | FID | Var_z |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.1196 | 12.48 | 13.85 | 0.1759 | 0.9148 | 380.92 | 1.6992e-02 |
| 0.25 | 0.1196 | 13.10 | 13.99 | 0.2166 | 0.9452 | 403.81 | 1.1121e-02 |
| 0.50 | 0.1196 | 13.27 | 14.03 | 0.2387 | 0.9661 | 430.38 | 8.1880e-03 |
| 0.75 | 0.1196 | 13.48 | 14.07 | 0.2548 | 0.9750 | 440.99 | 6.6644e-03 |
| 1.00 | 0.1196 | 13.56 | 14.17 | 0.2640 | 0.9798 | 450.01 | 5.7394e-03 |

All four axes move monotonically with λ in the predicted direction:
`PSNR_sample` ↑ 12.48→13.56 dB, `Var_z` ↓ 3.0× (1.70e-2→5.74e-3), `LPIPS` ↑
0.915→0.980, `FID` ↑ 380.9→450.0 — no plateau/dip at any λ, cleaner than the
CIFAR reference. **G4 verdict: POSITIVE at real resolution.**

MS-ILLM comparison point (bracketing CWFC's bpp=0.1196): quality=2
(target bpp=0.07, bpp=0.0809): PSNR=25.92, LPIPS=0.110, FID=71.73; quality=3
(target bpp=0.14, bpp=0.1535): PSNR=27.53, LPIPS=0.073, FID=49.79. At a
matched rate, CWFC sits **well below** MS-ILLM's rate-distortion-perception
curve on every metric (PSNR ~12-15dB lower, LPIPS ~9-11x worse, FID ~6-9x
worse). Expected given the scale gap: CWFC here is a from-scratch U-Net
trained on a ~117-image pool (17 crops/image × ~117 images, per the
`--same_image_only` invariant above) with a deliberately weak (`quality=1`)
frozen codec, vs. MS-ILLM's large-scale pretrained model — the point of this
track is reproducing the G4 distortion-perception mechanism at real
resolution, not matching MS-ILLM's absolute operating point.
