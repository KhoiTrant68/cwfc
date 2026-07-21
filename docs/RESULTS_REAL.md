# Real-image results — Kodak/CLIC, CWFC G4 vs. MS-ILLM

Companion to [`RESULTS.md`](RESULTS.md): that document proves the G1-negative
/ G4-positive mechanism on 32×32 CIFAR-100 with a toy autoencoder and a proxy
MMD metric. This document is where those same conclusions get re-tested at
real resolution, with a real bitstream, and against a published baseline —
the numbers below are placeholders until the training run in
[`real/flow.py`](../real/flow.py) has been executed on the target GPU; the
protocol, scripts, and pass/fail criteria are final.

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
# argument list — arch/quality, patch, npool, knn, steps, lams, seeds)
python real/flow.py --train_root data/clic/train --quality 1 \
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

*(pending: fill in after running the commands above on the target GPU)*

| λ | bpp | PSNR_sample | LPIPS | FID | Var_z |
|---|---|---|---|---|---|
| 0.00 | | | | | |
| 0.25 | | | | | |
| 0.50 | | | | | |
| 0.75 | | | | | |
| 1.00 | | | | | |

MS-ILLM comparison point (quality=_, target bpp=_): PSNR=_, LPIPS=_, FID=_.
