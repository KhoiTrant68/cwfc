# CWFC — Compression as Conditional Optimal Transport

De-risk experiments for framing learned image compression as **conditional
optimal transport**, and the central empirical result they produce: a
negative/positive pair showing *where* an aggregate OT formulation fails and *how*
a conditional one fixes it.

> **Headline.** An **aggregate** extended-cost OT objective matches the marginal
> `p(x)` but cannot steer the conditional `p(x | ŷ)` with its cost-mixing knob η
> (**G1, negative**). A **conditional** rectified-flow objective conditions the
> transport map on ŷ and traces the distortion–perception frontier with its knob λ
> (**G4, positive**). See [`docs/RESULTS.md`](docs/RESULTS.md).

## Repository map

The scripts are ordered as *de-risk gates* — cheap, falsifiable experiments run in
sequence (full gate table in [`docs/ROADMAP.md`](docs/ROADMAP.md)):

| file | gate | role | extra deps |
|---|---|---|---|
| `derisk_cot.py` | Toy | Extended-cost conditional Sinkhorn on a toy with a known bimodal posterior. Q1 = coupling geometry, Q2 = training frontier, `--cy` embedding hook. | — |
| `derisk_mixture_entropy.py` | Entropy (killed) | K-component mixture entropy head vs single Gaussian, with a real rANS roundtrip. Kept as a documented negative + coder infra. | `compressai`, `torchvision`, `pillow` |
| `derisk_dp.py` | D–P traversal | Is the optimal fixed-rate traversal non-linear on the PSNR–LPIPS plane? Toy and SD-VAE backbones. | `diffusers`, `lpips` (real mode only) |
| `wflow_endpoint.py` | G3 | W-Flow inversion to a quantisation-cell-consistent reconstruction (tested); `WFlowAdapter` is a skeleton to wire to a real checkout. | — (self-test) |
| `g1_pilot.py` | **G1** | Extended-cost OT lifted to small images with a spatial condition ŷ. **Negative result.** | — |
| `kaggle_g1.py` | G1 | Kaggle helper: train + freeze the AE once, reuse across sweeps. | — |
| `g4_wflow.py` | **G4** | Conditional rectified-flow endpoint with the λ knob. **Positive result.** | — |

Supporting: [`docs/`](docs/) (results + roadmap), [`results/`](results/) (JSON
outputs), `pyproject.toml`, `requirements.txt`, `Makefile`, `CITATION.cff`.

### Real-image track (`real/`)

The scripts above prove the G1/G4 mechanism at 32×32 CIFAR scale with a toy
autoencoder and a proxy MMD metric. `real/` scales G4 (the positive result)
to real photos, real bitrate, and a published baseline, on the same idea:

| file | role |
|---|---|
| `real/codec.py` | frozen `compressai` codec (replaces `TinyAE`); real bpp via actual entropy coding |
| `real/data.py` | Kodak/CLIC patch sampler + kNN conditional pool + a sanity-check tool for the conditional fibre |
| `real/flow.py` | `UNetVelocity` (replaces `CondVelocity`), single λ-conditioned model, patch=256 |
| `real/eval.py` | bpp/PSNR/MS-SSIM/LPIPS/FID on full Kodak/CLIC test images |
| `real/baselines/run_msillm.py` | MS-ILLM (ICML 2023) pretrained baseline, same metric protocol as `real/eval.py` |
| `real/viz_real.py` | CWFC λ-frontier vs. MS-ILLM on one bpp-matched D-P plane |
| `real/data/get_kodak.sh`, `real/data/get_clic.sh` | dataset download helpers |

Extra deps: `pip install -r real/requirements.txt`. See
[`docs/RESULTS_REAL.md`](docs/RESULTS_REAL.md) for the full protocol and
reproduce commands.

## Install

```bash
pip install -r requirements.txt          # or: pip install -e .[all]
```

The mechanism scripts (`derisk_cot.py`, `g1_pilot.py`, `g4_wflow.py`,
`wflow_endpoint.py`) need only `torch` + `numpy`. The extras above are only for the
entropy and SD-VAE experiments.

## Quickstart

```bash
# Plumbing check — synthetic data, no downloads (CPU-ok)
python g1_pilot.py --smoke
python g4_wflow.py --dataset synth --H 32 --N 96 --npool 300 --knn 8 \
  --steps 60 --ae_steps 100 --n_steps 8 --n_draws 4 --lams 0 0.5 1 --device cpu

# Toy extended-cost frontier (the core mechanism)
python derisk_cot.py --q both --N 1024 --steps 4000

# W-Flow inversion self-test (torch only)
python wflow_endpoint.py
```

`make help` lists canonical targets (`make smoke`, `make g1`, `make g4`, …).

## Reproduce the main result (Kaggle, 2× T4)

CIFAR-100 is read directly from the Kaggle `fedesoriano/cifar100` pickle; point
`--data_root` at the dataset root (**not** its `train/` subdir — the loader walks
for the pickle).

```bash
# G1 (negative): aggregate extended-cost OT, weak AE + K=4, potential loss
python g1_pilot.py --dataset cifar100 \
  --data_root /kaggle/input/datasets/fedesoriano/cifar100 \
  --H 32 --mode both --N 384 --steps 2000 --ae_steps 3000 \
  --ae_qscale 0.5 --ae_zc 2 --K 4 --eps 0.02 \
  --seeds 0 1 2 --etas 0 0.3 3 30 --embeds raw proj8 pool \
  --loss potential --out results/g1_potential_full.json

# G4 (positive): conditional rectified flow, lambda frontier
python g4_wflow.py --dataset cifar100 \
  --data_root /kaggle/input/datasets/fedesoriano/cifar100 \
  --H 32 --N 256 --npool 2000 --knn 16 --steps 1500 --ae_steps 3000 \
  --ae_zc 2 --ae_qscale 0.5 --n_steps 20 --n_draws 8 \
  --lams 0 0.25 0.5 0.75 1 --seeds 0 1 2 --embed pool --out results/g4_full.json
```

**Read.** G1 passes/kills on whether η moves `Var_z`/PSNR (it does **not** on real
data — the negative). G4 passes iff λ makes `Var_z` fall monotonically and PSNR
rise, with conditional-MMD low at the perception endpoint — the frontier η could
not produce. Full numbers and the paper-ready paragraphs are in
[`docs/RESULTS.md`](docs/RESULTS.md).

## Visualisation

`viz.py` turns the result JSONs into the argument figures (needs `matplotlib`;
`pip install -e .[viz]`). It is pure post-processing — no run is repeated:

```bash
python viz.py --g1 results/g1_potential_full.json \
              --g4 results/g4_full.json --out figs
```

produces `figs/g1_frontier.png` (flat in η — the negative), `figs/g4_frontier.png`
(monotone in λ — the positive), and `figs/dp_plane.png` (the PSNR-vs-MMD contrast:
G1's η cluster vs G4's λ frontier). Either `--g1`/`--g4` may be omitted.

For the qualitative *process* view, `g4_wflow.py --save_samples figs` dumps a grid
per λ (fixed ŷ, several draws + the MMSE mean) so the diversity collapse from λ=0
to λ=1 is visible directly.

## Key knobs (shared idea)

- **η / λ** — the distortion–perception knob. η mixes a condition cost into an
  aggregate OT plan (G1); λ blends the conditional-flow target between posterior
  sampling and the MMSE mean (G4).
- **`--ae_qscale` / `--ae_zc`** — weaken the frozen AE so a real residual exists
  (~22 dB). A strong AE pins the generator at the fidelity ceiling and flattens the
  frontier.
- **`--K`** (G1) — noise draws per condition, making the transport rectangular.
- **condition embedding** (`raw` / `proj{D}` / `pool`) — how `c_y` sees ŷ.

## Status

G1 is a **decisive negative** and G4 is **positive** — the pair is closed (see
[`docs/ROADMAP.md`](docs/ROADMAP.md)). Remaining hardening: multi-seed error bars +
conditional-MMD for the G4 table (both already supported by `g4_wflow.py`), and
wiring `WFlowAdapter` to a full-scale W-Flow checkout.

## Citation

See [`CITATION.cff`](CITATION.cff).
