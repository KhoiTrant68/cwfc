# scripts/

Thin wrappers around the canonical commands (same recipes as the `Makefile`).
Run from anywhere — each script `cd`s to the repo root so `import g1_pilot`
resolves and outputs land in `results/` and `figs/`.

```bash
bash scripts/<name>.sh
```

Two environment overrides (defaults in `_common.sh`):

- `DATA_ROOT` — CIFAR-100 dataset root (default the Kaggle
  `fedesoriano/cifar100` path).
- `PY` — Python interpreter (default `python`).

```bash
DATA_ROOT=/path/to/cifar100 PY=python3 bash scripts/run_g4_full.sh
```

| script | what it does | outputs |
|---|---|---|
| `setup.sh` | install deps (core+viz; `--all` adds entropy/DP extras) | — |
| `smoke.sh` | synth CPU plumbing check for both pilots | `figs/samples_*.png` |
| `run_toy.sh` | toy 2-D extended-cost Sinkhorn frontier | stdout |
| `run_g1.sh` | G1 decisive negative (pool, seed 0) | `results/g1_potential.json` |
| `run_g1_full.sh` | G1 robustness (3 embeds × 3 seeds) | `results/g1_potential_full.json` |
| `run_g4.sh` | G4 λ-frontier (seed 0) + sample grids | `results/g4_smoke.json`, `figs/` |
| `run_g4_full.sh` | G4 hardened (3 seeds + condMMD) + grids | `results/g4_full.json`, `figs/` |
| `make_figs.sh` | render frontier + D-P plane figures | `figs/*.png` |
| `reproduce_all.sh` | G1-full → G4-full → figures (full paper pipeline) | `results/`, `figs/` |

Any extra flags are forwarded, e.g. `bash scripts/run_g4.sh --steps 3000`.
Trailing `"$@"` passthrough is not available on `reproduce_all.sh`.
