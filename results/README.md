# Results directory

JSON/CSV outputs from the experiment scripts land here (pass `--out results/<name>.json`).
The runs are executed on Kaggle (2× T4); the headline numbers extracted from these
files are recorded in [`../docs/RESULTS.md`](../docs/RESULTS.md).

Expected artefacts:

| file | produced by | what it holds |
|---|---|---|
| `g1_potential.json` | `g1_pilot.py --loss potential` | 3-way G1 frontier (potential loss) |
| `g1_debiased.json` | `g1_pilot.py --loss debiased` | G1 collapse control (weak AE) |
| `g1_null.json` | `g1_pilot.py --loss debiased` (strong AE, K=1) | G1 collapse control (strong AE) |
| `g1_potential_full.json` | `g1_pilot.py` (3 embeds × 3 seeds) | G1 robustness lock |
| `g4_smoke.json` | `g4_wflow.py` (seed 0) | G4 λ-frontier verdict |
| `g4_full.json` | `g4_wflow.py --seeds 0 1 2` | G4 hardened table (mean±std + condMMD) |

Large result files are git-ignored by default (see `.gitignore`); commit a
specific JSON explicitly with `git add -f results/<name>.json` when it backs a
claim in the paper.
