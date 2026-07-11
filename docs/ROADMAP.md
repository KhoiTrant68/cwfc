# Roadmap — de-risk gates G1–G4

The project is organised as a sequence of *gates*: cheap, falsifiable experiments
run in order, each of which can kill or redirect the idea before the expensive
next stage. A gate that fails is not wasted — a clean negative localises exactly
where the contribution must live.

| Gate | Question | Script(s) | Status |
|---|---|---|---|
| Toy (E1–E6) | Does the extended-cost conditional Sinkhorn trace a D–P frontier on a toy with a known bimodal posterior? Does a low-dim `c_y` embedding reopen the high-dim cliff? | `derisk_cot.py` | **Passed** — knob works, `pool`/`proj` beat `raw` |
| Entropy (b) | Does a K-component mixture entropy head lower the rate vs a single Gaussian, all else equal? | `derisk_mixture_entropy.py` | **Killed** — mixture ≈ single Gaussian; kept as documented negative + real rANS infra |
| D–P traversal | At a fixed rate, does a curved (manifold) traversal dominate the pixel-straight line on the PSNR–LPIPS plane? | `derisk_dp.py` | **Passed** — non-linear optimal traversal → room for an OT/flow method |
| G3 endpoint | Can W-Flow inversion produce a reconstruction consistent with the transmitted quantised cell (perfect-perception endpoint)? | `wflow_endpoint.py` | **Passed** — inversion drives residual into the cell (~9500× reduction); `WFlowAdapter` is a skeleton to wire to a real checkout |
| **G1 pilot** | Can an **aggregate** extended-cost OT steer `p(x\|ŷ)` with η on real images? | `g1_pilot.py`, `kaggle_g1.py` | **Negative (decisive)** — diversity lifts 50× off the floor but η is flat; structural (`c_y` identical within a condition). See [RESULTS.md](RESULTS.md) §1 |
| **G4** | Does a **conditional** rectified-flow objective steer the frontier with λ where G1 could not? | `g4_wflow.py` | **Positive** — λ traces a clean D–P frontier (Var_z ↓4.2×, PSNR ↑2.4 dB, MMD ↑5.3×). See [RESULTS.md](RESULTS.md) §2 |

## The through-line

G1 and G4 are the same problem attacked two ways under an identical budget and
frozen weak AE:

- **G1 (aggregate):** mix a condition term into a batch-marginal OT cost. Fails to
  move the perception axis because the knob is blind to within-condition structure.
- **G4 (conditional):** condition the transport map itself on ŷ. The knob now acts
  inside each conditional fibre and moves the frontier.

This negative→positive pair is the intended narrative for the paper: it *localises*
the failure of the aggregate formulation and *delivers* the conditional one that
resolves it.

## Next

- Harden the G4 table: multi-seed error bars + conditional-MMD (both already
  supported by `g4_wflow.py`).
- Wire `WFlowAdapter` to a real W-Flow checkout to swap the tiny AE + toy flow for
  the full-scale conditional endpoint.
