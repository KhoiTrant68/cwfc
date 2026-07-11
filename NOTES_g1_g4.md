# G1 negative result + G4 (W-Flow conditional-OT) plan

## 1. G1 result — paragraph draft for the paper

> **Can an aggregate extended-cost OT objective realise the distortion–perception
> trade-off?**  We test whether a single batch-marginal optimal-transport objective,
> with a cost that mixes reconstruction distance and a condition-embedding distance
> `C_ij = c_x(x̂_i, x_j) + η·c_y(E(ŷ_i), E(ŷ_j))`, can move a generator along the
> distortion–perception (D–P) curve by sweeping the mixing weight η. We freeze a
> tiny quantised autoencoder to produce the conditioning code ŷ, deliberately
> weakening it (coarse quantisation, small latent) so a genuine unresolvable
> residual exists (21.7 dB), and we draw K=4 noise samples per condition so the
> transport is rectangular and *can* express within-condition spread. We report
> fidelity (PSNR), marginal realism (MMD on random features), and, crucially,
> within-condition diversity `Var_z[G(z,ŷ)]`.
>
> Three findings emerge. (i) The plan-detached objective (crude/debiased) leaves
> diversity fully collapsed (`Var_z ≈ 5×10⁻⁷`) at every η — the barycentric,
> mean-seeking bias of entropic OT dominates and the debias gradient vanishes near
> collapse. (ii) Replacing it with a non-detached Sinkhorn-divergence loss
> (dual-potential, envelope-theorem gradients) lifts diversity ~50× off that floor
> (`Var_z ≈ 2.6×10⁻⁵`) on real images — the generative-OT machinery *does* inject
> genuine within-condition spread that the plan-detached surrogate cannot. (iii)
> **Yet η controls neither axis on real data**: both `Var_z` and PSNR are flat
> across η∈[0,30] (a clean, monotone D–P curve appears only on a *synthetic*
> dataset with a designed two-mode residual). This is structural, not a tuning
> artefact: the K draws of a single condition ŷ_i share an *identical* condition
> embedding, so `c_y` is constant across those K rows and η — which only scales
> `c_y` — cannot differentiate within-condition draws. η therefore governs
> *between-condition* coupling (fidelity / marginal realism) only; within-condition
> diversity is fixed by the divergence term as an η-independent constant. An
> aggregate extended-cost OT objective can thus match the marginal `p(x)` but
> provably cannot *steer* the conditional `p(x|ŷ)` — the perception endpoint of the
> D–P trade-off — with the cost-mixing knob. This localises exactly where the
> contribution must live: a **conditional** optimal-transport formulation that
> shapes each `p(x|ŷ)` directly (Section: W-Flow / G4).

Backing numbers (CIFAR-100, N=384, K=4, pool embed, steps=2000, eps=0.02):

| Loss | frozen AE | Var_z(η=0→30) | PSNR | reading |
|---|---|---|---|---|
| potential (weak AE) | 21.7 dB | 2.56 → 2.64 → 2.81 → 2.73 ×10⁻⁵ | 22.4 flat | off-floor, **η-flat** |
| debiased (weak AE)  | 21.7 dB | ~5.2×10⁻⁷ flat | 22.2 flat | collapse |
| null (strong AE, K=1) | 24.0 dB | ~6.5×10⁻⁷ flat | 24.9 flat | collapse |

Synth reference (designed 2-mode residual) DID show η-ordering 8.2→2.8×10⁻⁵ — cited
as the *contrast* that proves the real-data flatness is about the natural conditional,
not the objective's plumbing.

Robustness confirmed (`g1_potential_full.json`, 3 embeds × 3 seeds × 4 eta, weak AE
22.05 dB, mean Var_z): eta {0,0.3,3,30} → {2.54, 2.66, 2.74, 2.70}×10⁻⁵ (flat, and the
tiny drift goes the WRONG way — up, not toward MMSE), PSNR flat 22.39→22.42 dB. raw ≈
proj8 ≈ pool to 2 sig figs → the diversity axis is independent of the c_y embedding,
exactly as predicted (eta acts only between-condition). The 50× off-floor lift + eta-flat
pattern is stable across all embeddings and seeds.

## 2. G4 — W-Flow conditional-OT scope

Goal: replace the aggregate marginal match `{x̂} ↔ {x}` with a **per-condition**
match `p(x | ŷ)`, so the perception endpoint (diverse, realistic samples that stay
faithful to ŷ) is reachable and *controllable*.

Why this is the right lever (from G1): the aggregate objective's failure is that
`c_y` cannot see within-condition structure. A conditional-OT / flow objective
conditions the transport map itself on ŷ, so the trade-off knob acts *inside* each
conditional fibre.

De-risked already (G3): `wflow_endpoint.py` showed W-Flow inversion drives the
residual down ~9500× — the flow machinery works; G4 = wire the *conditional* endpoint.

Proposed design:
- **Generator/flow**: reuse `CondGenImg(z, ŷ)` as the conditional velocity field /
  pushforward. Condition by concatenating (or FiLM-ing) ŷ into the flow at each step.
- **Training signal**: conditional flow-matching / rectified-flow objective so that
  the pushforward of the noise prior *given ŷ* equals `p(x|ŷ)`. Build conditional
  pairs by grouping real x by ŷ-neighbourhood (kNN in the pool embedding) so each
  minibatch fibre has several real targets — this is what gives non-degenerate
  within-condition transport (the thing K-samples faked but aggregate-OT couldn't use).
- **The knob**: fidelity–perception trade now = weight between the conditional-OT /
  flow-matching term (perception, sample p(x|ŷ)) and a direct reconstruction term
  (distortion, regress to E[x|ŷ]). This is the *conditional* analogue of η and, per
  the G1 analysis, is the version that should actually move Var_z.
- **Metrics**: same triple (PSNR / MMD / Var_z) plus a *conditional* realism check
  (MMD computed within ŷ-neighbourhood, not just on the marginal) — this is the
  metric G1 was missing and the one that will show G4 working where G1 could not.

Plumbing validated (CPU synth smoke, 150 steps, weak AE 20.26 dB, lams {0,0.5,1}):
Div FALLS monotonically with lam (6.96e-2 → 5.16e-2 → 4.74e-2) and PSNR_sample RISES
(8.72 → 9.64), PSNR_mmse > PSNR_sample throughout — the conditional D-P frontier's
correct SHAPE already appears (absolute values poor only from under-training). This is
exactly the lam-ordered diversity axis G1's eta could not produce. Full-budget Kaggle
run pending (`g4_smoke.json`).

Open questions to resolve when coding G4:
1. Grouping: fixed kNN in pool-embedding vs learned soft assignment.
2. Flow parameterisation: few-step rectified flow vs continuous (memory on Kaggle).
3. Whether to keep the frozen weak AE for ŷ or co-train — start frozen (isolate G4).
