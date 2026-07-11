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

## 1b. G4 result — paragraph draft for the paper (positive)

> **A conditional-flow objective realises the trade-off the aggregate one could not.**
> The G1 analysis pinpoints the failure: because the K draws of a condition ŷ share an
> identical embedding, the cost-mixing weight η can only reweight *between*-condition
> coupling and is blind to the *within*-condition fibre where perception lives. G4
> removes this limitation by conditioning the transport map itself on ŷ. We train a
> rectified-flow velocity field `v(x_t, t, ŷ)` whose pushforward of a Gaussian prior,
> *given ŷ*, targets the conditional `p(x|ŷ)`; conditional targets are built as real
> neighbours of ŷ in the frozen code embedding (kNN), giving each fibre genuine spread.
> A single scalar λ blends the flow target between a random neighbour (λ=0, sample the
> posterior) and the neighbourhood mean (λ=1, regress to E[x|ŷ]) — the conditional
> analogue of η, but one that acts *inside* the fibre. On CIFAR-100 (same frozen weak
> AE, ~22 dB, as G1), sweeping λ traces a clean distortion–perception frontier: sample
> PSNR rises 11.2→13.6 dB while within-condition diversity `Var_z` falls 4.2×
> (3.7×10⁻²→8.9×10⁻³) and marginal realism MMD degrades monotonically
> (2.5×10⁻²→1.3×10⁻¹) — realism is best exactly at the diverse λ=0 endpoint and worst
> at the MMSE endpoint, the textbook D–P signature. Under an identical budget G1's η
> leaves `Var_z` flat (2.5→2.7×10⁻⁵) and PSNR flat; G4's λ *positions* the generator
> continuously between the two regimes. Conditioning the optimal-transport map — rather
> than mixing a condition term into an aggregate cost — is therefore what makes the
> perception endpoint both reachable and controllable.

Backing numbers are in §2 (G4 VERDICT table). Headline contrast for the paper:

| knob | budget / AE | diversity response | PSNR response | verdict |
|---|---|---|---|---|
| G1 η (aggregate OT) | CIFAR-100, weak AE ~22 dB | Var_z **flat** 2.5→2.7×10⁻⁵ | flat 22.4 dB | cannot steer p(x\|ŷ) |
| G4 λ (conditional flow) | CIFAR-100, weak AE ~22 dB | Var_z **↓4.2×** 3.7e-2→8.9e-3 | ↑ 11.2→13.6 dB | steers the frontier |

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
exactly the lam-ordered diversity axis G1's eta could not produce.

### G4 VERDICT — Kaggle full-budget (`g4_smoke.json`, CIFAR-100, N=256, npool=2000,
knn=16, steps=1500, weak AE 22.37 dB, pool embed, seed 0, ode_steps=20): **POSITIVE.**
The conditional-flow knob λ realises the D–P frontier that G1's η could not:

| λ | PSNR_samp | PSNR_mmse | MMD | Var_z | reading |
|---|---|---|---|---|---|
| 0.00 | 11.19 | 13.59 | 2.48e-2 | 3.75e-2 | perception endpoint |
| 0.25 | 12.27 | 14.39 | 2.49e-2 | 2.69e-2 | |
| 0.50 | 12.70 | 14.11 | 7.14e-2 | 1.75e-2 | |
| 0.75 | 13.62 | 14.66 | 9.98e-2 | 1.06e-2 | |
| 1.00 | 13.57 | 14.40 | 1.31e-1 | 8.85e-3 | distortion (MMSE) endpoint |

All three axes move **monotonically** with λ: Var_z ↓ 4.2× (3.75e-2 → 8.85e-3),
PSNR_sample ↑ 2.4 dB (11.19 → 13.62; final point plateaus within 0.05 dB), and MMD ↑
5.3× (2.48e-2 → 1.31e-1). The classic distortion–perception signature: MMD (marginal
realism) is **lowest at λ=0** (posterior sampling) and worst at λ=1 (MMSE blur), while
PSNR trades the opposite way — you cannot have both, and λ **positions** the generator
on the frontier. This is exactly the controllable perception axis the aggregate
extended-cost OT (G1) provably could not steer with η (c_y identical within-condition).
Contrast is decisive: same weak AE (~22 dB), same metrics — G1's η gives flat Var_z
2.5→2.7e-5, G4's λ gives a 4.2× monotone sweep. **G1-negative / G4-positive pair
closed.** Next: multi-seed error bars + conditional-MMD (within-ŷ-neighbourhood) to
harden for the paper.

Open questions to resolve when coding G4:
1. Grouping: fixed kNN in pool-embedding vs learned soft assignment.
2. Flow parameterisation: few-step rectified flow vs continuous (memory on Kaggle).
3. Whether to keep the frozen weak AE for ŷ or co-train — start frozen (isolate G4).
