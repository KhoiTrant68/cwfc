#!/usr/bin/env python
"""
G4 — W-Flow conditional-OT endpoint.

Motivation (from the G1 finding): an AGGREGATE extended-cost OT objective matches
the marginal p(x) but its cost-mixing knob eta cannot steer the CONDITIONAL
p(x | y_hat) — the K draws of one condition share an identical condition
embedding, so c_y is constant across them and eta cannot differentiate
within-condition draws. Diversity comes off the floor (non-detached potential)
but is an eta-independent constant: no distortion-perception (D-P) frontier on
real images. See NOTES_g1_g4.md.

G4 fixes this by making the transport CONDITIONAL: a rectified-flow velocity field
v(x_t, t, y_hat) that pushes a noise prior to p(x | y_hat) directly. The perception
endpoint (diverse, sharp posterior samples) and its knob now live INSIDE each
conditional fibre, which is exactly what the aggregate objective could not reach.

Key pieces:
  * frozen weak TinyAE (reused from g1_pilot) -> spatial condition y_hat.
  * conditional targets by kNN in the pool embedding of y_hat: for an anchor
    condition y_hat_i, the real conditional set is the images whose codes are
    nearest to it. This gives p(x|y_hat_i) genuine multi-image SPREAD (the thing
    K-samples faked but aggregate-OT could not use).
  * rectified flow: x_t=(1-t)x_0+t x_1, target velocity x_1-x_0, regress v.
  * the D-P knob lam blends the flow target between a random neighbour (lam=0,
    perception: sample the posterior) and the neighbourhood MEAN (lam=1,
    distortion: MMSE). Target variance ~ (1-lam)^2 -> Var_z should DECREASE and
    PSNR INCREASE monotonically with lam. That eta-like frontier is precisely
    what G1's aggregate objective failed to produce; G4 PASSES iff it reappears.

Metrics (same family as G1):
  psnr_sample : PSNR of ONE stochastic draw vs the true source x_i (distortion of
                the coded sample; rises with lam as the sampler -> MMSE).
  psnr_mmse   : PSNR of the mean of draws vs x_i (the MMSE point; ~high, ~flat).
  mmd         : marginal realism, {samples} vs fresh real (lower = better).
  div         : Var_z over noise draws at fixed y_hat (diversity; the axis G1's
                eta could not move — here lam MUST move it).

Reuses g1_pilot for AE / data / embeddings / MMD, so it inherits --dataset
{synth,cifar,cifar100}, --data_root, and the AE-weakening knobs.

Runs on a single GPU (one flow model per lam); sweep lam sequentially.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import g1_pilot as g1


# ---------------------------------------------------------------------------
# Conditional velocity field  v(x_t, t, y_hat) -> velocity  (same-resolution CNN)
# ---------------------------------------------------------------------------


class CondVelocity(nn.Module):
    """
    Small same-resolution conv net. Inputs are concatenated along channels:
    the current state x_t (3), the spatial condition y_hat upsampled to image
    resolution (zc), and the time t as a broadcast channel (1). Predicts the
    rectified-flow velocity (3) at (x_t, t) given the condition.
    """

    def __init__(self, zc, ch=64):
        super().__init__()
        cin = 3 + zc + 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, ch, 3, 1, 1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.GELU(),
            nn.Conv2d(ch, 3, 3, 1, 1),
        )

    def forward(self, x, t, y):
        H = x.shape[-1]
        yup = F.interpolate(y, size=(H, H), mode="nearest")      # (B,zc,H,W)
        tch = t.expand(x.shape[0], 1, H, H)                       # (B,1,H,W)
        return self.net(torch.cat([x, yup, tch], dim=1))


# ---------------------------------------------------------------------------
# Precompute the frozen pool: images, their codes, and code embeddings
# ---------------------------------------------------------------------------


def build_pool(ae, data, embed, npool, device):
    """Sample a fixed pool of real images, encode to y_hat, embed for kNN."""
    with torch.no_grad():
        X = data.sample(npool)                        # (Np,3,H,W)
        Y = ae.condition(X)                           # (Np,zc,h,w)
        E = embed(Y)                                  # (Np,d)
    return X, Y, E


# ---------------------------------------------------------------------------
# Conditional rectified-flow training with the lam distortion-perception knob
# ---------------------------------------------------------------------------


def train_wflow(net, X, Y, E, lam, steps, N, knn, lr, device):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    Np = X.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, Np, (N,), device=device)
        ya = Y[idx]                                   # anchor condition (N,zc,h,w)
        with torch.no_grad():
            d = torch.cdist(E[idx], E)                # (N,Np) code-embedding dist
            d.scatter_(1, idx[:, None], float("inf")) # exclude self
            nn_idx = d.topk(knn, largest=False, dim=1).indices   # (N,knn)
            Xnb = X[nn_idx]                           # (N,knn,3,H,W) real neighbours
            pick = torch.randint(0, knn, (N,), device=device)
            x_samp = Xnb[torch.arange(N, device=device), pick]   # random neighbour
            x_mean = Xnb.mean(dim=1)                  # neighbourhood MMSE point
            # knob: lam=0 -> sample posterior (perception); lam=1 -> mean (distortion)
            x1 = (1.0 - lam) * x_samp + lam * x_mean
            x0 = torch.randn_like(x1)
            t = torch.rand(N, 1, 1, 1, device=device)
            xt = (1.0 - t) * x0 + t * x1
            target = x1 - x0
        v = net(xt, t, ya)
        loss = F.mse_loss(v, target)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return float(loss.detach())


@torch.no_grad()
def sample_wflow(net, y, n_steps=20, x0=None):
    """Integrate dx/dt = v(x,t,y) from t=0..1 with Euler. y:(B,zc,h,w)."""
    B, _, h, w = y.shape
    H = h * 4                                          # AE downsamples 4x
    if x0 is None:
        x0 = torch.randn(B, 3, H, H, device=y.device)
    x = x0
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((B, 1, 1, 1), i * dt, device=y.device)
        x = x + dt * net(x, t, y)
    return x.clamp(0, 1)


# ---------------------------------------------------------------------------
# Evaluation: the D-P metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def conditional_mmd(X, E, idx, draws, knn, seed, device):
    """
    CONDITIONAL realism: for each anchor condition y_hat_i, compare the generator's
    draws at that FIXED condition against the real conditional set (the knn real
    neighbours of y_hat_i in the code embedding), then average over anchors. Unlike
    the marginal MMD (samples vs fresh random real), this measures whether each
    p(x|y_hat) lands in the RIGHT fibre — the metric the aggregate objective (G1)
    could not produce. Lower = samples match the local conditional better.
    """
    d = torch.cdist(E[idx], E)
    d.scatter_(1, idx[:, None], float("inf"))
    nn_idx = d.topk(knn, largest=False, dim=1).indices        # (N,knn)
    N = idx.shape[0]
    vals = []
    for j in range(N):
        gen = draws[:, j].flatten(1)                          # (n_draws,D) samples @ y_hat_j
        real = X[nn_idx[j]].flatten(1)                        # (knn,D) real neighbours
        vals.append(g1.mmd_rff(gen, real, seed=seed))
    return float(np.mean(vals))


@torch.no_grad()
def evaluate(net, ae, data, X, Y, E, N, n_steps, n_draws, knn, seed, device):
    g1.set_seed(seed + 1)
    Np = X.shape[0]
    idx = torch.randint(0, Np, (N,), device=device)
    y_hat = Y[idx]
    x_true = X[idx]
    # n_draws stochastic samples at FIXED condition
    draws = torch.stack([sample_wflow(net, y_hat, n_steps) for _ in range(n_draws)])
    one = draws[0]                                     # a single coded sample
    mmse = draws.mean(dim=0)                           # the MMSE estimate
    mse_s = float((one - x_true).pow(2).mean())
    mse_m = float((mmse - x_true).pow(2).mean())
    psnr_sample = -10 * math.log10(mse_s) if mse_s > 0 else 99.0
    psnr_mmse = -10 * math.log10(mse_m) if mse_m > 0 else 99.0
    div = float(draws.flatten(2).var(dim=0).mean())    # Var_z[G(z,y_hat)]
    x_real = data.sample(N)
    mmd = g1.mmd_rff(one.flatten(1), x_real.flatten(1), seed=seed)   # marginal realism
    cmmd = conditional_mmd(X, E, idx, draws, knn, seed, device)      # conditional realism
    return psnr_sample, psnr_mmse, mmd, cmmd, div


# ---------------------------------------------------------------------------
# Qualitative process visualisation: a fixed set of conditions, several draws
# each, at a given lam. Across lam the SAME conditions are shown, so the eye
# sees the draws spread out at lam=0 (posterior sampling) and collapse onto the
# MMSE mean at lam=1 — the diversity axis, made visible. Optional (needs
# matplotlib); never touched by the metric path.
# ---------------------------------------------------------------------------


@torch.no_grad()
def save_sample_grid(net, ae, X, Y, lam, path, n_rows=6, n_draws=5,
                     n_steps=20, device="cpu"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[save_samples] matplotlib not installed — skipping grid"); return
    g = torch.Generator(device="cpu").manual_seed(12345)          # fixed rows
    idx = torch.randperm(X.shape[0], generator=g)[:n_rows].to(device)
    y_hat = Y[idx]
    x_true = X[idx]
    ae_dec = ae.decode(y_hat).clamp(0, 1)                          # AE recon of ŷ
    draws = [sample_wflow(net, y_hat, n_steps) for _ in range(n_draws)]
    mmse = torch.stack(draws).mean(0)

    cols = [("x_true", x_true), ("AE(ŷ)", ae_dec)]
    cols += [(f"draw {i+1}", draws[i]) for i in range(n_draws)]
    cols += [("MMSE", mmse)]
    ncol = len(cols)
    fig, axes = plt.subplots(n_rows, ncol, figsize=(1.4 * ncol, 1.4 * n_rows))
    axes = axes.reshape(n_rows, ncol)
    for r in range(n_rows):
        for c, (title, batch) in enumerate(cols):
            ax = axes[r, c]
            img = batch[r].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8)
    fig.suptitle(f"G4 samples at fixed ŷ — λ={lam:g} "
                 f"({'perception' if lam == 0 else 'MMSE' if lam == 1 else 'mid'})",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# lam sweep = the conditional D-P frontier
# ---------------------------------------------------------------------------


def frontier_lambda(ae, data, embed, y_shape, lams, steps, N, npool, knn,
                    n_steps, n_draws, lr, device, seeds, ch=64, save_dir=None):
    X, Y, E = build_pool(ae, data, embed, npool, device)
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
    ns = len(seeds)
    tag = f"mean±std over {ns} seeds" if ns > 1 else "seed " + str(seeds[0])
    print(f"\n=== G4 conditional-flow D-P frontier (knn={knn}, npool={npool}, "
          f"ode_steps={n_steps}, {tag}) — lam 0=perception .. 1=distortion(MMSE) ===")
    if ns > 1:
        print(f"{'lam':>6} {'PSNR_samp':>16} {'PSNR_mmse':>16} {'MMD':>14} "
              f"{'condMMD':>14} {'Div(Var_z)':>16}")
    else:
        print(f"{'lam':>6} {'PSNR_samp':>10} {'PSNR_mmse':>10} {'MMD':>12} "
              f"{'condMMD':>12} {'Div(Var_z)':>12}")
    out = []
    for lam in lams:
        ps, pm, md, cm, dv = [], [], [], [], []
        for sd in seeds:
            g1.set_seed(sd)
            net = CondVelocity(zc=y_shape[0], ch=ch).to(device)
            train_wflow(net, X, Y, E, lam, steps, N, knn, lr, device)
            a, b, c, cc, d_ = evaluate(net, ae, data, X, Y, E, N, n_steps,
                                       n_draws, knn, sd, device)
            ps.append(a); pm.append(b); md.append(c); cm.append(cc); dv.append(d_)
            if save_dir and sd == seeds[0]:        # qualitative grid, first seed
                import os
                save_sample_grid(net, ae, X, Y, lam,
                                 os.path.join(save_dir, f"samples_lam{lam:g}.png"),
                                 n_steps=n_steps, device=device)
        row = {"lam": lam,
               "psnr_sample": float(np.mean(ps)), "psnr_sample_std": float(np.std(ps)),
               "psnr_mmse": float(np.mean(pm)), "psnr_mmse_std": float(np.std(pm)),
               "mmd": float(np.mean(md)), "mmd_std": float(np.std(md)),
               "cond_mmd": float(np.mean(cm)), "cond_mmd_std": float(np.std(cm)),
               "div": float(np.mean(dv)), "div_std": float(np.std(dv))}
        out.append(row)
        if ns > 1:
            print(f"{lam:>6.2f} {np.mean(ps):>8.2f}±{np.std(ps):<6.2f} "
                  f"{np.mean(pm):>8.2f}±{np.std(pm):<6.2f} "
                  f"{np.mean(md):>10.3e} {np.mean(cm):>10.3e} "
                  f"{np.mean(dv):>10.3e}±{np.std(dv):<.1e}")
        else:
            print(f"{lam:>6.2f} {np.mean(ps):>10.2f} {np.mean(pm):>10.2f} "
                  f"{np.mean(md):>12.4e} {np.mean(cm):>12.4e} {np.mean(dv):>12.4e}")
    print("\nPASS iff: Div FALLS monotonically with lam (knob controls diversity — "
          "the axis G1's eta could NOT move) AND PSNR_sample RISES with lam. condMMD "
          "(realism WITHIN the y_hat fibre) should be LOW at small lam (posterior "
          "sampling lands in the right conditional) and rise toward the MMSE endpoint. "
          "That is the conditional D-P frontier.")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synth", "cifar", "cifar100"],
                    default="synth")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--npool", type=int, default=2000,
                    help="so anh trong pool co dinh de lay kNN dieu kien")
    ap.add_argument("--knn", type=int, default=16,
                    help="so lang gieng theo embedding y_hat = do rong cua p(x|y_hat)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--ae_steps", type=int, default=3000)
    ap.add_argument("--ae_zc", type=int, default=2)
    ap.add_argument("--ae_qscale", type=float, default=0.5)
    ap.add_argument("--n_steps", type=int, default=20, help="so buoc Euler khi lay mau")
    ap.add_argument("--n_draws", type=int, default=8, help="so mau z de do Var_z")
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed", type=str, default="pool",
                    help="embedding cua y_hat cho kNN dieu kien (pool/proj8/raw)")
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--save_samples", type=str, default=None,
                    help="thu muc de luu luoi mau dinh tinh moi lam (can matplotlib). "
                         "Tat mac dinh; khong anh huong metric.")
    args = ap.parse_args()

    if args.smoke:
        args.H, args.N, args.npool, args.steps, args.ae_steps = 32, 128, 512, 400, 800
        args.lams, args.seeds = [0.0, 0.5, 1.0], [0]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  dataset={args.dataset}  H={args.H}  embed={args.embed}  "
          f"lams={args.lams}  knn={args.knn}  ae_zc={args.ae_zc} ae_qscale={args.ae_qscale}")

    g1.set_seed(0)
    data = g1.make_dataset(args.dataset, args.H, seed=0, device=device,
                           root=args.data_root, download=args.download)
    ae = g1.TinyAE(zc=args.ae_zc, qscale=args.ae_qscale).to(device)
    rec = g1.train_ae(ae, data, steps=args.ae_steps, N=args.N, lr=1e-3, device=device)
    with torch.no_grad():
        x = data.sample(args.N)
        y_hat = ae.condition(x)
        y_shape = tuple(y_hat.shape[1:])
        ae_psnr = -10 * math.log10(float((ae.decode(y_hat) - x).pow(2).mean()))
    print(f"frozen AE: recon mse={rec:.4f}, quantized-recon PSNR={ae_psnr:.2f} dB, "
          f"y_hat shape={y_shape}")

    embed = g1.make_embed(args.embed, y_shape, device=device)
    results = frontier_lambda(ae, data, embed, y_shape, args.lams, args.steps,
                              args.N, args.npool, args.knn, args.n_steps,
                              args.n_draws, args.lr, device, args.seeds, ch=args.ch,
                              save_dir=args.save_samples)

    if args.out:
        import json
        payload = {"config": {k: getattr(args, k) for k in
                              ["dataset", "H", "N", "npool", "knn", "steps",
                               "ae_steps", "ae_zc", "ae_qscale", "n_steps",
                               "n_draws", "embed", "lams", "seeds"]},
                   "ae_recon_psnr": ae_psnr, "y_shape": list(y_shape),
                   "frontier": results}
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nsaved results -> {args.out}")


if __name__ == "__main__":
    main()
