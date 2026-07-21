#!/usr/bin/env python
"""
Tiny de-risk for "Compression as Conditional Optimal Transport".

The proposed training objective is a SINGLE Sinkhorn divergence on the extended
space (image, condition) with cost

    C_ij = c_x(x_hat_i, x_j) + eta * c_y(y_i, y_j)

where eta is the coupling-restriction strength = the distortion-perception knob:
    eta = 0    -> plain Sinkhorn (pure realism; perfect-perception endpoint)
    eta -> inf -> coupling forced diagonal (per-sample reconstruction; MMSE endpoint)

Two make-or-break questions, answered here on a toy problem with a KNOWN
bimodal posterior (so both endpoints have closed forms):

  Q1 (coupling geometry): as eta sweeps, does the Sinkhorn plan move from
     uniform to diagonal SMOOTHLY (a usable intermediate regime exists), or does
     it jump abruptly 0 -> diagonal?  Chemseddine et al. note that with finite
     empirical batches the Y-diagonal coupling is not recovered cleanly; in
     compression (every image has a distinct high-dim condition) this is the #1
     technical risk.  We probe it directly, including the high-dim-condition
     regime where all off-diagonal c_y distances concentrate.

  Q2 (does the knob trace a frontier?): train a small conditional generator
     with ONLY this objective at several eta, and measure
       distortion  = paired MSE  E||G(z,y_i) - x_i||^2
       perception  = off-manifold distance E[min over modes ||G(z,y_i) - mode||]
     Expected if the idea works: distortion decreases with eta while
     off-manifold distance increases -- i.e. eta traverses the D-P frontier
     inside one objective.  Sanity anchors from theory: distortion(MMSE) = s^2 +
     sigma^2, distortion(perfect posterior sampling) ~ 2 s^2 + 2 sigma^2 (the
     Blau-Michaeli <=2x law), off-manifold(MMSE) = s, off-manifold(sampler) ~ 0.

Toy data (closed-form posterior):
    y ~ U[-1,1]^dy,  mode m in {-1,+1} equally,  x = mu(y) + m*s*u(y) + noise
    mu, u fixed smooth maps.  Posterior of x|y is two Gaussians at mu +- s*u.

Everything is plain PyTorch; runs on CPU in minutes, GPU not required.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn as nn


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


# ---------------------------------------------------------------------------
# Log-domain Sinkhorn (uniform marginals) -> transport plan P
# ---------------------------------------------------------------------------


def sinkhorn_plan(C, eps=0.05, iters=300):
    """C: (N, M) cost. Returns P (N, M) with uniform marginals, entropic reg eps."""
    N, M = C.shape
    log_a = torch.full((N,), -math.log(N), device=C.device)
    log_b = torch.full((M,), -math.log(M), device=C.device)
    f = torch.zeros(N, device=C.device)
    g = torch.zeros(M, device=C.device)
    K = -C / eps
    for _ in range(iters):
        f = eps * log_a - eps * torch.logsumexp(K + g[None, :] / eps, dim=1)
        g = eps * log_b - eps * torch.logsumexp(K + f[:, None] / eps, dim=0)
    logP = K + f[:, None] / eps + g[None, :] / eps
    return torch.exp(logP)


# ---------------------------------------------------------------------------
# Toy conditional data with known bimodal posterior
# ---------------------------------------------------------------------------


class ToyPosterior:
    """x = mu(y) + m*s*u(y) + sigma*noise, m in {-1,+1}. dx-dim x, dy-dim y."""

    def __init__(self, dy=2, dx=2, s=1.0, sigma=0.05, seed=0, device="cpu"):
        g = torch.Generator().manual_seed(seed)
        self.A = torch.randn(dy, dx, generator=g).to(device) / math.sqrt(dy)
        self.B = torch.randn(dy, dx, generator=g).to(device) / math.sqrt(dy)
        self.dy, self.dx, self.s, self.sigma = dy, dx, s, sigma
        self.device = device

    def mu(self, y):
        return torch.tanh(y @ self.A)

    def u(self, y):
        v = torch.tanh(y @ self.B) + 0.5
        return v / v.norm(dim=1, keepdim=True).clamp(min=1e-6)

    def sample(self, n):
        y = torch.rand(n, self.dy, device=self.device) * 2 - 1
        m = (torch.randint(0, 2, (n, 1), device=self.device) * 2 - 1).float()
        x = (
            self.mu(y)
            + m * self.s * self.u(y)
            + self.sigma * torch.randn(n, self.dx, device=self.device)
        )
        return x, y

    def modes(self, y):
        mu, u = self.mu(y), self.u(y)
        return mu + self.s * u, mu - self.s * u

    def off_manifold(self, x_hat, y):
        m1, m2 = self.modes(y)
        d1 = (x_hat - m1).norm(dim=1)
        d2 = (x_hat - m2).norm(dim=1)
        return torch.minimum(d1, d2)


# ---------------------------------------------------------------------------
# Condition metric c_y: raw L2 vs low-dim embeddings (the design under test)
# ---------------------------------------------------------------------------


def make_cy(kind, dy, seed=0, device="cpu"):
    """
    Returns an embedding function E(y); c_y is squared L2 on E(y).
      'raw'    : identity (the high-dim cliff baseline)
      'projD'  : FIXED random Gaussian projection to D dims (e.g. 'proj4')
      'poolD'  : mean-pool dy into D blocks (e.g. 'pool8'; needs D | dy)
    The projection is fixed (not learned): what is being tested is the metric
    design, not metric learning.
    """
    if kind == "raw":
        return lambda y: y
    if kind.startswith("proj"):
        d = int(kind[4:])
        g = torch.Generator().manual_seed(seed + 777)
        W = (torch.randn(dy, d, generator=g) / math.sqrt(dy)).to(device)
        return lambda y: y @ W
    if kind.startswith("pool"):
        d = int(kind[4:])
        assert dy % d == 0, f"pool{d} needs {d} | dy={dy}"
        return lambda y: y.view(y.shape[0], d, dy // d).mean(-1)
    raise ValueError(kind)


def cond_cost(y, embed):
    e = embed(y)
    cy = torch.cdist(e, e).pow(2)
    return cy / cy[cy > 0].mean()


# ---------------------------------------------------------------------------
# Q1: coupling diagonality vs eta
# ---------------------------------------------------------------------------


def q1_coupling(
    dy_list=(2, 64), N=256, etas=None, eps=0.05, seed=0, device="cpu", cy_kind="raw"
):
    etas = etas if etas is not None else torch.logspace(-2, 2, 13).tolist()
    print(
        "\n=== Q1: Sinkhorn plan diagonality vs eta "
        f"(N={N}, eps={eps}, c_y='{cy_kind}') ==="
    )
    print(
        "diag_mass = sum_i P_ii  (uniform plan -> 1/N = "
        f"{1.0/N:.4f}; perfectly diagonal -> 1)"
    )
    results = {}
    for dy in dy_list:
        set_seed(seed)
        toy = ToyPosterior(dy=dy, dx=2, device=device)
        x, y = toy.sample(N)
        # fake batch: same conditions, generator imperfect -> x_hat = other mode
        # sample + noise (worst case for identity pairing)
        m1, m2 = toy.modes(y)
        pick = torch.rand(N, 1, device=device) < 0.5
        x_hat = torch.where(pick, m1, m2) + 0.15 * torch.randn(N, 2, device=device)

        cx = torch.cdist(x_hat, x).pow(2)
        embed = make_cy(cy_kind, dy, seed=seed, device=device)
        cy = cond_cost(y, embed)
        offdiag = cy[~torch.eye(N, dtype=bool, device=device)]
        conc = float(offdiag.std() / offdiag.mean())  # concentration of c_y

        row = []
        for eta in etas:
            P = sinkhorn_plan(cx + eta * cy, eps=eps)
            diag = float(torch.diagonal(P).sum())
            row.append(diag)
        results[dy] = (etas, row, conc)
        print(
            f"\n dy={dy:3d}  (off-diag c_y concentration std/mean = {conc:.3f}; "
            f"smaller = distances concentrate = harder)"
        )
        print("  eta:      " + " ".join(f"{e:8.3g}" for e in etas))
        print("  diagmass: " + " ".join(f"{d:8.3f}" for d in row))
    print("\nHow to read Q1: a SMOOTH monotone rise over ~2 decades of eta means a")
    print("usable intermediate regime exists. A near-step jump (flat ~0 then ~1")
    print("within one small eta interval) means the knob is degenerate there ->")
    print("the relaxation/metric design becomes a required contribution.")
    return results


# ---------------------------------------------------------------------------
# Q2: train a tiny conditional generator with ONLY the extended-cost Sinkhorn
# ---------------------------------------------------------------------------


class CondGen(nn.Module):
    def __init__(self, dy, dx, dz=4, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dy + dz, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, dx),
        )
        self.dz = dz

    def forward(self, y):
        z = torch.randn(y.shape[0], self.dz, device=y.device)
        return self.net(torch.cat([y, z], dim=1))


def train_one_eta(
    toy, eta, steps=1500, N=256, eps=0.05, lr=1e-3, device="cpu", seed=0, cy_kind="raw"
):
    set_seed(seed)  # same init across etas
    G = CondGen(toy.dy, toy.dx).to(device)
    opt = torch.optim.Adam(G.parameters(), lr=lr)
    embed = make_cy(cy_kind, toy.dy, seed=seed, device=device)
    for t in range(steps):
        x, y = toy.sample(N)
        x_hat = G(y)
        cx = torch.cdist(x_hat, x).pow(2)
        with torch.no_grad():
            cy = cond_cost(y, embed)
            P = sinkhorn_plan((cx + eta * cy).detach(), eps=eps, iters=150)
        loss = (P * cx).sum()  # transport-plan-weighted cost (OT-guided)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        x, y = toy.sample(4096)
        x_hat = G(y)
        distortion = float((x_hat - x).pow(2).sum(dim=1).mean())
        offman = float(toy.off_manifold(x_hat, y).mean())
        # mode balance: fraction assigned to +mode (0.5 = both modes covered)
        m1, m2 = toy.modes(y)
        frac_plus = float(
            ((x_hat - m1).norm(dim=1) < (x_hat - m2).norm(dim=1)).float().mean()
        )
    return distortion, offman, frac_plus


def q2_frontier(
    dy=2,
    etas=(0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0),
    steps=1500,
    N=256,
    eps=0.05,
    device="cpu",
    seed=0,
    cy_kind="raw",
):
    toy = ToyPosterior(dy=dy, dx=2, device=device, seed=seed)
    # theory anchors
    d_mmse = toy.s**2 + toy.sigma**2 * toy.dx
    d_pp = 2 * toy.s**2 + 2 * toy.sigma**2 * toy.dx
    print(
        f"\n=== Q2: trained-generator frontier vs eta (dy={dy}, "
        f"c_y='{cy_kind}') ==="
    )
    print(
        f"theory anchors: distortion(MMSE)={d_mmse:.3f}  "
        f"distortion(posterior sampling)~{d_pp:.3f} (the <=2x law)"
    )
    print(
        f"                off-manifold(MMSE)={toy.s:.3f}  "
        f"off-manifold(sampler)~{toy.sigma * math.sqrt(toy.dx):.3f}"
    )
    print(f"{'eta':>8} {'distortion':>11} {'off-manifold':>13} {'frac+mode':>10}")
    rows = []
    for eta in etas:
        d, o, fp = train_one_eta(
            toy,
            eta,
            steps=steps,
            N=N,
            eps=eps,
            device=device,
            seed=seed,
            cy_kind=cy_kind,
        )
        rows.append((eta, d, o, fp))
        print(f"{eta:>8.3g} {d:>11.4f} {o:>13.4f} {fp:>10.2f}")
    print("\nHow to read Q2 (the frontier test):")
    print(" * eta small  -> distortion near the ~2x sampler level, off-manifold")
    print("   near sigma (realistic samples on the modes).")
    print(" * eta large  -> distortion falls toward the MMSE anchor, off-manifold")
    print("   rises toward s (blurry mode-average).")
    print(" * MONOTONE movement between the two anchors across intermediate eta")
    print("   = the knob traverses the D-P frontier inside ONE objective. PASS.")
    print(" * If instead metrics jump between the two extremes with no usable")
    print("   middle, or off-manifold is large at ALL eta (mode collapse /")
    print("   averaging even at eta=0), the plain extended cost is not enough ->")
    print("   the relaxation/metric design is the research work. That is not a")
    print("   kill: it localises exactly where the contribution must live.")
    return rows


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", choices=["1", "2", "both"], default="both")
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--dy_q2", type=int, default=2)
    ap.add_argument(
        "--cy",
        type=str,
        default="raw",
        help="condition metric: raw | projD (e.g. proj4) | poolD",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    if args.q in ("1", "both"):
        q1_coupling(
            N=args.N, eps=args.eps, seed=args.seed, device=device, cy_kind=args.cy
        )
    if args.q in ("2", "both"):
        q2_frontier(
            dy=args.dy_q2,
            steps=args.steps,
            N=args.N,
            eps=args.eps,
            device=device,
            seed=args.seed,
            cy_kind=args.cy,
        )


if __name__ == "__main__":
    main()
