#!/usr/bin/env python
"""
Visualisation of the experiment results — turns the metric JSONs into the
figures that carry the paper's argument. Pure post-processing: it reads the
result files produced by g1_pilot.py / g4_wflow.py and draws, so no experiment
has to be re-run.

Three figures:

  1. G1 frontier (the NEGATIVE) — Var_z / PSNR / MMD vs eta, one line per
     condition embedding. The point is that they are FLAT: eta does not move the
     perception axis. (`--g1 results/g1_potential_full.json`)

  2. G4 frontier (the POSITIVE) — Var_z / PSNR_sample / MMD vs lambda, with
     mean+/-std error bars when multiple seeds are present. The point is that all
     three move MONOTONICALLY: lambda steers the conditional D-P frontier.
     (`--g4 results/g4_full.json`)

  3. D-P plane contrast — PSNR vs MMD, G1's eta sweep (a cluster: no frontier)
     against G4's lambda sweep (a curve: the frontier). This is the one-figure
     summary of the whole G1-negative / G4-positive story.

Usage:
    python viz.py --g1 results/g1_potential_full.json \
                  --g4 results/g4_full.json --out figs
Any argument may be omitted; only the figures whose inputs are present are drawn.
"""
from __future__ import annotations

import argparse
import json
import os

try:
    import matplotlib
    matplotlib.use("Agg")               # headless (Kaggle / servers)
    import matplotlib.pyplot as plt
except ImportError:                     # pragma: no cover
    raise SystemExit(
        "viz.py needs matplotlib. Install it with:  pip install matplotlib\n"
        "(or: pip install -e .[viz])")


def _load(path):
    with open(path) as f:
        return json.load(f)


def _ae_psnr(payload):
    return payload.get("ae_recon_psnr", payload.get("ae_psnr"))


# ---------------------------------------------------------------------------
# 1. G1 frontier — flat in eta = the negative
# ---------------------------------------------------------------------------


def plot_g1(path, out):
    p = _load(path)
    frontier = p["frontier"]            # {embed_name: [ {eta,psnr,mmd,div,...}, ]}
    embeds = list(frontier.keys())
    # x axis: categorical eta positions (etas include 0 -> no log axis)
    etas = [r["eta"] for r in frontier[embeds[0]]]
    xs = list(range(len(etas)))
    xlabels = [f"{e:g}" for e in etas]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [("div", "Var_z  (within-condition diversity)", True),
              ("psnr", "PSNR (dB)", False),
              ("mmd", "MMD (marginal realism)", True)]
    for ax, (key, title, logy) in zip(axes, panels):
        for name in embeds:
            ys = [r[key] for r in frontier[name]]
            ax.plot(xs, ys, marker="o", label=name)
        ax.set_xticks(xs); ax.set_xticklabels(xlabels)
        ax.set_xlabel(r"$\eta$ (cost-mixing knob)")
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].legend(title="c_y embedding", fontsize=8)
    ae = _ae_psnr(p)
    fig.suptitle(
        f"G1 (aggregate extended-cost OT) — flat in η = NEGATIVE"
        + (f"  |  frozen AE {ae:.1f} dB" if ae else ""), fontweight="bold")
    fig.tight_layout()
    dst = os.path.join(out, "g1_frontier.png")
    fig.savefig(dst, dpi=140); plt.close(fig)
    print(f"saved {dst}")


# ---------------------------------------------------------------------------
# 2. G4 frontier — monotone in lambda = the positive
# ---------------------------------------------------------------------------


def plot_g4(path, out):
    p = _load(path)
    rows = p["frontier"]                # [ {lam, psnr_sample, mmd, div, ...}, ]
    lams = [r["lam"] for r in rows]

    def col(key):
        return [r[key] for r in rows]

    def std(key):
        return [r.get(key, 0.0) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [("div", "div_std", "Var_z  (diversity ↓)", True),
              ("psnr_sample", "psnr_sample_std", "PSNR sample (dB, ↑)", False),
              ("mmd", "mmd_std", "MMD (marginal realism ↑ = worse)", False)]
    for ax, (key, skey, title, logy) in zip(axes, panels):
        ax.errorbar(lams, col(key), yerr=std(skey), marker="o", capsize=3)
        ax.set_xlabel(r"$\lambda$  (0 = perception, 1 = MMSE)")
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    # conditional-MMD overlay if present (the fibre-level metric G1 lacked)
    if "cond_mmd" in rows[0]:
        ax2 = axes[2].twinx()
        ax2.plot(lams, col("cond_mmd"), color="tab:red", marker="s",
                 linestyle="--", label="conditional MMD")
        ax2.set_ylabel("conditional MMD", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.legend(fontsize=8, loc="upper left")
    ae = _ae_psnr(p)
    fig.suptitle(
        "G4 (conditional rectified flow) — monotone in λ = POSITIVE"
        + (f"  |  frozen AE {ae:.1f} dB" if ae else ""), fontweight="bold")
    fig.tight_layout()
    dst = os.path.join(out, "g4_frontier.png")
    fig.savefig(dst, dpi=140); plt.close(fig)
    print(f"saved {dst}")


# ---------------------------------------------------------------------------
# 3. D-P plane — the one-figure contrast
# ---------------------------------------------------------------------------


def plot_dp_plane(g1_path, g4_path, out, g1_embed=None):
    fig, ax = plt.subplots(figsize=(6.5, 5))

    if g1_path and os.path.exists(g1_path):
        p = _load(g1_path)
        frontier = p["frontier"]
        name = g1_embed or ("pool" if "pool" in frontier else list(frontier)[0])
        rows = frontier[name]
        xs = [r["mmd"] for r in rows]; ys = [r["psnr"] for r in rows]
        ax.plot(xs, ys, marker="o", color="tab:gray",
                label=f"G1 η-sweep ({name}) — a cluster, no frontier")
        for r in rows:
            ax.annotate(f"η={r['eta']:g}", (r["mmd"], r["psnr"]),
                        fontsize=7, color="tab:gray",
                        xytext=(3, 3), textcoords="offset points")

    if g4_path and os.path.exists(g4_path):
        p = _load(g4_path)
        rows = p["frontier"]
        xs = [r["mmd"] for r in rows]; ys = [r["psnr_sample"] for r in rows]
        ax.plot(xs, ys, marker="s", color="tab:blue",
                label="G4 λ-sweep — the D-P frontier")
        for r in rows:
            ax.annotate(f"λ={r['lam']:g}", (r["mmd"], r["psnr_sample"]),
                        fontsize=7, color="tab:blue",
                        xytext=(3, -8), textcoords="offset points")

    ax.set_xlabel("MMD  (marginal realism → worse)")
    ax.set_ylabel("PSNR (dB)  (fidelity → better)")
    ax.set_title("Distortion–Perception plane: aggregate vs conditional OT",
                 fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    dst = os.path.join(out, "dp_plane.png")
    fig.savefig(dst, dpi=140); plt.close(fig)
    print(f"saved {dst}")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1", type=str, default=None,
                    help="g1_pilot result JSON (e.g. results/g1_potential_full.json)")
    ap.add_argument("--g4", type=str, default=None,
                    help="g4_wflow result JSON (e.g. results/g4_full.json)")
    ap.add_argument("--g1_embed", type=str, default=None,
                    help="which G1 embedding to plot on the D-P plane (default: pool)")
    ap.add_argument("--out", type=str, default="figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.g1 and os.path.exists(args.g1):
        plot_g1(args.g1, args.out)
    elif args.g1:
        print(f"[skip] G1 file not found: {args.g1}")
    if args.g4 and os.path.exists(args.g4):
        plot_g4(args.g4, args.out)
    elif args.g4:
        print(f"[skip] G4 file not found: {args.g4}")
    if args.g1 or args.g4:
        plot_dp_plane(args.g1, args.g4, args.out, g1_embed=args.g1_embed)


if __name__ == "__main__":
    main()
