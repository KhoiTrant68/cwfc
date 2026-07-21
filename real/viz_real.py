#!/usr/bin/env python
"""
Real-scale D-P plane -- the analogue of viz.py's dp_plane.png, but rate (bpp)
on the x-axis instead of MMD, and with an actual published baseline (MS-ILLM)
overlaid instead of just G1 vs G4. Two panels sharing one bpp axis:

  1. bpp vs PSNR (distortion)   -- lower-left-to-upper-right tradeoff surface
  2. bpp vs LPIPS (perception)  -- lower is better; this is the axis the
                                    G1-negative/G4-positive story is about

Reads:
  --baseline results/baseline_msillm.json   (real/baselines/run_msillm.py)
  --cwfc     results/eval_real_kodak.json   (real/eval.py)

Either may be omitted; only the series whose input is present are drawn.
Pure post-processing, same spirit as viz.py -- no run is repeated here.
"""

from __future__ import annotations

import argparse
import json
import os

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("real/viz_real.py needs matplotlib: pip install matplotlib")


def _load(path):
    with open(path) as f:
        return json.load(f)


def plot_dp_real(baseline_path, cwfc_path, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    if baseline_path and os.path.exists(baseline_path):
        p = _load(baseline_path)
        rows = sorted(p["frontier"], key=lambda r: r["bpp"])
        bpp = [r["bpp"] for r in rows]
        axes[0].plot(
            bpp,
            [r["psnr"] for r in rows],
            marker="o",
            color="tab:gray",
            label="MS-ILLM (published)",
        )
        axes[1].plot(
            bpp,
            [r["lpips"] for r in rows],
            marker="o",
            color="tab:gray",
            label="MS-ILLM (published)",
        )
        for r in rows:
            for ax, key in ((axes[0], "psnr"), (axes[1], "lpips")):
                ax.annotate(
                    f"q{r['quality']}",
                    (r["bpp"], r[key]),
                    fontsize=7,
                    color="tab:gray",
                    xytext=(3, 3),
                    textcoords="offset points",
                )
    elif baseline_path:
        print(f"[skip] baseline file not found: {baseline_path}")

    if cwfc_path and os.path.exists(cwfc_path):
        p = _load(cwfc_path)
        rows = sorted(p["frontier"], key=lambda r: r["lam"])
        bpp = [r["bpp"] for r in rows]  # constant across lam (same frozen codec)
        axes[0].plot(
            bpp,
            [r["psnr_sample"] for r in rows],
            marker="s",
            color="tab:blue",
            label="CWFC G4 (this work) -- λ sweep",
        )
        axes[1].plot(
            bpp,
            [r["lpips"] for r in rows],
            marker="s",
            color="tab:blue",
            label="CWFC G4 (this work) -- λ sweep",
        )
        for r in rows:
            for ax, key in ((axes[0], "psnr_sample"), (axes[1], "lpips")):
                ax.annotate(
                    f"λ={r['lam']:g}",
                    (r["bpp"], r[key]),
                    fontsize=7,
                    color="tab:blue",
                    xytext=(3, -8),
                    textcoords="offset points",
                )
    elif cwfc_path:
        print(f"[skip] cwfc eval file not found: {cwfc_path}")

    axes[0].set_xlabel("bpp")
    axes[0].set_ylabel("PSNR (dB, ↑ better)")
    axes[0].set_title("Rate-Distortion")
    axes[1].set_xlabel("bpp")
    axes[1].set_ylabel("LPIPS (↓ better)")
    axes[1].set_title("Rate-Perception")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Real-image D-P plane: CWFC G4 vs MS-ILLM", fontweight="bold")
    fig.tight_layout()
    dst = os.path.join(out, "dp_plane_real.png")
    fig.savefig(dst, dpi=140)
    plt.close(fig)
    print(f"saved {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="results/baseline_msillm.json from real/baselines/run_msillm.py",
    )
    ap.add_argument(
        "--cwfc",
        type=str,
        default=None,
        help="results/eval_real_*.json from real/eval.py",
    )
    ap.add_argument("--out", type=str, default="figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plot_dp_real(args.baseline, args.cwfc, args.out)


if __name__ == "__main__":
    main()
