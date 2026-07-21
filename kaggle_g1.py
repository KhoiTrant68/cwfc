#!/usr/bin/env python
"""
Kaggle notebook runner for the G1 pilot.

Two ways to use on Kaggle (GPU):

A) CLI (simplest) — add g1_pilot.py to the notebook and run cells:

    !python g1_pilot.py --dataset cifar --H 32 --mode both \
        --N 1024 --steps 4000 --ae_steps 3000 \
        --seeds 0 1 2 --etas 0 0.1 0.3 1 3 10 100 \
        --embeds raw proj8 pool --loss debiased --out g1_cifar.json

B) Programmatic (train the frozen AE ONCE, reuse across sweeps in later cells):

    from kaggle_g1 import setup, run_geom, run_frontier
    ctx = setup(dataset="cifar", H=32, N=1024, ae_steps=3000)   # cell 1 (once)
    run_geom(ctx, etas=[0,0.1,0.3,1,3,10,100])                  # cell 2 (cheap)
    res = run_frontier(ctx, etas=[0,0.1,0.3,1,3,10,100],        # cell 3 (heavy)
                       embeds=["raw","proj8","pool"],
                       seeds=[0,1,2], steps=4000, loss="debiased",
                       out="g1_cifar.json")

Everything is the same code as g1_pilot.py — this only avoids retraining the AE.
"""

from __future__ import annotations

import math

import torch

import g1_pilot as g1


def setup(
    dataset="cifar",
    H=32,
    N=1024,
    ae_steps=3000,
    seed=0,
    device=None,
    data_root="./data",
    download=False,
    ae_zc=4,
    ae_qscale=1.0,
):
    """Build data + train-and-freeze the tiny quantized AE once. Returns a ctx.

    Weaken the AE (smaller ae_zc, coarser ae_qscale ~0.5) so a real residual
    exists and a D-P frontier is measurable — an AE at ~25 dB pins the generator
    at the fidelity ceiling and flattens the frontier (see the CIFAR-100 null).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g1.set_seed(seed)
    data = g1.make_dataset(
        dataset, H, seed=seed, device=device, root=data_root, download=download
    )
    ae = g1.TinyAE(zc=ae_zc, qscale=ae_qscale).to(device)
    rec = g1.train_ae(ae, data, steps=ae_steps, N=N, lr=1e-3, device=device)
    with torch.no_grad():
        x = data.sample(N)
        y_hat = ae.condition(x)
        y_shape = tuple(y_hat.shape[1:])
        ae_psnr = -10 * math.log10(float((ae.decode(y_hat) - x).pow(2).mean()))
    print(
        f"[setup] device={device} dataset={dataset} H={H} "
        f"AE recon mse={rec:.4f} quantized-recon PSNR={ae_psnr:.2f}dB "
        f"condition y_hat shape={y_shape}"
    )
    return {
        "data": data,
        "ae": ae,
        "y_shape": y_shape,
        "device": device,
        "N": N,
        "ae_psnr": ae_psnr,
    }


def run_geom(ctx, etas, embeds=("raw", "proj8", "pool"), eps=0.05):
    embeds = [(e, e) for e in embeds]
    g1.geom_probe(ctx["ae"], ctx["data"], embeds, etas, ctx["N"], eps, ctx["device"])


def run_frontier(
    ctx,
    etas,
    embeds=("raw", "proj8", "pool"),
    seeds=(0, 1, 2),
    steps=4000,
    eps=0.05,
    lr=1e-3,
    loss="debiased",
    debias_w=0.5,
    K=1,
    out=None,
):
    embeds = [(e, e) for e in embeds]
    res = g1.frontier(
        ctx["ae"],
        ctx["data"],
        embeds,
        etas,
        ctx["y_shape"],
        steps,
        ctx["N"],
        eps,
        lr,
        ctx["device"],
        list(seeds),
        loss_kind=loss,
        debias_w=debias_w,
        K=K,
    )
    if out:
        import json

        with open(out, "w") as f:
            json.dump(
                {
                    "ae_psnr": ctx["ae_psnr"],
                    "y_shape": list(ctx["y_shape"]),
                    "loss": loss,
                    "frontier": res,
                },
                f,
                indent=2,
            )
        print(f"saved -> {out}")
    return res


if __name__ == "__main__":
    # quick self-check on synth (no download)
    ctx = setup(dataset="synth", H=16, N=256, ae_steps=500)
    run_frontier(
        ctx,
        etas=[0, 0.3, 1, 3, 30],
        embeds=["pool"],
        seeds=[0],
        steps=400,
        loss="debiased",
    )
