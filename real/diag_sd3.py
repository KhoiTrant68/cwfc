#!/usr/bin/env python
"""Cheap (~seconds, NO training) isolation test for real/flow_sd3.py's SD3
integration.

The full --target residual run produces ~unconditional-quality output
(PSNR_samp ~12.6dB, ~17dB BELOW the decode floor, LPIPS ~0.70) with ZERO
lambda effect even after 1500 steps -- the signature of the ControlNet
conditioning and/or the lambda pathway being silently ignored (a wrong
diffusers kwarg is the prime suspect; several call sites in flow_sd3.py are
tagged VERIFY because the installed diffusers API could not be checked
offline). This isolates that WITHOUT a training run:

  1. prints the ACTUAL forward() signatures of the installed diffusers
     transformer + controlnet, so the VERIFY-tagged kwarg names
     (block_controlnet_hidden_states / controlnet_cond) can be confirmed;
  2. feeds the untrained backbone a fixed xt and checks whether the predicted
     velocity RESPONDS to the conditioning latent -- if v(cond) == v(0) the
     ControlNet residuals are not reaching the transformer output;
  3. nudges the (zero-initialised) lambda_mlp a few steps, then checks whether
     v(lam=0) != v(lam=1) -- if not, the lambda->pooled_projections offset
     does not reach the output either.

Run (on the A40 box, same env as the training run):
    python real/diag_sd3.py --dtype bf16
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from real.flow_sd3 import SD3LatentFlow  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--hf_token", default=None)
    ap.add_argument("--hf_variant", default=None)
    ap.add_argument("--controlnet_layers", type=int, default=12)
    ap.add_argument("--patch", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    net = SD3LatentFlow(
        controlnet_layers=args.controlnet_layers, device=device, dtype=dtype,
        hf_token=args.hf_token, hf_variant=args.hf_variant,
    )

    # --- 1) the actual installed-diffusers forward signatures -----------------
    print("\n=== transformer.forward parameters ===")
    print(list(inspect.signature(net.transformer.forward).parameters))
    print("=== controlnet.forward parameters ===")
    print(list(inspect.signature(net.controlnet.forward).parameters))
    # how many residual blocks the controlnet emits vs how many the transformer
    # has -- a count mismatch is a classic silent-misalignment source.
    print(f"transformer depth (num_layers) = {net.transformer.config.num_layers}")
    print(f"controlnet num_layers          = {args.controlnet_layers}")

    h = args.patch // 8
    B = 2
    torch.manual_seed(0)
    xt = torch.randn(B, 16, h, h, device=device, dtype=dtype)
    cond = torch.randn(B, 16, h, h, device=device, dtype=dtype)
    sigma = torch.full((B,), 0.5, device=device)

    # --- 2) does the velocity respond to the conditioning latent? -------------
    with torch.no_grad():
        v_cond = net.velocity(xt, sigma, cond, 0.0).float()
        v_zero = net.velocity(xt, sigma, torch.zeros_like(cond), 0.0).float()
        v_cond2 = net.velocity(xt, sigma, cond * 3.0, 0.0).float()
    scale = v_cond.abs().mean().item()
    d_zero = (v_cond - v_zero).abs().mean().item()
    d_c2 = (v_cond - v_cond2).abs().mean().item()
    print(f"\n[COND TEST]  mean|v| = {scale:.4e}")
    print(f"  mean|v(cond) - v(0*cond)| = {d_zero:.4e}  (rel {d_zero / scale:.2%})")
    print(f"  mean|v(cond) - v(3*cond)| = {d_c2:.4e}  (rel {d_c2 / scale:.2%})")
    if d_zero / scale < 1e-3 and d_c2 / scale < 1e-3:
        print("  >> CONTROLNET IS DEAD: velocity ignores the conditioning latent.")
    else:
        print("  >> conditioning reaches the output (ControlNet wired OK).")

    # --- 3) does the velocity respond to lambda? ------------------------------
    # lambda_mlp is zero-initialised (identity prior), so v(lam=0)==v(lam=1) at
    # init BY DESIGN -- nudge the MLP a few steps so its output differs across
    # lambda, then test whether that difference propagates to the velocity.
    opt = torch.optim.SGD(net.lambda_mlp.parameters(), lr=1.0)
    for _ in range(30):
        off0 = net.lambda_mlp(torch.zeros(B, device=device))
        off1 = net.lambda_mlp(torch.ones(B, device=device))
        loss = ((off1 - off0 - 1.0) ** 2).mean()  # push the two apart
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        v_l0 = net.velocity(xt, sigma, cond, 0.0).float()
        v_l1 = net.velocity(xt, sigma, cond, 1.0).float()
    d_lam = (v_l0 - v_l1).abs().mean().item()
    print(f"\n[LAMBDA TEST] (after nudging lambda_mlp apart)")
    print(f"  mean|v(lam=0) - v(lam=1)| = {d_lam:.4e}  (rel {d_lam / scale:.2%})")
    if d_lam / scale < 1e-3:
        print("  >> LAMBDA PATH IS DEAD: the pooled-projection offset never reaches v.")
    else:
        print("  >> lambda reaches the output (pooled-projection path wired OK).")


if __name__ == "__main__":
    main()
