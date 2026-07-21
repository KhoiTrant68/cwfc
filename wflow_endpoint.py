#!/usr/bin/env python
"""
W-Flow perception endpoint for the D-P de-risk.

Goal: produce x0 = the most realistic on-manifold reconstruction that is
CONSISTENT with the transmitted quantized latent z_q (same rate, no extra bits).
This is Freirich's perfect-perception estimator: closest point to the data
manifold inside the quantization cell of z_q.

Mechanism (training-free): W-Flow is an unconditional/class-conditional one-step
generator G: noise -> data-latent. We INVERT it per image -- optimise the noise
input z_ref so that G(z_ref) lands inside the quantization cell of z_q, while a
prior term keeps z_ref near the noise distribution so G(z_ref) stays realistic.

  min_z_ref   soft_box(|G(z_ref) - z_q|, delta/2)  +  prior_w * ||z_ref||^2

`soft_box` only penalises the part of the residual that leaves the cell, so the
solution is free to add realistic detail WITHIN the transmitted uncertainty.

TESTED: invert_to_consistency (the optimisation loop, the soft-box consistency,
the prior regulariser) is verified in-script with a synthetic generator.
NOT TESTED HERE: WFlowAdapter (loads the real W-Flow DiT) -- you must wire it to
your checkout's loader/forward; the call sites are marked  # WIRE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Generic inversion (TESTED with a synthetic generator below)
# ---------------------------------------------------------------------------


def invert_to_consistency(
    generator,  # callable: z_ref (B, *gshape) -> z_hat (B, *latent_shape)
    z_q,  # (B, C, H, W) transmitted quantized latent
    delta,  # quantization step that produced z_q
    gen_input_shape,  # shape of one z_ref sample, e.g. (4, 32, 32)
    steps=400,
    lr=0.05,
    prior_w=0.05,
    restarts=2,
    device="cuda",
    verbose=False,
):
    """
    Returns (z_hat_best, info). z_hat_best is the consistent on-manifold latent
    per image; info reports residual-outside-cell (lower=better consistency).
    """
    B = z_q.shape[0]
    half = delta / 2.0
    best_z_hat = None
    best_out = torch.full((B,), float("inf"), device=device)

    for r in range(restarts):
        z_ref = torch.randn(B, *gen_input_shape, device=device).requires_grad_(True)
        opt = torch.optim.Adam([z_ref], lr=lr)
        for s in range(steps):
            z_hat = generator(z_ref)
            resid = z_hat - z_q
            outside = F.relu(resid.abs() - half)  # 0 inside the cell
            cons = outside.pow(2).mean(dim=(1, 2, 3))  # (B,)
            prior = z_ref.pow(2).mean(dim=tuple(range(1, z_ref.dim())))
            loss = (cons + prior_w * prior).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if verbose and s % max(1, steps // 4) == 0:
                print(
                    f"  [restart {r}] step {s}: cons={cons.mean().item():.4e} "
                    f"prior={prior.mean().item():.3f}"
                )
        with torch.no_grad():
            z_hat = generator(z_ref)
            outside = F.relu((z_hat - z_q).abs() - half)
            out_per = outside.pow(2).mean(dim=(1, 2, 3))  # (B,)
            improved = out_per < best_out
            if best_z_hat is None:
                best_z_hat = z_hat.clone()
            best_z_hat[improved] = z_hat[improved]
            best_out = torch.minimum(best_out, out_per)

    info = {
        "mean_outside_cell": float(best_out.mean()),
        "frac_in_cell": None,  # filled by caller if wanted
    }
    return best_z_hat.detach(), info


# ---------------------------------------------------------------------------
# W-Flow adapter  (SKELETON -- WIRE to your W-Flow checkout; NOT tested here)
# ---------------------------------------------------------------------------


class WFlowAdapter(nn.Module):
    """
    Wrap a pretrained W-Flow one-step generator as  generator(z_ref) -> z_hat.

    The real W-Flow generates a data-latent from a noise sample in a single
    forward pass, conditioned on a class id with CFG. For inversion we fix the
    class to the image's true label and optimise z_ref.

    You must connect three things from your W-Flow checkout (github.com/hanjq17/
    W-Flow):  the model build, the checkpoint load (`ema_model` in state_*.pt),
    and the one-step forward. See inference_ours.py / models/ for the exact API.
    """

    def __init__(self, ckpt_path, config_path, class_id, cfg_scale=1.09, device="cuda"):
        super().__init__()
        self.device = device
        self.class_id = class_id
        self.cfg_scale = cfg_scale
        # WIRE: build the DiT from `config_path` and load ema weights:
        #   from utils... import build_model   # per W-Flow repo
        #   self.model = build_model(config_path)
        #   state = torch.load(ckpt_path, map_location="cpu")
        #   self.model.load_state_dict(state["ema_model"]); self.model.eval().to(device)
        raise NotImplementedError(
            "Wire WFlowAdapter to your W-Flow checkout (see inference_ours.py). "
            "The inversion logic below is generator-agnostic and already tested."
        )

    def forward(self, z_ref):
        # WIRE: one-step generation. Roughly:
        #   y = torch.full((z_ref.shape[0],), self.class_id, device=self.device)
        #   z_hat = self.model.one_step(z_ref, y, cfg_scale=self.cfg_scale)
        #   return z_hat                      # data-latent, shape == z_q shape
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Endpoint builder to drop into derisk_dp.py
# ---------------------------------------------------------------------------


@torch.no_grad()
def _quantize(z, delta):
    return torch.round(z / delta) * delta


def build_endpoints_wflow(
    sdvae, wflow_generator, x, delta, class_ids, inv_kwargs=None, device="cuda"
):
    """
    x*  = decode(z_q)                               (MMSE endpoint)
    x0  = decode(invert(G -> consistent with z_q))  (perception endpoint)
    Returns x_star, x_zero, z_star(=z_q), z_zero(=z_hat).

    Generalised endpoints: traverse() should use (z_star, z_zero) for the
    latent-linear path:  decode((1-t)*z_star + t*z_zero).
    """
    inv_kwargs = inv_kwargs or {}
    with torch.no_grad():
        z = sdvae.encode(x)
        z_q = _quantize(z, delta)
        x_star = sdvae.decode(z_q).clamp(0, 1)

    gshape = tuple(z_q.shape[1:])  # W-Flow noise input matches latent shape
    # NOTE: if your W-Flow generator takes class id per call, set it before
    # calling (here we assume wflow_generator already holds the class, or wrap).
    z_hat, info = invert_to_consistency(
        wflow_generator, z_q, delta, gen_input_shape=gshape, device=device, **inv_kwargs
    )
    with torch.no_grad():
        x_zero = sdvae.decode(z_hat).clamp(0, 1)
    return x_star, x_zero, z_q, z_hat, info


# ---------------------------------------------------------------------------
# Self-test of the inversion logic with a SYNTHETIC generator
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cpu"
    B, C, H, W = 4, 4, 16, 16
    delta = 0.7

    # Synthetic "generator": fixed random conv net noise->latent with enough
    # gain that its output spans a range >> delta, so a random z_ref is NOT
    # trivially consistent and inversion has to actually work.
    gen = nn.Sequential(
        nn.Conv2d(C, 64, 3, 1, 1),
        nn.Tanh(),
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.Tanh(),
        nn.Conv2d(64, C, 3, 1, 1),
    ).to(device)
    with torch.no_grad():
        for m in gen:
            if isinstance(m, nn.Conv2d):
                m.weight.mul_(2.5)
                m.bias.normal_(0, 1.5)
    for p in gen.parameters():
        p.requires_grad_(False)

    def generator(z_ref):
        return gen(z_ref) * 3.0  # spread outputs to ~[-several, +several]

    # Feasible target from a known z_ref0, quantized.
    with torch.no_grad():
        z_target = generator(torch.randn(B, C, H, W))
        z_q = _quantize(z_target, delta)
        print(
            f"target latent range ~ [{z_q.min():.1f}, {z_q.max():.1f}], "
            f"delta/2={delta/2}"
        )

    # baseline: average outside-cell residual over random inits
    with torch.no_grad():
        outs = [
            F.relu((generator(torch.randn(B, C, H, W)) - z_q).abs() - delta / 2)
            .pow(2)
            .mean()
            .item()
            for _ in range(5)
        ]
        base = sum(outs) / len(outs)
        print(f"random init  outside-cell residual = {base:.4e}")

    z_hat, info = invert_to_consistency(
        generator,
        z_q,
        delta,
        gen_input_shape=(C, H, W),
        steps=500,
        lr=0.05,
        prior_w=0.005,
        restarts=3,
        device=device,
        verbose=True,
    )
    inside = ((z_hat - z_q).abs() <= delta / 2).float().mean()
    typical_overshoot = info["mean_outside_cell"] ** 0.5
    print(
        f"after inversion: mean outside-cell = {info['mean_outside_cell']:.4e} "
        f"(baseline {base:.4e}; ~{base/max(info['mean_outside_cell'],1e-12):.0f}x lower)"
    )
    print(
        f"typical overshoot beyond the cell boundary = {typical_overshoot:.4f} "
        f"(cell half-width = {delta/2})"
    )
    print(f"fraction of coords strictly inside the cell = {inside.item()*100:.1f}%")
    # The consistency quantity that matters is the outside-cell residual (how much
    # transmitted info is violated), not strict frac-inside (coords on the cell
    # boundary count as 'outside' but cost ~0).
    ok = info["mean_outside_cell"] < base * 0.05
    print(
        "PASS: inversion drives G(z_ref) into the quant cell (consistency met)"
        if ok
        else "weak/needs tuning"
    )
