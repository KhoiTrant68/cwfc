#!/usr/bin/env python
"""
De-risk for "OT Decoding of the Distortion-Perception Frontier".

KILL CRITERION
--------------
At a FIXED rate, between an MMSE-type endpoint x*  (low distortion, flat) and a
perception-type endpoint x0 (realistic, higher distortion), does a CURVED
traversal that follows the data manifold trace a better PSNR-vs-LPIPS curve than
the straight pixel-space line between the same endpoints?

  * If pixel-linear (A) is already as good as the curved paths (B, C) under
    LPIPS  -> the optimal D-P traversal is ~linear -> NO room for an OT/flow
    method -> the idea collapses to (b) -> KILL.
  * If a curved path (B latent-linear, or C manifold-projected) clearly
    DOMINATES the straight line on the PSNR-LPIPS plane -> the optimal traversal
    is non-linear -> there IS room for an OT/flow method to exploit -> PROCEED
    to build the conditional Wasserstein-flow traversal.

This tests the NECESSARY condition (non-linearity of the optimal traversal)
cheaply and training-free. It does not by itself prove OT specifically wins
(that needs the full conditional flow), but if even this necessary condition
fails, the idea is dead and you save weeks.

Three traversals between x* (t=0) and x0 (t=1):
  A  pixel-linear      : (1-t)*x* + t*x0                  (straight in pixels)
  B  latent-linear     : decode(z_q + t*dither)           (straight in VAE latent
                                                           -> curved in pixels,
                                                           follows the manifold)
  C  manifold-projected: decode(encode((1-t)*x* + t*x0))  (project the straight
                                                           pixel path back onto
                                                           the manifold; cheap
                                                           proxy for an OT path)

Backbones:
  --backbone vae   real run: Stable-Diffusion VAE (stabilityai/sd-vae-ft-mse),
                   the same VAE W-Flow uses. Needs diffusers + HF download.
  --backbone toy   sandbox/quick run: a tiny conv AE trained in-script. Verifies
                   the pipeline logic. NOTE: a near-lossless toy AE has almost no
                   manifold prior, so C~A and B is the only curved signal there;
                   the real manifold effect needs --backbone vae.

Perceptual metric:
  --percep lpips   real run: LPIPS (needs the `lpips` package + download).
  --percep feat    no-download proxy: L2 in the backbone's own latent/feature
                   space (used for the toy sandbox test).

Run (real):
  python derisk_dp.py --data /path/to/images --backbone vae --percep lpips \
                      --delta 0.7 --n_t 9 --max_images 200
"""

from __future__ import annotations

import argparse
import glob
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Backbones: must expose encode(x)->z and decode(z)->x in [0,1] pixel space
# ---------------------------------------------------------------------------


class ToyAE(nn.Module):
    """Tiny conv autoencoder, trained briefly in-script. For pipeline testing."""

    def __init__(self, ch=64, zc=8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(ch, zc, 3, 1, 1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(zc, ch, 3, 1, 1),
            nn.GELU(),
            nn.ConvTranspose2d(ch, ch, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(ch, ch, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(ch, 3, 4, 2, 1),
        )

    def encode(self, x):
        return self.enc(x * 2 - 1)

    def decode(self, z):
        return (self.dec(z).tanh() + 1) / 2


class VAEBackbone(nn.Module):
    """Stable-Diffusion VAE wrapper (the VAE W-Flow uses)."""

    def __init__(self, name="stabilityai/sd-vae-ft-mse", device="cuda"):
        super().__init__()
        from diffusers import AutoencoderKL

        self.vae = AutoencoderKL.from_pretrained(name).to(device).eval()
        self.scale = 0.18215
        for p in self.vae.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, x):
        return self.vae.encode(x * 2 - 1).latent_dist.mean * self.scale

    @torch.no_grad()
    def decode(self, z):
        x = self.vae.decode(z / self.scale).sample
        return ((x + 1) / 2).clamp(0, 1)


# ---------------------------------------------------------------------------
# Perceptual metric
# ---------------------------------------------------------------------------


class FeatPercep:
    """No-download proxy: L2 distance in the backbone's latent space."""

    def __init__(self, backbone):
        self.b = backbone

    def __call__(self, a, b):
        za, zb = self.b.encode(a), self.b.encode(b)
        return (za - zb).pow(2).mean(dim=(1, 2, 3)).sqrt()


class LPIPSPercep:
    def __init__(self, device, net="alex"):
        import lpips

        self.fn = lpips.LPIPS(net=net).to(device).eval()

    @torch.no_grad()
    def __call__(self, a, b):
        return self.fn(a * 2 - 1, b * 2 - 1).flatten()


# ---------------------------------------------------------------------------
# Endpoints + traversals
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_endpoints(backbone, x, delta):
    """
    Fixed-rate endpoints from a coarse-quantized latent.
      z_q   = round(z/delta)*delta           (rate set by delta; coarser=lower rate)
      x*    = decode(z_q)                    (deterministic center -> MMSE-like)
      x0    = decode(z_q + dither)           (stochastic dequant -> perception-ish)
    NOTE: the dither endpoint is a training-free PROXY. For the decisive run,
    replace x0 with a real perceptual-codec / generative reconstruction at the
    same rate (see README).
    """
    z = backbone.encode(x)
    z_q = torch.round(z / delta) * delta
    dither = (torch.rand_like(z_q) - 0.5) * delta
    x_star = backbone.decode(z_q).clamp(0, 1)
    x_zero = backbone.decode(z_q + dither).clamp(0, 1)
    return x_star, x_zero, z_q, dither


@torch.no_grad()
def traverse(backbone, x_star, x_zero, z_q, dither, t, kind):
    if kind == "A_pixel_linear":
        return ((1 - t) * x_star + t * x_zero).clamp(0, 1)
    if kind == "B_latent_linear":
        return backbone.decode(z_q + t * dither).clamp(0, 1)
    if kind == "C_manifold_proj":
        mix = (1 - t) * x_star + t * x_zero
        return backbone.decode(backbone.encode(mix)).clamp(0, 1)
    raise ValueError(kind)


def psnr(a, b):
    mse = (a - b).pow(2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
    return -10 * torch.log10(mse)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class Imgs(torch.utils.data.Dataset):
    def __init__(
        self, root, size=256, n=None, exts=(".png", ".jpg", ".jpeg", ".bmp", ".webp")
    ):
        self.paths = sorted(
            p
            for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
            if os.path.splitext(p)[1].lower() in exts
        )
        if n:
            self.paths = self.paths[:n]
        if not self.paths:
            raise RuntimeError(f"No images under {root}")
        self.t = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.t(Image.open(self.paths[i]).convert("RGB"))


# ---------------------------------------------------------------------------
# Toy AE quick-train (so --backbone toy actually reconstructs)
# ---------------------------------------------------------------------------


def quick_train_toy(ae, loader, device, steps=300):
    opt = torch.optim.Adam(ae.parameters(), 1e-3)
    ae.train()
    it = iter(loader)
    for s in range(steps):
        try:
            x = next(it)
        except StopIteration:
            it = iter(loader)
            x = next(it)
        x = x.to(device)
        opt.zero_grad()
        loss = F.mse_loss(ae.decode(ae.encode(x)), x)
        loss.backward()
        opt.step()
    ae.eval()
    return float(loss.detach())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--backbone", choices=["vae", "toy"], default="toy")
    ap.add_argument("--percep", choices=["lpips", "feat"], default="feat")
    ap.add_argument(
        "--delta", type=float, default=0.7, help="latent quant step (rate knob)"
    )
    ap.add_argument("--n_t", type=int, default=9, help="traversal points in [0,1]")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_images", type=int, default=200)
    ap.add_argument("--toy_steps", type=int, default=300)
    ap.add_argument("--out", type=str, default="derisk_dp_result.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Imgs(args.data, size=args.size, n=args.max_images)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.bs, shuffle=False, num_workers=2
    )
    print(
        f"device={device} backbone={args.backbone} percep={args.percep} "
        f"images={len(ds)} delta={args.delta}"
    )

    if args.backbone == "vae":
        backbone = VAEBackbone(device=device)
    else:
        backbone = ToyAE().to(device)
        tl = torch.utils.data.DataLoader(
            ds, batch_size=args.bs, shuffle=True, num_workers=2
        )
        rec = quick_train_toy(backbone, tl, device, steps=args.toy_steps)
        print(f"toy AE trained, recon mse={rec:.5f}")

    percep = LPIPSPercep(device) if args.percep == "lpips" else FeatPercep(backbone)

    kinds = ["A_pixel_linear", "B_latent_linear", "C_manifold_proj"]
    ts = torch.linspace(0, 1, args.n_t)
    # accumulators: per kind, per t -> sum psnr, sum percep, count
    sp = {k: torch.zeros(args.n_t) for k in kinds}
    sl = {k: torch.zeros(args.n_t) for k in kinds}
    n = 0

    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            x_star, x_zero, z_q, dither = build_endpoints(backbone, x, args.delta)
            for k in kinds:
                for ti, t in enumerate(ts):
                    xt = traverse(backbone, x_star, x_zero, z_q, dither, float(t), k)
                    sp[k][ti] += psnr(xt, x).sum().cpu()
                    sl[k][ti] += percep(xt, x).sum().cpu()
            n += x.shape[0]

    P = {k: (sp[k] / n).numpy() for k in kinds}  # PSNR(t)
    L = {k: (sl[k] / n).numpy() for k in kinds}  # LPIPS(t)

    # ---- report ----
    print(
        f"\n(rate proxy: latent quant step delta={args.delta}; "
        f"averaged over {n} images)"
    )
    print("\nPSNR(t) [dB] / perceptual(t) [lower=better]  along the traversal:")
    hdr = "  t   " + "".join(f"|{k[:14]:>16}" for k in kinds)
    print(hdr)
    for ti, t in enumerate(ts):
        row = f"{float(t):.2f}  "
        for k in kinds:
            row += f"|{P[k][ti]:7.2f}/{L[k][ti]:7.4f}"
        print(row)

    # ---- dominance verdict ----
    # For each traversal, interpolate perceptual as a function of PSNR onto a
    # common PSNR grid, then compare perceptual at equal PSNR. Lower = better.
    def curve(k):
        order = np.argsort(P[k])
        return P[k][order], L[k][order]

    pa, la = curve("A_pixel_linear")
    grid = np.linspace(
        max(
            P["A_pixel_linear"].min(),
            P["B_latent_linear"].min(),
            P["C_manifold_proj"].min(),
        ),
        min(
            P["A_pixel_linear"].max(),
            P["B_latent_linear"].max(),
            P["C_manifold_proj"].max(),
        ),
        25,
    )
    la_g = np.interp(grid, pa, la)
    print(
        "\n--- Dominance on the PSNR-perceptual plane (perceptual at equal PSNR, "
        "lower=better) ---"
    )
    verdicts = {}
    for k in ["B_latent_linear", "C_manifold_proj"]:
        pk, lk = curve(k)
        lk_g = np.interp(grid, pk, lk)
        gain = la_g - lk_g  # positive => k beats A (lower perceptual at same PSNR)
        mean_gain = float(np.mean(gain))
        frac_better = float(np.mean(gain > 0))
        verdicts[k] = (mean_gain, frac_better)
        print(
            f"{k:>18}: mean perceptual gain vs A = {mean_gain:+.4f} "
            f"(>0 means curved path beats the straight line), "
            f"better at {frac_better*100:.0f}% of PSNR levels"
        )

    best_gain = max(v[0] for v in verdicts.values())
    print("\nVERDICT:")
    if best_gain > 0.02 * abs(np.mean(la_g) + 1e-9) and best_gain > 1e-4:
        print(" * A curved (manifold) traversal DOMINATES the straight line on the")
        print("   D-P plane -> the optimal traversal is non-linear -> THERE IS ROOM")
        print("   for an OT/flow method. PROCEED to the conditional Wasserstein flow.")
    else:
        print(" * Curved traversals ~= pixel-linear -> the optimal traversal is")
        print("   approximately linear under this metric -> NO room -> the idea")
        print("   collapses to (b). Reconsider before building the OT machinery.")
    print(" * NB: with --backbone toy this only checks the pipeline logic")
    print("   (a near-lossless toy AE has no manifold prior). Run --backbone vae")
    print("   --percep lpips for the decisive signal, and ideally replace the")
    print("   dither x0 with a real perceptual-codec endpoint at the same rate.")

    # save raw curves
    import csv

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["traversal", "t", "psnr", "perceptual"])
        for k in kinds:
            for ti, t in enumerate(ts):
                w.writerow([k, float(ts[ti]), P[k][ti], L[k][ti]])
    print(f"\nsaved curves -> {args.out}")


if __name__ == "__main__":
    main()
