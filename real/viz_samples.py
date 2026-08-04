#!/usr/bin/env python
"""
Visual sample grid for a trained real/flow.py checkpoint -- shows the
original, the codec's own decode(y_hat) (the lam=1 floor by construction
under --target residual, see real/flow.py's module docstring), and flow
samples across a handful of lambda values, with multiple independent draws
at lambda<1 so posterior diversity can be eyeballed directly instead of
read off aggregate metrics (PSNR/LPIPS/Div) only. Complements
real/viz_real.py (which plots the bpp/PSNR/LPIPS curves, not images).

Usage:
    python real/viz_samples.py --test_root data/kodak \
        --checkpoint ckpts/g4_real_seed0.pt --target residual \
        --lams 0 0.5 1 --n_draws 2 --n_images 4 \
        --out figs/samples_residual.png
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image
from torchvision.transforms import ToTensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("real/viz_samples.py needs matplotlib: pip install matplotlib")

from real.codec import CompressAICodec
from real.data import list_images
from real.flow import UNetVelocity, sample_flow, sample_flow_residual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--test_root", required=True,
        help="Kodak/CLIC folder to draw sample images from -- prefer a "
        "held-out folder (e.g. Kodak) over the training folder so the grid "
        "reflects generalization, not memorization",
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target", choices=["knn", "residual"], default="residual")
    ap.add_argument("--arch", default="cheng2020-anchor")
    ap.add_argument("--quality", type=int, default=1)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument(
        "--n_draws", type=int, default=2,
        help="independent draws shown for each lam<1 (diversity check); "
        "lam=1 always shows just 1 draw since its target is deterministic",
    )
    ap.add_argument("--n_images", type=int, default=4)
    ap.add_argument("--n_steps", type=int, default=20)
    ap.add_argument(
        "--crop", type=int, default=None,
        help="center-crop to this size before sampling (faster grid); "
        "default runs the full image (fine for unet, no tiling needed)",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="figs/samples.png")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    codec = CompressAICodec(arch=args.arch, quality=args.quality, device=device)

    with torch.no_grad():
        zc = codec.condition(torch.rand(1, 3, 256, 256, device=device)).shape[1]
    net = UNetVelocity(zc=zc, ch=args.ch).to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()

    paths = list_images(args.test_root)[: args.n_images]
    totensor = ToTensor()

    # column plan: original, decode(y_hat), then n_draws columns per lam
    # (1 column if lam>=1.0, since its target is deterministic -- draws
    # would just show sampling/Euler-integration noise, not real diversity)
    titles = ["original", "decode(y_hat)"]
    col_specs = [(None, None), (None, None)]  # placeholders for the two above
    for lam in args.lams:
        ndraw = 1 if lam >= 1.0 else args.n_draws
        for d in range(ndraw):
            titles.append(f"λ={lam:g}" + (f" #{d + 1}" if ndraw > 1 else ""))
            col_specs.append((lam, d))

    n_rows, n_cols = len(paths), len(col_specs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.8 * n_cols, 1.8 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for r, p in enumerate(paths):
        img = totensor(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        if args.crop:
            c = args.crop
            _, _, H, W = img.shape
            y0, x0 = max(0, (H - c) // 2), max(0, (W - c) // 2)
            img = img[:, :, y0 : y0 + c, x0 : x0 + c]
        # codec needs H,W divisible by its 16x downsample
        _, _, H, W = img.shape
        img = img[:, :, : (H // 16) * 16, : (W // 16) * 16]

        with torch.no_grad():
            y_hat = codec.condition(img)
            x_dec = codec.decode(y_hat)

        row_imgs = [img[0], x_dec[0]]
        for lam, d in col_specs[2:]:
            torch.manual_seed(1000 * r + 17 * d + int(lam * 1000))
            with torch.no_grad():
                if args.target == "residual":
                    out = sample_flow_residual(net, codec, y_hat, lam, args.n_steps)
                else:
                    out = sample_flow(net, y_hat, lam, args.n_steps)
            row_imgs.append(out[0])

        for c_i, im in enumerate(row_imgs):
            ax = axes[r, c_i]
            ax.imshow(im.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c_i], fontsize=8)

    fig.suptitle(os.path.basename(args.checkpoint), fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
