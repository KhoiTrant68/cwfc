#!/usr/bin/env python
"""
Real-scale D-P evaluation -- scores a trained real/flow.py checkpoint on a
full Kodak/CLIC test folder using the *same* metrics module
(neuralcompression.metrics + torchmetrics) that real/baselines/run_msillm.py
uses for MS-ILLM, so the two are on identical footing rather than computed
under different recipes (Kodak's 24 images are too few for a vanilla FID;
`update_patch_fid` -- the same patch-FID trick MS-ILLM's own eval script
uses -- sidesteps that).

UNetVelocity is fully convolutional, so no tiling is needed at eval time:
Kodak's 768x512 (and most CLIC images) are already divisible by 16, matching
the codec's downsample factor, so the network trained on `--patch` crops
just runs directly on the full test image.

Usage (after training with real/flow.py, saving the model's state_dict):
    python real/eval.py --test_root data/kodak --checkpoint ckpt.pt \
        --arch cheng2020-anchor --quality 1 --lams 0 0.25 0.5 0.75 1 \
        --out results/eval_real_kodak.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from real.baselines.run_msillm import _ensure_neuralcompression_on_path, rescale_image
from real.codec import CompressAICodec
from real.data import load_images
from real.flow import UNetVelocity, sample_flow


@torch.no_grad()
def eval_lambda(net, codec, test_root, lam, n_draws, n_steps, device):
    _ensure_neuralcompression_on_path()
    from neuralcompression.metrics import (
        MultiscaleStructuralSimilarity,
        calc_psnr,
        update_patch_fid,
    )
    from torchmetrics.image import (
        FrechetInceptionDistance,
        LearnedPerceptualImagePatchSimilarity,
    )

    msssim_metric = MultiscaleStructuralSimilarity(data_range=255.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    fid_metric = FrechetInceptionDistance().to(device)

    bpp_vals, psnr_sample_vals, psnr_mmse_vals, div_vals = [], [], [], []
    for path, image in load_images(test_root, device=device):
        y_hat = codec.condition(image)
        bpp_vals.append(codec.real_bpp(image))

        draws = torch.stack(
            [sample_flow(net, y_hat, lam, n_steps) for _ in range(n_draws)]
        )
        sample, mmse = draws[0], draws.mean(dim=0)
        div_vals.append(float(draws.flatten(2).var(dim=0).mean()))

        mse_s = float((sample - image).pow(2).mean().clamp(min=1e-12))
        mse_m = float((mmse - image).pow(2).mean().clamp(min=1e-12))
        psnr_sample_vals.append(-10 * torch.log10(torch.tensor(mse_s)).item())
        psnr_mmse_vals.append(-10 * torch.log10(torch.tensor(mse_m)).item())

        orig, pred = rescale_image(image), rescale_image(sample)
        msssim_metric(pred, orig)
        lpips_metric(sample, image)
        update_patch_fid(image, sample, fid_metric)

    return {
        "lam": lam,
        "bpp": sum(bpp_vals) / len(bpp_vals),
        "psnr_sample": sum(psnr_sample_vals) / len(psnr_sample_vals),
        "psnr_mmse": sum(psnr_mmse_vals) / len(psnr_mmse_vals),
        "div": sum(div_vals) / len(div_vals),
        "msssim": float(msssim_metric.compute()),
        "lpips": float(lpips_metric.compute()),
        "fid": float(fid_metric.compute()),
        "n_images": len(bpp_vals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--test_root", required=True, help="Kodak dir or CLIC valid/test dir"
    )
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="state_dict saved from real/flow.py training",
    )
    ap.add_argument("--arch", default="cheng2020-anchor")
    ap.add_argument("--quality", type=int, default=1)
    ap.add_argument(
        "--zc",
        type=int,
        default=None,
        help="y_hat channels; auto-detected from --arch/--quality if omitted",
    )
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument(
        "--lams", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    ap.add_argument("--n_draws", type=int, default=8)
    ap.add_argument("--n_steps", type=int, default=20)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default="results/eval_real.json")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    codec = CompressAICodec(arch=args.arch, quality=args.quality, device=device)
    zc = args.zc
    if zc is None:
        with torch.no_grad():
            zc = codec.condition(torch.rand(1, 3, 256, 256, device=device)).shape[1]
    net = UNetVelocity(zc=zc, ch=args.ch).to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()

    rows = []
    print(
        f"{'lam':>6} {'bpp':>8} {'PSNR_samp':>10} {'PSNR_mmse':>10} "
        f"{'MS-SSIM':>9} {'LPIPS':>8} {'FID':>8} {'Div':>10}"
    )
    for lam in args.lams:
        row = eval_lambda(
            net, codec, args.test_root, lam, args.n_draws, args.n_steps, device
        )
        print(
            f"{lam:>6.2f} {row['bpp']:>8.4f} {row['psnr_sample']:>10.2f} "
            f"{row['psnr_mmse']:>10.2f} {row['msssim']:>9.4f} {row['lpips']:>8.4f} "
            f"{row['fid']:>8.2f} {row['div']:>10.4e}"
        )
        rows.append(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "model": "cwfc_g4_real",
                "test_root": args.test_root,
                "checkpoint": args.checkpoint,
                "frontier": rows,
            },
            f,
            indent=2,
        )
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
