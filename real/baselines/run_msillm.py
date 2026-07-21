#!/usr/bin/env python
"""
MS-ILLM baseline — real bpp/PSNR/MS-SSIM/LPIPS/FID on Kodak/CLIC.

Reproduces facebookresearch/NeuralCompression's own
`projects/illm/eval_folder_example.py` protocol (real entropy-coded bpp via
`model.compress()`/`pickle_size_of`, patch-FID via `update_patch_fid`) so
these numbers sit on the exact same footing the CWFC real-scale eval in
`real/eval.py` will use — that shared metric module is what makes the two
comparable at all.

Pretrained weights load via torch.hub (no training): quality_1..6 target
0.035/0.07/0.14/0.3/0.45/0.9 bpp; vlo1/vlo2 target 0.00218/0.00438 bpp
(per the docstrings in facebookresearch/NeuralCompression's hubconf.py).

Needs (beyond this repo's core deps): torchmetrics, lpips, DISTS_pytorch,
torch-fidelity, fvcore, pillow, tqdm — see real/requirements.txt.

Usage:
    python real/baselines/run_msillm.py --data_root /path/to/kodak \
        --qualities 1 2 3 4 5 6 --out results/baseline_msillm_kodak.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

TARGET_BPP = {
    "vlo1": 0.00218,
    "vlo2": 0.00438,
    1: 0.035,
    2: 0.07,
    3: 0.14,
    4: 0.3,
    5: 0.45,
    6: 0.9,
}


def _ensure_neuralcompression_on_path():
    """
    torch.hub.load clones facebookresearch/NeuralCompression into the torch
    hub cache but does not leave `neuralcompression` importable afterwards —
    a plain `pip install` of the repo fails in many environments (it pulls an
    old pinned scipy that needs a meson/cython toolchain to build). Point
    sys.path at the already-cloned cache dir instead; torch.hub.load must run
    at least once first so the clone exists.
    """
    hub_dir = torch.hub.get_dir()
    matches = glob.glob(os.path.join(hub_dir, "facebookresearch_NeuralCompression_*"))
    if not matches:
        raise RuntimeError(
            "NeuralCompression not found in torch hub cache — call "
            "torch.hub.load('facebookresearch/NeuralCompression', ...) first."
        )
    path = matches[0]
    if path not in sys.path:
        sys.path.insert(0, path)


def rescale_image(image, back_to_float=True):
    dtype = image.dtype
    image = (image * 255 + 0.5).to(torch.uint8)
    if back_to_float:
        image = image.to(dtype)
    return image


def load_images(data_root, device):
    paths = sorted(
        p
        for ext in ("*.png", "*.jpg", "*.jpeg")
        for p in glob.glob(os.path.join(data_root, ext))
    )
    if not paths:
        raise FileNotFoundError(f"No .png/.jpg images found under {data_root}")
    totensor = ToTensor()
    for p in paths:
        img = Image.open(p).convert("RGB")
        yield p, totensor(img).unsqueeze(0).to(device)


@torch.no_grad()
def eval_quality(quality, data_root, device):
    from neuralcompression.metrics import (
        MultiscaleStructuralSimilarity,
        calc_psnr,
        pickle_size_of,
        update_patch_fid,
    )
    from torchmetrics.image import (
        FrechetInceptionDistance,
        LearnedPerceptualImagePatchSimilarity,
    )

    name = (
        f"msillm_quality_{quality}"
        if quality not in ("vlo1", "vlo2")
        else f"msillm_quality_{quality}"
    )
    model = torch.hub.load("facebookresearch/NeuralCompression", name)
    model = model.to(device).eval()
    model.update()
    model.update_tensor_devices("compress")

    msssim_metric = MultiscaleStructuralSimilarity(data_range=255.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    fid_metric = FrechetInceptionDistance().to(device)

    bpp_vals, psnr_vals = [], []
    for path, image in tqdm(
        list(load_images(data_root, device)), desc=f"quality={quality}"
    ):
        compressed = model.compress(image, force_cpu=(device == "cpu"))
        decompressed = model.decompress(compressed, force_cpu=(device == "cpu")).clamp(
            0.0, 1.0
        )

        num_bytes = pickle_size_of(compressed)
        bpp = num_bytes * 8 / (image.shape[0] * image.shape[-2] * image.shape[-1])
        bpp_vals.append(float(bpp))

        orig = rescale_image(image)
        pred = rescale_image(decompressed)
        psnr_vals.append(float(calc_psnr(pred, orig)))
        msssim_metric(pred, orig)
        lpips_metric(decompressed, image)
        update_patch_fid(image, decompressed, fid_metric)

    return {
        "quality": quality,
        "target_bpp": TARGET_BPP.get(quality),
        "bpp": sum(bpp_vals) / len(bpp_vals),
        "psnr": sum(psnr_vals) / len(psnr_vals),
        "msssim": float(msssim_metric.compute()),
        "lpips": float(lpips_metric.compute()),
        "fid": float(fid_metric.compute()),
        "n_images": len(bpp_vals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="folder of Kodak or CLIC test PNGs/JPGs",
    )
    ap.add_argument(
        "--qualities",
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        help="msillm_quality_{q} levels to run (int, or vlo1/vlo2)",
    )
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default="results/baseline_msillm.json")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    qualities = [int(q) if q not in ("vlo1", "vlo2") else q for q in args.qualities]

    # trigger the clone once so _ensure_neuralcompression_on_path can find it
    torch.hub.load(
        "facebookresearch/NeuralCompression",
        (
            f"msillm_quality_{qualities[0]}"
            if qualities[0] not in ("vlo1", "vlo2")
            else f"msillm_quality_{qualities[0]}"
        ),
    )
    _ensure_neuralcompression_on_path()

    rows = []
    for q in qualities:
        print(f"\n=== MS-ILLM quality={q} (target {TARGET_BPP.get(q)} bpp) ===")
        row = eval_quality(q, args.data_root, device)
        print(
            f"  bpp={row['bpp']:.4f} psnr={row['psnr']:.2f} "
            f"msssim={row['msssim']:.4f} lpips={row['lpips']:.4f} fid={row['fid']:.2f}"
        )
        rows.append(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {"model": "msillm", "data_root": args.data_root, "frontier": rows},
            f,
            indent=2,
        )
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
