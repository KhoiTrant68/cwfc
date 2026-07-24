#!/usr/bin/env python
"""
Real-image data pipeline for the large-scale G4 track -- replaces
g1_pilot.SynthImages / the CIFAR-100 pickle loader.

Two things live here:
  * `load_images`       -- full-resolution loader for eval folders (Kodak,
                            CLIC valid/test). No cropping; used by
                            real/baselines/run_msillm.py and real/eval.py so
                            both the baseline and CWFC are scored on exactly
                            the same images.
  * `PatchFolderDataset` -- random 256x256 (configurable) crop sampler over a
                            large train folder (CLIC train), exposing the
                            same `.sample(n) -> (n,3,H,H)` interface
                            g1_pilot/g4_wflow already assume, so it drops
                            into `build_pool`-style code unchanged. Loads
                            images lazily per crop (a CLIC train split is a
                            few hundred MB of full-res photos -- too much to
                            hold fully decoded in RAM at once).

Run this file directly for the M2 sanity check the plan calls for: pull a
few anchor patches, find their kNN neighbours in a frozen codec's condition
embedding, and save a grid -- confirming the conditional fibre looks sane
*before* spending GPU-hours training the flow on it. Real photos are not
CIFAR: a pooled-embedding neighbour on a natural image can land on
unrelated content in a way it never did on tiny synthetic/CIFAR patches, so
this check is not optional.
"""

from __future__ import annotations

import glob
import os
import random

import torch
from PIL import Image
from torchvision.transforms import ToTensor


def list_images(root, max_images=None):
    paths = sorted(
        p
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
        for p in glob.glob(os.path.join(root, "**", ext), recursive=True)
    )
    if not paths:
        raise FileNotFoundError(f"No .png/.jpg images found under {root}")
    return paths[:max_images] if max_images else paths


def load_images(root, device="cpu"):
    """Yield (path, tensor) for every image under root, full resolution,
    no cropping/resizing -- for eval, not training."""
    totensor = ToTensor()
    for p in list_images(root):
        img = Image.open(p).convert("RGB")
        yield p, totensor(img).unsqueeze(0).to(device)


class PatchFolderDataset:
    """Random `patch`x`patch` crops from a folder of (larger) real photos.

    `.sample(n)` matches g1_pilot's dataset interface (SynthImages/_Wrap),
    so it's a drop-in replacement wherever those were passed as `data`.
    `source_id(n)` additionally returns which source image each of the last
    `n` sampled crops came from -- needed for a same-image-restricted kNN
    fallback in real/flow.py if the cross-image pooled-embedding kNN turns
    out to pick unrelated neighbours (see the sanity check below).
    """

    def __init__(self, root, patch=256, device="cpu", seed=0, max_images=None):
        self.paths = list_images(root, max_images=max_images)
        self.patch = patch
        self.device = device
        self.totensor = ToTensor()
        self._rng = random.Random(seed)
        self._last_source_ids = None

    def _random_crop(self, img: Image.Image):
        w, h = img.size
        p = self.patch
        if w < p or h < p:
            # upscale small images just enough to admit one crop
            scale = p / min(w, h)
            img = img.resize((max(p, int(w * scale)), max(p, int(h * scale))))
            w, h = img.size
        x0 = self._rng.randint(0, w - p)
        y0 = self._rng.randint(0, h - p)
        return img.crop((x0, y0, x0 + p, y0 + p))

    def sample(self, n):
        crops, source_ids = [], []
        for _ in range(n):
            idx = self._rng.randrange(len(self.paths))
            img = Image.open(self.paths[idx]).convert("RGB")
            crops.append(self.totensor(self._random_crop(img)))
            source_ids.append(idx)
        self._last_source_ids = torch.tensor(source_ids, device=self.device)
        return torch.stack(crops).to(self.device)

    def sample_grouped(self, n, crops_per_image):
        """Like `.sample(n)` but drawn as groups of `crops_per_image` crops
        from the same source image, so every crop has genuine same-image
        neighbours in the pool.

        `.sample(n)` picks a source image independently at random for each
        of the n crops; when the number of distinct source images is
        comparable to or larger than n (true for both CLIC train's ~1.6k
        images and ImageNet-style folders), same-image collisions across
        independent draws are Poisson-rare (P(none) ~= exp(-n/n_images)),
        so most anchors end up with *no* same-image candidate at all. That
        makes `real/flow.py`'s `--same_image_only` kNN mask degenerate (an
        all-inf row, so `topk` returns arbitrary unrelated indices) for a
        large fraction of anchors -- silently reproducing the exact
        unrelated-neighbour problem `--same_image_only` exists to fix. This
        grouped sampler is the fallback: only load `n // crops_per_image`
        images, but draw enough independent crops from each that every
        anchor's own image is guaranteed to contribute `crops_per_image - 1`
        other pool members.
        """
        n_images = max(1, n // crops_per_image)
        crops, source_ids = [], []
        for _ in range(n_images):
            idx = self._rng.randrange(len(self.paths))
            img = Image.open(self.paths[idx]).convert("RGB")
            for _ in range(crops_per_image):
                crops.append(self.totensor(self._random_crop(img)))
                source_ids.append(idx)
        while len(crops) < n:  # top up if crops_per_image doesn't divide n
            idx = self._rng.randrange(len(self.paths))
            img = Image.open(self.paths[idx]).convert("RGB")
            crops.append(self.totensor(self._random_crop(img)))
            source_ids.append(idx)
        self._last_source_ids = torch.tensor(source_ids, device=self.device)
        return torch.stack(crops).to(self.device)

    def last_source_ids(self):
        """Source-image index for each crop in the most recent `.sample()`
        call -- same length/order as that call's returned batch."""
        if self._last_source_ids is None:
            raise RuntimeError("call .sample(n) before .last_source_ids()")
        return self._last_source_ids


@torch.no_grad()
def sanity_check_knn(
    image_root,
    codec,
    embed,
    patch=256,
    n_anchor=6,
    knn=5,
    npool=200,
    out_path="figs/real_knn_sanity.png",
    device="cpu",
):
    """The M2 verification step: sample a pool of real patches, embed their
    codec condition ŷ, and for a handful of anchors show their kNN real
    neighbours -- so a human can eyeball whether "nearest in code-embedding
    space" still means "plausible alternative residual for this patch" once
    images are real photos instead of CIFAR/synthetic textures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = PatchFolderDataset(image_root, patch=patch, device=device)
    X = data.sample(npool)
    Y = codec.condition(X)
    E = embed(Y)

    anchors = torch.randperm(npool)[:n_anchor]
    fig, axes = plt.subplots(
        n_anchor, knn + 1, figsize=(1.6 * (knn + 1), 1.6 * n_anchor)
    )
    for r, a in enumerate(anchors):
        d = torch.cdist(E[a : a + 1], E).squeeze(0)
        d[a] = float("inf")
        nn_idx = d.topk(knn, largest=False).indices
        row_imgs = [X[a]] + [X[j] for j in nn_idx]
        for c, im in enumerate(row_imgs):
            ax = axes[r, c]
            ax.imshow(im.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title("anchor" if c == 0 else f"nn{c}", fontsize=8)
    fig.suptitle(
        "Anchor patch (col 0) vs its kNN real neighbours in the "
        "codec condition embedding -- do these look like plausible "
        "alternative residuals, or unrelated content?",
        fontsize=9,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path} -- inspect before training the flow on this pool")


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--arch", default="cheng2020-anchor")
    ap.add_argument("--quality", type=int, default=1)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--npool", type=int, default=200)
    ap.add_argument("--out", default="figs/real_knn_sanity.png")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import g1_pilot as g1
    from real.codec import CompressAICodec

    codec = CompressAICodec(arch=args.arch, quality=args.quality, device=args.device)
    x0 = ToTensor()(Image.open(list_images(args.image_root)[0]).convert("RGB"))
    x0 = x0[:, : args.patch, : args.patch].unsqueeze(0).to(args.device)
    y_shape = tuple(codec.condition(x0).shape[1:])
    embed = g1.make_embed("pool", y_shape, device=args.device)

    sanity_check_knn(
        args.image_root,
        codec,
        embed,
        patch=args.patch,
        knn=args.knn,
        npool=args.npool,
        out_path=args.out,
        device=args.device,
    )
