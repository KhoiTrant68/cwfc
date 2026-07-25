#!/usr/bin/env python
"""
Real-scale conditional rectified flow -- replaces g4_wflow.CondVelocity.

g4_wflow's CondVelocity is a 4-layer *same-resolution* CNN: it upsamples the
spatial condition y_hat to full image resolution with nearest-neighbour and
concatenates it at the input. That is adequate for 32x32 CIFAR texture but
has no capacity to model photographic detail at patch=256.

UNetVelocity instead downsamples x_t through 4 stride-2 stages -- by design,
exactly matching a compressai codec's typical 16x downsample (real/codec.py),
so y_hat concatenates directly into the bottleneck at its *native* resolution
with no interpolation. Skip connections carry high-frequency detail back up.

Architecture change vs. g4_wflow.py, per the approved plan: ONE model is
trained across the whole lambda range (lambda concatenated as a broadcast
channel at the bottleneck, same trick as t) instead of one model per lambda
point. g4_wflow trains 5 separate models per lambda sweep; at 256x256 that is
5x the GPU-hours for a functionally equivalent frontier. This file trains a
single lambda-conditioned model instead.

Everything else -- the rectified-flow loss (x_t=(1-t)x0+t*x1, regress
x1-x0), the Euler sampler, the kNN-conditional-target construction, the
Var_z/PSNR/MMD/conditional-MMD metrics -- is resolution-independent and is
reused from g1_pilot / g4_wflow rather than reimplemented.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import g1_pilot as g1  # noqa: E402  (set_seed, make_embed, mmd_rff)

# ---------------------------------------------------------------------------
# U-Net velocity field v(x_t, t, y_hat, lam) -> velocity
# ---------------------------------------------------------------------------


def _groups(c, max_groups=8):
    for g in range(min(max_groups, c), 0, -1):
        if c % g == 0:
            return g
    return 1


def conv_block(cin, cout):
    return nn.Sequential(
        nn.GroupNorm(_groups(cin), cin),
        nn.SiLU(),
        nn.Conv2d(cin, cout, 3, 1, 1),
        nn.GroupNorm(_groups(cout), cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, 1, 1),
    )


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = conv_block(cin, cout)
        self.pool = nn.Conv2d(cout, cout, 4, 2, 1)  # learned downsample

    def forward(self, x):
        h = self.block(x)
        return self.pool(h), h  # downsampled, skip


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin, 4, 2, 1)
        self.block = conv_block(cin + cskip, cout)

    def forward(self, x, skip):
        x = self.up(x)
        return self.block(torch.cat([x, skip], dim=1))


class UNetVelocity(nn.Module):
    """
    4-stage down/up U-Net for `patch`x`patch` inputs. `zc` = y_hat channel
    count (from CompressAICodec.condition); y_hat's spatial size must equal
    the bottleneck size, i.e. patch / 16 (four /2 stages) -- true for any
    16x-downsampling compressai codec at patch=256, 128, 64, ...
    """

    def __init__(self, zc, ch=64, in_ch=3):
        super().__init__()
        self.t_ch, self.lam_ch = 1, 1
        self.down1 = Down(in_ch + self.t_ch, ch)
        self.down2 = Down(ch, ch * 2)
        self.down3 = Down(ch * 2, ch * 4)
        self.down4 = Down(ch * 4, ch * 4)
        self.bottleneck = conv_block(ch * 4 + zc + self.t_ch + self.lam_ch, ch * 4)
        self.up4 = Up(ch * 4, ch * 4, ch * 4)
        self.up3 = Up(ch * 4, ch * 4, ch * 2)
        self.up2 = Up(ch * 2, ch * 2, ch)
        self.up1 = Up(ch, ch, ch)
        self.out = nn.Conv2d(ch, in_ch, 3, 1, 1)

    def forward(self, x, t, y, lam):
        """x:(B,3,H,W) t:(B,1,1,1) y:(B,zc,H/16,W/16) lam:(B,1,1,1) or scalar."""
        B, _, H, W = x.shape
        tch = t.expand(B, 1, H, W)
        x1, s1 = self.down1(torch.cat([x, tch], dim=1))
        x2, s2 = self.down2(x1)
        x3, s3 = self.down3(x2)
        x4, s4 = self.down4(x3)

        hb, wb = x4.shape[-2:]
        tb = t.expand(B, 1, hb, wb)
        if not torch.is_tensor(lam):
            lam = torch.full((B, 1, 1, 1), float(lam), device=x.device)
        lb = lam.expand(B, 1, hb, wb)
        b = self.bottleneck(torch.cat([x4, y, tb, lb], dim=1))

        u4 = self.up4(b, s4)
        u3 = self.up3(u4, s3)
        u2 = self.up2(u3, s2)
        u1 = self.up1(u2, s1)
        return self.out(u1)


# ---------------------------------------------------------------------------
# Conditional pool + kNN targets (real-photo analogue of g4_wflow.build_pool)
# ---------------------------------------------------------------------------


def build_real_pool(codec, patch_data, npool, device, crops_per_image=None):
    """Sample a fixed pool of real patches + their codec condition + a pool
    embedding for kNN. Mirrors g4_wflow.build_pool; `patch_data` is a
    real/data.PatchFolderDataset (or anything with .sample(n)).

    `crops_per_image`, when set, uses `patch_data.sample_grouped` instead of
    `.sample` -- required for `--same_image_only` to have genuine same-image
    candidates rather than a mostly-empty mask (see sample_grouped's
    docstring in real/data.py)."""
    with torch.no_grad():
        X = (
            patch_data.sample_grouped(npool, crops_per_image)
            if crops_per_image
            else patch_data.sample(npool)
        )
        Y = codec.condition(X)
        source_ids = (
            patch_data.last_source_ids()
            if hasattr(patch_data, "last_source_ids")
            else None
        )
    return X, Y, source_ids


def _knn_indices(E, idx, knn, same_image_only, source_ids):
    """
    kNN in embedding space E for anchor rows idx, excluding self.
    If same_image_only, restrict the candidate set to patches sharing the
    anchor's source image (cross-image distances set to +inf) -- the
    fallback flagged in the plan: real/data.py's sanity_check_knn showed
    pooled-embedding kNN can land on tonally-similar but semantically
    unrelated photos once images are real (unlike CIFAR); restricting to
    crops of the same source photo guarantees a genuinely related fibre at
    the cost of less diverse conditional sets.
    """
    d = torch.cdist(E[idx], E)
    d.scatter_(1, idx[:, None], float("inf"))
    if same_image_only:
        assert source_ids is not None
        same = source_ids[idx][:, None] == source_ids[None, :]
        d = d.masked_fill(~same, float("inf"))
    return d.topk(knn, largest=False, dim=1).indices


def train_flow(
    net,
    X,
    Y,
    E,
    lams,
    steps,
    N,
    knn,
    lr,
    device,
    same_image_only=False,
    source_ids=None,
):
    """Single model trained across the full lambda range -- lam is sampled
    per-batch-element and fed to the net as conditioning, rather than one
    model per lambda point (see module docstring)."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    Np = X.shape[0]
    lams_t = torch.tensor(lams, device=device)
    for _ in range(steps):
        idx = torch.randint(0, Np, (N,), device=device)
        ya = Y[idx]
        with torch.no_grad():
            nn_idx = _knn_indices(E, idx, knn, same_image_only, source_ids)
            Xnb = X[nn_idx]
            pick = torch.randint(0, knn, (N,), device=device)
            x_samp = Xnb[torch.arange(N, device=device), pick]
            x_mean = Xnb.mean(dim=1)
            lam = lams_t[torch.randint(0, len(lams_t), (N,), device=device)]
            lam_b = lam.view(N, 1, 1, 1)
            x1 = (1.0 - lam_b) * x_samp + lam_b * x_mean
            x0 = torch.randn_like(x1)
            t = torch.rand(N, 1, 1, 1, device=device)
            xt = (1.0 - t) * x0 + t * x1
            target = x1 - x0
        v = net(xt, t, ya, lam_b)
        loss = F.mse_loss(v, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return float(loss.detach())


@torch.no_grad()
def sample_flow(net, y, lam, n_steps=20, x0=None):
    B, _, h, w = y.shape
    H, W = h * 16, w * 16  # 4-stage U-Net matches a 16x-downsampling codec
    if x0 is None:
        x0 = torch.randn(B, 3, H, W, device=y.device)
    x = x0
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((B, 1, 1, 1), i * dt, device=y.device)
        x = x + dt * net(x, t, y, lam)
    return x.clamp(0, 1)


@torch.no_grad()
def evaluate(
    net,
    codec,
    X,
    Y,
    E,
    N,
    lam,
    n_steps,
    n_draws,
    knn,
    seed,
    device,
    same_image_only=False,
    source_ids=None,
):
    g1.set_seed(seed + 1)
    Np = X.shape[0]
    idx = torch.randint(0, Np, (N,), device=device)
    y_hat, x_true = Y[idx], X[idx]
    draws = torch.stack([sample_flow(net, y_hat, lam, n_steps) for _ in range(n_draws)])
    one, mmse = draws[0], draws.mean(dim=0)
    mse_s = float((one - x_true).pow(2).mean())
    mse_m = float((mmse - x_true).pow(2).mean())
    psnr_sample = -10 * math.log10(mse_s) if mse_s > 0 else 99.0
    psnr_mmse = -10 * math.log10(mse_m) if mse_m > 0 else 99.0
    div = float(draws.flatten(2).var(dim=0).mean())

    nn_idx = _knn_indices(E, idx, knn, same_image_only, source_ids)
    cmmd_vals = []
    for j in range(N):
        gen = draws[:, j].flatten(1)
        real = X[nn_idx[j]].flatten(1)
        cmmd_vals.append(g1.mmd_rff(gen, real, seed=seed))
    cmmd = float(sum(cmmd_vals) / len(cmmd_vals))
    return psnr_sample, psnr_mmse, cmmd, div


# ---------------------------------------------------------------------------
# Multi-GPU (DDP) -- launch with torchrun, e.g. on Kaggle's 2xT4:
#   torchrun --standalone --nproc_per_node=2 real/flow.py --train_root ...
# Plain `python real/flow.py ...` (no torchrun) still works unchanged: WORLD_SIZE
# is unset, _is_distributed() is False, everything runs single-process as before.
#
# Design: each rank builds its OWN local pool (independent `.sample()` calls,
# rank offset into the RNG seed) instead of building one pool on rank 0 and
# broadcasting it -- simpler, and the one-time per-rank pool-encoding cost
# (a few thousand codec forward passes, done once before the step loop, not
# per-step) is cheap enough on 2 GPUs that it isn't worth the extra
# gather/broadcast code. DDP itself only needs the model wrapped: the
# training loop (train_flow) is untouched -- `net(...)`, `net.parameters()`,
# `net.train()` all work the same through a DDP wrapper, since DDP hooks
# gradient sync into `.backward()` transparently. Only logging/eval/checkpoint
# saving are guarded to rank 0, to avoid duplicate console spam, redundant
# eval compute, and concurrent writes to the same output file.
# ---------------------------------------------------------------------------


def _is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _setup_distributed():
    import torch.distributed as dist

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    return rank, world_size, local_rank, device


def _cleanup_distributed():
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI: train ONE lambda-conditioned model, then sweep lambda at eval time
# (cheaper than g4_wflow's per-lambda retraining -- see module docstring)
# ---------------------------------------------------------------------------


def main():
    import argparse
    import json

    from real.codec import CompressAICodec
    from real.data import PatchFolderDataset

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_root",
        required=True,
        help="CLIC train folder (or any large photo folder)",
    )
    ap.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="cap the pool to the first N images found under --train_root "
        "-- for a quick mini-subset stability run before a full CLIC train",
    )
    ap.add_argument("--arch", default="cheng2020-anchor")
    ap.add_argument("--quality", type=int, default=1)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--npool", type=int, default=2000)
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument(
        "--same_image_only",
        action="store_true",
        help="restrict kNN targets to crops of the same source photo "
        "(use if real/data.py's sanity_check_knn shows cross-image "
        "neighbours landing on unrelated content)",
    )
    ap.add_argument(
        "--crops_per_image",
        type=int,
        default=8,
        help="with --same_image_only, draw the pool as groups of this many "
        "crops per source image (via PatchFolderDataset.sample_grouped) so "
        "every anchor actually has same-image candidates -- independent "
        "per-crop sampling makes same-image collisions rare whenever the "
        "folder has as many/more images as --npool",
    )
    ap.add_argument(
        "--embed",
        type=str,
        default="pool",
        help="g1_pilot.make_embed kind for the kNN retrieval embedding: "
        "'pool' (per-channel spatial average -- fast, but spatially blind, "
        "the diagnosed cause of tonally-similar-but-content-unrelated cross-"
        "image kNN matches, see docs/RESULTS_REAL.md) or 'proj<d>' (random "
        "linear projection of the flattened y_hat, e.g. proj64 -- preserves "
        "spatial structure, recommended once training data scales beyond a "
        "--same_image_only-sized pool) or 'raw' (no projection, high-dim)",
    )
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument(
        "--N", type=int, default=32, help="batch size per step, PER GPU under DDP"
    )
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--lams", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    ap.add_argument("--n_steps", type=int, default=20, help="Euler steps when sampling")
    ap.add_argument("--n_draws", type=int, default=8)
    ap.add_argument("--eval_N", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default="results/g4_real.json")
    ap.add_argument(
        "--save_checkpoint_dir",
        type=str,
        default=None,
        help="if set, saves <dir>/g4_real_seed{seed}.pt state_dicts "
        "for later use with real/eval.py on full test images",
    )
    args = ap.parse_args()

    distributed = _is_distributed()
    if distributed:
        rank, world_size, local_rank, device = _setup_distributed()
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    if args.same_image_only and args.crops_per_image <= args.knn:
        # A source image only contributes (crops_per_image - 1) genuine
        # same-image candidates besides the anchor itself; if knn asks for
        # more than that, _knn_indices's topk fills the rest from masked
        # (+inf, tied) entries -- i.e. arbitrary crops, not same-image ones,
        # silently defeating part of --same_image_only. Keep the invariant
        # crops_per_image > knn so every requested neighbour is genuine.
        old = args.crops_per_image
        args.crops_per_image = args.knn + 1
        if is_main:
            print(
                f"[warn] --crops_per_image={old} <= --knn={args.knn}; bumping "
                f"--crops_per_image to {args.crops_per_image} so every kNN "
                f"slot has a genuine same-image candidate"
            )

    if is_main:
        print(
            f"distributed={distributed} world_size={world_size} device={device} "
            f"arch={args.arch} q={args.quality} patch={args.patch} "
            f"npool={args.npool} knn={args.knn} embed={args.embed} "
            f"same_image_only={args.same_image_only}"
        )

    # rank 0 downloads the pretrained codec first so concurrent torch.hub cache
    # writes on a first-ever run can't race; other ranks then hit a warm cache.
    if distributed and rank != 0:
        torch.distributed.barrier()
    codec = CompressAICodec(arch=args.arch, quality=args.quality, device=device)
    if distributed and rank == 0:
        torch.distributed.barrier()

    data = PatchFolderDataset(
        args.train_root,
        patch=args.patch,
        device=device,
        seed=rank,
        max_images=args.max_images,
    )

    rows = []
    for seed in args.seeds:
        g1.set_seed(seed * 1000 + rank)  # distinct local pool/batches per rank
        X, Y, source_ids = build_real_pool(
            codec,
            data,
            args.npool,
            device,
            crops_per_image=args.crops_per_image if args.same_image_only else None,
        )
        y_shape = tuple(Y.shape[1:])
        embed = g1.make_embed(args.embed, y_shape, seed=seed, device=device)
        E = embed(Y)

        net = UNetVelocity(zc=y_shape[0], ch=args.ch).to(device)
        if distributed:
            ddp_kwargs = (
                {"device_ids": [local_rank]} if torch.cuda.is_available() else {}
            )
            net = torch.nn.parallel.DistributedDataParallel(net, **ddp_kwargs)
        loss = train_flow(
            net,
            X,
            Y,
            E,
            args.lams,
            args.steps,
            args.N,
            args.knn,
            args.lr,
            device,
            args.same_image_only,
            source_ids,
        )
        raw_net = net.module if distributed else net  # unwrap DDP for save/eval
        if is_main:
            print(
                f"seed={seed} final training loss={loss:.4f} (y_hat channels zc={y_shape[0]})"
            )

        if args.save_checkpoint_dir and is_main:
            os.makedirs(args.save_checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_checkpoint_dir, f"g4_real_seed{seed}.pt")
            torch.save(raw_net.state_dict(), ckpt_path)
            print(f"saved checkpoint -> {ckpt_path}")

        # eval only on rank 0, against rank 0's own local pool -- other ranks'
        # pools were only there to diversify training batches, not to be scored
        if is_main:
            print(
                f"{'lam':>6} {'PSNR_samp':>10} {'PSNR_mmse':>10} {'condMMD':>12} {'Div(Var_z)':>12}"
            )
            for lam in args.lams:
                ps, pm, cm, dv = evaluate(
                    raw_net,
                    codec,
                    X,
                    Y,
                    E,
                    args.eval_N,
                    lam,
                    args.n_steps,
                    args.n_draws,
                    args.knn,
                    seed,
                    device,
                    args.same_image_only,
                    source_ids,
                )
                print(f"{lam:>6.2f} {ps:>10.2f} {pm:>10.2f} {cm:>12.4e} {dv:>12.4e}")
                rows.append(
                    {
                        "seed": seed,
                        "lam": lam,
                        "psnr_sample": ps,
                        "psnr_mmse": pm,
                        "cond_mmd": cm,
                        "div": dv,
                    }
                )

    if args.out and is_main:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        payload = {
            "config": {
                k: getattr(args, k)
                for k in [
                    "train_root",
                    "max_images",
                    "arch",
                    "quality",
                    "patch",
                    "npool",
                    "knn",
                    "embed",
                    "same_image_only",
                    "steps",
                    "N",
                    "lams",
                    "seeds",
                ]
            },
            "world_size": world_size,
            "y_shape": list(y_shape),
            "frontier": rows,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nsaved -> {args.out}")

    if distributed:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
