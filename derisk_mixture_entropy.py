#!/usr/bin/env python
"""
De-risk experiment (b): does a K-component MIXTURE entropy head lower the rate
versus a single Gaussian, all else equal?

Design principle
----------------
The Gaussian baseline and the mixture are the SAME code path with different K
(K=1 == single Gaussian).  Backbone, hyperprior, channel-slice context, data,
schedule and lambda are identical across runs.  The ONLY variable is K.  This
isolates the hypothesis "is p(y_hat | context) multimodal enough that a mixture
helps?" — the question the whole OT entropy-model idea rests on.

What it measures
----------------
  * estimated bpp  = mean over elements of  -log2 P_discrete(symbol)
                     where P_discrete is the (mixture) PMF integrated over the
                     unit bin actually coded.  This IS the arithmetic coder's
                     cross-entropy (= achievable real rate minus negligible
                     coder overhead), so it is the correct quantity for the
                     hypothesis test.
  * PSNR
  * component-usage entropy  H_used = H(mean mixture weights) in bits.
                     If a mixture lowers rate but H_used collapses toward 0
                     (a few components dominate, dead modes), THAT is exactly
                     the gap the OT marginal constraint is designed to fix —
                     the best-case outcome for the OT story.
  * a real rANS roundtrip on a few images (verify_real_coder) that confirms the
    mixture PMF codes losslessly and that real bpp ~= estimated bpp, so the
    estimated number is trustworthy.

This script does NOT implement a full compress/decompress codec for the whole
pipeline — that is milestone 2 and only matters once the hypothesis passes.

Run on Kaggle
-------------
    # 2x T4 (Kaggle "GPU T4 x2"): runs one K per GPU in parallel, automatically
    python derisk.py --data /kaggle/input/<your-image-folder> --ks 1 3 5

    # force sequential on a single GPU
    python derisk.py --data /kaggle/input/<your-image-folder> --ks 1 3 5 --parallel 0
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import queue
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf

# ---------------------------------------------------------------------------
# Repro
# ---------------------------------------------------------------------------


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
# Conv helpers
# ---------------------------------------------------------------------------


def conv(in_ch, out_ch, k=5, s=2):
    return nn.Conv2d(in_ch, out_ch, k, stride=s, padding=k // 2)


def deconv(in_ch, out_ch, k=5, s=2):
    return nn.ConvTranspose2d(
        in_ch, out_ch, k, stride=s, padding=k // 2, output_padding=s - 1
    )


def ste_round(x):
    return torch.round(x) - x.detach() + x


# Standardized Gaussian CDF via erfc (matches CompressAI numerics).
_INV_SQRT2 = float(2 ** -0.5)


def std_cdf(x):
    return 0.5 * torch.erfc(-_INV_SQRT2 * x)


# ---------------------------------------------------------------------------
# Mixture conditional entropy head  (K=1 -> single Gaussian baseline)
# ---------------------------------------------------------------------------


class MixtureConditional(nn.Module):
    """
    Per-element K-component Gaussian-mixture conditional.

    params layout (channel dim): [w_logits (K), means (K), log_scales (K)]
    broadcast per latent channel, i.e. the entropy net outputs slice_ch * 3K
    channels which we reshape to (B, slice_ch, 3K, H, W).
    """

    SCALE_MIN = 0.11

    def __init__(self, K: int):
        super().__init__()
        self.K = K

    def _split(self, params, slice_ch):
        B, C, H, W = params.shape
        params = params.view(B, slice_ch, 3 * self.K, H, W)
        w_logits = params[:, :, : self.K]
        means = params[:, :, self.K : 2 * self.K]
        log_scales = params[:, :, 2 * self.K :]
        weights = torch.softmax(w_logits, dim=2)
        scales = F.softplus(log_scales).clamp(min=self.SCALE_MIN)
        return weights, means, scales  # each (B, slice_ch, K, H, W)

    def mixture_mean(self, weights, means):
        return (weights * means).sum(dim=2)  # (B, slice_ch, H, W)

    def likelihood_bits(self, y, weights, means, scales):
        """
        Discretized mixture PMF mass in the unit bin around y, in bits:
            -log2( sum_k w_k [Phi((y+.5-m_k)/s_k) - Phi((y-.5-m_k)/s_k)] )
        y: (B, slice_ch, H, W) -> broadcast over K.
        """
        yk = y.unsqueeze(2)  # (B, slice_ch, 1, H, W)
        upper = std_cdf((yk + 0.5 - means) / scales)
        lower = std_cdf((yk - 0.5 - means) / scales)
        comp = (upper - lower).clamp(min=1e-9)  # (B, slice_ch, K, H, W)
        mix = (weights * comp).sum(dim=2).clamp(min=1e-9)  # (B, slice_ch, H, W)
        return -torch.log2(mix)

    @torch.no_grad()
    def discrete_pmf_on_grid(self, weights, means, scales, grid):
        """
        Mixture PMF over an integer grid of residual values, for the real coder.
        grid: 1-D LongTensor of residual integers (same for the whole element
              batch passed). Returns pmf (..., len(grid)). Caller supplies a grid
              wide enough to cover the mass.
        weights/means/scales: (..., K)  (already squeezed per element)
        """
        g = grid.to(means.dtype).view(*([1] * (means.dim() - 1)), 1, -1)  # (...,1,G)
        m = means.unsqueeze(-1)  # (...,K,1)
        s = scales.unsqueeze(-1)
        w = weights.unsqueeze(-1)
        upper = std_cdf((g + 0.5 - m) / s)
        lower = std_cdf((g - 0.5 - m) / s)
        comp = (upper - lower).clamp(min=0.0)  # (...,K,G)
        pmf = (w * comp).sum(dim=-2)  # (...,G)
        return pmf


# ---------------------------------------------------------------------------
# Small LIC model with channel-wise slice context + pluggable K
# ---------------------------------------------------------------------------


class SmallLIC(nn.Module):
    def __init__(self, N=128, M=192, num_slices=4, K=1):
        super().__init__()
        assert M % num_slices == 0
        self.N, self.M = N, M
        self.num_slices = num_slices
        self.slice_ch = M // num_slices
        self.K = K
        self.cond = MixtureConditional(K)

        self.g_a = nn.Sequential(
            conv(3, N), GDN(N), conv(N, N), GDN(N), conv(N, N), GDN(N), conv(N, M)
        )
        self.g_s = nn.Sequential(
            deconv(M, N), GDN(N, inverse=True),
            deconv(N, N), GDN(N, inverse=True),
            deconv(N, N), GDN(N, inverse=True),
            deconv(N, 3),
        )
        self.h_a = nn.Sequential(
            conv(M, N, 3, 1), nn.LeakyReLU(inplace=True),
            conv(N, N), nn.LeakyReLU(inplace=True),
            conv(N, N),
        )
        self.h_s = nn.Sequential(
            deconv(N, N), nn.LeakyReLU(inplace=True),
            deconv(N, N), nn.LeakyReLU(inplace=True),
            conv(N, 2 * M, 3, 1),
        )
        self.entropy_bottleneck = EntropyBottleneck(N)

        # per-slice param predictor: input = 2M (hyper) + i*slice_ch (prev slices)
        out_ch = self.slice_ch * 3 * K
        self.param_nets = nn.ModuleList()
        for i in range(num_slices):
            in_ch = 2 * M + i * self.slice_ch
            self.param_nets.append(
                nn.Sequential(
                    conv(in_ch, 224, 3, 1), nn.GELU(),
                    conv(224, 128, 3, 1), nn.GELU(),
                    conv(128, out_ch, 3, 1),
                )
            )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_lik = self.entropy_bottleneck(z)
        hyper = self.h_s(z_hat)  # (B, 2M, Hy, Wy)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices, bits_list, weight_acc = [], [], []
        for i, ys in enumerate(y_slices):
            ctx = torch.cat([hyper] + y_hat_slices, dim=1)
            params = self.param_nets[i](ctx)
            w, m, s = self.cond._split(params, self.slice_ch)
            mu_bar = self.cond.mixture_mean(w, m)

            if self.training:
                y_noisy = ys + (torch.rand_like(ys) - 0.5)
                bits = self.cond.likelihood_bits(y_noisy, w, m, s)
                y_hat = ste_round(ys - mu_bar) + mu_bar
            else:
                y_hat = torch.round(ys - mu_bar) + mu_bar
                bits = self.cond.likelihood_bits(y_hat, w, m, s)

            bits_list.append(bits)
            y_hat_slices.append(y_hat)
            weight_acc.append(w.mean(dim=(0, 1, 3, 4)))  # (K,) avg usage

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat)

        N_pix = x.shape[0] * x.shape[2] * x.shape[3]
        y_bits = torch.stack([b.sum() for b in bits_list]).sum()
        z_bits = torch.log2(z_lik.clamp(min=1e-9)).sum().neg()
        return {
            "x_hat": x_hat,
            "y_bpp": y_bits / N_pix,
            "z_bpp": z_bits / N_pix,
            "bpp": (y_bits + z_bits) / N_pix,
            "weights": torch.stack(weight_acc).mean(0),  # (K,) mean usage
        }


# ---------------------------------------------------------------------------
# Real rANS roundtrip for the mixture head (verification on a few images)
# ---------------------------------------------------------------------------


@torch.no_grad()
def verify_real_coder(model, x, device, grid_cap=384, precision=16):
    """
    Encode/decode the y latent with a real rANS coder using per-element mixture
    PMFs, and report (lossless?, real_bpp, est_bpp, clamped).

    The grid radius adapts to the actual residual range so NO residual is
    clamped -- only then does a lossless symbol roundtrip equal a lossless
    reconstruction of y_hat, and only then is real_y_bpp ~= est_y_bpp a
    meaningful statement.  If residuals exceed grid_cap (undertrained model),
    `clamped` is reported > 0 and the numbers should be ignored.
    """
    model.eval()
    x = x.to(device)
    y = model.g_a(x)
    z = model.h_a(y)
    z_hat, _ = model.entropy_bottleneck(z)
    hyper = model.h_s(z_hat)
    B = y.shape[0]
    assert B == 1, "verify on one image at a time"

    y_slices = y.chunk(model.num_slices, 1)

    # First pass: compute all residuals to size the grid so nothing clamps.
    resids, mus, ws, ms, ss = [], [], [], [], []
    y_hat_slices = []
    for i, ys in enumerate(y_slices):
        ctx = torch.cat([hyper] + y_hat_slices, dim=1)
        params = model.param_nets[i](ctx)
        w, m, s = model.cond._split(params, model.slice_ch)
        mu_bar = model.cond.mixture_mean(w, m)
        resid = torch.round(ys - mu_bar).to(torch.long)
        y_hat_slices.append(resid.float() + mu_bar)
        resids.append(resid); ws.append(w); ms.append(m); ss.append(s)

    max_abs = int(max(int(r.abs().max()) for r in resids))
    grid_radius = min(max_abs + 1, grid_cap)
    clamped = sum(int((r.abs() > grid_radius).sum()) for r in resids)
    grid = torch.arange(-grid_radius, grid_radius + 1, device=device)
    G = grid.numel()

    symbols, indexes, cdfs, cdf_lengths, offsets = [], [], [], [], []
    est_bits = 0.0
    for resid, w, m, s in zip(resids, ws, ms, ss):
        wf = w.permute(0, 1, 3, 4, 2).reshape(-1, model.K)
        mf = m.permute(0, 1, 3, 4, 2).reshape(-1, model.K)
        sf = s.permute(0, 1, 3, 4, 2).reshape(-1, model.K)
        rf = resid.reshape(-1)
        pmf = model.cond.discrete_pmf_on_grid(wf, mf, sf, grid)  # (Nel, G)
        pmf = pmf / pmf.sum(dim=1, keepdim=True).clamp(min=1e-12)

        sym_idx = (rf + grid_radius).clamp(0, G - 1)
        p_at = pmf[torch.arange(pmf.shape[0]), sym_idx].clamp(min=1e-12)
        est_bits += float((-torch.log2(p_at)).sum())

        pmf_np = pmf.cpu().numpy().astype(np.float64)
        sidx = sym_idx.cpu().tolist()
        for j in range(pmf_np.shape[0]):
            q = _pmf_to_quantized_cdf(pmf_np[j].tolist(), precision)
            cdfs.append(q); cdf_lengths.append(len(q)); offsets.append(0)
            indexes.append(len(cdfs) - 1); symbols.append(int(sidx[j]))

    enc = BufferedRansEncoder()
    enc.encode_with_indexes(symbols, indexes, cdfs, cdf_lengths, offsets)
    stream = enc.flush()
    real_bits = len(stream) * 8

    dec = RansDecoder()
    dec.set_stream(stream)
    decoded = dec.decode_stream(indexes, cdfs, cdf_lengths, offsets)

    N_pix = x.shape[2] * x.shape[3]
    return {
        "lossless": decoded == symbols,
        "clamped": clamped,
        "real_y_bpp": real_bits / N_pix,
        "est_y_bpp": est_bits / N_pix,
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class PatchDataset(Dataset):
    def __init__(self, root, patch=256, exts=(".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        self.paths = [
            p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
            if os.path.splitext(p)[1].lower() in exts
        ]
        if not self.paths:
            raise RuntimeError(f"No images under {root}")
        self.patch = patch
        self.to_tensor = transforms.ToTensor()  # picklable; no module objects stored

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        w, h = img.size
        p = self.patch
        if w < p or h < p:
            img = img.resize((max(w, p), max(h, p)))
            w, h = img.size
        x = random.randint(0, w - p)
        y = random.randint(0, h - p)
        return self.to_tensor(img.crop((x, y, x + p, y + p)))


# ---------------------------------------------------------------------------
# Train / eval one K
# ---------------------------------------------------------------------------


def run_one(K, args, device):
    set_seed(args.seed)  # identical init/data order across K -> fair
    model = SmallLIC(N=args.N, M=args.M, num_slices=args.slices, K=K).to(device)

    ds = PatchDataset(args.data, patch=args.patch)
    n_val = max(1, int(0.05 * len(ds)))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(ds), generator=g).tolist()
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr = torch.utils.data.Subset(ds, tr_idx)
    va = torch.utils.data.Subset(ds, val_idx)
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=True)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=args.workers)

    main_p = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
    aux_p = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]
    opt = torch.optim.Adam(main_p, lr=args.lr)
    aux_opt = torch.optim.Adam(aux_p, lr=1e-3)

    step = 0
    model.train()
    t0 = time.time()
    while step < args.steps:
        for x in tl:
            if step >= args.steps:
                break
            x = x.to(device)
            opt.zero_grad(); aux_opt.zero_grad()
            out = model(x)
            mse = F.mse_loss(out["x_hat"], x)
            loss = args.lmbda * 255.0 ** 2 * mse + out["bpp"]
            loss.backward()
            nn.utils.clip_grad_norm_(main_p, 1.0)
            opt.step()
            aux_loss = model.entropy_bottleneck.loss()
            aux_loss.backward()
            aux_opt.step()
            step += 1
            if step % args.log_every == 0:
                print(f"[K={K}] step {step}/{args.steps} "
                      f"loss {loss.item():.3f} bpp {out['bpp'].item():.4f} "
                      f"mse {mse.item():.5f}")

    # eval
    model.eval()
    bpps, psnrs, ybpps = [], [], []
    wsum = torch.zeros(K, device=device)
    with torch.no_grad():
        for x in vl:
            x = x.to(device)
            out = model(x)
            mse = F.mse_loss(out["x_hat"].clamp(0, 1), x).item()
            psnr = -10 * math.log10(mse) if mse > 0 else 99.0
            bpps.append(out["bpp"].item()); ybpps.append(out["y_bpp"].item())
            psnrs.append(psnr); wsum += out["weights"]
    w_mean = (wsum / len(vl)).cpu()
    H_used = float(-(w_mean.clamp(1e-9) * w_mean.clamp(1e-9).log2()).sum())

    # real-coder verification on up to N images
    real = []
    with torch.no_grad():
        for j, x in enumerate(va):
            if j >= args.verify_imgs:
                break
            r = verify_real_coder(model, x.unsqueeze(0), device)
            real.append(r)

    res = {
        "K": K,
        "val_bpp": float(np.mean(bpps)),
        "val_y_bpp": float(np.mean(ybpps)),
        "val_psnr": float(np.mean(psnrs)),
        "H_used_bits": H_used,
        "max_H_bits": math.log2(K),
        "minutes": (time.time() - t0) / 60,
        "real_check": real,
    }
    return res


# ---------------------------------------------------------------------------
# Parallel scheduler: one K per GPU (each run stays a clean single-GPU run)
# ---------------------------------------------------------------------------


def _worker(gpu_id, k_queue, res_queue, args):
    """Pull K values off the queue and run each on this worker's GPU."""
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    # We already parallelise across GPUs with these processes; nesting DataLoader
    # worker subprocesses on top is the fragile part on Kaggle (spawn-in-spawn,
    # pickling). Load data inline in each GPU process instead.
    args.workers = 0
    while True:
        try:
            K = k_queue.get_nowait()
        except queue.Empty:
            break
        try:
            res = run_one(K, args, device)
        except Exception as e:  # never let the parent hang on a crashed worker
            import traceback
            res = {"K": K, "error": f"{e}", "trace": traceback.format_exc()}
        res_queue.put(res)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="folder of training images")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lmbda", type=float, default=0.013)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--M", type=int, default=192)
    ap.add_argument("--slices", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--verify_imgs", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=1,
                    help="1 = run one K per GPU in parallel when >=2 GPUs; 0 = sequential")
    ap.add_argument("--gpus", type=int, nargs="+", default=None,
                    help="GPU ids to use (default: all visible)")
    args = ap.parse_args()

    n_gpu = torch.cuda.device_count()
    gpus = args.gpus if args.gpus is not None else list(range(n_gpu))
    use_parallel = bool(args.parallel) and len(gpus) >= 2 and len(args.ks) > 1

    if use_parallel:
        n_workers = min(len(gpus), len(args.ks))
        print(f"PARALLEL: {len(args.ks)} runs over {n_workers} GPUs {gpus[:n_workers]} "
              f"(one K per GPU; λ={args.lmbda}, steps={args.steps})")
        ctx = mp.get_context("spawn")
        kq, rq = ctx.Queue(), ctx.Queue()
        for K in args.ks:
            kq.put(K)
        procs = [ctx.Process(target=_worker, args=(gpus[i], kq, rq, args))
                 for i in range(n_workers)]
        for p in procs:
            p.start()
        results = [rq.get() for _ in args.ks]  # one result per K
        for p in procs:
            p.join()
    else:
        device = f"cuda:{gpus[0]}" if n_gpu else "cpu"
        print(f"SEQUENTIAL on {device}  λ={args.lmbda}  steps={args.steps}")
        results = []
        for K in args.ks:
            print(f"\n===== Training K={K} =====")
            results.append(run_one(K, args, device))

    # surface any worker errors, then sort by K
    for r in results:
        if "error" in r:
            print(f"\n[!] K={r['K']} FAILED: {r['error']}\n{r.get('trace','')}")
    results = sorted([r for r in results if "error" not in r], key=lambda r: r["K"])
    if not results:
        print("No successful runs."); return

    print("\n================ SUMMARY ================")
    print(f"{'K':>3} {'val_bpp':>9} {'val_psnr':>9} {'H_used/maxH':>14} "
          f"{'lossless':>9} {'real-est':>9} {'clamped':>8}")
    base = next((r for r in results if r["K"] == 1), results[0])
    for r in results:
        rc = r["real_check"]
        loss_ok = all(c["lossless"] for c in rc) if rc else None
        clamped = sum(c["clamped"] for c in rc) if rc else 0
        if rc:
            gap = np.mean([c["real_y_bpp"] - c["est_y_bpp"] for c in rc])
            gap_s = f"{gap:+.4f}"
        else:
            gap_s = "n/a"
        hus = f"{r['H_used_bits']:.2f}/{r['max_H_bits']:.2f}" if r["K"] > 1 else "-"
        d_bpp = (r["val_bpp"] - base["val_bpp"]) / base["val_bpp"] * 100
        print(f"{r['K']:>3} {r['val_bpp']:>9.4f} {r['val_psnr']:>9.2f} {hus:>14} "
              f"{str(loss_ok):>9} {gap_s:>9} {clamped:>8}   (Δbpp vs K=1: {d_bpp:+.1f}%)")

    print("\nHow to read this:")
    print(" * Δbpp vs K=1 NEGATIVE at similar PSNR  -> mixture helps. Hypothesis PASSES.")
    print(" * Δbpp ~ 0                               -> conditional ~unimodal. PIVOT.")
    print(" * Mixture helps BUT H_used << max_H      -> dead modes / collapse:")
    print("     best case for the OT idea (marginal constraint is the fix).")
    print(" * real_lossless=True and real-est gap ~0 -> estimated bpp is trustworthy.")


if __name__ == "__main__":
    main()
