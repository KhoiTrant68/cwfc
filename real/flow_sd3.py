#!/usr/bin/env python
"""
SD3-backbone conditional rectified flow -- replaces real/flow.py's
from-scratch UNetVelocity with a pretrained flow-matching (rectified-flow)
transformer, so the G4 lambda/kNN mechanism runs on top of an internet-scale
image prior instead of a ~2000-crop from-scratch pool.

WHY: real/flow.py's UNetVelocity proved the mechanism (docs/RESULTS_REAL.md
-- monotonic PSNR/Var_z/LPIPS/FID vs lambda) but at PSNR~13dB / LPIPS~0.95,
~12-15dB and ~9-11x below MS-ILLM at matched bpp -- not a "different regime",
a CVPR/ICCV/ECCV reviewer reads that as broken. Stable Diffusion 3 (MMDiT,
~2B params) is trained natively with flow matching (not DDPM diffusion), so
swapping to it both (a) is a straight upgrade in prior strength/capacity and
(b) keeps this project's stated "flow matching over diffusion" preference.
See the plan this file was built from
(C:\\Users\\khoit\\.claude\\plans\\piped-enchanting-storm.md) for the full
literature positioning (arXiv:2405.16817, arXiv:2606.13366).

WHAT STAYS THE SAME (the actual paper contribution, unchanged):
  * The lambda/kNN retrieval mechanism -- real/flow.py's `_knn_indices` and
    `build_real_pool` are imported and reused verbatim, not reimplemented.
  * The lambda-blended target construction
    `x1 = (1-lam)*x_samp + lam*x_mean` -- identical formula, just applied to
    VAE latents here instead of pixels (see `train_flow_sd3`).
  * The rectified-flow *shape* of the objective (linear interpolation between
    a Gaussian and a data point, regress the connecting velocity) -- only the
    interpolation direction/sign convention changes, to match how SD3 itself
    was pretrained (see the sigma-convention note below) so the pretrained
    weights are used the way they were trained, not fought against.

WHAT'S NEW:
  * `SD3LatentFlow` -- loads a frozen SD3Transformer2DModel + AutoencoderKL
    + a *trainable* SD3ControlNetModel (diffusers' own ControlNet-for-SD3,
    reused rather than hand-rolled -- see
    https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sd3).
    Its conditioning image is `codec.decode(y_hat)` -- the frozen compressai
    codec's own degraded pixel reconstruction -- i.e. the task the
    ControlNet+ transformer jointly solve is "hallucinate a photorealistic
    reconstruction from this degraded codec decode", the same pattern as
    ControlNet-tile/restoration usage.
  * lambda conditioning is injected into SD3's *existing* pooled-projection
    conditioning pathway (`pooled_projections`, normally derived from CLIP
    text pooling) instead of a new modulation branch: since this project
    uses a fixed empty text prompt, the pooled-text channel is otherwise
    idle, so a small lambda-MLP's output is added to the (frozen, cached)
    empty-prompt pooled embedding before it reaches
    `SD3Transformer2DModel`'s own `time_text_embed` submodule. This needs no
    subclassing/copying of diffusers' forward() -- it reuses the model's own
    conditioning entry point.
  * Training scope: transformer, VAE, and text encoders are frozen; only the
    ControlNet branch (diffusers initialises it from a subset of the
    transformer's blocks -- a few hundred M params, not the full ~2B) and
    the lambda-MLP are trained. Checkpoints therefore only save those two
    submodules (`save_trainable`/`load_trainable`), not the frozen backbone.

SIGMA / VELOCITY-SIGN CONVENTION -- read before touching the math below:
real/flow.py uses x0=noise, x1=data, t: 0->1 (noise->data),
xt=(1-t)x0+t*x1, target=x1-x0 (velocity points noise->data).
SD3 was pretrained with the *opposite* convention (matches diffusers'
`FlowMatchEulerDiscreteScheduler` / `examples/dreambooth/train_dreambooth_
lora_sd3.py`): sigma: 0->1 is data->noise, xt=(1-sigma)*x1+sigma*x0,
target=x0-x1 (velocity points data->noise). This file follows SD3's own
convention (sigma, target=x0-x1) rather than real/flow.py's, because the
whole point of this backbone swap is to use the pretrained weights the way
they were trained -- feeding them real/flow.py's convention would silently
fight the pretrained prior. The lambda-blended *target construction* is
still identical in spirit: `x1_latent` is built by the same
`(1-lam)*x_samp + lam*x_mean` formula as real/flow.py, just fed into SD3's
xt/target formulas instead of real/flow.py's.

Places tagged `# VERIFY:` depend on the exact installed `diffusers` version's
API surface (forward() signatures, config attribute names, the timestep
scaling constant) and could not be checked in this offline environment (no
local Python/diffusers here) -- run the smoke test in
docs/RESULTS_REAL.md-style (`--max_images`, tiny `--npool`, `--steps 20`)
first and grep this file for VERIFY before a full run.
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
from real.flow import (  # noqa: E402  (unchanged mechanism + DDP helpers)
    _cleanup_distributed,
    _is_distributed,
    _knn_indices,
    _setup_distributed,
    build_real_pool,
)

SD3_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"


# ---------------------------------------------------------------------------
# lambda -> pooled-projection offset (reuses SD3's own conditioning pathway,
# see module docstring -- no transformer subclass/forward() copy needed)
# ---------------------------------------------------------------------------


def _sinusoidal_scalar_embed(x, dim):
    """x: (B,) in [0,1] -> (B, dim) sinusoidal embedding, same recipe as a
    diffusion timestep embedding (log-spaced frequencies)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=x.device, dtype=torch.float32) / half
    )
    args = x[:, None].float() * freqs[None, :] * 1000.0
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class LambdaEmbedding(nn.Module):
    """lam:(B,) in [0,1] -> (B, pooled_dim) offset added to SD3's (cached,
    empty-prompt) pooled_projections before it reaches time_text_embed."""

    def __init__(self, pooled_dim, hidden=256):
        super().__init__()
        self.sin_dim = hidden
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, pooled_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)  # start as a no-op (identity prior)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, lam):
        if not torch.is_tensor(lam):
            lam = torch.full((1,), float(lam))
        lam = lam.to(next(self.parameters()).device).view(-1)
        return self.mlp(_sinusoidal_scalar_embed(lam, self.sin_dim))


# ---------------------------------------------------------------------------
# SD3 backbone wrapper: frozen transformer/VAE/text-embeds + trainable
# ControlNet + lambda-MLP
# ---------------------------------------------------------------------------


class SD3LatentFlow(nn.Module):
    """Bundles the frozen SD3 transformer+VAE+cached-empty-prompt embeds
    with the trainable ControlNet + lambda-MLP. Plays the role `net` plays
    in real/flow.py's train_flow/sample_flow/evaluate (same call shape:
    velocity-in, velocity-out), just backed by a pretrained transformer
    instead of UNetVelocity.
    """

    def __init__(
        self,
        model_id=SD3_MODEL_ID,
        controlnet_layers=12,
        device="cuda",
        dtype=torch.float32,
        hf_token=None,
        hf_variant=None,
    ):
        # hf_variant="fp16" downloads the repo's fp16-precision *files* (half
        # the disk footprint -- matters on small container disks; SD3-medium's
        # T5-XXL text encoder alone is ~19GB at fp32 vs ~9.5GB at fp16 on
        # disk) independently of `dtype` (the *compute* dtype weights get
        # cast to once loaded into memory) -- e.g. hf_variant="fp16",
        # dtype=torch.bfloat16 is a fine combination.
        # dtype defaults to float32 rather than bf16: this file's dtype-cast
        # points (see the `.to(...)` calls below and in train_flow_sd3/
        # sample_flow_sd3) should make bf16 work too, but that combination
        # (frozen bf16 backbone + fp32-trained adapter, mixed at each forward
        # call) could not be exercised on real hardware in this offline
        # environment -- fp32 removes that whole bug surface for the first
        # correctness pass. On an A100 (40-80GB) SD3-medium comfortably fits
        # in fp32; switch to bf16 as a memory/speed optimisation only after
        # the fp32 smoke test confirms the pipeline itself is correct.
        super().__init__()
        from diffusers import AutoencoderKL, SD3ControlNetModel, SD3Transformer2DModel, StableDiffusion3Pipeline

        self.device_ = device
        self.dtype = dtype

        # Text encoders are only ever run once (fixed empty prompt) --
        # load the full pipeline transiently just to get the empty-prompt
        # embeddings, then drop it so its (large, esp. T5-XXL, ~4.7B params)
        # text encoders don't sit in GPU memory for the rest of training.
        # Peak memory during this transient load (pipeline's own transformer
        # + T5-XXL + CLIP encoders + VAE, all at `dtype`) can be the highest
        # watermark of the whole run -- if this OOMs on a smaller GPU before
        # even reaching training, the fix is loading T5 separately at lower
        # precision (`text_encoder_3=None` + a manually-loaded fp16 T5) or
        # calling `pipe.enable_model_cpu_offload()` first; not implemented
        # here since it wasn't reachable to test in this offline environment.
        #
        # NOTE: stabilityai/stable-diffusion-3-medium-diffusers is a *gated*
        # HF repo -- accept the license at
        # https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
        # and pass a token (this arg, or `huggingface_hub.login(token=...)`,
        # or an HF_TOKEN env var) before this will download successfully.
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, torch_dtype=dtype, token=hf_token, variant=hf_variant
        )
        pipe.to(device)
        with torch.no_grad():
            # VERIFY: encode_prompt's exact positional/keyword signature and
            # return-tuple order against the installed diffusers version.
            (
                prompt_embeds,
                _neg_prompt_embeds,
                pooled_prompt_embeds,
                _neg_pooled_prompt_embeds,
            ) = pipe.encode_prompt(prompt="", prompt_2="", prompt_3="", device=device)
        self.register_buffer("prompt_embeds", prompt_embeds.detach())
        self.register_buffer("pooled_prompt_embeds", pooled_prompt_embeds.detach())
        del pipe
        torch.cuda.empty_cache()

        self.transformer = SD3Transformer2DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=dtype, token=hf_token, variant=hf_variant
        ).to(device)
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=dtype, token=hf_token, variant=hf_variant
        ).to(device)
        self.transformer.eval().requires_grad_(False)
        self.vae.eval().requires_grad_(False)

        # VERIFY: from_transformer's exact kwargs (num_layers vs num_control
        # layers) against the installed diffusers version; num_layers=12 is
        # a common ControlNet-depth choice (half of SD3-medium's 24 blocks)
        # but check examples/controlnet/train_controlnet_sd3.py's own
        # default/recommendation before a full run.
        #
        # from_transformer is a plain constructor-style factory, not a
        # from_pretrained call -- it has no torch_dtype kwarg, and layers it
        # initialises fresh (not copied from `self.transformer`, e.g.
        # pos_embed) default to float32 regardless of what dtype the source
        # transformer's copied weights are in. `.to(device)` alone only
        # moves device, not dtype, which crashes the very first conv
        # (`RuntimeError: Input type (BFloat16) and bias type (float) should
        # be the same`) as soon as a non-fp32 `dtype` is used. Cast
        # explicitly here instead of relying on from_transformer/`.to()`.
        self.controlnet = SD3ControlNetModel.from_transformer(
            self.transformer, num_layers=controlnet_layers
        ).to(device=device, dtype=dtype)
        self.controlnet.train()

        pooled_dim = self.transformer.config.pooled_projection_dim
        self.lambda_mlp = LambdaEmbedding(pooled_dim).to(device)

        # VAE latent normalisation constants (SD3's VAE, unlike SD1.5/SDXL's,
        # has a shift_factor as well as a scaling_factor).
        self.vae_scale = self.vae.config.scaling_factor
        self.vae_shift = getattr(self.vae.config, "shift_factor", 0.0)
        # VERIFY: 1000 is SD3's standard num_train_timesteps; the transformer
        # expects `timestep` scaled to roughly this range (not raw sigma in
        # [0,1]) for its sinusoidal time_proj to see pretraining-scale
        # inputs -- confirm against the scheduler config
        # (FlowMatchEulerDiscreteScheduler.config.num_train_timesteps).
        self.num_train_timesteps = 1000

    def trainable_parameters(self):
        return list(self.controlnet.parameters()) + list(self.lambda_mlp.parameters())

    def _unwrap(self, m):
        """Strip a DistributedDataParallel wrapper if present, so checkpoints
        have the same key layout regardless of whether training used DDP
        (DDP stores the real module under `.module`, which would otherwise
        add a `module.` prefix to every saved key)."""
        return m.module if hasattr(m, "module") else m

    def save_trainable(self, path):
        controlnet, lambda_mlp = self._unwrap(self.controlnet), self._unwrap(self.lambda_mlp)
        torch.save(
            {"controlnet": controlnet.state_dict(), "lambda_mlp": lambda_mlp.state_dict()},
            path,
        )

    def load_trainable(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self._unwrap(self.controlnet).load_state_dict(ckpt["controlnet"])
        self._unwrap(self.lambda_mlp).load_state_dict(ckpt["lambda_mlp"])

    @torch.no_grad()
    def encode_pixels(self, x_pixel):
        """x_pixel: (B,3,H,W) in [0,1] -> VAE latent (B,16,H/8,W/8).
        Uses the encoder's mode (not a stochastic sample) since these latents
        serve as regression targets/conditioning, not VAE-training data."""
        x = x_pixel.to(self.dtype) * 2.0 - 1.0
        latent = self.vae.encode(x).latent_dist.mode()
        return (latent - self.vae_shift) * self.vae_scale

    @torch.no_grad()
    def decode_latent(self, latent):
        """(B,16,H/8,W/8) -> pixel (B,3,H,W) in [0,1]."""
        latent = latent.to(self.dtype) / self.vae_scale + self.vae_shift
        x = self.vae.decode(latent).sample
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)

    def velocity(self, xt_latent, sigma, cond_latent, lam):
        """xt_latent:(B,16,h,w) sigma:(B,) cond_latent:(B,16,h,w)
        lam:(B,) or scalar -> predicted velocity (B,16,h,w), SD3's own
        data->noise convention (see module docstring)."""
        B = xt_latent.shape[0]
        timestep = sigma.to(xt_latent.dtype) * self.num_train_timesteps
        # lambda_mlp runs in its own (float32) parameter dtype for training
        # stability -- cast its output to match pooled_prompt_embeds (net.dtype,
        # e.g. bf16) before adding, same reasoning as the sigma/lam casts in
        # train_flow_sd3/sample_flow_sd3.
        lam_offset = self.lambda_mlp(lam).to(self.pooled_prompt_embeds.dtype)
        pooled = self.pooled_prompt_embeds.expand(B, -1) + lam_offset
        prompt_embeds = self.prompt_embeds.expand(B, -1, -1)

        # VERIFY: SD3ControlNetModel.forward's exact kwarg names
        # (controlnet_cond / conditioning_scale) and return type against the
        # installed diffusers version.
        block_samples = self.controlnet(
            hidden_states=xt_latent,
            controlnet_cond=cond_latent,
            conditioning_scale=1.0,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled,
            return_dict=False,
        )[0]

        # VERIFY: SD3Transformer2DModel.forward's block_controlnet_hidden_
        # states kwarg name against the installed diffusers version.
        v_pred = self.transformer(
            hidden_states=xt_latent,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled,
            block_controlnet_hidden_states=block_samples,
            return_dict=False,
        )[0]
        return v_pred


# ---------------------------------------------------------------------------
# Train / sample -- same shape as real/flow.py::train_flow/sample_flow, same
# lambda/kNN retrieval mechanism (imported, not reimplemented), only the
# latent space + sigma convention differ (see module docstring).
# ---------------------------------------------------------------------------


def train_flow_sd3(
    net: SD3LatentFlow,
    X,
    X_latent,
    Ycond_latent,
    E,
    lams,
    steps,
    N,
    knn,
    lr,
    device,
    same_image_only=False,
    source_ids=None,
    grad_accum=1,
    is_main=True,
    log_every=5,
):
    """X/X_latent/Ycond_latent are the pool's pixel crops / VAE latents of
    those crops / VAE latents of the codec's decoded reconstruction --
    precomputed once by `build_real_pool_latent` below, mirroring
    real/flow.py's train_flow signature (X, Y, E, ...) with the extra
    *_latent tensors standing in for pixel-space X/Y where the flow math
    itself runs (see module docstring's sigma-convention note).

    Unlike real/flow.py's train_flow (fast enough per-step that only the
    final loss is worth printing), this prints every `log_every` steps: SD3
    is heavy enough (a 50-step run with a large batch/patch can take many
    minutes) that silence until the very end is indistinguishable from a
    hang -- exactly what happened on the first real multi-GPU run."""
    import time

    opt = torch.optim.AdamW(net.trainable_parameters(), lr=lr)
    net.controlnet.train()
    Np = X_latent.shape[0]
    lams_t = torch.tensor(lams, device=device)
    last_loss = 0.0
    t0 = time.time()
    opt.zero_grad()
    for step in range(steps):
        idx = torch.randint(0, Np, (N,), device=device)
        cond = Ycond_latent[idx]
        with torch.no_grad():
            nn_idx = _knn_indices(E, idx, knn, same_image_only, source_ids)
            Xnb = X_latent[nn_idx]
            pick = torch.randint(0, knn, (N,), device=device)
            x_samp = Xnb[torch.arange(N, device=device), pick]
            x_mean = Xnb.mean(dim=1)
            lam = lams_t[torch.randint(0, len(lams_t), (N,), device=device)]
            # cast the blend/interpolation weights to the flow tensors' own
            # dtype (net.dtype, e.g. bf16) explicitly -- mixing a float32
            # scalar-derived weight with a bf16 tensor is ambiguous under
            # PyTorch's type promotion and can silently upcast xt to float32,
            # which then mismatches the bf16 transformer's expected input.
            lam_b = lam.view(N, 1, 1, 1).to(x_samp.dtype)
            x1 = (1.0 - lam_b) * x_samp + lam_b * x_mean  # same blend as real/flow.py

            # SD3's own convention (data->noise), see module docstring.
            x0 = torch.randn_like(x1)
            sigma = torch.rand(N, device=device)
            sigma_b = sigma.view(N, 1, 1, 1).to(x1.dtype)
            xt = (1.0 - sigma_b) * x1 + sigma_b * x0
            target = x0 - x1

        v = net.velocity(xt, sigma, cond, lam)
        loss = F.mse_loss(v.float(), target.float()) / grad_accum
        loss.backward()
        if (step + 1) % grad_accum == 0:
            opt.step()
            opt.zero_grad()
        last_loss = float(loss.detach()) * grad_accum
        if is_main and (step == 0 or (step + 1) % log_every == 0 or step + 1 == steps):
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            eta = (steps - step - 1) / rate if rate > 0 else float("inf")
            print(
                f"  step {step + 1:>5}/{steps} loss={last_loss:.4f} "
                f"{rate:.2f} it/s elapsed={elapsed:.0f}s eta={eta:.0f}s"
            )
    if steps % grad_accum != 0:
        opt.step()  # flush any accumulated grads from a partial final batch
        opt.zero_grad()
    net.controlnet.eval()
    return last_loss


@torch.no_grad()
def sample_flow_sd3(net: SD3LatentFlow, cond_latent, lam, n_steps=20, x0=None):
    """cond_latent: (B,16,h,w) VAE latent of the codec's decoded
    reconstruction. Returns pixel-space (B,3,H,W) in [0,1] -- same output
    contract as real/flow.py::sample_flow, so real/eval.py's downstream
    metrics code needs no changes beyond the model-call site."""
    B = cond_latent.shape[0]
    if x0 is None:
        x0 = torch.randn_like(cond_latent)
    x = x0
    sigmas = torch.linspace(1.0, 0.0, n_steps + 1, device=cond_latent.device)
    for i in range(n_steps):
        sigma = sigmas[i].expand(B)
        # positive: sigma decreasing toward data; cast to x's dtype for the
        # same reason as train_flow_sd3's lam_b/sigma_b cast above.
        d_sigma = (sigmas[i] - sigmas[i + 1]).to(x.dtype)
        v = net.velocity(x, sigma, cond_latent, lam)
        x = x - d_sigma * v  # target=x0-x1 (data->noise); step *toward* data
    return net.decode_latent(x)


def build_real_pool_latent(net: SD3LatentFlow, codec, patch_data, npool, device, crops_per_image=None, vae_batch=16):
    """Extends real/flow.py::build_real_pool: also VAE-encodes the pool's
    pixel crops (X) and the codec's decoded reconstruction of each crop's
    y_hat, batched to avoid VAE-encoding the whole pool in one forward pass."""
    X, Y, source_ids = build_real_pool(codec, patch_data, npool, device, crops_per_image=crops_per_image)
    with torch.no_grad():
        x_lat_chunks, ycond_lat_chunks = [], []
        for i in range(0, X.shape[0], vae_batch):
            x_lat_chunks.append(net.encode_pixels(X[i : i + vae_batch]))
            y_decoded = codec.decode(Y[i : i + vae_batch])
            ycond_lat_chunks.append(net.encode_pixels(y_decoded))
        X_latent = torch.cat(x_lat_chunks, dim=0)
        Ycond_latent = torch.cat(ycond_lat_chunks, dim=0)
    return X, Y, X_latent, Ycond_latent, source_ids


@torch.no_grad()
def evaluate_sd3(net, codec, X, X_latent, Ycond_latent, E, N, lam, n_steps, n_draws, knn, seed, device, same_image_only=False, source_ids=None):
    """Mirrors real/flow.py::evaluate's quick internal-pool eval (printed
    during training) -- same PSNR_sample/PSNR_mmse/condMMD/Div(Var_z)
    metrics, same kNN-based conditional-MMD construction."""
    g1.set_seed(seed + 1)
    Np = X.shape[0]
    idx = torch.randint(0, Np, (N,), device=device)
    cond, x_true = Ycond_latent[idx], X[idx]
    draws = torch.stack([sample_flow_sd3(net, cond, lam, n_steps) for _ in range(n_draws)])
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
# CLI -- same shape as real/flow.py::main() (quality, patch, npool, knn,
# lams, seeds, save_checkpoint_dir, embed), including the same DDP pattern
# for multi-GPU (torchrun --nproc_per_node=N): each rank builds its OWN
# local pool (independent .sample() calls, rank offset into the RNG seed)
# rather than building one pool on rank 0 and broadcasting it -- see
# real/flow.py's module-level comment for the full rationale. Unlike
# real/flow.py (which DDP-wraps its single `net` directly), SD3LatentFlow
# bundles frozen (transformer/vae) and trainable (controlnet/lambda_mlp)
# submodules behind non-forward() methods (`velocity`, `encode_pixels`, ...),
# so DDP wraps `net.controlnet`/`net.lambda_mlp` individually instead of the
# whole wrapper -- see the DDP-wrap block in main() below.
# ---------------------------------------------------------------------------


def main():
    import argparse
    import json
    import time

    from real.codec import CompressAICodec
    from real.data import PatchFolderDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--model_id", default=SD3_MODEL_ID)
    ap.add_argument(
        "--hf_token",
        default=None,
        help="Hugging Face access token -- stabilityai/stable-diffusion-3-"
        "medium-diffusers is gated, accept its license on huggingface.co "
        "first, then pass a token here (or set HF_TOKEN / run "
        "huggingface_hub.login() beforehand instead)",
    )
    ap.add_argument("--controlnet_layers", type=int, default=12)
    ap.add_argument(
        "--dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="compute dtype once weights are loaded. fp32 is the "
        "verified-safe default (see SD3LatentFlow's docstring/comment); "
        "bf16 roughly halves activation memory and is faster on "
        "Ampere/Ada+ tensor cores -- worth using on a GPU with real "
        "headroom (e.g. RTX 6000 Ada, A100) once the fp32 smoke test has "
        "confirmed the pipeline is otherwise correct",
    )
    ap.add_argument(
        "--hf_variant",
        default=None,
        help="HF repo file variant to download, e.g. 'fp16' -- halves the "
        "on-*disk* download size independently of --dtype (SD3-medium's "
        "full-precision files are ~31GB total, dominated by the T5-XXL "
        "text encoder; matters on small container disks). Set this "
        "whenever disk space is tighter than ~35GB free",
    )
    ap.add_argument("--arch", default="cheng2020-anchor")
    ap.add_argument("--quality", type=int, default=1)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--npool", type=int, default=2000)
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--embed", type=str, default="proj64")
    ap.add_argument("--same_image_only", action="store_true")
    ap.add_argument("--crops_per_image", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--N", type=int, default=4, help="batch size per step (SD3 is much heavier than UNetVelocity)")
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--n_steps", type=int, default=20)
    ap.add_argument("--n_draws", type=int, default=4)
    ap.add_argument("--eval_N", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", type=str, default=None, help="only used when not launched via torchrun")
    ap.add_argument("--out", type=str, default="results/g4_real_sd3.json")
    ap.add_argument("--save_checkpoint_dir", type=str, default=None)
    args = ap.parse_args()

    distributed = _is_distributed()
    if distributed:
        rank, world_size, local_rank, device = _setup_distributed()
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    if args.same_image_only and args.crops_per_image <= args.knn:
        old = args.crops_per_image
        args.crops_per_image = args.knn + 1
        if is_main:
            print(f"[warn] --crops_per_image={old} <= --knn={args.knn}; bumping to {args.crops_per_image}")

    if is_main:
        print(
            f"distributed={distributed} world_size={world_size} device={device} "
            f"backbone=sd3({args.model_id}) dtype={args.dtype} arch={args.arch} q={args.quality} "
            f"patch={args.patch} npool={args.npool} knn={args.knn} embed={args.embed} "
            f"same_image_only={args.same_image_only}"
        )

    codec = CompressAICodec(arch=args.arch, quality=args.quality, device=device)
    # each rank builds its own local pool from its own RNG seed offset, not
    # one shared/broadcast pool -- see the DDP comment block above main().
    data = PatchFolderDataset(args.train_root, patch=args.patch, device=device, seed=rank, max_images=args.max_images)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    # Deliberately NO barrier around this (unlike real/flow.py's small
    # compressai codec download): SD3's weights are large enough that a
    # first-ever download can take many minutes, and an earlier version of
    # this file that barriered here made rank 1 idle-wait on GPU for that
    # whole duration -- wasted GPU-hour on a paid box, and it also blew
    # NCCL's collective timeout outright (both ranks aborted with a 600s
    # ALLREDUCE timeout). huggingface_hub's own hf_hub_download already
    # takes a per-file lock, so concurrent from_pretrained calls to the same
    # repo across ranks on the same machine are safe without an extra
    # barrier -- let every rank download/build in parallel instead.
    if is_main:
        print(f"rank{rank}: loading/downloading SD3 backbone (can take a while on a cold HF cache)...")
    _load_t0 = time.time()
    net = SD3LatentFlow(
        model_id=args.model_id, controlnet_layers=args.controlnet_layers,
        device=device, dtype=dtype, hf_token=args.hf_token, hf_variant=args.hf_variant,
    )
    if is_main:
        print(f"rank{rank}: backbone loaded in {time.time() - _load_t0:.0f}s")

    if distributed:
        # only the trainable submodules need gradient sync; SD3LatentFlow
        # itself is never called via a single forward() (velocity() invokes
        # frozen transformer/controlnet calls directly), so DDP wraps these
        # two submodules individually rather than the whole wrapper -- see
        # the DDP comment block above main(). net.velocity()'s existing
        # `self.controlnet(...)`/`self.lambda_mlp(...)` calls work unchanged
        # against the DDP-wrapped versions (DDP is transparently callable).
        ddp_kwargs = {"device_ids": [local_rank]} if torch.cuda.is_available() else {}
        net.controlnet = torch.nn.parallel.DistributedDataParallel(net.controlnet, **ddp_kwargs)
        net.lambda_mlp = torch.nn.parallel.DistributedDataParallel(net.lambda_mlp, **ddp_kwargs)

    if is_main:
        print(
            f"trainable params: controlnet={sum(p.numel() for p in net.controlnet.parameters()):,} "
            f"lambda_mlp={sum(p.numel() for p in net.lambda_mlp.parameters()):,} "
            f"(frozen transformer={sum(p.numel() for p in net.transformer.parameters()):,})"
        )

    rows = []
    for seed in args.seeds:
        g1.set_seed(seed * 1000 + rank)  # distinct local pool/batches per rank
        if is_main:
            print(f"seed={seed}: building pool (npool={args.npool}, VAE-encoding in batches)...")
        _pool_t0 = time.time()
        X, Y, X_latent, Ycond_latent, source_ids = build_real_pool_latent(
            net, codec, data, args.npool, device,
            crops_per_image=args.crops_per_image if args.same_image_only else None,
        )
        if is_main:
            print(f"seed={seed}: pool built in {time.time() - _pool_t0:.0f}s, starting training...")
        y_shape = tuple(Y.shape[1:])
        embed = g1.make_embed(args.embed, y_shape, seed=seed, device=device)
        E = embed(Y)

        loss = train_flow_sd3(
            net, X, X_latent, Ycond_latent, E, args.lams, args.steps, args.N, args.knn,
            args.lr, device, args.same_image_only, source_ids, grad_accum=args.grad_accum,
            is_main=is_main,
        )
        if is_main:
            print(f"seed={seed} final training loss={loss:.4f}")

        if args.save_checkpoint_dir and is_main:
            os.makedirs(args.save_checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_checkpoint_dir, f"g4_real_sd3_seed{seed}.pt")
            net.save_trainable(ckpt_path)
            print(f"saved checkpoint -> {ckpt_path}")

        # eval only on rank 0, against rank 0's own local pool -- other
        # ranks' pools were only there to diversify training batches.
        if is_main:
            print(f"{'lam':>6} {'PSNR_samp':>10} {'PSNR_mmse':>10} {'condMMD':>12} {'Div(Var_z)':>12}")
            for lam in args.lams:
                ps, pm, cm, dv = evaluate_sd3(
                    net, codec, X, X_latent, Ycond_latent, E, args.eval_N, lam,
                    args.n_steps, args.n_draws, args.knn, seed, device,
                    args.same_image_only, source_ids,
                )
                print(f"{lam:>6.2f} {ps:>10.2f} {pm:>10.2f} {cm:>12.4e} {dv:>12.4e}")
                rows.append({"seed": seed, "lam": lam, "psnr_sample": ps, "psnr_mmse": pm, "cond_mmd": cm, "div": dv})

    if args.out and is_main:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        payload = {
            "config": {
                k: getattr(args, k)
                for k in [
                    "train_root", "max_images", "model_id", "controlnet_layers", "arch",
                    "quality", "patch", "npool", "knn", "embed", "same_image_only",
                    "steps", "N", "lams", "seeds",
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
