#!/usr/bin/env python
"""
Real frozen codec for the large-scale G4 track — replaces g1_pilot.TinyAE.

TinyAE was a 2-layer toy conv net with a hand-rolled round-to-grid
"quantizer" trained from scratch per run; it has no real bitstream, so G1/G4
could only report a synthetic "quantized-recon PSNR" as a proxy for rate.
CompressAICodec wraps a *pretrained, frozen* compressai model instead and
exposes the same two methods G4's training loop needs (`condition`, `decode`)
plus a `real_bpp` path that goes through the model's actual entropy coder —
closing the "no real rate axis" gap called out in docs/RESULTS.md.

Two separate paths, deliberately:
  * `condition`/`decode` — round(g_a(x)) / g_s(y_hat). Fast, differentiable
    end-to-end (no entropy coding), used for building the kNN conditional
    pool and as the flow's conditioning tensor. This mirrors exactly what
    TinyAE did (encode -> round -> decode) so g4_wflow's training loop is a
    drop-in swap.
  * `real_bpp` — model.compress()/decompress(), an actual arithmetic-coded
    bitstream (like real/baselines/run_msillm.py uses for MS-ILLM), for
    reporting the rate axis a reader can trust and for rate-matching against
    an MS-ILLM quality level.

Pick a compressai quality level whose `real_bpp` lands near the MS-ILLM
quality point you want to compare against (see TARGET_BPP in
real/baselines/run_msillm.py) -- e.g. quality=1 on cheng2020_anchor is a weak,
low-bitrate codec, leaving a genuine residual for the flow to model, same
spirit as TinyAE's --ae_qscale/--ae_zc weakening knobs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from compressai.zoo import image_models


class CompressAICodec(nn.Module):
    """Frozen wrapper around a pretrained compressai model.

    condition(x) -> y_hat : (B, M, h, w) quantized latent (spatial condition)
    decode(y_hat) -> x_hat : (B, 3, H, W) reconstruction in [0, 1]
    real_bpp(x) -> float   : true bits-per-pixel via the model's entropy coder
    """

    def __init__(
        self, arch: str = "cheng2020-anchor", quality: int = 1, device: str = "cpu"
    ):
        super().__init__()
        if arch not in image_models:
            raise ValueError(
                f"unknown compressai arch {arch!r}; "
                f"available: {sorted(image_models)}"
            )
        self.model = image_models[arch](quality=quality, pretrained=True).to(device)
        self.model.eval()
        self.model.update()  # builds the entropy-coding CDF tables
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.arch, self.quality = arch, quality

    @torch.no_grad()
    def condition(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model.g_a(x)
        return torch.round(y)

    @torch.no_grad()
    def decode(self, y_hat: torch.Tensor) -> torch.Tensor:
        return self.model.g_s(y_hat).clamp(0.0, 1.0)

    @torch.no_grad()
    def real_bpp(self, x: torch.Tensor) -> float:
        """True bpp for a single image (B=1) via the actual bitstream."""
        out = self.model.compress(x)
        n_bits = sum(len(s[0]) * 8 for s in out["strings"])
        return n_bits / (x.shape[-2] * x.shape[-1])

    @torch.no_grad()
    def quantized_recon_psnr(self, x: torch.Tensor) -> float:
        """Cheap proxy (no entropy coding) -- same role as g1_pilot's
        ae_recon_psnr, for a fast sanity check while picking a quality level."""
        y_hat = self.condition(x)
        x_hat = self.decode(y_hat)
        mse = (x_hat - x).pow(2).mean().clamp(min=1e-12)
        return float(-10 * torch.log10(mse))


def probe_quality_ladder(
    arch="cheng2020-anchor",
    qualities=(1, 2, 3, 4),
    image_path=None,
    H=256,
    device="cpu",
):
    """Print bpp/PSNR for each quality level -- run once to pick the quality
    level that rate-matches an MS-ILLM comparison point.

    IMPORTANT: pass --image_path to a real photo (Kodak/CLIC). Codecs are
    trained on natural-image statistics; i.i.d. random noise (the fallback
    if no image is given) is incompressible and wildly overstates bpp /
    understates PSNR relative to any real operating point -- it's a
    functional smoke test only, not a calibration reading.
    """
    if image_path is not None:
        from PIL import Image
        from torchvision.transforms import ToTensor

        x = ToTensor()(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        x = x[:, :, :H, :H]
    else:
        print(
            "[probe_quality_ladder] no --image_path given -- using random "
            "noise, functional check only (see docstring)"
        )
        x = torch.rand(1, 3, H, H, device=device)
    print(f"{'quality':>8} {'bpp':>8} {'psnr':>8}")
    for q in qualities:
        codec = CompressAICodec(arch=arch, quality=q, device=device)
        bpp = codec.real_bpp(x)
        psnr = codec.quantized_recon_psnr(x)
        print(f"{q:>8} {bpp:>8.4f} {psnr:>8.2f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="cheng2020-anchor")
    ap.add_argument("--qualities", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--image_path", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    probe_quality_ladder(
        args.arch, tuple(args.qualities), args.image_path, device=args.device
    )
