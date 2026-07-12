#!/usr/bin/env python
"""
G1 pilot — CWFC on small images with a SPATIALLY-STRUCTURED condition.

This is the next gate after the toy de-risk (derisk_cot.py). It lifts the exact
extended-cost conditional-Sinkhorn objective from the 2-D toy to real small
images, where the condition y_hat is now a QUANTIZED latent from a tiny AE (it
has spatial structure, like the transmitted code of a real codec).

Objective (unchanged from the toy, per batch {(x_i, y_hat_i)}):

    x_hat_i = G(z_i, y_hat_i)
    C_ij    = c_x(x_hat_i, x_j) + eta * c_y(E(y_hat_i), E(y_hat_j))
    P*      = Sinkhorn(C_normalised, eps)          (P detached, plan-weighted)
    L       = <P*, c_x>

    eta = 0    -> plain distribution matching  (perception endpoint, sharp)
    eta -> inf -> diagonal coupling            (per-sample MMSE, blurry)

What G1 must show to PASS (Phan 5 of idea.md):
  * a usable intermediate eta band exists (not a binary cliff) on IMAGES;
  * eta's DUAL ROLE reappears: below threshold, the condition term is a pairing
    regulariser that improves BOTH axes; above it, it is the D-P knob;
  * a SEMANTIC embedding of y_hat (spatial mean-pool) is >= random projection.

Three condition embeddings under test (the design variable):
    raw   : flatten y_hat, identity            (the high-dim cliff baseline)
    proj  : fixed random projection to D dims   (random-proj)
    pool  : per-channel global average pool     (SEMANTIC embedding of y_hat)

Metrics:
    fidelity  = PSNR from paired MSE  E||x_hat_i - x_i||^2   (higher = better)
    realism   = MMD on FIXED random Fourier features between {x_hat} and a fresh
                real batch {x} (lower = better; dependency-free image analogue of
                the toy's off-manifold distance)
    diversity = Var_z[G(z, y_hat)] at fixed condition (catches mode collapse:
                ~0 everywhere => the generator ignores its noise)

TWO controls that decide whether a D-P frontier is even MEASURABLE (added after
the CIFAR-100 null run, where the AE was too strong and aggregate-OT could not
force within-condition diversity):
    --ae_qscale / --ae_zc : WEAKEN the frozen AE so a real unresolvable residual
                exists. If AE quantized-recon already ~25 dB the generator hits
                the fidelity ceiling at eta=0 and the frontier collapses to a
                point; drop AE to ~18-20 dB (coarser qscale ~0.5, smaller zc) to
                open room for the knob and for the perception endpoint.
    --K       : draw K noise samples PER CONDITION. Aggregate-OT over the marginal
                {x_hat}<->{x} does NOT force diversity WITHIN a condition (a
                deterministic per-condition map already matches the marginal). With
                K>1 each y_hat is repeated K times and the extended-cost plan maps
                its K draws onto real neighbours (similar y_hat, different residual
                mode) via c_y, so z must be USED to cover them. K=1 == old behaviour.

Datasets:
    synth : procedurally generated small images with a KNOWN residual mode the
            coarse quantized condition cannot resolve -> a genuine D-P frontier,
            zero downloads. This is the smoke/sanity dataset.
    cifar : torchvision CIFAR-10. NO auto-download by default — reads data already
            present at --data_root (use download_cifar.py or a Kaggle dataset).
            Add --download if you still want torchvision to fetch it.
    cifar100 : reads the pickle 'train' directly (Kaggle fedesoriano/cifar100).

Runs on CPU. Use --smoke for a fast end-to-end check.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


# ---------------------------------------------------------------------------
# Log-domain Sinkhorn (uniform marginals) -> transport plan P  (same as toy)
# ---------------------------------------------------------------------------


def sinkhorn_plan(C, eps=0.05, iters=200):
    N, M = C.shape
    log_a = torch.full((N,), -math.log(N), device=C.device)
    log_b = torch.full((M,), -math.log(M), device=C.device)
    f = torch.zeros(N, device=C.device)
    g = torch.zeros(M, device=C.device)
    K = -C / eps
    for _ in range(iters):
        f = eps * log_a - eps * torch.logsumexp(K + g[None, :] / eps, dim=1)
        g = eps * log_b - eps * torch.logsumexp(K + f[:, None] / eps, dim=0)
    logP = K + f[:, None] / eps + g[None, :] / eps
    return torch.exp(logP)


# ---------------------------------------------------------------------------
# Realism proxy: MMD on FIXED random Fourier features  (image analogue of the
# toy's off-manifold distance; low = {x_hat} distribution matches real {x})
# ---------------------------------------------------------------------------


def mmd_rff(a, b, n_feat=512, gamma=None, seed=0):
    """
    RBF-kernel MMD^2 between two equal-N flattened image sets, estimated with
    fixed Random Fourier Features phi(x)=sqrt(2/D)cos(Wx+b), W,b frozen by seed.
    Lower = the generated distribution is closer to the real one.
    """
    device = a.device
    D = a.shape[1]
    if gamma is None:  # median heuristic on the real set
        m = min(128, a.shape[0])
        with torch.no_grad():
            d = torch.cdist(a[:m], a[:m])
            med = d[d > 0].median()
        gamma = 1.0 / (2.0 * med.pow(2).clamp(min=1e-8))
    g = torch.Generator().manual_seed(seed + 12345)
    W = (torch.randn(D, n_feat, generator=g) * math.sqrt(2.0 * float(gamma))).to(device)
    bph = (torch.rand(n_feat, generator=g) * 2 * math.pi).to(device)

    def phi(x):
        return math.sqrt(2.0 / n_feat) * torch.cos(x @ W + bph)

    return float((phi(a).mean(0) - phi(b).mean(0)).pow(2).sum())


def condition_diversity(G, y_hat, K=8):
    """
    Var_z[G(z, y_hat)] averaged over pixels and conditions: how much the output
    varies over noise draws at FIXED condition. ~0 at every eta => the generator
    ignores z (mode collapse); should be higher at low eta (samples the posterior)
    and shrink toward 0 as eta -> inf (deterministic MMSE).
    """
    with torch.no_grad():
        outs = torch.stack([G(y_hat).flatten(1) for _ in range(K)])  # (K,N,D)
        return float(outs.var(dim=0).mean())


# ---------------------------------------------------------------------------
# Synthetic small images with a KNOWN unresolvable residual mode
# ---------------------------------------------------------------------------


class SynthImages:
    """
    Each image is a smooth low-frequency base (set by a latent u) PLUS one of two
    high-frequency texture overlays (a binary mode m the coarse condition cannot
    resolve) PLUS small noise. A tiny quantized AE captures the base but loses
    the mode -> residual uncertainty -> a real distortion-perception frontier.
    """

    def __init__(self, H=16, ku=8, s=0.9, sigma=0.03, seed=0, device="cpu"):
        self.H, self.ku, self.s, self.sigma, self.device = H, ku, s, sigma, device
        g = torch.Generator().manual_seed(seed)
        # smooth base: linear map from u to a few low-freq 2-D cosine coeffs / colour
        self.Wbase = torch.randn(ku, 3 * 3, generator=g) / math.sqrt(ku)
        # two fixed high-freq texture patterns (the modes)
        yy, xx = torch.meshgrid(torch.linspace(0, math.pi, H),
                                torch.linspace(0, math.pi, H), indexing="ij")
        t0 = torch.sin(4 * xx) * torch.cos(4 * yy)
        t1 = torch.cos(5 * xx + 5 * yy)
        self.tex = torch.stack([t0, t1]).to(device)          # (2, H, W)
        self.to(device)

    def to(self, device):
        self.Wbase = self.Wbase.to(device)
        return self

    def _base(self, u):
        # u: (N, ku) -> 3 low-freq cosine coeffs per channel -> (N,3,H,W)
        H = self.H
        c = (u @ self.Wbase).view(-1, 3, 3)                  # (N,3,3) coeffs
        yy, xx = torch.meshgrid(torch.linspace(0, 1, H, device=self.device),
                                torch.linspace(0, 1, H, device=self.device),
                                indexing="ij")
        basis = torch.stack([torch.ones_like(xx),
                             torch.cos(math.pi * xx),
                             torch.cos(math.pi * yy)])        # (3,H,W)
        img = torch.einsum("ncj,jhw->nchw", c, basis)
        return torch.sigmoid(1.5 * img)

    def sample(self, n):
        u = torch.randn(n, self.ku, device=self.device)
        m = torch.randint(0, 2, (n,), device=self.device)
        base = self._base(u)                                 # (N,3,H,W) in (0,1)
        overlay = self.s * self.tex[m].unsqueeze(1)          # (N,1,H,W)
        x = (base + overlay + self.sigma *
             torch.randn(n, 3, self.H, self.H, device=self.device)).clamp(0, 1)
        return x


class _Wrap:
    """Wrap a fixed image pool (N,3,H,W) as a dataset with .sample(n)."""

    def __init__(self, pool):
        self.pool = pool

    def sample(self, n):
        idx = torch.randint(0, self.pool.shape[0], (n,))
        return self.pool[idx]


def _load_cifar100_images(root, max_n=5000):
    """
    Read CIFAR-100 images directly from the pickle 'train' file (Krizhevsky's
    original python version — the format Kaggle datasets like fedesoriano/cifar100
    use). It searches recursively for a file named 'train' under root, so pointing
    --data_root straight at /kaggle/input/... works without copying. Labels are
    ignored (the experiment only needs images). If only a tar.gz is present, it is
    extracted to ./data and read from there.
    Returns a float tensor (N,3,32,32) in [0,1].
    """
    import os
    import pickle
    import tarfile

    train_path = None
    for dirpath, _, files in os.walk(root):
        if "train" in files:
            train_path = os.path.join(dirpath, "train")
            break
    if train_path is None:
        # the dataset may only contain cifar-100-python.tar.gz -> extract to a writable dir
        tar = None
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if fn.endswith(".tar.gz"):
                    tar = os.path.join(dirpath, fn)
        if tar is not None:
            os.makedirs("./data", exist_ok=True)
            with tarfile.open(tar, "r:gz") as t:
                t.extractall("./data")
            return _load_cifar100_images("./data", max_n)
        raise FileNotFoundError(
            f"No pickle 'train' (or .tar.gz) found under {root}. "
            f"Check the --data_root path.")

    with open(train_path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    data = d[b"data"][:max_n]                      # (n, 3072) uint8, row-major RGB
    x = torch.from_numpy(data.reshape(-1, 3, 32, 32).copy()).float() / 255.0
    print(f"cifar100: doc {x.shape[0]} anh tu {train_path}")
    return x


def make_dataset(name, H, seed, device, root="./data", download=False):
    if name == "synth":
        return SynthImages(H=H, seed=seed, device=device)
    if name == "cifar":
        import os
        from torchvision import datasets, transforms
        if not download and not os.path.isdir(
                os.path.join(root, "cifar-10-batches-py")):
            raise FileNotFoundError(
                f"{root}/cifar-10-batches-py not found.\n"
                f"Download it first:  python download_cifar.py --root {root}\n"
                f"or on Kaggle: add the CIFAR-10 python dataset and extract the "
                f"tar.gz into {root}/, "
                f"or re-run with --download to let torchvision fetch it.")
        tf = transforms.Compose([transforms.Resize(H), transforms.ToTensor()])
        ds = datasets.CIFAR10(root=root, train=True, download=download,
                              transform=tf)
        imgs = torch.stack([ds[i][0] for i in range(min(len(ds), 5000))]).to(device)
        return _Wrap(imgs)
    if name == "cifar100":
        x = _load_cifar100_images(root)            # (N,3,32,32) in [0,1]
        if H != 32:
            x = F.interpolate(x, size=(H, H), mode="bilinear", align_corners=False)
        return _Wrap(x.to(device))
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Tiny quantized AE -> the spatial condition y_hat  (trained once, frozen)
# ---------------------------------------------------------------------------


class TinyAE(nn.Module):
    def __init__(self, ch=32, zc=4, qscale=1.0):
        super().__init__()
        self.zc = zc
        self.enc = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.GELU(),
            nn.Conv2d(ch, ch, 4, 2, 1), nn.GELU(),   # 4x downsample -> drops high-freq
            nn.Conv2d(ch, zc, 3, 1, 1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(zc, ch, 3, 1, 1), nn.GELU(),
            nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(ch, 3, 4, 2, 1),
        )
        # coarse quantization step -> loses the texture mode. SMALLER qscale =
        # coarser grid (step 1/qscale) = weaker AE = more residual for G to model.
        self.qscale = qscale

    def encode(self, x):
        return self.enc(x * 2 - 1)

    def quantize(self, z):
        return torch.round(z * self.qscale) / self.qscale

    def decode(self, z):
        return torch.sigmoid(self.dec(z))

    def condition(self, x):
        return self.quantize(self.encode(x))


def train_ae(ae, data, steps, N, lr, device):
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    ae.train()
    for _ in range(steps):
        x = data.sample(N)
        z = ae.encode(x)
        zq = z + (ae.quantize(z) - z).detach()   # STE through the round
        rec = ae.decode(zq)
        loss = F.mse_loss(rec, x)
        opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return float(loss.detach())


# ---------------------------------------------------------------------------
# Conditional generator  G(y_hat, noise) -> image
# ---------------------------------------------------------------------------


class CondGenImg(nn.Module):
    def __init__(self, zc=8, dz=4, ch=48):
        super().__init__()
        self.dz = dz
        self.net = nn.Sequential(
            nn.Conv2d(zc + dz, ch, 3, 1, 1), nn.GELU(),
            nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.GELU(),   # 2x
            nn.Conv2d(ch, ch, 3, 1, 1), nn.GELU(),
            nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.GELU(),   # 2x  -> matches AE 4x
            nn.Conv2d(ch, 3, 3, 1, 1),
        )

    def forward(self, y_hat):
        B, _, h, w = y_hat.shape
        z = torch.randn(B, self.dz, h, w, device=y_hat.device)
        return torch.sigmoid(self.net(torch.cat([y_hat, z], dim=1)))


# ---------------------------------------------------------------------------
# Condition embeddings E(y_hat) for c_y  (the design variable under test)
# ---------------------------------------------------------------------------


def make_embed(kind, y_shape, seed=0, device="cpu"):
    """y_shape = (C, h, w). Returns E: (B,C,h,w) -> (B,d)."""
    C, h, w = y_shape
    flat = C * h * w
    if kind == "raw":
        return lambda y: y.flatten(1)
    if kind.startswith("proj"):
        d = int(kind[4:])
        g = torch.Generator().manual_seed(seed + 777)
        W = (torch.randn(flat, d, generator=g) / math.sqrt(flat)).to(device)
        return lambda y: y.flatten(1) @ W
    if kind == "pool":                       # semantic: per-channel global avg
        return lambda y: y.mean(dim=(2, 3))
    raise ValueError(kind)


def _cy(e_a, e_b):
    """Normalised squared condition cost between two embedded condition sets."""
    cy = torch.cdist(e_a, e_b).pow(2)
    return cy / cy[cy > 0].mean().clamp(min=1e-9)


def cond_cost(y_hat, embed):
    """Square self condition cost (used by the training-free geom probe)."""
    return _cy(embed(y_hat), embed(y_hat))


# ---------------------------------------------------------------------------
# Q1-style geometry probe: coupling diagonality vs eta (no training)
# ---------------------------------------------------------------------------


def geom_probe(ae, data, embeds, etas, N, eps, device):
    x = data.sample(N)
    y_hat = ae.condition(x)
    y_shape = y_hat.shape[1:]
    # imperfect x_hat: decode a mode-swapped / noised code (worst case pairing)
    x_hat = (ae.decode(y_hat) + 0.1 * torch.randn_like(x)).clamp(0, 1)
    xf_hat, xf = x_hat.flatten(1), x.flatten(1)
    cx = torch.cdist(xf_hat, xf).pow(2) / xf.shape[1]
    cx = cx / cx[cx > 0].mean().clamp(min=1e-9)
    print(f"\n=== Q1 geometry (diag_mass vs eta) — condition y_hat shape {tuple(y_shape)} ===")
    print("  eta:  " + " ".join(f"{e:7.3g}" for e in etas))
    for name, kind in embeds:
        embed = make_embed(kind, y_shape, device=device)
        cy = cond_cost(y_hat, embed)
        off = cy[~torch.eye(N, dtype=bool, device=device)]
        conc = float(off.std() / off.mean())
        row = []
        for eta in etas:
            P = sinkhorn_plan(cx + eta * cy, eps=eps)
            row.append(float(torch.diagonal(P).sum()))
        print(f"  {name:>6} (conc {conc:.3f}): " + " ".join(f"{d:7.3f}" for d in row))


# ---------------------------------------------------------------------------
# Q2-style training frontier: train one generator per (embedding, eta, seed)
# ---------------------------------------------------------------------------


def _plan_and_cost(xf_a, xf_b, cy, eta, eps, iters=150):
    """Per-pixel MSE cost cx (differentiable) + detached extended-cost plan P."""
    cx = torch.cdist(xf_a, xf_b).pow(2) / xf_a.shape[1]
    with torch.no_grad():
        cx_n = cx / cx[cx > 0].mean().clamp(min=1e-9)
        P = sinkhorn_plan((cx_n + eta * cy).detach(), eps=eps, iters=iters)
    return P, cx


# ---------------------------------------------------------------------------
# Potential-based (NON-detached) Sinkhorn divergence  — the geomloss-style loss
# the G1 finding points to for the perception/diversity axis. Unlike the
# plan-detach <P, c_x> loss above (which only barycentrically matches means),
# gradient here flows through the entropic dual potentials, so the gen->gen term
# is a PROPER distribution divergence that actively penalises mode collapse.
# ---------------------------------------------------------------------------


def _ot_eps(C, eps, iters=100):
    """
    Entropic OT value in dual form, differentiable w.r.t. the cost C.

    Sinkhorn iterations run detached (cheap, no unrolled graph); a single final
    update through the grad-carrying C then makes the potentials depend on C, so
    autodiff recovers the correct envelope-theorem gradient  d OT / d C = P
    without holding the whole recursion in memory. Uniform marginals.
    """
    N, M = C.shape
    log_a = torch.full((N,), -math.log(N), device=C.device)
    log_b = torch.full((M,), -math.log(M), device=C.device)
    Cd = C.detach()
    with torch.no_grad():
        f = torch.zeros(N, device=C.device)
        g = torch.zeros(M, device=C.device)
        for _ in range(iters):
            f = eps * log_a - eps * torch.logsumexp((-Cd + g[None, :]) / eps, dim=1)
            g = eps * log_b - eps * torch.logsumexp((-Cd + f[:, None]) / eps, dim=0)
    # one differentiable pass through the real (grad-carrying) cost C
    f = eps * log_a - eps * torch.logsumexp((-C + g[None, :]) / eps, dim=1)
    g = eps * log_b - eps * torch.logsumexp((-C + f[:, None]) / eps, dim=0)
    return (torch.exp(log_a) * f).sum() + (torch.exp(log_b) * g).sum()


def _ext_cost(a, b, cy, eta):
    """Normalised image cost + eta * (detached) condition cost, differentiable in a,b."""
    cx = torch.cdist(a, b).pow(2) / a.shape[1]
    with torch.no_grad():
        scale = cx[cx > 0].mean().clamp(min=1e-9)
    return cx / scale + eta * cy


def sinkhorn_div_loss(xf_hat, xf, cy_ab, cy_aa, eta, eps, iters=100):
    """
    Debiased Sinkhorn divergence with the extended (image + condition) cost:

        S = OT_eps(x_hat, x) - 1/2 OT_eps(x_hat, x_hat)      (+ const OT(x, x))

    Zero iff the generated distribution equals the real one, so minimising it
    matches SPREAD as well as mean -> gives the diversity/perception endpoint a
    real gradient (the endpoint the plan-detach losses structurally collapse).
    cy_ab is the gen<->real condition cost, cy_aa the gen<->gen one (they differ
    once K>1, i.e. the generated batch is larger than the real batch).
    """
    ot_ab = _ot_eps(_ext_cost(xf_hat, xf, cy_ab, eta), eps, iters)
    ot_aa = _ot_eps(_ext_cost(xf_hat, xf_hat, cy_aa, eta), eps, iters)  # OT(x,x) const
    return ot_ab - 0.5 * ot_aa


def train_one(ae, data, embed_kind, eta, y_shape, steps, N, eps, lr, device, seed,
              loss_kind="debiased", debias_w=0.5, K=1):
    set_seed(seed)
    G = CondGenImg(zc=y_shape[0]).to(device)
    opt = torch.optim.Adam(G.parameters(), lr=lr)
    embed = make_embed(embed_kind, y_shape, seed=seed, device=device)
    for _ in range(steps):
        x = data.sample(N)                                # real batch (N)
        y = ae.condition(x)
        # K noise draws per condition: repeat each y_hat K times so the generated
        # batch is (N*K), each row an independent z. K=1 -> identical to before.
        y_gen = y if K == 1 else y.repeat_interleave(K, dim=0)
        x_hat = G(y_gen)                                  # (N*K,3,H,W), indep z
        xf_hat, xf = x_hat.flatten(1), x.flatten(1)
        with torch.no_grad():
            e_gen = embed(y_gen)                          # (N*K,d)
            e_real = e_gen if K == 1 else embed(y)        # (N,d)
            cy_gr = _cy(e_gen, e_real)                    # gen x real  (N*K,N)
            need_gg = loss_kind in ("debiased", "potential")
            cy_gg = _cy(e_gen, e_gen) if need_gg else None   # gen x gen (N*K,N*K)
        if loss_kind == "potential":
            # NON-detached Sinkhorn divergence: gradient flows through the dual
            # potentials, so the gen->gen term is a proper distribution divergence
            # (defeats collapse by matching spread, not just the mean). The fix
            # the G1 finding localises to G4 / W-Flow generative-OT machinery.
            loss = sinkhorn_div_loss(xf_hat, xf, cy_gr, cy_gg, eta, eps)
        else:
            # gen -> real transport (the fidelity-driving term, == crude A1 loss)
            P_gr, cx_gr = _plan_and_cost(xf_hat, xf, cy_gr, eta, eps)
            loss = (P_gr * cx_gr).sum()
            if loss_kind == "debiased":
                # -1/2 * gen->gen transport: the Sinkhorn-divergence debiasing term.
                # It rewards SPREAD among generated samples -> defeats the regress-to-
                # mean collapse of squared-cost matching -> restores the perception
                # (diverse-sampling) endpoint at low eta. (W-Flow gen-to-gen map.)
                P_gg, cx_gg = _plan_and_cost(xf_hat, xf_hat.detach(), cy_gg, eta, eps)
                loss = loss - debias_w * (P_gg * cx_gg).sum()
        opt.zero_grad(); loss.backward(); opt.step()

    # eval  (single sample per condition -> paired PSNR / MMD; diversity over K=8 z)
    with torch.no_grad():
        x = data.sample(N)
        y_hat = ae.condition(x)
        x_hat = G(y_hat)
        mse = float((x_hat - x).pow(2).mean())
        psnr = -10 * math.log10(mse) if mse > 0 else 99.0
        x_real = data.sample(N)
        mmd = mmd_rff(x_hat.flatten(1), x_real.flatten(1), seed=seed)
    div = condition_diversity(G, y_hat, K=8)
    return psnr, mse, mmd, div


def frontier(ae, data, embeds, etas, y_shape, steps, N, eps, lr, device, seeds,
             loss_kind="debiased", debias_w=0.5, K=1):
    print(f"\n=== Q2 frontier (PSNR up=fidelity, MMD down=realism, Div=Var_z) — "
          f"steps={steps}, N={N}, K={K}, seeds={list(seeds)}, loss={loss_kind} ===")
    out = {}
    for name, kind in embeds:
        print(f"\n embedding = {name}")
        print(f"{'eta':>8} {'PSNR(dB)':>10} {'MMD':>12} {'Div(Var_z)':>12}")
        rows = []
        for eta in etas:
            ps, md, dv = [], [], []
            for sd in seeds:
                psnr, _, mmd, div = train_one(ae, data, kind, eta, y_shape,
                                              steps, N, eps, lr, device, sd,
                                              loss_kind=loss_kind, debias_w=debias_w,
                                              K=K)
                ps.append(psnr); md.append(mmd); dv.append(div)
            row = {"eta": eta, "psnr": float(np.mean(ps)), "psnr_std": float(np.std(ps)),
                   "mmd": float(np.mean(md)), "div": float(np.mean(dv))}
            rows.append(row)
            print(f"{eta:>8.3g} {np.mean(ps):>10.2f} {np.mean(md):>12.4e} "
                  f"{np.mean(dv):>12.4e}")
        out[name] = rows
    print("\nRead: within one embedding, PSNR should RISE and MMD FALL-then-RISE")
    print("as eta grows (dual role: mid-eta best realism, high-eta best fidelity).")
    print("Div(Var_z) should be high at low eta and shrink toward 0 as eta->inf")
    print("(deterministic MMSE). Div~0 at ALL eta = generator ignores z = collapse.")
    print("A monotone usable band (not a cliff) on images = mechanism survives to")
    print("real spatial conditions. 'pool' (semantic) should be >= 'proj'.")
    return out


# ---------------------------------------------------------------------------
# Multi-GPU: the (embedding, eta, seed) training runs are fully independent ->
# split jobs round-robin across GPUs, one process per GPU (spawn).
# NOT DataParallel: the model is tiny, so splitting jobs is the right parallelism.
# ---------------------------------------------------------------------------


def _rebuild_data(data_spec, device):
    kind = data_spec[0]
    if kind == "synth":
        return SynthImages(H=data_spec[1], seed=0, device=device)
    if kind == "pool":
        return _Wrap(data_spec[1].to(device))
    raise ValueError(kind)


def _gpu_worker(dev, jobs, ae_state, data_spec, cfg, out_q):
    """Runs in a child process: rebuild AE + data on this GPU, then train."""
    try:
        data = _rebuild_data(data_spec, dev)
        ae = TinyAE(zc=cfg["ae_zc"], qscale=cfg["ae_qscale"]).to(dev)
        ae.load_state_dict(ae_state)
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        for (name, kind, eta, sd) in jobs:
            psnr, _, mmd, div = train_one(
                ae, data, kind, eta, tuple(cfg["y_shape"]), cfg["steps"],
                cfg["N"], cfg["eps"], cfg["lr"], dev, sd,
                loss_kind=cfg["loss_kind"], debias_w=cfg["debias_w"], K=cfg["K"])
            out_q.put(("ok", name, eta, sd, psnr, mmd, div, dev))
    except Exception as e:
        out_q.put(("err", repr(e), dev, None, None, None, None, None))


def frontier_parallel(ae, data, embeds, etas, y_shape, steps, N, eps, lr,
                      devices, seeds, loss_kind="debiased", debias_w=0.5, K=1,
                      H=16, ae_zc=4, ae_qscale=1.0):
    import torch.multiprocessing as mp

    if isinstance(data, SynthImages):
        data_spec = ("synth", H)
    else:
        data_spec = ("pool", data.pool.cpu())
    ae_state = {k: v.cpu() for k, v in ae.state_dict().items()}
    cfg = dict(y_shape=list(y_shape), steps=steps, N=N, eps=eps, lr=lr,
               loss_kind=loss_kind, debias_w=debias_w, K=K,
               ae_zc=ae_zc, ae_qscale=ae_qscale)

    jobs = [(name, kind, eta, sd)
            for name, kind in embeds for eta in etas for sd in seeds]
    chunks = [jobs[i::len(devices)] for i in range(len(devices))]
    print(f"\n=== Q2 frontier PARALLEL on {devices} — {len(jobs)} jobs "
          f"(steps={steps}, N={N}, K={K}, loss={loss_kind}) ===")

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for dev, ch in zip(devices, chunks):
        p = ctx.Process(target=_gpu_worker,
                        args=(dev, ch, ae_state, data_spec, cfg, q))
        p.start()
        procs.append(p)

    res = {}
    for i in range(len(jobs)):
        msg = q.get()
        if msg[0] == "err":
            for p in procs:
                p.terminate()
            raise RuntimeError(f"worker {msg[2]} error: {msg[1]}")
        _, name, eta, sd, psnr, mmd, div, dev = msg
        res.setdefault((name, eta), []).append((psnr, mmd, div))
        print(f"  [{i+1:>3}/{len(jobs)}] {dev}  {name:>6} eta={eta:<7g} seed={sd} "
              f"-> PSNR={psnr:6.2f}  MMD={mmd:.3e}  Div={div:.3e}")
    for p in procs:
        p.join()

    # aggregate results back into frontier()'s format
    out = {}
    for name, _ in embeds:
        print(f"\n embedding = {name}")
        print(f"{'eta':>8} {'PSNR(dB)':>10} {'MMD':>12} {'Div(Var_z)':>12}")
        rows = []
        for eta in etas:
            vals = res[(name, eta)]
            ps = [v[0] for v in vals]; md = [v[1] for v in vals]
            dv = [v[2] for v in vals]
            rows.append({"eta": eta, "psnr": float(np.mean(ps)),
                         "psnr_std": float(np.std(ps)),
                         "mmd": float(np.mean(md)), "div": float(np.mean(dv))})
            print(f"{eta:>8.3g} {np.mean(ps):>10.2f} {np.mean(md):>12.4e} "
                  f"{np.mean(dv):>12.4e}")
        out[name] = rows
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synth", "cifar", "cifar100"],
                    default="synth")
    ap.add_argument("--data_root", type=str, default="./data",
                    help="directory holding cifar-10-batches-py (pre-downloaded data)")
    ap.add_argument("--download", action="store_true",
                    help="let torchvision download CIFAR-10 (OFF by default; "
                         "use download_cifar.py to fetch it first)")
    ap.add_argument("--mode", choices=["geom", "train", "both"], default="both")
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--ae_steps", type=int, default=800)
    ap.add_argument("--ae_zc", type=int, default=4,
                    help="AE latent channels. Smaller => weaker AE => more residual "
                         "for a frontier (the 2nd lever to lower AE recon).")
    ap.add_argument("--ae_qscale", type=float, default=1.0,
                    help="AE quantization fineness (step = 1/qscale). SMALLER => coarser "
                         "=> weaker AE. Set ~0.5 to pull AE recon down to ~18-20 dB.")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--K", type=int, default=1,
                    help="z draws per condition. >1 forces WITHIN-condition diversity "
                         "(cost: the generated batch becomes N*K).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--etas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0])
    ap.add_argument("--embeds", type=str, nargs="+",
                    default=["raw", "proj8", "pool"])
    ap.add_argument("--loss", choices=["debiased", "crude", "potential"],
                    default="debiased",
                    help="crude = A1 plan-weighted P-detach (regress-to-mean baseline); "
                         "debiased = adds detached gen-gen debias term; "
                         "potential = NON-detached Sinkhorn divergence (geomloss-style, "
                         "gradient through potentials -> best shot at the diversity axis)")
    ap.add_argument("--debias_w", type=float, default=0.5,
                    help="weight of the gen-gen debiasing term (raise if diversity "
                         "stays collapsed at full budget)")
    ap.add_argument("--devices", type=str, nargs="+", default=None,
                    help="devices to train on in parallel, e.g. cuda:0 cuda:1. "
                         "Default: automatically use ALL visible GPUs (or cpu).")
    ap.add_argument("--out", type=str, default=None,
                    help="path to save results JSON (e.g. g1_result.json)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast end-to-end run (overrides sizes)")
    args = ap.parse_args()

    if args.smoke:
        args.H, args.N, args.steps, args.ae_steps = 16, 128, 250, 300
        args.seeds = [0]
        args.etas = [0.0, 0.3, 1.0, 3.0, 30.0]

    if args.devices is not None:
        devices = args.devices
    elif torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]
    device = devices[0]          # AE + geom probe run on the first device
    print(f"devices={devices}  dataset={args.dataset}  H={args.H}  "
          f"embeds={args.embeds}  etas={args.etas}  K={args.K}  "
          f"ae_zc={args.ae_zc} ae_qscale={args.ae_qscale}")

    set_seed(0)
    data = make_dataset(args.dataset, args.H, seed=0, device=device,
                        root=args.data_root, download=args.download)

    ae = TinyAE(zc=args.ae_zc, qscale=args.ae_qscale).to(device)
    rec = train_ae(ae, data, steps=args.ae_steps, N=args.N, lr=1e-3, device=device)
    with torch.no_grad():
        x = data.sample(args.N)
        y_hat = ae.condition(x)
        y_shape = tuple(y_hat.shape[1:])
        ae_psnr = -10 * math.log10(float((ae.decode(y_hat) - x).pow(2).mean()))
    print(f"tiny AE trained: recon mse={rec:.4f}, quantized-recon PSNR={ae_psnr:.2f} dB, "
          f"condition y_hat shape={y_shape}")

    embeds = [(e, e) for e in args.embeds]

    if args.mode in ("geom", "both"):
        geom_probe(ae, data, embeds, args.etas, args.N, args.eps, device)
    results = None
    if args.mode in ("train", "both"):
        if len(devices) > 1:
            results = frontier_parallel(ae, data, embeds, args.etas, y_shape,
                                        args.steps, args.N, args.eps, args.lr,
                                        devices, args.seeds,
                                        loss_kind=args.loss,
                                        debias_w=args.debias_w, K=args.K, H=args.H,
                                        ae_zc=args.ae_zc, ae_qscale=args.ae_qscale)
        else:
            results = frontier(ae, data, embeds, args.etas, y_shape, args.steps,
                               args.N, args.eps, args.lr, device, args.seeds,
                               loss_kind=args.loss, debias_w=args.debias_w, K=args.K)

    if args.out and results is not None:
        import json
        payload = {"config": {k: getattr(args, k) for k in
                              ["dataset", "H", "N", "steps", "ae_steps", "ae_zc",
                               "ae_qscale", "eps", "lr", "K", "seeds", "etas",
                               "embeds", "loss", "debias_w"]},
                   "ae_recon_psnr": ae_psnr, "y_shape": list(y_shape),
                   "frontier": results}
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nsaved results -> {args.out}")


if __name__ == "__main__":
    main()
