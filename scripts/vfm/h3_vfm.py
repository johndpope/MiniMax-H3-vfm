#!/usr/bin/env python3
"""Isolated VFM validation path for MiniMax-H3. No SCD.

Claim
-----
Conditioning is initial noise, not a sampling-path trick.
  y  = (text, optional refs)          # observation
  z  ~ qφ(z | y)                      # packed video+audio noise
  x  = fθ(z, y, t≈1 → 0)              # frozen H3 DiT as the flow map
  x ≈ x0 in 1 (or few) NFE

This file is the cheapest falsifier of that claim:

  1. ToyPackedMap  — tiny packed AV velocity field with H3's two σ-shifts.
                     Swap for a real MiniMaxH3 forward later; the adapter
                     interface does not change.
  2. H3NoiseAdapter — qφ, dual heads (24-ch video / 32-ch audio).
  3. vfm_loss / one_step — train + measure.

qφ is diagonal Gaussian, same family H3 samples at t=1:
  z = μ(y) + exp(logσ(y)) ⊙ ε,  ε ~ N(0, I)

Run
---
    WANDB_MODE=offline python scripts/vfm/h3_vfm.py
    WANDB_MODE=offline python scripts/vfm/h3_vfm.py --steps 200 --device cuda

Success bar (toy): recon drops while KL stays bounded; audio recon does not
explode relative to video (the dual-clock failure mode).
Fail bar: adapter μ collapses to 0 and you still need many NFEs on the map.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

H3_TEXT_DIM = 5120
H3_VIDEO_CH = 24
H3_AUDIO_CH = 32
H3_VIDEO_SHIFT = 12.0
H3_AUDIO_SHIFT = 3.0

TASK = {"t2va": 0, "i2va": 1, "fl2va": 2, "ref2va": 3, "talking_head": 4}


def shift_sigma(t: torch.Tensor, shift: float) -> torch.Tensor:
    # t = 1 (noise) → 0 (data). H3 released shifts: video 12, audio 3.
    return shift * t / (1.0 + (shift - 1.0) * t)


def kl_to_standard_normal(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mu.pow(2) + torch.exp(2 * log_sigma) - 2 * log_sigma - 1).mean()


# ---------------------------------------------------------------------------
# Adapter  qφ(z | y)
# ---------------------------------------------------------------------------

class _Pos(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        half = max(dim // 6, 1)
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, pos):  # [B,S,3]
        ang = pos.unsqueeze(-1) * self.freqs
        enc = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)
        if enc.shape[-1] < self.dim:
            enc = F.pad(enc, (0, self.dim - enc.shape[-1]))
        return enc[..., : self.dim]


class _Block(nn.Module):
    def __init__(self, h, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(h)
        self.sa = nn.MultiheadAttention(h, heads, batch_first=True)
        self.n2 = nn.LayerNorm(h)
        self.ca = nn.MultiheadAttention(h, heads, batch_first=True)
        self.n3 = nn.LayerNorm(h)
        self.ff = nn.Sequential(nn.Linear(h, h * 4), nn.GELU(), nn.Linear(h * 4, h))

    def forward(self, x, kv, pad=None):
        h = self.n1(x)
        x = x + self.sa(h, h, h, need_weights=False)[0]
        h = self.n2(x)
        x = x + self.ca(h, kv, kv, key_padding_mask=pad, need_weights=False)[0]
        return x + self.ff(self.n3(x))


class H3NoiseAdapter(nn.Module):
    """Per-token diagonal Gaussian qφ(z|y): μ, logσ for video and audio rows."""

    def __init__(self, text_dim=H3_TEXT_DIM, hidden=256, heads=4, layers=2, pos_dim=128):
        super().__init__()
        self.pos = _Pos(pos_dim)
        self.task = nn.Embedding(len(TASK), 64)
        self.mod = nn.Embedding(2, 64)
        self.in_proj = nn.Linear(pos_dim + 128, hidden)
        self.text_proj = nn.Linear(text_dim, hidden)
        self.blocks = nn.ModuleList([_Block(hidden, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.v_mu = nn.Linear(hidden, H3_VIDEO_CH)
        self.v_log = nn.Linear(hidden, H3_VIDEO_CH)
        self.a_mu = nn.Linear(hidden, H3_AUDIO_CH)
        self.a_log = nn.Linear(hidden, H3_AUDIO_CH)
        for lin in (self.v_mu, self.v_log, self.a_mu, self.a_log):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, text, text_mask, pos, task, modality):
        B, S, _ = pos.shape
        x = torch.cat(
            [
                self.pos(pos.float()),
                self.task(task).unsqueeze(1).expand(-1, S, -1),
                self.mod(modality),
            ],
            dim=-1,
        )
        x = self.in_proj(x)
        kv = self.text_proj(text.float())
        pad = None if text_mask is None else ~text_mask.bool()
        for blk in self.blocks:
            x = blk(x, kv, pad)
        x = self.norm(x)
        v_sel = (modality == 0).unsqueeze(-1).to(x.dtype)
        a_sel = (modality == 1).unsqueeze(-1).to(x.dtype)
        return {
            "v_mu": self.v_mu(x) * v_sel,
            "v_log": self.v_log(x).clamp(-1, 2) * v_sel,
            "a_mu": self.a_mu(x) * a_sel,
            "a_log": self.a_log(x).clamp(-1, 2) * a_sel,
        }

    def sample(self, text, text_mask, pos, task, modality, temperature=1.0):
        o = self.forward(text, text_mask, pos, task, modality)
        z_v = o["v_mu"] + torch.exp(o["v_log"]) * torch.randn_like(o["v_mu"]) * temperature
        z_a = o["a_mu"] + torch.exp(o["a_log"]) * torch.randn_like(o["a_mu"]) * temperature
        o["kl_v"] = kl_to_standard_normal(o["v_mu"], o["v_log"])
        o["kl_a"] = kl_to_standard_normal(o["a_mu"], o["a_log"])
        return z_v, z_a, o


# ---------------------------------------------------------------------------
# Frozen map stand-in. Replace `velocity()` with real H3 DiT later.
# ---------------------------------------------------------------------------

class ToyPackedMap(nn.Module):
    """Deterministic packed AV velocity field.

    Mimics H3's contract, not its quality:
      video tokens [B, Sv, 24], audio tokens [B, Sa, 32]
      two σ-shifts, one forward, velocity ≈ x0 - z at t=1.

    Real swap:
        class RealH3Map:
            def velocity(self, z_v, z_a, text, t):
                return dit(... packed sequence ..., t_video=shift(t,12), t_audio=shift(t,3))
    """

    def __init__(self, text_dim=H3_TEXT_DIM):
        super().__init__()
        self.v_from_z = nn.Linear(H3_VIDEO_CH, H3_VIDEO_CH)
        self.a_from_z = nn.Linear(H3_AUDIO_CH, H3_AUDIO_CH)
        self.v_from_text = nn.Linear(text_dim, H3_VIDEO_CH)
        self.a_from_text = nn.Linear(text_dim, H3_AUDIO_CH)
        nn.init.eye_(self.v_from_z.weight)
        nn.init.eye_(self.a_from_z.weight)
        nn.init.zeros_(self.v_from_z.bias)
        nn.init.zeros_(self.a_from_z.bias)

    def velocity(self, z_v, z_a, text, t):
        """Imperfect map: x̂ = α z + (1-α) text_bias. Better z ⇒ better x̂.

        A perfect -z + x0 velocity would make the adapter irrelevant (any z
        reconstructs). H3 at 1 NFE is not that map. Keep a residual on z.
        """
        pooled = text.mean(dim=1)
        sv = shift_sigma(t, H3_VIDEO_SHIFT).view(-1, 1, 1)
        sa = shift_sigma(t, H3_AUDIO_SHIFT).view(-1, 1, 1)
        # residual mix: most of the 1-step output is still z (α≈0.85)
        alpha_v = 0.85 * sv.clamp(max=1.0)
        alpha_a = 0.85 * sa.clamp(max=1.0)
        bias_v = self.v_from_text(pooled).unsqueeze(1)
        bias_a = self.a_from_text(pooled).unsqueeze(1)
        x_v = alpha_v * self.v_from_z(z_v) + (1.0 - alpha_v) * torch.tanh(bias_v)
        x_a = alpha_a * self.a_from_z(z_a) + (1.0 - alpha_a) * torch.tanh(bias_a)
        return x_v - z_v, x_a - z_a

    @torch.no_grad()
    def one_step(self, z_v, z_a, text, t=None):
        if t is None:
            t = torch.ones(z_v.shape[0], device=z_v.device)
        v_v, v_a = self.velocity(z_v, z_a, text, t)
        return z_v + v_v, z_a + v_a


class RealH3Map:
    """Adapter-facing wrapper. Fill in when you have a loaded FL2VA/Ref2VA module.

    Expected: `dit` exposes the same kwargs Fizgig / Comfy use
    (video_latent, audio_latent, text_embeds, t) and returns packed velocity
    that you split back into video / audio heads.
    """

    def __init__(self, dit):
        self.dit = dit
        self.dit.eval()
        for p in self.dit.parameters():
            p.requires_grad_(False)

    def velocity(self, z_v, z_a, text, t):
        raise NotImplementedError(
            "Wire your loaded MiniMax H3 here: pack z_v/z_a + text, "
            "call dit once with video shift 12 / audio shift 3, split velocity."
        )


# ---------------------------------------------------------------------------
# Batch + loss
# ---------------------------------------------------------------------------

@dataclass
class PackedBatch:
    text: torch.Tensor          # [B, L, text_dim]
    text_mask: torch.Tensor     # [B, L]
    pos: torch.Tensor           # [B, S, 3]  S = Sv + Sa
    modality: torch.Tensor      # [B, S] 0 video, 1 audio
    task: torch.Tensor          # [B]
    x0_v: torch.Tensor          # [B, Sv, 24]
    x0_a: torch.Tensor          # [B, Sa, 32]
    Sv: int
    Sa: int


def make_synthetic_batch(B=4, Sv=16, Sa=8, L=32, text_dim=H3_TEXT_DIM, device="cpu"):
    """Structured dummy data: video x0 depends on text + (t,h,w); audio on text + t only."""
    text = torch.randn(B, L, text_dim, device=device)
    text_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    pos_v = torch.stack(
        [
            torch.arange(Sv, device=device).float() / max(Sv - 1, 1),
            (torch.arange(Sv, device=device) // max(Sv // 4, 1)).float() / 4,
            (torch.arange(Sv, device=device) % max(Sv // 4, 1)).float() / 4,
        ],
        dim=-1,
    )
    pos_a = torch.stack(
        [
            torch.arange(Sa, device=device).float() / max(Sa - 1, 1),
            torch.zeros(Sa, device=device),
            torch.zeros(Sa, device=device),
        ],
        dim=-1,
    )
    pos = torch.cat([pos_v, pos_a], dim=0).unsqueeze(0).expand(B, -1, -1).contiguous()
    modality = torch.cat(
        [
            torch.zeros(B, Sv, dtype=torch.long, device=device),
            torch.ones(B, Sa, dtype=torch.long, device=device),
        ],
        dim=1,
    )
    task = torch.zeros(B, dtype=torch.long, device=device)
    # x0 is a linear function of pooled text — adapter can learn the matching z
    pooled = text.mean(dim=1)
    x0_v = torch.tanh(pooled[:, :H3_VIDEO_CH]).unsqueeze(1) + 0.15 * pos_v.mean(-1).view(1, Sv, 1)
    x0_v = x0_v.expand(B, Sv, H3_VIDEO_CH).contiguous()
    x0_a = torch.tanh(pooled[:, :H3_AUDIO_CH]).unsqueeze(1) + 0.15 * pos_a[:, 0].view(1, Sa, 1)
    x0_a = x0_a.expand(B, Sa, H3_AUDIO_CH).contiguous()
    return PackedBatch(text, text_mask, pos, modality, task, x0_v, x0_a, Sv, Sa)


def split_rows(z_v_rows, z_a_rows, Sv, Sa):
    return z_v_rows[:, :Sv], z_a_rows[:, Sv : Sv + Sa]


def vfm_loss(adapter, fmap, batch: PackedBatch, kl_w=1e-3, audio_kl_w=3e-3):
    z_rows_v, z_rows_a, stats = adapter.sample(
        batch.text, batch.text_mask, batch.pos, batch.task, batch.modality
    )
    z_v, _ = split_rows(z_rows_v, z_rows_v, batch.Sv, batch.Sa)
    _, z_a = split_rows(z_rows_a, z_rows_a, batch.Sv, batch.Sa)
    t = torch.ones(batch.text.shape[0], device=batch.text.device)
    v_v, v_a = fmap.velocity(z_v, z_a, batch.text, t)
    x_v, x_a = z_v + v_v, z_a + v_a
    recon_v = (x_v - batch.x0_v).square().mean()
    recon_a = (x_a - batch.x0_a).square().mean()
    loss = recon_v + recon_a + kl_w * stats["kl_v"] + audio_kl_w * stats["kl_a"]
    return loss, {
        "loss": float(loss.detach()),
        "recon_v": float(recon_v.detach()),
        "recon_a": float(recon_a.detach()),
        "kl_v": float(stats["kl_v"].detach()),
        "kl_a": float(stats["kl_a"].detach()),
        "z_v_std": float(z_v.std().detach()),
        "z_a_std": float(z_a.std().detach()),
        "mu_v_norm": float(stats["v_mu"].norm(dim=-1).mean().detach()),
        "mu_a_norm": float(stats["a_mu"].norm(dim=-1).mean().detach()),
    }


@torch.no_grad()
def evaluate(adapter, fmap, batch):
    adapter.eval()
    _, metrics = vfm_loss(adapter, fmap, batch)
    # Gaussian-init baseline on the same map (the thing VFM has to beat).
    z_v = torch.randn_like(batch.x0_v)
    z_a = torch.randn_like(batch.x0_a)
    t = torch.ones(batch.text.shape[0], device=batch.text.device)
    g_v, g_a = fmap.one_step(z_v, z_a, batch.text, t)
    metrics["base_recon_v"] = float((g_v - batch.x0_v).square().mean())
    metrics["base_recon_a"] = float((g_a - batch.x0_a).square().mean())
    adapter.train()
    return metrics


def train(steps=150, lr=2e-3, device="cpu", seed=0, wandb_run=None):
    torch.manual_seed(seed)
    adapter = H3NoiseAdapter(text_dim=H3_TEXT_DIM).to(device)
    fmap = ToyPackedMap(text_dim=H3_TEXT_DIM).to(device)
    fmap.eval()
    for p in fmap.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr)
    batch = make_synthetic_batch(device=device)
    history = []
    for i in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss, m = vfm_loss(adapter, fmap, batch)
        loss.backward()
        opt.step()
        if i == 1 or i % 25 == 0 or i == steps:
            ev = evaluate(adapter, fmap, batch)
            history.append((i, ev))
            print(
                f"step {i:4d}  recon_v {ev['recon_v']:.4f} (base {ev['base_recon_v']:.4f})  "
                f"recon_a {ev['recon_a']:.4f} (base {ev['base_recon_a']:.4f})  "
                f"kl_v {ev['kl_v']:.4f}  kl_a {ev['kl_a']:.4f}  "
                f"||μ_v|| {ev['mu_v_norm']:.4f}  ||μ_a|| {ev['mu_a_norm']:.4f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({**ev, "step": i}, step=i)
        elif wandb_run is not None:
            wandb_run.log(m, step=i)
    ok_v = history[-1][1]["recon_v"] < 0.85 * history[-1][1]["base_recon_v"]
    ok_a = history[-1][1]["recon_a"] < 0.85 * history[-1][1]["base_recon_a"]
    passed = bool(ok_v and ok_a)
    print(
        "PASS" if passed else "FAIL",
        "— adapter 1-step beats Gaussian init on both streams" if passed
        else "— adapter did not beat Gaussian 1-step (claim not validated on toy map)",
    )
    if wandb_run is not None:
        wandb_run.summary["pass"] = passed
        wandb_run.summary["ok_v"] = bool(ok_v)
        wandb_run.summary["ok_a"] = bool(ok_a)
        wandb_run.summary["final_recon_v"] = history[-1][1]["recon_v"]
        wandb_run.summary["final_recon_a"] = history[-1][1]["recon_a"]
        wandb_run.summary["base_recon_v"] = history[-1][1]["base_recon_v"]
        wandb_run.summary["base_recon_a"] = history[-1][1]["base_recon_a"]
    return adapter, fmap, history


def main():
    ap = argparse.ArgumentParser(description="Isolated VFM validation (no SCD, no 33B)")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="h3-vfm")
    ap.add_argument("--wandb-name", default="toy-packed-map")
    args = ap.parse_args()

    os.environ.setdefault("WANDB_MODE", "offline")
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        mode=os.environ.get("WANDB_MODE", "offline"),
        config={
            "steps": args.steps,
            "lr": args.lr,
            "qphi": "diagonal_gaussian",
            "device": args.device,
            "seed": args.seed,
            "text_dim": H3_TEXT_DIM,
            "video_ch": H3_VIDEO_CH,
            "audio_ch": H3_AUDIO_CH,
            "video_shift": H3_VIDEO_SHIFT,
            "audio_shift": H3_AUDIO_SHIFT,
            "map": "ToyPackedMap",
        },
    )
    try:
        train(
            steps=args.steps,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
            wandb_run=run,
        )
    finally:
        run.finish()


if __name__ == "__main__":
    main()
