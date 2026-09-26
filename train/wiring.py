"""Shared wiring checks and TODO text for the stage scaffolds."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAGE_A_TODOS = (
    "Load a local Ref2VA MiniMaxH3DiTModel (Ref2VA/transformer) or "
    "MiniMaxH3Transformer3DModel (transformer_ref/). Do not download weights.",
    "Attach attention LoRA from vfm_fd.components.LORA_TARGETS. Leave every adaln_proj frozen.",
    "Train scripts/vfm/h3_vfm.py:H3NoiseAdapter with that LoRA under vfm_nfe_loss "
    "(mf 0.5, cos 4, std 8, obs 4, grad 2). Do not generalize endpoint vfm_loss.",
    "Observation y = A(x) = x[:,:,0] (I2VA first frame). "
    "vfm_fd.losses.observation_loss computes ||A(fθ(z)) - y||. "
    "Text and Qwen states are conditioning, not y.",
    "Mix N(0, I) with vfm_fd.adapter.unconditional_mask at --alpha. One mask for video and audio.",
    "Freeze MiniMaxH3VideoVAE / AutoencoderKLMiniMaxH3, MiniMaxH3AudioVAE / "
    "AutoencoderKLMiniMaxH3Audio, and the Qwen3-VL encoder.",
)

STAGE_B_TODOS = (
    "Load the Stage A H3NoiseAdapter and LoRA. Freeze the adapter unless --adapter-lr > 0.",
    "Draw z from the adapter and integrate K=1 through scripts/vfm/h3_vfm.py:RealH3Map. Shifts stay 12 / 3.",
    "Decode with the frozen VAEs. Run frozen towers. Keep the activation graph so FD can reach the LoRA.",
    "fd_loss.MultiTowerFD.step on the current batch. Real (mu, sigma) from --real-stats "
    "(python -m fd_loss.offline_real_stats). Population ema (beta 0.999) or queue (8k-32k).",
    "Keep ||A(fθ(z)) - y|| in the objective. fd_weight is 1 at K=1. Do not replace the first-frame observation with FD.",
    "Eval the per-tower FDr vector at fixed 1 NFE on a fresh population, not the training queue.",
)

STAGE_C_TODOS = (
    "Decode 17-frame chunks (vae/config.json clip_length, Ref2VA/video_vae vae_clip_length).",
    "Spatially pool each chunk to a coarse canvas (layout, parts, sync before fine detail).",
    "Light FD against teacher canvases from one source only: --canvas-source real or teacher.",
    "Add the canvas term on top of Stage B. Do not replace L_obs or the full-detail towers.",
)


def run_checks() -> None:
    from fd_loss.selfcheck import run_selfcheck as fd_check
    from vfm_fd.selfcheck import run_selfcheck as vfm_check

    vfm_check(ROOT)
    fd_check()


def format_plan(title: str, todos: tuple[str, ...], notes: tuple[str, ...]) -> str:
    lines = [title, ""]
    lines.append("Components:")
    from vfm_fd.components import COMPONENT_INDEX, LORA_TARGETS

    for name, target in COMPONENT_INDEX.items():
        lines.append(f"  {name}: {target}")
    lines.append("LoRA targets (attention only):")
    for layout, keys in LORA_TARGETS.items():
        lines.append(f"  {layout}: {', '.join(keys)}")
    lines.append("TODO:")
    for index, item in enumerate(todos, start=1):
        lines.append(f"  {index}. {item}")
    if notes:
        lines.append("This run:")
        for note in notes:
            lines.append(f"  {note}")
    return "\n".join(lines)
