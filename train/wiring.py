"""Shared wiring checks and TODO text for the stage scaffolds."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAGE_A_TODOS = (
    "Load a local Ref2VA MiniMaxH3DiTModel (Ref2VA/transformer) or "
    "MiniMaxH3Transformer3DModel (transformer_ref/). Do not download weights.",
    "Attach attention LoRA from vfm.components.LORA_TARGETS. Leave every adaln_proj frozen.",
    "Fill scripts/vfm/h3_vfm.py:RealH3Map.velocity: pack z_v/z_a, one DiT forward, "
    "shift_sigma video 12 / audio 3 (MiniMaxH3Scheduler).",
    "Train scripts/vfm/h3_vfm.py:H3NoiseAdapter with that LoRA. Generalize vfm_loss from K=1 to --k.",
    "Mix N(0, I) with vfm.adapter.unconditional_mask at --alpha. One mask for video and audio.",
    "L_obs = vfm.losses.observation_loss (identity, line, sync) on vfm.ema.ParameterEMA. "
    "L_MF updates the online LoRA. KL via kl_to_standard_normal (--kl-video, --kl-audio).",
    "Freeze MiniMaxH3VideoVAE / AutoencoderKLMiniMaxH3, MiniMaxH3AudioVAE / "
    "AutoencoderKLMiniMaxH3Audio, and the Qwen3-VL encoder.",
)

STAGE_B_TODOS = (
    "Load the Stage A H3NoiseAdapter and LoRA. Freeze the adapter unless --adapter-lr > 0.",
    "Draw z ~ q_phi(z|y) and integrate K=1 through RealH3Map. Shifts stay 12 / 3.",
    "Decode with the frozen VAEs. Run frozen towers. Keep the activation graph so FD can reach the LoRA.",
    "fd_loss.MultiTowerFD.step on the current batch. Real (mu, sigma) from --real-stats "
    "(python -m fd_loss.offline_real_stats). Population ema (beta 0.999) or queue (8k-32k).",
    "Keep L_obs inside vfm.losses.total_loss. fd_weight is 1 at K=1. Do not drop identity, line, or sync.",
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
    from vfm.selfcheck import run_selfcheck as vfm_check

    vfm_check(ROOT)
    fd_check()


def format_plan(title: str, todos: tuple[str, ...], notes: tuple[str, ...]) -> str:
    lines = [title, ""]
    lines.append("Components:")
    from vfm.components import COMPONENT_INDEX, LORA_TARGETS

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
