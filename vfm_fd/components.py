"""In-repo names the stage scripts attach to. No weights are loaded."""

from __future__ import annotations

import json
from pathlib import Path


# Attention-only LoRA. Keys match the safetensors indexes in this repo.
# AdaLN (adaln_proj, ~13B of the 33B) stays frozen in v1.
LORA_TARGETS = {
    "ref2va_dit": (
        "blocks.*.attn.qkv_proj",
        "blocks.*.attn.out_proj",
    ),
    "diffusers_transformer_ref": (
        "transformer_blocks.*.attn.to_q",
        "transformer_blocks.*.attn.to_k",
        "transformer_blocks.*.attn.to_v",
        "transformer_blocks.*.attn.to_out.0",
    ),
}

FROZEN_SUBSTRINGS = (
    "adaln_proj",
    "proj_in",
    "audio_proj_in",
    "proj_out",
    "audio_proj_out",
    "audio_patch_proj",
)

COMPONENT_INDEX = {
    "noise_adapter": "scripts/vfm/h3_vfm.py:H3NoiseAdapter",
    "split_rows": "scripts/vfm/h3_vfm.py:split_rows",
    "shift_sigma": "scripts/vfm/h3_vfm.py:shift_sigma",
    "kl": "scripts/vfm/h3_vfm.py:kl_to_standard_normal",
    "vfm_loss": "scripts/vfm/h3_vfm.py:vfm_loss (endpoint MSE; not the Stage A objective)",
    "vfm_nfe_loss": "vfm_fd.losses.vfm_nfe_loss mf=0.5 cos=4 std=8 obs=4 grad=2",
    "observation": "y = A(x) = x[:,:,0]; text/Qwen states are conditioning",
    "toy_map": "scripts/vfm/h3_vfm.py:ToyPackedMap",
    "real_map": "scripts/vfm/h3_vfm.py:RealH3Map",
    "ref2va_dit": "Ref2VA/transformer MiniMaxH3DiTModel",
    "fl2va_dit": "FL2VA/transformer MiniMaxH3DiTModel",
    "diffusers_transformer": "transformer/ MiniMaxH3Transformer3DModel",
    "diffusers_transformer_ref": "transformer_ref/ MiniMaxH3Transformer3DModel",
    "ref2va_video_vae": "Ref2VA/video_vae MiniMaxH3VideoVAE",
    "ref2va_audio_vae": "Ref2VA/audio_vae MiniMaxH3AudioVAE",
    "diffusers_video_vae": "vae/ AutoencoderKLMiniMaxH3",
    "diffusers_audio_vae": "audio_vae/ AutoencoderKLMiniMaxH3Audio",
    "video_scheduler": "scheduler/ MiniMaxH3Scheduler shift 12",
    "audio_scheduler": "audio_scheduler/ MiniMaxH3Scheduler shift 3",
    "text_encoder_ref2va": "Ref2VA/text_encoder MiniMaxH3Qwen3VLHFEncoder",
    "text_encoder_diffusers": "text_encoder/ Qwen3VLForConditionalGeneration",
    "pipeline_ref2va": "Ref2VA MiniMaxH3Pipeline",
    "pipeline_modular": "MiniMaxH3ModularPipeline",
}

_REQUIRED = (
    "scripts/vfm/h3_vfm.py",
    "Ref2VA/transformer/config.json",
    "Ref2VA/model_index.json",
    "Ref2VA/video_vae/config.json",
    "Ref2VA/video_vae/minimax_h3_video_vae.py",
    "Ref2VA/audio_vae/config.json",
    "Ref2VA/audio_vae/minimax_h3_audio_vae.py",
    "FL2VA/transformer/config.json",
    "vae/config.json",
    "audio_vae/config.json",
    "transformer/config.json",
    "transformer_ref/config.json",
    "scheduler/scheduler_config.json",
    "audio_scheduler/scheduler_config.json",
    "model_index.json",
)

_SCRIPT_MARKERS = (
    "class H3NoiseAdapter",
    "class ToyPackedMap",
    "class RealH3Map",
    "def vfm_loss",
    "def shift_sigma",
    "def kl_to_standard_normal",
    "def split_rows",
    "H3_TEXT_DIM = 5120",
    "H3_VIDEO_CH = 24",
    "H3_AUDIO_CH = 32",
    "H3_VIDEO_SHIFT = 12.0",
    "H3_AUDIO_SHIFT = 3.0",
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_repo_layout(root: Path) -> list[str]:
    """Return a list of problems. Empty means the wiring targets are on disk."""
    problems: list[str] = []
    for rel in _REQUIRED:
        if not (root / rel).is_file():
            problems.append(f"missing {rel}")
    if problems:
        return problems

    script = (root / "scripts/vfm/h3_vfm.py").read_text(encoding="utf-8")
    for marker in _SCRIPT_MARKERS:
        if marker not in script:
            problems.append(f"scripts/vfm/h3_vfm.py missing {marker!r}")

    ref_dit = _load(root / "Ref2VA/transformer/config.json")
    if ref_dit.get("_class_name") != "MiniMaxH3DiTModel":
        problems.append("Ref2VA transformer class is not MiniMaxH3DiTModel")
    if ref_dit.get("latents_dim") != 24 or ref_dit.get("audio_latents_dim") != 32:
        problems.append("Ref2VA DiT latent dims are not 24/32")
    if ref_dit.get("text_dim") != 5120:
        problems.append("Ref2VA DiT text_dim is not 5120")

    fl_dit = _load(root / "FL2VA/transformer/config.json")
    if fl_dit.get("_class_name") != "MiniMaxH3DiTModel":
        problems.append("FL2VA transformer class is not MiniMaxH3DiTModel")

    for rel in ("transformer/config.json", "transformer_ref/config.json"):
        cfg = _load(root / rel)
        if cfg.get("_class_name") != "MiniMaxH3Transformer3DModel":
            problems.append(f"{rel} class is not MiniMaxH3Transformer3DModel")
        if cfg.get("in_channels") != 24 or cfg.get("audio_in_channels") != 32:
            problems.append(f"{rel} channels are not 24/32")
        if cfg.get("text_dim") != 5120:
            problems.append(f"{rel} text_dim is not 5120")

    ref_index = _load(root / "Ref2VA/model_index.json")
    shifts = ref_index.get("_minimax_h3", {}).get("sigma_shift_scales", {})
    if shifts.get("video") != 12.0 or shifts.get("audio") != 3.0:
        problems.append("Ref2VA sigma_shift_scales are not video 12 / audio 3")
    if ref_index.get("_class_name") != "MiniMaxH3Pipeline":
        problems.append("Ref2VA pipeline class is not MiniMaxH3Pipeline")

    video_vae = _load(root / "vae/config.json")
    if video_vae.get("_class_name") != "AutoencoderKLMiniMaxH3":
        problems.append("diffusers video VAE class changed")
    if video_vae.get("clip_length") != 17 or video_vae.get("latent_channels") != 24:
        problems.append("diffusers video VAE is not clip_length 17 / 24 channels")

    ref_video = _load(root / "Ref2VA/video_vae/config.json")
    if ref_video.get("_class_name") != "MiniMaxH3VideoVAE":
        problems.append("Ref2VA video VAE class is not MiniMaxH3VideoVAE")
    if ref_video.get("vae_clip_length") != 17 or ref_video.get("latent_channels") != 24:
        problems.append("Ref2VA video VAE is not vae_clip_length 17 / 24 channels")

    audio = _load(root / "audio_vae/config.json")
    if audio.get("_class_name") != "AutoencoderKLMiniMaxH3Audio":
        problems.append("diffusers audio VAE class changed")
    if audio.get("latent_channels") != 32 or audio.get("sampling_rate") != 32000:
        problems.append("diffusers audio VAE is not 32 ch / 32000 Hz")

    ref_audio = _load(root / "Ref2VA/audio_vae/config.json")
    if ref_audio.get("_class_name") != "MiniMaxH3AudioVAE":
        problems.append("Ref2VA audio VAE class is not MiniMaxH3AudioVAE")
    if ref_audio.get("latent_channels") != 32 or ref_audio.get("sample_rate") != 32000:
        problems.append("Ref2VA audio VAE is not 32 ch / 32000 Hz")

    if _load(root / "scheduler/scheduler_config.json").get("shift") != 12.0:
        problems.append("video scheduler shift is not 12")
    if _load(root / "audio_scheduler/scheduler_config.json").get("shift") != 3.0:
        problems.append("audio scheduler shift is not 3")

    root_index = _load(root / "model_index.json")
    if root_index.get("_class_name") != "MiniMaxH3ModularPipeline":
        problems.append("root pipeline class is not MiniMaxH3ModularPipeline")
    return problems
