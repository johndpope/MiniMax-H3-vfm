"""VFM contracts for MiniMax-H3 Ref2VA.

The trainable noise adapter is ``scripts/vfm/h3_vfm.py:H3NoiseAdapter``.
This package is the interface around it: shapes, unconditional mix, loss
split, parameter EMA, and the K / FD-weight anneal. It does not load weights.
"""

from vfm_fd.adapter import (
    SAMPLE_STAT_KEYS,
    apply_unconditional_mix,
    kl_standard_normal,
    reparameterize,
    unconditional_mask,
)
from vfm_fd.components import LORA_TARGETS, COMPONENT_INDEX, verify_repo_layout
from vfm_fd.constants import (
    H3_AUDIO_CH,
    H3_AUDIO_SHIFT,
    H3_TEXT_DIM,
    H3_VIDEO_CH,
    H3_VIDEO_SHIFT,
    TASK,
    VAE_CLIP_LENGTH,
)
from vfm_fd.ema import ParameterEMA
from vfm_fd.losses import NFE_WEIGHTS, observation_loss, total_loss, vfm_nfe_loss
from vfm_fd.particles import frontier_step
from vfm_fd.schedule import AnnealPhase, fd_weight_for_k, stage_a_ks

__all__ = [
    "SAMPLE_STAT_KEYS",
    "AnnealPhase",
    "COMPONENT_INDEX",
    "H3_AUDIO_CH",
    "H3_AUDIO_SHIFT",
    "H3_TEXT_DIM",
    "H3_VIDEO_CH",
    "H3_VIDEO_SHIFT",
    "LORA_TARGETS",
    "ParameterEMA",
    "TASK",
    "VAE_CLIP_LENGTH",
    "apply_unconditional_mix",
    "fd_weight_for_k",
    "frontier_step",
    "kl_standard_normal",
    "NFE_WEIGHTS",
    "observation_loss",
    "reparameterize",
    "stage_a_ks",
    "total_loss",
    "vfm_nfe_loss",
    "unconditional_mask",
    "verify_repo_layout",
]
