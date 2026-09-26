"""Noise-adapter contract.

``H3NoiseAdapter.sample`` in ``scripts/vfm/h3_vfm.py`` is the implementation.
Call it as ``sample(text, text_mask, pos, task, modality, temperature=1.0)``.
It returns ``(z_video_rows, z_audio_rows, stats)`` with stats keys
``v_mu``, ``v_log``, ``a_mu``, ``a_log``, ``kl_v``, ``kl_a``.

Rows are packed: video tokens occupy ``[:Sv]`` on the video head, audio
tokens occupy ``[Sv:Sv+Sa]`` on the audio head. Split with ``split_rows``
in that same file. Modality 0 is video, 1 is audio.

``q_phi`` is a diagonal Gaussian, not ``DiagonalGaussianDistribution`` in
``Ref2VA/video_vae/vae_module.py`` (that is the VAE posterior).

``text`` passed to ``sample`` is conditioning (Qwen / H3-Encoder states).
The Stage A observation is ``y = A(x) = x[:, :, 0]``, not those states.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

SAMPLE_STAT_KEYS = ("v_mu", "v_log", "a_mu", "a_log", "kl_v", "kl_a")
SAMPLE_PARAMS = ("text", "text_mask", "pos", "task", "modality")


def reparameterize(
    mu: np.ndarray,
    log_sigma: np.ndarray,
    *,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """``z = μ + exp(log σ) ⊙ ε``, matching ``H3NoiseAdapter.sample``."""
    rng = rng or np.random.default_rng()
    mu = np.asarray(mu, dtype=np.float64)
    log_sigma = np.asarray(log_sigma, dtype=np.float64)
    eps = rng.standard_normal(mu.shape)
    return mu + np.exp(log_sigma) * eps * float(temperature)


def kl_standard_normal(mu: np.ndarray, log_sigma: np.ndarray) -> float:
    """Mean KL to N(0, I). Same expression as ``kl_to_standard_normal``."""
    mu = np.asarray(mu, dtype=np.float64)
    log_sigma = np.asarray(log_sigma, dtype=np.float64)
    return float(0.5 * np.mean(mu**2 + np.exp(2.0 * log_sigma) - 2.0 * log_sigma - 1.0))


def unconditional_mask(
    batch_size: int,
    alpha: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Per-example Bernoulli. True means "use N(0, I)" with probability ``alpha``.

    Share one mask across video and audio. A sample is fully conditional or
    fully unconditional.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    rng = rng or np.random.default_rng()
    return rng.random(batch_size) < alpha


def apply_unconditional_mix(
    z: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Replace masked examples with standard normal noise. ``mask`` is shape ``[B]``."""
    rng = rng or np.random.default_rng()
    z = np.asarray(z, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (z.shape[0],):
        raise ValueError(f"mask shape {mask.shape} != batch {(z.shape[0],)}")
    eps = rng.standard_normal(z.shape)
    view = mask.reshape((z.shape[0],) + (1,) * (z.ndim - 1))
    return np.where(view, eps, z)


def assert_sample_stats(stats: Mapping) -> None:
    missing = [key for key in SAMPLE_STAT_KEYS if key not in stats]
    if missing:
        raise KeyError(f"adapter stats missing {missing}; expected {SAMPLE_STAT_KEYS}")
