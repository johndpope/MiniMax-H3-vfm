"""Numeric checks that do not need towers, weights, or a GPU."""

from __future__ import annotations

import numpy as np

from fd_loss.frechet import frechet_distance, frozen_norm_scale
from fd_loss.multi_tower import MultiTowerFD
from fd_loss.stats import EMAFeatureStats, QueueFeatureStats
from fd_loss.towers import DEFAULT_TOWERS, enabled_towers


def run_selfcheck() -> None:
    dim = 4
    eye = np.eye(dim)
    zero = np.zeros(dim)
    identical = frechet_distance(zero, eye, zero, eye)
    if identical > 1e-5:
        raise AssertionError(f"identical Gaussians FD={identical}")

    mu_g = np.zeros(dim)
    mu_g[0] = 3.0
    shifted = frechet_distance(zero, eye, mu_g, eye)
    if abs(shifted - 9.0) > 1e-4:
        raise AssertionError(f"mean-shift FD={shifted}, expected 9")

    scale = frozen_norm_scale(mu_g, eye)
    if scale <= 0:
        raise AssertionError("norm scale must be positive")

    rng = np.random.default_rng(0)
    batch = rng.standard_normal((32, dim))
    ema = EMAFeatureStats(dim, beta=0.5)
    mu, _, _ = ema.surrogate(batch)
    ema.commit(batch)
    if not np.allclose(ema.mu, batch.mean(axis=0)):
        raise AssertionError("first EMA commit should match the batch mean")
    if not np.allclose(mu, batch.mean(axis=0)):
        raise AssertionError("empty EMA surrogate should be the batch")

    # β = 0.5, batch at 0 then at 10. Within-batch variance is 0.
    # Mixture variance is β(1-β)(μ - μ_b)² = 0.25 * 100 = 25, not 0.
    probe = EMAFeatureStats(1, beta=0.5)
    probe.commit(np.zeros((8, 1)))
    mu_mix, sigma_mix, _ = probe.surrogate(np.full((8, 1), 10.0))
    if abs(float(mu_mix[0]) - 5.0) > 1e-8 or abs(float(sigma_mix[0, 0]) - 25.0) > 1e-8:
        raise AssertionError(f"EMA mixture mu={mu_mix} sigma={sigma_mix}, expected 5 and 25")

    queue = QueueFeatureStats(dim, capacity=4)
    first = rng.standard_normal((3, dim))
    second = rng.standard_normal((3, dim))
    queue.commit(first)
    mu_s, sigma_s, _ = queue.surrogate(second)
    combined = np.concatenate([first, second], axis=0)
    mu_c = combined.mean(axis=0)
    centered = combined - mu_c
    sigma_c = (centered.T @ centered) / combined.shape[0]
    if not np.allclose(mu_s, mu_c) or not np.allclose(sigma_s, sigma_c):
        raise AssertionError("queue surrogate is not the concatenated covariance")
    queue.commit(second)
    if queue.filled != 4:
        raise AssertionError(f"queue filled={queue.filled}, expected 4")

    names = ("video_frame_dinov2", "audio_clap", "sync_imagebind")
    towers = enabled_towers(list(names))
    weight_sum = sum(spec.weight for spec in towers)
    if abs(weight_sum - 1.0) > 1e-8:
        raise AssertionError(weight_sum)
    if abs(sum(spec.weight for spec in DEFAULT_TOWERS) - 1.0) > 1e-8:
        raise AssertionError("default tower weights must sum to 1")

    real = {}
    features = {}
    for spec in towers:
        cloud = rng.standard_normal((64, dim)) + 0.5
        mu_r = cloud.mean(axis=0)
        xc = cloud - mu_r
        real[spec.name] = {
            "mu": mu_r,
            "sigma": (xc.T @ xc) / cloud.shape[0],
            "dim": dim,
        }
        features[spec.name] = rng.standard_normal((16, dim)) + 0.5
    model = MultiTowerFD(towers, real, population="ema", ema_beta=0.5)
    held = {name: stats.count for name, stats in model.stats.items()}
    held_mu = {name: stats.mu.copy() for name, stats in model.stats.items()}
    preview = model.step(features, commit=False)
    if not np.isfinite(preview["loss"]):
        raise AssertionError(preview)
    for name, stats in model.stats.items():
        if stats.count != held[name] or not np.allclose(stats.mu, held_mu[name]):
            raise AssertionError(f"commit=False changed {name}")

    bad = dict(features)
    bad["audio_clap"] = np.zeros((16, dim + 1))
    try:
        model.step(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("a later tower with the wrong dim should fail")
    for name, stats in model.stats.items():
        if stats.count != held[name]:
            raise AssertionError(f"partial commit on {name}")

    saved_mu = model.real["sync_imagebind"]["mu"]
    model.real["sync_imagebind"]["mu"] = np.zeros(dim + 3)
    try:
        model.step(features)
    except ValueError:
        pass
    else:
        raise AssertionError("frechet_distance shape mismatch should fail")
    model.real["sync_imagebind"]["mu"] = saved_mu
    for name, stats in model.stats.items():
        if stats.count != held[name]:
            raise AssertionError(f"frechet failure committed {name}")

    out = model.step(features)
    if not np.isfinite(out["loss"]) or set(out["towers"]) != set(names):
        raise AssertionError(out)
    if any(stats.count == 0 for stats in model.stats.values()):
        raise AssertionError("successful step should commit every tower")
    vector = {name: out["towers"][name]["raw"] for name in names}
    if len(vector) != 3:
        raise AssertionError("FDr vector collapsed")
