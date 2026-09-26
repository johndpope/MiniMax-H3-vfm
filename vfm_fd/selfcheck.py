"""Checks for the adapter contract, loss split, anneal, and on-disk H3 names."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vfm_fd.adapter import (
    apply_unconditional_mix,
    kl_standard_normal,
    reparameterize,
    unconditional_mask,
)
from vfm_fd.components import verify_repo_layout
from vfm_fd.constants import TASK
from vfm_fd.ema import ParameterEMA
from vfm_fd.losses import NFE_WEIGHTS, observation_loss, total_loss, vfm_nfe_loss
from vfm_fd.schedule import fd_weight_for_k, stage_a_ks


def run_selfcheck(root: Path | None = None) -> None:
    root = root or Path(__file__).resolve().parents[1]
    problems = verify_repo_layout(root)
    if problems:
        raise AssertionError("\n".join(problems))

    zeros = np.zeros((8, 4))
    if kl_standard_normal(zeros, zeros) != 0.0:
        raise AssertionError("KL of N(0, I) must be 0")

    rng = np.random.default_rng(1)
    mu = np.ones((6, 4))
    log_sigma = np.full((6, 4), -30.0)
    drawn = reparameterize(mu, log_sigma, rng=rng)
    if not np.allclose(drawn, mu, atol=1e-6):
        raise AssertionError("tiny log-sigma should reproduce mu")

    mask = unconditional_mask(5, 0.0, rng)
    if mask.any():
        raise AssertionError("alpha 0 must never take the prior")
    base = np.arange(10, dtype=np.float64).reshape(5, 2)
    if not np.array_equal(apply_unconditional_mix(base, mask, rng), base):
        raise AssertionError("alpha 0 mix must keep z")
    prior = unconditional_mask(5, 1.0, rng)
    video = apply_unconditional_mix(base, prior, np.random.default_rng(2))
    audio = apply_unconditional_mix(base + 3, prior, np.random.default_rng(3))
    if np.allclose(video, base) or np.allclose(audio, base + 3):
        raise AssertionError("alpha 1 must replace both streams")

    if NFE_WEIGHTS != {"mf": 0.5, "cos": 4.0, "std": 8.0, "obs": 4.0, "grad": 2.0}:
        raise AssertionError(NFE_WEIGHTS)
    clip = np.zeros((2, 3, 4, 5))
    clip[:, :, 0] = 1.0
    y = np.ones((2, 3, 5))
    if observation_loss(clip, y) != 0.0:
        raise AssertionError("first-frame observation should be 0 when A(x) == y")
    obs = observation_loss(clip, np.zeros_like(y))
    if abs(obs - float(np.linalg.norm(np.ones_like(y)))) > 1e-8:
        raise AssertionError(obs)
    nfe = vfm_nfe_loss(mf=1.0, cos=1.0, std=1.0, obs=obs, grad=1.0)
    expected = 0.5 + 4.0 + 8.0 + 4.0 * obs + 2.0
    if abs(nfe - expected) > 1e-8:
        raise AssertionError(nfe)

    if total_loss(l_nfe=3.5, fd=10.0, fd_weight=0.0) != 3.5:
        raise AssertionError("Stage A must drop FD when fd_weight is 0")
    if total_loss(l_nfe=3.5, fd=10.0, fd_weight=1.0) != 13.5:
        raise AssertionError("Stage B FD weight")

    if stage_a_ks() != (4, 2):
        raise AssertionError(stage_a_ks())
    if fd_weight_for_k(4) != 0.0 or fd_weight_for_k(1) != 1.0:
        raise AssertionError("anneal weights")
    if fd_weight_for_k(2, transition=True) != 0.1:
        raise AssertionError("transition FD weight")
    if 1 in stage_a_ks():
        raise AssertionError("Stage A must not start at 1 NFE")

    ema = ParameterEMA(beta=0.5)
    ema.update({"lora": np.array([0.0, 0.0])})
    ema.update({"lora": np.array([2.0, 4.0])})
    shadow = ema.weights()["lora"]
    if not np.allclose(shadow, [1.0, 2.0]):
        raise AssertionError(shadow)

    if TASK["ref2va"] != 3:
        raise AssertionError("ref2va task id drifted from scripts/vfm/h3_vfm.py")

    from vfm_fd.particles import run_selfcheck as particle_check

    particle_check()
