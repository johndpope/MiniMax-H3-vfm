"""Checks for the adapter contract, loss split, anneal, and on-disk H3 names."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vfm.adapter import (
    apply_unconditional_mix,
    kl_standard_normal,
    reparameterize,
    unconditional_mask,
)
from vfm.components import verify_repo_layout
from vfm.constants import TASK
from vfm.ema import ParameterEMA
from vfm.losses import observation_loss, total_loss
from vfm.schedule import fd_weight_for_k, stage_a_ks


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

    try:
        observation_loss({"identity": 1.0, "line": 1.0, "sync": 1.0, "fd": 1.0})
    except ValueError:
        pass
    else:
        raise AssertionError("L_obs accepted an FD term")
    if observation_loss({"identity": 0.1, "line": 0.2, "sync": 0.3}) != 0.6:
        raise AssertionError("observation_loss sum")

    if total_loss(l_mf=1.0, l_obs=2.0, l_kl=0.5, fd=10.0, fd_weight=0.0) != 3.5:
        raise AssertionError("Stage A must drop FD when fd_weight is 0")
    if total_loss(l_mf=1.0, l_obs=2.0, l_kl=0.5, fd=10.0, fd_weight=1.0) != 13.5:
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
