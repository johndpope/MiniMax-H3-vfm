"""Weighted sum of per-tower Fréchet distances, each divided by a frozen scale."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from fd_loss.frechet import frechet_distance, frozen_norm_scale
from fd_loss.stats import EMAFeatureStats, QueueFeatureStats
from fd_loss.towers import TowerSpec


RealStats = Mapping[str, Mapping[str, np.ndarray | float | int]]


def attach_scales(
    towers: tuple[TowerSpec, ...],
    real: RealStats,
) -> tuple[TowerSpec, ...]:
    scaled = []
    for spec in towers:
        if spec.name not in real:
            raise KeyError(f"no real stats for tower {spec.name}")
        blob = real[spec.name]
        mu = np.asarray(blob["mu"], dtype=np.float64)
        sigma = np.asarray(blob["sigma"], dtype=np.float64)
        dim = int(blob.get("dim", mu.shape[0]))
        scale = spec.norm_scale
        if scale is None:
            stored = blob.get("norm_scale")
            scale = float(stored) if stored is not None else frozen_norm_scale(mu, sigma)
        scaled.append(spec.with_scale(float(scale), dim))
    return tuple(scaled)


class MultiTowerFD:
    """``Σ_i w_i * FD_i / scale_i``.

    ``population`` is ``\"ema\"`` (default, β=0.999) or ``\"queue\"``.
    ``step`` builds the surrogate, optionally commits detached features, and
    returns the scalar plus the per-tower FDr vector.
    """

    def __init__(
        self,
        towers: tuple[TowerSpec, ...],
        real: RealStats,
        *,
        population: str = "ema",
        ema_beta: float = 0.999,
        queue_size: int = 8192,
        population_factory: Callable[[TowerSpec], EMAFeatureStats | QueueFeatureStats] | None = None,
    ):
        if population not in {"ema", "queue"}:
            raise ValueError(f"population must be ema or queue, got {population!r}")
        self.towers = attach_scales(towers, real)
        self.real = real
        self.population = population
        self.stats: dict[str, EMAFeatureStats | QueueFeatureStats] = {}
        for spec in self.towers:
            assert spec.feature_dim is not None
            if population_factory is not None:
                self.stats[spec.name] = population_factory(spec)
            elif population == "ema":
                self.stats[spec.name] = EMAFeatureStats(spec.feature_dim, beta=ema_beta)
            else:
                self.stats[spec.name] = QueueFeatureStats(spec.feature_dim, capacity=queue_size)

    def step(
        self,
        features_by_tower: Mapping[str, np.ndarray],
        *,
        commit: bool = True,
    ) -> dict:
        total = 0.0
        parts: dict[str, dict] = {}
        missing = [spec.name for spec in self.towers if spec.name not in features_by_tower]
        if missing:
            raise KeyError(f"features missing for {missing}")
        for spec in self.towers:
            feats = np.asarray(features_by_tower[spec.name], dtype=np.float64)
            mu_g, sigma_g, aux = self.stats[spec.name].surrogate(feats)
            blob = self.real[spec.name]
            raw = frechet_distance(
                np.asarray(blob["mu"]),
                np.asarray(blob["sigma"]),
                mu_g,
                sigma_g,
            )
            assert spec.norm_scale is not None
            normalized = raw / spec.norm_scale
            weighted = spec.weight * normalized
            total += weighted
            parts[spec.name] = {
                "raw": raw,
                "normalized": normalized,
                "weight": spec.weight,
                "weighted": weighted,
                "modality": spec.modality,
                **aux,
            }
            if commit:
                self.stats[spec.name].commit(feats)
        return {"loss": total, "towers": parts}
