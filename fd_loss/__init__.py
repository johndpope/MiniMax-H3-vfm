"""Multi-tower representation Fréchet loss for 1-NFE Ref2VA post-training.

Population stats are EMA (β ≈ 0.999) or a feature queue. Real (μ, Σ) are
loaded from an offline file. Gradients, when this is ported to autograd,
flow only through the current batch. Towers stay frozen.
"""

from fd_loss.frechet import frechet_distance, frozen_norm_scale
from fd_loss.multi_tower import MultiTowerFD
from fd_loss.stats import EMAFeatureStats, QueueFeatureStats
from fd_loss.towers import DEFAULT_TOWERS, TowerSpec

__all__ = [
    "DEFAULT_TOWERS",
    "EMAFeatureStats",
    "MultiTowerFD",
    "QueueFeatureStats",
    "TowerSpec",
    "frechet_distance",
    "frozen_norm_scale",
]
