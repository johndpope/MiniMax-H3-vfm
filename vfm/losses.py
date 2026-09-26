"""Loss split. FD is never folded into L_obs."""

from __future__ import annotations

OBS_KEYS = ("identity", "line", "sync")


def observation_loss(parts: dict[str, float]) -> float:
    """Identity + spoken line + sync.

    Refuses an ``fd`` key. Population Fréchet distance is a separate term.
    """
    if "fd" in parts or any(key.startswith("fd_") for key in parts):
        raise ValueError("L_obs must not contain FD terms; pass them to total_loss")
    missing = [key for key in OBS_KEYS if key not in parts]
    if missing:
        raise KeyError(f"L_obs missing {missing}; required {OBS_KEYS}")
    return float(sum(parts[key] for key in OBS_KEYS))


def total_loss(
    *,
    l_mf: float,
    l_obs: float,
    l_kl: float,
    fd: float = 0.0,
    fd_weight: float = 0.0,
) -> float:
    """``L_MF + L_obs + L_KL + fd_weight * FD``.

    ``l_obs`` is already ``λ_obs * observation_loss``. ``l_kl`` is already
    the weighted sum of the video and audio KLs. ``fd`` is the weighted
    multi-tower sum ``Σ w_i FD̂_i``. ``fd_weight`` is the anneal multiplier
    (0 in Stage A, 1 at Stage B).
    """
    return float(l_mf) + float(l_obs) + float(l_kl) + float(fd_weight) * float(fd)
