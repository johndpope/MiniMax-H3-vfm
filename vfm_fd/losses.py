"""Stage A objective and the first-frame observation.

``y`` is the I2VA measurement ``A(x) = x[:, :, 0]``. Text and Qwen states are
conditioning. They are not ``y``.

``vfm_nfe_loss`` uses the trainer weights mf 0.5, cos 4, std 8, obs 4, grad 2.
Endpoint latent MSE (``scripts/vfm/h3_vfm.py:vfm_loss``) is a different
function: ``||v - (x0 - z)||²`` is the same as ``||xhat - x0||²``, and
``v = -z`` is a black frame. Do not generalize that endpoint loss.
"""

from __future__ import annotations

import numpy as np

# Weights settled for the trainer. Do not retune them in this scaffold.
NFE_WEIGHTS = {
    "mf": 0.5,
    "cos": 4.0,
    "std": 8.0,
    "obs": 4.0,
    "grad": 2.0,
}


def first_frame(x: np.ndarray) -> np.ndarray:
    """``A(x) = x[:, :, 0]``. Time is axis 2 (``[B, C, T, ...]``)."""
    arr = np.asarray(x)
    if arr.ndim < 3:
        raise ValueError(f"A(x) = x[:,:,0] needs time at axis 2, got shape {arr.shape}")
    return arr[:, :, 0]


def observation_loss(prediction, y) -> float:
    """``||A(fθ(z)) - y||``.

    ``prediction`` is the map output ``fθ(z)``. ``y`` is the first-frame
    measurement, already ``A(x)``, not text hidden states.
    """
    measured = np.asarray(first_frame(prediction), dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    if measured.shape != target.shape:
        raise ValueError(
            f"A(fθ(z)) shape {measured.shape} != y shape {target.shape}. "
            "y is x[:,:,0], not text or Qwen states."
        )
    return float(np.linalg.norm(measured - target))


def vfm_nfe_loss(
    mf: float,
    cos: float,
    std: float,
    obs: float,
    grad: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted NFE objective. ``obs`` should be ``observation_loss(...)``."""
    w = NFE_WEIGHTS if weights is None else weights
    missing = [key for key in NFE_WEIGHTS if key not in w]
    if missing:
        raise KeyError(f"vfm_nfe_loss weights missing {missing}")
    return float(
        w["mf"] * mf + w["cos"] * cos + w["std"] * std + w["obs"] * obs + w["grad"] * grad
    )


def total_loss(
    *,
    l_nfe: float,
    fd: float = 0.0,
    fd_weight: float = 0.0,
) -> float:
    """``vfm_nfe_loss + fd_weight * FD``.

    ``fd_weight`` is 0 in Stage A. Stage B passes the anneal multiplier.
    FD is not part of ``vfm_nfe_loss``.
    """
    return float(l_nfe) + float(fd_weight) * float(fd)
