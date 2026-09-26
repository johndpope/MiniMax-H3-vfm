"""Closed-form Fréchet distance between two Gaussians.

    ||mu_r - mu_g||^2
    + Tr(sigma_r + sigma_g - 2 (sigma_r^{1/2} sigma_g sigma_r^{1/2})^{1/2})

Used both as the training surrogate and as eval FDr^k. Square roots are
symmetric eigendecompositions with a ridge, so the routine runs on CPU
without SciPy.
"""

from __future__ import annotations

import numpy as np


def _sym(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _spd_sqrtm(matrix: np.ndarray, eps: float) -> np.ndarray:
    sym = _sym(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    eigenvalues = np.maximum(eigenvalues, eps)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def frechet_distance(
    mu_r: np.ndarray,
    sigma_r: np.ndarray,
    mu_g: np.ndarray,
    sigma_g: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    mu_r = np.asarray(mu_r, dtype=np.float64).reshape(-1)
    mu_g = np.asarray(mu_g, dtype=np.float64).reshape(-1)
    sigma_r = np.asarray(sigma_r, dtype=np.float64)
    sigma_g = np.asarray(sigma_g, dtype=np.float64)
    dim = mu_r.shape[0]
    if mu_g.shape != (dim,) or sigma_r.shape != (dim, dim) or sigma_g.shape != (dim, dim):
        raise ValueError(
            f"shape mismatch mu {mu_r.shape}/{mu_g.shape} "
            f"sigma {sigma_r.shape}/{sigma_g.shape}"
        )
    eye = np.eye(dim)
    sigma_r = _sym(sigma_r) + eps * eye
    sigma_g = _sym(sigma_g) + eps * eye
    mean_term = float((mu_r - mu_g) @ (mu_r - mu_g))
    root_r = _spd_sqrtm(sigma_r, eps)
    mid = root_r @ sigma_g @ root_r
    cov_term = float(np.trace(sigma_r) + np.trace(sigma_g) - 2.0 * np.trace(_spd_sqrtm(mid, eps)))
    # Negative values are numerical dust when the two Gaussians match.
    return mean_term + max(cov_term, 0.0)


def frozen_norm_scale(
    mu_r: np.ndarray,
    sigma_r: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    """FD-SIM-style divisor: FD between the real Gaussian and N(0, I).

    Computed once from offline stats and then held fixed. This is the
    scaffold normalizer, not a claim about a published formula.
    """
    dim = int(np.asarray(mu_r).reshape(-1).shape[0])
    scale = frechet_distance(mu_r, sigma_r, np.zeros(dim), np.eye(dim), eps=eps)
    return float(max(scale, eps))
