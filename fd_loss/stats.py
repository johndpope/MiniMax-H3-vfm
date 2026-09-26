"""Population feature stats for FD-loss.

Call ``surrogate(batch)`` to form the moments the loss sees, then
``commit(batch)`` with features that will not receive gradients. The
population accumulated before this batch is the detached branch.

EMA (β ≈ 0.999) uses the two-component mixture. ``μ`` in the covariance
term is the EMA mean before the batch:

``Σ ← β Σ + (1-β) Σ_b + β(1-β) (μ - μ_b)(μ - μ_b)ᵀ``

Queue (8k–32k slots) uses the exact mixture of the detached buffer and the
current batch. The batch's share is ``n_batch / (n_queue + n_batch)``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


def _batch_moments(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"features must be [N, D], got {features.shape}")
    if features.shape[0] < 1:
        raise ValueError("batch is empty")
    mu = features.mean(axis=0)
    centered = features - mu
    sigma = (centered.T @ centered) / features.shape[0]
    return mu, sigma


class PopulationStats(Protocol):
    def surrogate(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Moments for the loss. Does not mutate the population."""

    def commit(self, features: np.ndarray) -> None:
        """Fold detached features into the population after the loss is built."""


class EMAFeatureStats:
    def __init__(self, dim: int, beta: float = 0.999, eps: float = 1e-6):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.beta = float(beta)
        self.eps = float(eps)
        self.mu = np.zeros(self.dim, dtype=np.float64)
        self.sigma = np.eye(self.dim, dtype=np.float64) * self.eps
        self.count = 0

    def surrogate(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        mu_b, sigma_b = _batch_moments(features)
        if mu_b.shape != (self.dim,):
            raise ValueError(f"expected dim {self.dim}, got {mu_b.shape[0]}")
        if self.count == 0:
            mu, sigma = mu_b, sigma_b
        else:
            beta = self.beta
            # μ is the EMA mean before this batch (not the updated mean).
            mu_old = self.mu
            delta = (mu_old - mu_b).reshape(-1, 1)
            mu = beta * mu_old + (1.0 - beta) * mu_b
            sigma = (
                beta * self.sigma
                + (1.0 - beta) * sigma_b
                + (beta * (1.0 - beta)) * (delta @ delta.T)
            )
        aux = {
            "population": "ema",
            "beta": self.beta,
            "count": self.count,
        }
        return mu, sigma, aux

    def commit(self, features: np.ndarray) -> None:
        mu, sigma, _ = self.surrogate(features)
        self.mu = mu
        self.sigma = sigma
        self.count += int(np.asarray(features).shape[0])


class QueueFeatureStats:
    def __init__(self, dim: int, capacity: int = 8192):
        if dim < 1:
            raise ValueError("dim must be positive")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.dim = int(dim)
        self.capacity = int(capacity)
        self.buf = np.zeros((self.capacity, self.dim), dtype=np.float64)
        self.filled = 0
        self.ptr = 0

    def _stored(self) -> np.ndarray:
        if self.filled < self.capacity:
            return self.buf[: self.filled]
        return self.buf

    def surrogate(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        mu_b, sigma_b = _batch_moments(features)
        if mu_b.shape != (self.dim,):
            raise ValueError(f"expected dim {self.dim}, got {mu_b.shape[0]}")
        n_b = int(np.asarray(features).shape[0])
        stored = self._stored()
        n_q = int(stored.shape[0])
        if n_q == 0:
            mu, sigma = mu_b, sigma_b
        else:
            mu_q, sigma_q = _batch_moments(stored)
            n = n_q + n_b
            mu = (n_q * mu_q + n_b * mu_b) / n
            d_q = (mu_q - mu).reshape(-1, 1)
            d_b = (mu_b - mu).reshape(-1, 1)
            sigma = (n_q * (sigma_q + d_q @ d_q.T) + n_b * (sigma_b + d_b @ d_b.T)) / n
        aux = {
            "population": "queue",
            "capacity": self.capacity,
            "count": n_q,
        }
        return mu, sigma, aux

    def commit(self, features: np.ndarray) -> None:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.dim:
            raise ValueError(f"expected [N, {self.dim}], got {features.shape}")
        n = int(features.shape[0])
        if n >= self.capacity:
            self.buf[:] = features[-self.capacity :]
            self.ptr = 0
            self.filled = self.capacity
            return
        end = self.ptr + n
        if end <= self.capacity:
            self.buf[self.ptr : end] = features
        else:
            first = self.capacity - self.ptr
            self.buf[self.ptr :] = features[:first]
            self.buf[: n - first] = features[first:]
        self.ptr = end % self.capacity
        self.filled = min(self.capacity, self.filled + n)
