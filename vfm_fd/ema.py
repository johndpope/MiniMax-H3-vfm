"""EMA of trainable parameters (θ), used inside L_obs.

Distinct from the FD feature EMA in ``fd_loss.stats``. This one tracks LoRA
(and, in Stage A, adapter) weights. Decay 0.999.

Scaffold rule, from docs/VFM_FD_REDESIGN.md:

- L_obs reads θ_ema and updates the adapter through z.
- L_MF updates the online LoRA.
- Call ``update`` after the optimizer step. Do not step the shadow buffers
  with L_obs.
"""

from __future__ import annotations

import numpy as np


class ParameterEMA:
    def __init__(self, beta: float = 0.999):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        self.beta = float(beta)
        self.shadow: dict[str, np.ndarray] = {}

    def update(self, named_arrays: dict[str, np.ndarray]) -> None:
        for name, value in named_arrays.items():
            arr = np.asarray(value, dtype=np.float64)
            if name not in self.shadow:
                self.shadow[name] = arr.copy()
            else:
                self.shadow[name] = self.beta * self.shadow[name] + (1.0 - self.beta) * arr

    def weights(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self.shadow.items()}
