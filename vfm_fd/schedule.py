"""K anneal 8 → 4 → 2 → 1, with FD weight rising only after the constraint lock."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnnealPhase:
    name: str
    k: int
    fd_weight: float
    trains_adapter: bool
    trains_lora: bool


# fd_weight on the transition row is a fraction of the Stage B multiplier.
SCHEDULE: tuple[AnnealPhase, ...] = (
    AnnealPhase("warmup", 8, 0.0, True, True),
    AnnealPhase("stage_a_early", 4, 0.0, True, True),
    AnnealPhase("stage_a_late", 2, 0.0, True, True),
    AnnealPhase("transition", 2, 0.1, True, True),
    AnnealPhase("stage_b", 1, 1.0, False, True),
)


def stage_a_ks() -> tuple[int, ...]:
    """Entry NFEs for Stage A. Not 1."""
    return (4, 2)


def fd_weight_for_k(k: int, *, transition: bool = False) -> float:
    """FD anneal multiplier. Stage A (including K=2) stays at 0.

    Pass ``transition=True`` for the K=2 hand-off into Stage B.
    """
    if k not in {8, 4, 2, 1}:
        raise ValueError(f"K must be one of 8, 4, 2, 1; got {k}")
    if k == 1:
        return 1.0
    if k == 2 and transition:
        return 0.1
    return 0.0


def phase_for(k: int, *, transition: bool = False) -> AnnealPhase:
    if k == 2 and transition:
        return SCHEDULE[3]
    for phase in SCHEDULE:
        if phase.k == k and phase.name != "transition":
            return phase
    raise ValueError(f"no phase for K={k}")
