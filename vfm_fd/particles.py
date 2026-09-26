"""Test-time particle merge in tower-feature space.

Numeric step only. Does not run the H3 map, the VAEs, or the towers.
See docs/STAGE_B_PARTICLE_MERGE.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit_rows(embeddings: np.ndarray) -> np.ndarray:
    rows = np.asarray(embeddings, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 1:
        raise ValueError(f"embeddings must be [N, D], got {rows.shape}")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


def cosine_cluster(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy leader clustering. Same cluster if cosine to the leader is at least ``threshold``.

    Returns an int id per row. This is a soft merge key, not pixel equality.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    units = _unit_rows(embeddings)
    ids = np.empty(units.shape[0], dtype=np.int64)
    leaders: list[int] = []
    next_id = 0
    for index in range(units.shape[0]):
        if leaders:
            sims = units[leaders] @ units[index]
            best = int(np.argmax(sims))
            if float(sims[best]) >= threshold:
                ids[index] = ids[leaders[best]]
                continue
        ids[index] = next_id
        leaders.append(index)
        next_id += 1
    return ids


def quantize_key(embedding: np.ndarray, decimals: int = 2) -> tuple:
    """Stable memory key for one representative. Not an exact state hash."""
    unit = _unit_rows(np.asarray(embedding, dtype=np.float64).reshape(1, -1))[0]
    return tuple(np.round(unit, decimals).tolist())


def judge_scores(
    obs: np.ndarray,
    fd_proxy: np.ndarray,
    *,
    obs_weight: float = 1.0,
    fd_weight: float = 1.0,
) -> np.ndarray:
    """Higher is better. ``obs`` is ``||A(fθ(z)) - y||``. ``fd_proxy`` is a cost, not L_obs."""
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    fd_proxy = np.asarray(fd_proxy, dtype=np.float64).reshape(-1)
    if obs.shape != fd_proxy.shape:
        raise ValueError(f"obs shape {obs.shape} != fd_proxy shape {fd_proxy.shape}")
    return -(obs_weight * obs + fd_weight * fd_proxy)


def best_of_n(scores: np.ndarray) -> int:
    """Ablation: one winner, no merge. Matched-N protocol is in the Stage B sketch."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size < 1:
        raise ValueError("scores are empty")
    return int(np.argmax(scores))


@dataclass(frozen=True)
class FrontierResult:
    survivor_index: np.ndarray
    scores: np.ndarray
    cluster_ids: np.ndarray
    seen_keys: frozenset
    n_merged_away: int
    n_dead: int
    n_revisits: int


def frontier_step(
    embeddings: np.ndarray,
    obs: np.ndarray,
    fd_proxy: np.ndarray,
    *,
    width: int,
    threshold: float,
    tau: float,
    seen_keys: set | frozenset | None = None,
    obs_weight: float = 1.0,
    fd_weight: float = 1.0,
    key_decimals: int = 2,
) -> FrontierResult:
    """One level: merge, drop expanded keys, cancel scores below ``tau``, keep ``width``.

    Dead and kept keys are both written into ``seen_keys``. A cancelled mode
    is not proposed again. ``tau`` is on the judge score. Set it high and
    rare good particles disappear.
    """
    if width < 1:
        raise ValueError(f"width must be positive, got {width}")
    embeddings = np.asarray(embeddings, dtype=np.float64)
    scores = judge_scores(obs, fd_proxy, obs_weight=obs_weight, fd_weight=fd_weight)
    if scores.shape[0] != embeddings.shape[0]:
        raise ValueError("one score per embedding row is required")
    seen = set() if seen_keys is None else set(seen_keys)
    cluster_ids = cosine_cluster(embeddings, threshold)
    ranked: list[tuple[float, int, tuple]] = []
    n_dead = 0
    n_revisits = 0
    n_rows = int(cluster_ids.shape[0])
    for cluster_id in range(int(cluster_ids.max()) + 1 if n_rows else 0):
        members = np.flatnonzero(cluster_ids == cluster_id)
        rep = int(members[np.argmax(scores[members])])
        key = quantize_key(embeddings[rep], decimals=key_decimals)
        if key in seen:
            n_revisits += 1
            continue
        if float(scores[rep]) < tau:
            n_dead += 1
            seen.add(key)
            continue
        ranked.append((float(scores[rep]), rep, key))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    kept = ranked[:width]
    new_seen = set(seen)
    for _, _, key in kept:
        new_seen.add(key)
    survivors = np.array([row[1] for row in kept], dtype=np.int64)
    n_clusters = int(cluster_ids.max()) + 1 if n_rows else 0
    return FrontierResult(
        survivor_index=survivors,
        scores=scores,
        cluster_ids=cluster_ids,
        seen_keys=frozenset(new_seen),
        n_merged_away=n_rows - n_clusters,
        n_dead=n_dead,
        n_revisits=n_revisits,
    )


def run_selfcheck() -> None:
    same = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    ids = cosine_cluster(same, threshold=0.9)
    if ids[0] != ids[1] or ids[0] == ids[2]:
        raise AssertionError(f"cluster ids {ids}")

    obs = np.zeros(3)
    proxy = np.array([0.1, 0.1, 0.2])
    step = frontier_step(same, obs, proxy, width=2, threshold=0.9, tau=-1.0)
    if step.survivor_index.size != 2:
        raise AssertionError(f"width should keep two clusters, got {step.survivor_index}")
    if step.n_merged_away != 1:
        raise AssertionError(f"duplicate row should merge away, got {step.n_merged_away}")

    again = frontier_step(
        same, obs, proxy, width=2, threshold=0.9, tau=-1.0, seen_keys=step.seen_keys
    )
    if again.survivor_index.size != 0 or again.n_revisits != 2:
        raise AssertionError(f"expanded keys should drop the next level, got {again}")

    # Score = -(obs + fd). tau above every score cancels the pool.
    tight = frontier_step(same, obs, proxy, width=2, threshold=0.9, tau=0.0)
    if tight.survivor_index.size != 0 or tight.n_dead != 2:
        raise AssertionError(f"high tau should cancel clusters, got {tight}")

    if best_of_n(np.array([1.0, 3.0, 2.0])) != 1:
        raise AssertionError("best-of-N should return the max score, without merging")
