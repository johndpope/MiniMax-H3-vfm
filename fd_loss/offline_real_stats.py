"""Offline (μ_r, Σ_r) for real Ref2VA-domain clips.

Does not load MiniMaxH3DiTModel or MiniMaxH3Transformer3DModel. The real
tower forward is ``encode_manifest`` and is intentionally unimplemented.
``--synthetic`` writes Gaussian moments so the file format can be checked
without weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fd_loss.frechet import frozen_norm_scale
from fd_loss.towers import DEFAULT_TOWERS, enabled_towers


def _moments(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = features.mean(axis=0)
    centered = features - mu
    sigma = (centered.T @ centered) / features.shape[0]
    return mu, sigma


def synthetic_stats(
    towers: tuple,
    *,
    n: int,
    dim: int,
    seed: int,
) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    blobs = {}
    for index, spec in enumerate(towers):
        # Shifted clouds so the frozen norm scale is not ~0.
        features = rng.standard_normal((n, dim)) + (index + 1)
        mu, sigma = _moments(features)
        blobs[spec.name] = {
            "mu": mu,
            "sigma": sigma,
            "n": n,
            "dim": dim,
            "family": spec.family,
            "modality": spec.modality,
            "norm_scale": frozen_norm_scale(mu, sigma),
            "source": "synthetic",
        }
    return blobs


def save_stats(path: Path, blobs: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    meta = {"towers": {}}
    for name, blob in blobs.items():
        arrays[f"{name}.mu"] = np.asarray(blob["mu"], dtype=np.float64)
        arrays[f"{name}.sigma"] = np.asarray(blob["sigma"], dtype=np.float64)
        meta["towers"][name] = {
            "n": int(blob["n"]),
            "dim": int(blob["dim"]),
            "family": blob["family"],
            "modality": blob["modality"],
            "norm_scale": float(blob["norm_scale"]),
            "source": blob["source"],
        }
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_stats(path: Path) -> dict[str, dict]:
    archive = np.load(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    blobs = {}
    for name, info in meta["towers"].items():
        blobs[name] = {
            "mu": archive[f"{name}.mu"],
            "sigma": archive[f"{name}.sigma"],
            **info,
        }
    return blobs


def encode_manifest(manifest: Path, towers: tuple) -> dict[str, dict]:
    """TODO: walk a Ref2VA-domain manifest and encode it with frozen towers.

    Intended inputs are decoded real clips (video frames, 32 kHz audio), not
    generated samples. One source per file: real clips, or frozen high-NFE
    teacher canvases, never both. Do not construct
    ``MiniMaxH3DiTModel`` or ``MiniMaxH3Transformer3DModel`` here.

    Tower families to plug in (weights not downloaded by this stub):
    V-JEPA or InternVideo tubes, DINOv2 frames, SigLIP frames, CLAP, BEATs,
    ImageBind or a small AV-sync head.
    """
    raise NotImplementedError(
        f"encode_manifest is not implemented ({manifest}, {len(towers)} towers). "
        "Use --synthetic to write a format fixture, or fill this TODO locally."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute real Ref2VA-domain FD stats. No H3 weight download."
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Real-clip manifest. Not read by --synthetic.")
    parser.add_argument("--out", type=Path, required=True, help="Output .npz path.")
    parser.add_argument(
        "--towers",
        nargs="*",
        default=None,
        help="Tower names. Default: the full video/audio/sync mix.",
    )
    parser.add_argument("--synthetic", action="store_true", help="Write RNG moments. No tower weights.")
    parser.add_argument("--synthetic-n", type=int, default=256)
    parser.add_argument("--synthetic-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out.suffix != ".npz":
        args.out = args.out.with_suffix(".npz")
    towers = enabled_towers(args.towers)
    if args.synthetic:
        blobs = synthetic_stats(
            towers,
            n=args.synthetic_n,
            dim=args.synthetic_dim,
            seed=args.seed,
        )
        save_stats(args.out, blobs)
        print(f"wrote synthetic stats for {len(blobs)} towers to {args.out}")
        return 0
    if args.manifest is None:
        raise SystemExit("pass --manifest, or --synthetic to write a fixture without weights")
    blobs = encode_manifest(args.manifest, towers)
    save_stats(args.out, blobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
