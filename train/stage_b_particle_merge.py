#!/usr/bin/env python3
"""Stage B test-time particle merge.

Numeric merge and judge cutoff only. Does not load weights or call RealH3Map.
Protocol: docs/STAGE_B_PARTICLE_MERGE.md.

  python train/stage_b_particle_merge.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train.wiring import format_plan, run_checks

TODOS = (
    "Draw W noises from scripts/vfm/h3_vfm.py:H3NoiseAdapter for one session y = x[:,:,0]. Text stays conditioning.",
    "Execute one shared NFE through RealH3Map (video shift 12, audio shift 3). Do not give each particle its own schedule.",
    "Decode with frozen MiniMaxH3VideoVAE / AutoencoderKLMiniMaxH3 and MiniMaxH3AudioVAE / AutoencoderKLMiniMaxH3Audio.",
    "Embed with the frozen fd_loss towers. Cluster with vfm_fd.particles.cosine_cluster. Do not merge on pixels or captions.",
    "Score representatives with observation_loss plus an FD proxy. Cancel scores below --tau. Keep width W via frontier_step.",
    "Ablate against best_of_n at matched map calls N = W * levels. Do not claim a video result from this stub.",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage B test-time particle merge. Numeric stub; no DiT call."
    )
    parser.add_argument("--width", type=int, default=8, help="Frontier width W.")
    parser.add_argument("--levels", type=int, default=1, help="Shared map steps. 1 is the 1-NFE port.")
    parser.add_argument("--merge-threshold", type=float, default=0.9, help="Cosine merge threshold.")
    parser.add_argument(
        "--tau",
        type=float,
        default=-1.0,
        help="Judge score cutoff. Higher cancels more particles, including rare good ones.",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.width < 1 or args.levels < 1:
        raise SystemExit("--width and --levels must be positive")
    notes = (
        f"width={args.width}  levels={args.levels}  merge_threshold={args.merge_threshold}  tau={args.tau}",
        f"matched best-of-N budget = {args.width * args.levels} map calls",
        "y = A(x) = x[:,:,0]. FD towers are the merge metric. FD does not replace observation_loss.",
        "frontier_step is numeric. map_one_nfe is not wired.",
    )
    if args.check or not args.run:
        run_checks()
        print(format_plan("Stage B — particle merge (test time)", TODOS, notes))
    if not args.run:
        print("Scaffold only. --run stops before any DiT call. No weights downloaded.")
        return 0
    print(format_plan("Stage B — particle merge (test time)", TODOS, notes))
    raise SystemExit(
        "TODO: particle merge is not wired to RealH3Map. "
        "Use frontier_step on tower features after a local 1-NFE decode. Do not download weights."
    )


if __name__ == "__main__":
    raise SystemExit(main())
