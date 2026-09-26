#!/usr/bin/env python3
"""Stage C — optional canvas FD.

Light Fréchet penalty on coarse 17-frame boards (layout, parts, sync),
compared with real or frozen high-NFE teacher canvases. Does not replace
Stage B or L_obs.

  python train/stage_c_canvas_fd.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train.wiring import STAGE_C_TODOS, format_plan, run_checks
from vfm.constants import VAE_CLIP_LENGTH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage C: coarse 17-frame canvas FD.")
    parser.add_argument("--clip-length", type=int, default=VAE_CLIP_LENGTH)
    parser.add_argument("--lambda-canvas", type=float, default=0.1)
    parser.add_argument("--canvas-source", choices=("real", "teacher"), default="real")
    parser.add_argument("--population", choices=("ema", "queue"), default="ema")
    parser.add_argument("--ema-beta", type=float, default=0.999)
    parser.add_argument("--queue-size", type=int, default=8192)
    parser.add_argument("--real-stats", type=Path, default=None, help="Canvas (mu_r, sigma_r). One source only.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.clip_length != VAE_CLIP_LENGTH:
        raise SystemExit(
            f"--clip-length {args.clip_length} does not match the on-disk VAE clip ({VAE_CLIP_LENGTH})."
        )
    notes = (
        f"clip_length={args.clip_length}  lambda_canvas={args.lambda_canvas}",
        f"canvas_source={args.canvas_source}  population={args.population}",
        f"real_stats={args.real_stats}",
        "Decode MiniMaxH3VideoVAE / AutoencoderKLMiniMaxH3 in 17-frame chunks, then pool.",
        "Do not mix real clips and many-step teacher decodes in one stats file.",
    )
    if args.check or not args.run:
        run_checks()
        print(format_plan("Stage C — canvas FD", STAGE_C_TODOS, notes))
    if not args.run:
        print("Scaffold only. --run stops until the TODOs are implemented. No weights downloaded.")
        return 0
    print(format_plan("Stage C — canvas FD", STAGE_C_TODOS, notes))
    raise SystemExit(
        "TODO: Stage C canvas FD is not implemented. "
        "Pool 17-frame decodes and call MultiTowerFD against a single teacher source."
    )


if __name__ == "__main__":
    raise SystemExit(main())
