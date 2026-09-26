#!/usr/bin/env python3
"""Stage A — constraint lock.

Train H3NoiseAdapter plus a thin Ref2VA LoRA at K in {4, 2}.
FD weight is 0. This scaffold does not load weights.

  python train/stage_a_vfm.py --check
  python train/stage_a_vfm.py --k 4 --alpha 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train.wiring import STAGE_A_TODOS, format_plan, run_checks
from vfm_fd.losses import NFE_WEIGHTS
from vfm_fd.schedule import fd_weight_for_k, stage_a_ks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage A: VFM adapter + thin LoRA (no FD).")
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        choices=sorted(set(stage_a_ks()) | {8}),
        help="NFEs. Default 4. 8 is an optional warmup. 1 belongs to Stage B.",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="P(z ~ N(0, I)). Default 0.5.")
    parser.add_argument("--lambda-obs", type=float, default=1.0)
    parser.add_argument("--kl-video", type=float, default=1e-3)
    parser.add_argument("--kl-audio", type=float, default=3e-3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lr-adapter", type=float, default=1e-4)
    parser.add_argument("--lr-lora", type=float, default=1e-4)
    parser.add_argument("--ema-beta", type=float, default=0.999, help="Parameter EMA decay for L_obs.")
    parser.add_argument("--ref2va-root", type=Path, default=_ROOT / "Ref2VA")
    parser.add_argument("--steps", type=int, default=0, help="Reserved for the real loop. Unused.")
    parser.add_argument("--check", action="store_true", help="Run numeric and wiring checks, then exit.")
    parser.add_argument("--run", action="store_true", help="Enter the training TODO (not implemented).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.k == 1:
        raise SystemExit("Stage A starts at K in {4, 2}, not 1. Use train/stage_b_fd_turbo.py for 1 NFE.")
    fd_weight = fd_weight_for_k(args.k)
    notes = (
        f"K={args.k}  fd_weight={fd_weight}  alpha={args.alpha}",
        f"adapter lr={args.lr_adapter}  lora lr={args.lr_lora}  rank={args.lora_rank}",
        f"kl_video={args.kl_video}  kl_audio={args.kl_audio}",
        f"theta EMA beta={args.ema_beta}  ref2va={args.ref2va_root}",
        "Objective: vfm_nfe_loss "
        + " ".join(f"{name}={weight}" for name, weight in NFE_WEIGHTS.items())
        + ". y = A(x) = x[:,:,0]. Text is conditioning. fd_weight stays 0. Not endpoint vfm_loss.",
    )
    if args.check or not args.run:
        run_checks()
        print(format_plan("Stage A — constraint lock", STAGE_A_TODOS, notes))
    if not args.run:
        print("Scaffold only. --run stops until the TODOs are implemented. No weights downloaded.")
        return 0
    print(format_plan("Stage A — constraint lock", STAGE_A_TODOS, notes))
    raise SystemExit(
        "TODO: Stage A training is not implemented. "
        "Use vfm_nfe_loss and y = x[:,:,0]. Do not download weights."
    )


if __name__ == "__main__":
    raise SystemExit(main())
