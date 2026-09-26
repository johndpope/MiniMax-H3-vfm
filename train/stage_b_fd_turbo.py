#!/usr/bin/env python3
"""Stage B — FD turbo.

Post-train the Ref2VA LoRA at 1 NFE. The adapter stays frozen (tiny LR optional).
FD-loss is the population term that is meant to make that single step match
real audio-video features. L_obs stays in the objective.

  python train/stage_b_fd_turbo.py --check
  python train/stage_b_fd_turbo.py --population queue --queue-size 8192
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train.wiring import STAGE_B_TODOS, format_plan, run_checks
from vfm_fd.schedule import fd_weight_for_k


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage B: 1-NFE LoRA post-train with FD-loss.")
    parser.add_argument("--k", type=int, default=1, choices=[1], help="Fixed at 1 NFE.")
    parser.add_argument("--population", choices=("ema", "queue"), default="ema")
    parser.add_argument("--ema-beta", type=float, default=0.999, help="Feature-stat EMA decay.")
    parser.add_argument("--queue-size", type=int, default=8192, help="Queue slots. Use 8192-32768.")
    parser.add_argument("--adapter-lr", type=float, default=0.0, help="0 freezes q_phi. Tiny LR is optional.")
    parser.add_argument("--lr-lora", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lambda-obs", type=float, default=1.0)
    parser.add_argument("--fd-weight", type=float, default=None, help="Default: anneal value at K=1 (1.0).")
    parser.add_argument("--real-stats", type=Path, default=None, help="Offline (mu_r, sigma_r) .npz.")
    parser.add_argument("--towers", nargs="*", default=None, help="Subset of fd_loss.towers.DEFAULT_TOWERS.")
    parser.add_argument("--stage-a-checkpoint", type=Path, default=None, help="Local Stage A dir. Not loaded.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 8192 <= args.queue_size <= 32768 and args.population == "queue":
        raise SystemExit("--queue-size must sit in [8192, 32768] for a Stage B queue")
    fd_weight = fd_weight_for_k(1) if args.fd_weight is None else args.fd_weight
    notes = (
        f"K=1  fd_weight={fd_weight}  population={args.population}  ema_beta={args.ema_beta}",
        f"queue_size={args.queue_size}  adapter_lr={args.adapter_lr}  lora_lr={args.lr_lora}",
        f"lambda_obs={args.lambda_obs}  real_stats={args.real_stats}  towers={args.towers or 'all'}",
        "Samples: z ~ q_phi, one NFE, frozen VAE decode, frozen towers, grad through the decode into the LoRA.",
        "Loss: vfm_nfe_loss + fd_weight * sum_i w_i FD_i. y stays the first frame.",
    )
    if args.check or not args.run:
        run_checks()
        print(format_plan("Stage B — FD turbo (1 NFE)", STAGE_B_TODOS, notes))
    if not args.run:
        print("Scaffold only. --run stops until the TODOs are implemented. No weights downloaded.")
        return 0
    if args.real_stats is None:
        raise SystemExit("TODO: pass --real-stats from fd_loss.offline_real_stats before a real Stage B run.")
    print(format_plan("Stage B — FD turbo (1 NFE)", STAGE_B_TODOS, notes))
    raise SystemExit(
        "TODO: Stage B training is not implemented. "
        "Freeze H3NoiseAdapter, post-train the LoRA with MultiTowerFD at 1 NFE."
    )


if __name__ == "__main__":
    raise SystemExit(main())
