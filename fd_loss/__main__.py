"""``python -m fd_loss`` runs the numeric self-check. Stats job: ``python -m fd_loss.offline_real_stats``."""

from __future__ import annotations

from fd_loss.selfcheck import run_selfcheck


def main() -> int:
    run_selfcheck()
    print("fd_loss selfcheck ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
