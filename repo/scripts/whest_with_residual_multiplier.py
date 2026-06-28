#!/usr/bin/env python3
"""Run the WhestBench CLI with a local residual-wall-time multiplier.

WhestBench 0.10.0 has a fixed residual conversion rate of 1e11 FLOPs/sec.
This wrapper lets repo-local Make targets pessimistically scale that residual
charge when local subprocess timing is known to understate the server path.
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args(argv: list[str]) -> tuple[float, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--residual-wall-time-multiplier",
        type=float,
        default=float(os.environ.get("WHEST_RESIDUAL_WALL_TIME_MULTIPLIER", "1.0")),
    )
    parser.add_argument("cli_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    cli_args = list(args.cli_args)
    if cli_args and cli_args[0] == "--":
        cli_args = cli_args[1:]
    if args.residual_wall_time_multiplier <= 0:
        raise SystemExit("--residual-wall-time-multiplier must be positive")
    return args.residual_wall_time_multiplier, cli_args


def main(argv: list[str] | None = None) -> int:
    multiplier, cli_args = _parse_args(list(sys.argv[1:] if argv is None else argv))

    import whestbench.budget as budget
    from whestbench.cli import main as whest_main

    lambda_flops_per_second = budget.LAMBDA_FLOPS_PER_SECOND * multiplier
    if cli_args and cli_args[0] == "run" and "--lambda-flops-per-second" not in cli_args:
        cli_args = [
            "run",
            "--lambda-flops-per-second",
            str(lambda_flops_per_second),
            *cli_args[1:],
        ]
    if multiplier != 1.0:
        print(
            "whest wrapper: residual wall time charged at "
            f"{multiplier:g}x ({lambda_flops_per_second:.3g} FLOPs/sec)",
            file=sys.stderr,
        )
    return int(whest_main(cli_args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
