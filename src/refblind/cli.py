from __future__ import annotations

import argparse

from .errors import decompose_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RefBlind scientific bookkeeping utilities")
    parser.add_argument("--mlip", type=float, required=True)
    parser.add_argument("--dft", type=float, required=True)
    parser.add_argument("--high-level", dest="high_level", type=float, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = decompose_error(args.mlip, args.dft, args.high_level)
    print(f"surrogate_error={result.surrogate_error:.8g}")
    print(f"reference_error={result.reference_error:.8g}")
    print(f"total_error={result.total_error:.8g}")


if __name__ == "__main__":
    main()
