from __future__ import annotations

import argparse
from pathlib import Path

from las_classifier.self_test import run_self_test


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    return run_self_test(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
