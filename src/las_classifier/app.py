from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .logging_utils import configure_logging
from .self_test import run_self_test


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="LAS-CAFIISICA")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--self-test-output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.self_test:
        return run_self_test(args.self_test_output)

    log_path = configure_logging(debug=args.debug)
    logging.getLogger("las_cafiisica").info("LOG_FILE=%s", log_path)

    from PySide6.QtWidgets import QApplication
    from .main_window import MainWindow

    app = QApplication.instance() or QApplication([sys.argv[0]])
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
