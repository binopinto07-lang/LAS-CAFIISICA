from __future__ import annotations

import importlib
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from . import __version__
from .cloud.loader import (
    IGNORE_INPUT_CLASSIFICATION,
    load_cloud,
)
from .cloud.statistics import calculate_statistics


REPORT_NAME = "LAS_CAFIISICA_SELF_TEST.txt"


def _ok(lines: list[str], key: str) -> None:
    lines.append(f"{key}=OK")


def run_self_test(output_dir: str | Path | None = None) -> int:
    destination = Path(output_dir or Path.cwd()).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / REPORT_NAME
    lines = [
        "LAS-CAFIISICA SELF TEST",
        "",
        f"VERSION={__version__}",
    ]

    try:
        import scipy  # noqa: F401
        import laspy
        import pyproj  # noqa: F401
        import PySide6  # noqa: F401

        _ok(lines, "PYTHON")
        _ok(lines, "NUMPY")
        _ok(lines, "SCIPY")
        _ok(lines, "LASPY")
        _ok(lines, "PYPROJ")
        importlib.import_module("lazrs")
        _ok(lines, "LAZ")
        _ok(lines, "PYSIDE6")

        if not IGNORE_INPUT_CLASSIFICATION:
            raise RuntimeError(
                "Input classification must be ignored by default"
            )

        with TemporaryDirectory(
            prefix="las_cafiisica_selftest_"
        ) as tmp:
            temp = Path(tmp)
            header = laspy.LasHeader(
                point_format=3,
                version="1.2",
            )
            las = laspy.LasData(header)
            las.x = np.array([0.0, 1.0, 2.0, 3.0])
            las.y = np.array([0.0, 1.0, 0.0, 1.0])
            las.z = np.array([10.0, 10.5, 11.0, 11.5])
            las.classification = np.array(
                [5, 2, 1, 7],
                dtype=np.uint8,
            )

            las_path = temp / "synthetic.las"
            laz_path = temp / "synthetic.laz"
            las.write(las_path)
            las.write(laz_path)
            _ok(lines, "WRITE_LAS")
            _ok(lines, "WRITE_LAZ")

            cloud_las = load_cloud(las_path)
            cloud_laz = load_cloud(laz_path)
            np.testing.assert_array_equal(
                cloud_las.original_class,
                [5, 2, 1, 7],
            )
            np.testing.assert_array_equal(
                cloud_las.working_class,
                [0, 0, 0, 0],
            )
            np.testing.assert_array_equal(
                cloud_laz.original_class,
                [5, 2, 1, 7],
            )
            np.testing.assert_array_equal(
                cloud_laz.working_class,
                [0, 0, 0, 0],
            )
            _ok(lines, "READ_LAS")
            _ok(lines, "READ_LAZ")
            _ok(lines, "ORIGINAL_CLASS")
            _ok(lines, "IGNORE_INPUT_CLASSIFICATION")

            stats = calculate_statistics(cloud_las)
            if (
                stats.point_count != 4
                or stats.height_range != 1.5
            ):
                raise RuntimeError(
                    "Cloud statistics self-test failed"
                )
            _ok(lines, "STATISTICS")

        lines.extend(["", "RESULT=PASS"])
        report_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        return 0
    except Exception:
        lines.extend(
            ["", "RESULT=FAIL", "", traceback.format_exc()]
        )
        report_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )
        return 1
