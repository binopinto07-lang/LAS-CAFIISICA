"""Regression for circular imports between terrain and cloud packages.

Run imports in separate processes: pytest may have already cached the modules.
"""
from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "first,second",
    (
        (
            "from las_classifier.terrain.dense_spatial_evidence import DenseSpatialEvidenceGrid",
            "from las_classifier.cloud.statistics import calculate_statistics",
        ),
        (
            "from las_classifier.cloud.statistics import calculate_statistics",
            "from las_classifier.terrain.dense_spatial_evidence import DenseSpatialEvidenceGrid",
        ),
        (
            "from las_classifier.terrain import inspect_source",
            "from las_classifier.cloud.model import CloudModel",
        ),
    ),
)
def test_cloud_terrain_imports_are_order_independent(first: str, second: str) -> None:
    code = "\n".join((first, second, "from las_classifier.terrain.source_inspector import inspect_source", "assert callable(inspect_source)"))
    result = subprocess.run(
        (sys.executable, "-c", code),
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert result.returncode == 0, (
        f"Import order: {first!r} then {second!r}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
