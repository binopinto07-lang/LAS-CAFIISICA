from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt

import numpy as np

from .model import CloudModel


@dataclass(frozen=True, slots=True)
class CloudStatistics:
    file_name: str
    las_version: str
    point_format: int
    point_count: int
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]
    height_range: float
    crs: str
    dimensions: tuple[str, ...]
    original_class_histogram: dict[int, int]
    approximate_xy_density: float | None
    approximate_point_spacing: float | None

    def as_display_lines(self) -> list[str]:
        density = "n/a" if self.approximate_xy_density is None else f"{self.approximate_xy_density:.3f} pts/m²"
        spacing = "n/a" if self.approximate_point_spacing is None else f"{self.approximate_point_spacing:.3f} m"
        histogram = ", ".join(f"{key}: {value}" for key, value in sorted(self.original_class_histogram.items())) or "empty"
        return [
            f"File: {self.file_name}",
            f"LAS version: {self.las_version}",
            f"Point format: {self.point_format}",
            f"Point count: {self.point_count:,}",
            f"Min XYZ: {self.min_xyz}",
            f"Max XYZ: {self.max_xyz}",
            f"Height range: {self.height_range:.3f} m",
            f"CRS: {self.crs}",
            f"Approx. XY density: {density}",
            f"Approx. point spacing: {spacing}",
            f"Original classification: {histogram}",
            f"Dimensions: {', '.join(self.dimensions)}",
        ]


def calculate_statistics(cloud: CloudModel) -> CloudStatistics:
    xyz = cloud.xyz
    if cloud.point_count:
        minimum = np.min(xyz, axis=0)
        maximum = np.max(xyz, axis=0)
        min_xyz = tuple(float(v) for v in minimum)
        max_xyz = tuple(float(v) for v in maximum)
        height_range = float(maximum[2] - minimum[2])
        area = float((maximum[0] - minimum[0]) * (maximum[1] - minimum[1]))
    else:
        min_xyz = max_xyz = (0.0, 0.0, 0.0)
        height_range = 0.0
        area = 0.0

    density = (cloud.point_count / area) if cloud.point_count and area > 0 else None
    spacing = (sqrt(1.0 / density)) if density and density > 0 else None

    parsed_crs = cloud.las.header.parse_crs()
    crs = "Unknown" if parsed_crs is None else parsed_crs.to_string()
    histogram = dict(Counter(int(v) for v in cloud.original_class.tolist()))

    return CloudStatistics(
        file_name=cloud.path.name,
        las_version=str(cloud.las.header.version),
        point_format=int(cloud.las.header.point_format.id),
        point_count=cloud.point_count,
        min_xyz=min_xyz,
        max_xyz=max_xyz,
        height_range=height_range,
        crs=crs,
        dimensions=tuple(cloud.las.point_format.dimension_names),
        original_class_histogram=histogram,
        approximate_xy_density=density,
        approximate_point_spacing=spacing,
    )
