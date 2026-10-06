"""R20.4 Ground -> MDT export with explicit observation provenance."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import laspy
import numpy as np
from scipy.ndimage import distance_transform_edt

LOGGER = logging.getLogger("las_cafiisica.terrain.mdt_export")
ProgressCallback = Callable[[int, str], None]
GROUND_CLASS = np.uint8(2)
OBSERVED_GROUND = np.uint8(1)
INTERPOLATED_MDT = np.uint8(2)
NO_GROUND_OBSERVATION = np.uint8(0)


def _scaled(raw, scale: float, offset: float) -> np.ndarray:
    return np.asarray(raw, dtype=np.float64) * float(scale) + float(offset)


def _classify(model, points, x, y, z) -> np.ndarray:
    evaluate = getattr(model, "evaluate_points", None)
    if evaluate is not None:
        return evaluate(points, x, y, z).classifications()
    method = getattr(model, "classify_points", None)
    if method is not None:
        return method(points, x, y, z)
    return model.classify_xyz(x, y, z)


def fill_small_mdt_gaps(
    elevation: np.ndarray,
    observed: np.ndarray,
    *,
    resolution_m: float,
    max_gap_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest support fills MDT cells only; it never creates measured Ground."""
    elevation = np.asarray(elevation, dtype=np.float32)
    observed = np.asarray(observed, dtype=np.bool_)
    if elevation.shape != observed.shape:
        raise ValueError("MDT elevation/observation shape mismatch")
    result = elevation.copy()
    state = np.full(observed.shape, NO_GROUND_OBSERVATION, dtype=np.uint8)
    state[observed] = OBSERVED_GROUND
    if not np.any(observed) or max_gap_m <= 0:
        return result, state

    distance, nearest = distance_transform_edt(~observed, return_indices=True)
    fill = (~observed) & (distance * float(resolution_m) <= float(max_gap_m))
    if np.any(fill):
        result[fill] = result[tuple(nearest[:, fill])]
        state[fill] = INTERPOLATED_MDT
    return result, state


@dataclass(slots=True)
class MDTPreview:
    """Computed terrain in RAM: no GeoTIFF exists until explicit EXPORT."""

    source: Path
    elevation: np.ndarray
    state: np.ndarray
    xmin: float
    ymax: float
    resolution_m: float
    max_gap_m: float
    ground_points: int

    def browser_payload(self, max_side: int = 256) -> dict:
        """Bound the viewport mesh size, independent of the full raster size."""
        if max_side < 2:
            raise ValueError("max_side must be >=2")
        h, w = self.elevation.shape
        stride = max(1, int(np.ceil(max(h, w) / max_side)))
        z = self.elevation[::stride, ::stride]
        state = self.state[::stride, ::stride]
        # Null, never a fake 0-height terrain vertex, for unknown cells.
        flat = z.ravel()
        values = [float(v) if np.isfinite(v) else None for v in flat]
        return {
            "width": int(z.shape[1]),
            "height": int(z.shape[0]),
            "xmin": self.xmin,
            "ymax": self.ymax,
            "resolution_m": self.resolution_m * stride,
            "z": values,
            "state": state.ravel().astype(np.uint8).tolist(),
        }


def build_ground_mdt_preview(
    source_path: str | Path,
    model,
    progress: ProgressCallback | None = None,
    *,
    resolution_m: float = 0.25,
    max_gap_m: float = 0.75,
    max_cells: int = 20_000_000,
) -> MDTPreview:
    """Compute Ground-derived MDT without opening a save dialog or writing files."""
    if resolution_m <= 0 or max_gap_m < 0:
        raise ValueError("Invalid MDT resolution/gap settings")
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    with laspy.open(source) as reader:
        crs = reader.header.parse_crs()
        if crs is None or crs.to_epsg() != 3763:
            raise ValueError("R20.4 MDT requires declared EPSG:3763")
        xmin, ymin, _ = map(float, reader.header.mins)
        xmax, ymax, _ = map(float, reader.header.maxs)
        width = max(1, int(np.floor((xmax - xmin) / resolution_m)) + 1)
        height = max(1, int(np.floor((ymax - ymin) / resolution_m)) + 1)
        cells = int(width) * int(height)
        if cells > int(max_cells):
            raise MemoryError(
                f"MDT grid has {cells:,} cells; increase resolution_m "
                f"or tile the raster (limit {max_cells:,})."
            )

        z_sum = np.zeros(cells, dtype=np.float64)
        count = np.zeros(cells, dtype=np.uint32)
        total = int(reader.header.point_count)
        scales = reader.header.scales
        offsets = reader.header.offsets
        chunk_size = min(int(getattr(model.params, "chunk_size", 750_000)), 750_000)
        processed = 0
        ground_points = 0

        for points in reader.chunk_iterator(chunk_size):
            x = _scaled(points.X, scales[0], offsets[0])
            y = _scaled(points.Y, scales[1], offsets[1])
            z = _scaled(points.Z, scales[2], offsets[2])
            classes = _classify(model, points, x, y, z)
            keep = (classes == GROUND_CLASS) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            if np.any(keep):
                col = np.floor((x[keep] - xmin) / resolution_m).astype(np.int64)
                row = np.floor((ymax - y[keep]) / resolution_m).astype(np.int64)
                np.clip(col, 0, width - 1, out=col)
                np.clip(row, 0, height - 1, out=row)
                flat = row * width + col
                z_sum += np.bincount(flat, weights=z[keep], minlength=cells)
                count += np.bincount(flat, minlength=cells).astype(np.uint32)
                ground_points += int(np.count_nonzero(keep))
            processed += len(points)
            if progress is not None and total:
                progress(
                    int(75 * processed / total),
                    f"R20.4 MDT: Ground {processed:,}/{total:,}",
                )

    count2 = count.reshape(height, width)
    observed = count2 > 0
    if not np.any(observed):
        raise ValueError("Current Ground model produced no MDT support cells")
    elevation = np.full((height, width), np.nan, dtype=np.float32)
    sum2 = z_sum.reshape(height, width)
    elevation[observed] = (sum2[observed] / count2[observed]).astype(np.float32)
    elevation, state = fill_small_mdt_gaps(
        elevation, observed, resolution_m=resolution_m, max_gap_m=max_gap_m
    )

    return MDTPreview(
        source=source,
        elevation=elevation,
        state=state,
        xmin=xmin,
        ymax=ymax,
        resolution_m=resolution_m,
        max_gap_m=max_gap_m,
        ground_points=ground_points,
    )


def write_ground_mdt(
    preview: MDTPreview,
    output_path: str | Path,
    progress: ProgressCallback | None = None,
) -> dict:
    """Export the exact MDT the operator previewed; no recomputation."""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError as exc:
        raise RuntimeError("R20.5 MDT export requires rasterio") from exc

    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("MDT output must be GeoTIFF (.tif/.tiff)")
    output.parent.mkdir(parents=True, exist_ok=True)
    state_path = output.with_name(output.stem + "_OBSERVATION_STATE.tif")
    report_path = output.with_suffix(".json")
    elevation = preview.elevation
    state = preview.state
    nodata = np.float32(-9999.0)
    transform = from_origin(
        preview.xmin, preview.ymax,
        preview.resolution_m, preview.resolution_m,
    )
    profile = {
        "driver": "GTiff", "width": int(elevation.shape[1]),
        "height": int(elevation.shape[0]), "count": 1,
        "dtype": "float32", "crs": "EPSG:3763",
        "transform": transform, "nodata": float(nodata),
        "compress": "deflate", "tiled": True,
    }
    if progress is not None:
        progress(50, "R20.5 MDT: escrever GeoTIFF")
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(
            np.where(np.isfinite(elevation), elevation, nodata).astype(np.float32),
            1,
        )
        dst.set_band_description(1, "MDT elevation metres")

    if progress is not None:
        progress(75, "R20.5 MDT: escrever mapa Observado/Interpolado")
    state_profile = dict(profile, dtype="uint8", nodata=255)
    with rasterio.open(state_path, "w", **state_profile) as dst:
        dst.write(state.astype(np.uint8), 1)
        dst.set_band_description(
            1, "0=NO_GROUND_OBSERVATION;1=MEASURED_GROUND;2=INTERPOLATED_MDT"
        )

    result = {
        "algorithm": "LAS_CAFIISICA_MDT_R20_5",
        "source": str(preview.source),
        "mdt": str(output),
        "observation_state": str(state_path),
        "crs": "EPSG:3763",
        "resolution_m": float(preview.resolution_m),
        "max_gap_m": float(preview.max_gap_m),
        "ground_points": int(preview.ground_points),
        "observed_cells": int(np.count_nonzero(state == OBSERVED_GROUND)),
        "interpolated_cells": int(np.count_nonzero(state == INTERPOLATED_MDT)),
        "no_ground_observation_cells": int(np.count_nonzero(state == NO_GROUND_OBSERVATION)),
        "interpolated_is_measured_ground": False,
    }
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("R20_5_MDT=%s", output)
    if progress is not None:
        progress(100, "R20.5 MDT exportado")
    return result


def export_ground_mdt(
    source_path: str | Path,
    output_path: str | Path,
    model,
    progress: ProgressCallback | None = None,
    *,
    resolution_m: float = 0.25,
    max_gap_m: float = 0.75,
    max_cells: int = 20_000_000,
    preview: MDTPreview | None = None,
) -> dict:
    """Compatibility API; optionally export a previously previewed MDT."""
    if preview is None:
        preview = build_ground_mdt_preview(
            source_path, model, progress,
            resolution_m=resolution_m, max_gap_m=max_gap_m,
            max_cells=max_cells,
        )
    elif Path(source_path).resolve() != preview.source:
        raise ValueError("Preview belongs to another LAS source")
    return write_ground_mdt(preview, output_path, progress)
