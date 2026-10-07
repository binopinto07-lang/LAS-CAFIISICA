"""R20.6 Ground Complete -> MDT preview with measured/reconstructed provenance."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import laspy
import numpy as np
from scipy.ndimage import distance_transform_edt, zoom

LOGGER = logging.getLogger("las_cafiisica.terrain.mdt_export")
ProgressCallback = Callable[[int, str], None]
GROUND_CLASS = np.uint8(2)
OBSERVED_GROUND = np.uint8(1)
RECONSTRUCTED_GROUND = np.uint8(2)
INTERPOLATED_MDT = np.uint8(3)
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
    support: np.ndarray,
    *,
    resolution_m: float,
    max_gap_m: float,
    base_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill tiny raster-only gaps without changing Ground provenance."""
    elevation = np.asarray(elevation, dtype=np.float32)
    support = np.asarray(support, dtype=np.bool_)
    if elevation.shape != support.shape:
        raise ValueError("MDT elevation/support shape mismatch")
    result = elevation.copy()
    if base_state is None:
        state = np.full(support.shape, NO_GROUND_OBSERVATION, dtype=np.uint8)
        state[support] = OBSERVED_GROUND
    else:
        state = np.asarray(base_state, dtype=np.uint8).copy()
        if state.shape != support.shape:
            raise ValueError("MDT state/support shape mismatch")
    if not np.any(support) or max_gap_m <= 0:
        return result, state

    distance, nearest = distance_transform_edt(~support, return_indices=True)
    fill = (~support) & (distance * float(resolution_m) <= float(max_gap_m))
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
    reconstructed_points: int = 0

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


def _model_surface_mdt_preview(
    source: Path,
    model,
    *,
    resolution_m: float,
    max_gap_m: float,
    max_cells: int,
) -> MDTPreview | None:
    """Fast R20.7 path: build MDT from the already solved terrain model.

    No second pass over the 318M-point source is required. Measured Ground uses
    the preserved mantle surface in cells where points survived FINAL GROUND;
    reconstructed Ground uses the separate reconstruction surface.
    """
    mantle = getattr(model, "mantle", None)
    reconstruction = getattr(model, "reconstruction", None)
    measured_cells = getattr(model, "measured_ground_cells", None)
    if mantle is None or reconstruction is None or measured_cells is None:
        return None

    measured = np.asarray(measured_cells, dtype=np.bool_)
    reconstructed = np.asarray(reconstruction.fill_mask, dtype=np.bool_)
    shape = (int(mantle.ny), int(mantle.nx))
    if measured.shape != shape or reconstructed.shape != shape:
        raise ValueError("R20.7 model-surface MDT shape mismatch")

    support = measured | reconstructed
    if not np.any(support):
        raise ValueError("R20.7 model contains no Ground support cells")

    source_surface = np.full(shape, np.nan, dtype=np.float32)
    mantle_surface = np.asarray(mantle.surface, dtype=np.float32).reshape(shape)
    reconstruction_surface = np.asarray(
        reconstruction.surface, dtype=np.float32
    ).reshape(shape)
    source_surface[measured] = mantle_surface[measured]
    source_surface[reconstructed & ~measured] = (
        reconstruction_surface[reconstructed & ~measured]
    )

    state_src = np.full(shape, NO_GROUND_OBSERVATION, dtype=np.uint8)
    state_src[measured] = OBSERVED_GROUND
    state_src[reconstructed & ~measured] = RECONSTRUCTED_GROUND

    cell = float(mantle.cell_size)
    xmin = float(mantle.origin[0])
    ymin = float(mantle.origin[1])
    xmax = xmin + int(mantle.nx) * cell
    ymax = ymin + int(mantle.ny) * cell

    width = max(1, int(np.floor((xmax - xmin) / resolution_m)) + 1)
    height = max(1, int(np.floor((ymax - ymin) / resolution_m)) + 1)
    cells = int(width) * int(height)
    if cells > int(max_cells):
        raise MemoryError(
            f"MDT grid has {cells:,} cells; increase resolution_m "
            f"or tile the raster (limit {max_cells:,})."
        )

    # Fill numeric support only for interpolation, then reapply the categorical
    # support mask. This never turns an unsupported cell into Ground.
    nearest = distance_transform_edt(
        ~support,
        return_distances=False,
        return_indices=True,
    )
    numeric = source_surface[tuple(nearest)].astype(np.float32, copy=False)

    scale_y = height / shape[0]
    scale_x = width / shape[1]
    elevation = zoom(
        numeric,
        (scale_y, scale_x),
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32, copy=False)
    state = zoom(
        state_src,
        (scale_y, scale_x),
        order=0,
        mode="nearest",
        prefilter=False,
    ).astype(np.uint8, copy=False)

    # scipy may round shape by one cell depending on scale.
    elevation = elevation[:height, :width]
    state = state[:height, :width]
    if elevation.shape != (height, width) or state.shape != (height, width):
        padded_elevation = np.full((height, width), np.nan, dtype=np.float32)
        padded_state = np.zeros((height, width), dtype=np.uint8)
        hh = min(height, elevation.shape[0])
        ww = min(width, elevation.shape[1])
        padded_elevation[:hh, :ww] = elevation[:hh, :ww]
        padded_state[:hh, :ww] = state[:hh, :ww]
        elevation, state = padded_elevation, padded_state

    elevation[state == NO_GROUND_OBSERVATION] = np.nan

    # Mantle rows start at ymin; GeoTIFF rows start at ymax.
    elevation = np.flipud(elevation)
    state = np.flipud(state)

    elevation, state = fill_small_mdt_gaps(
        elevation,
        state > NO_GROUND_OBSERVATION,
        resolution_m=resolution_m,
        max_gap_m=max_gap_m,
        base_state=state,
    )

    return MDTPreview(
        source=source,
        elevation=elevation,
        state=state,
        xmin=xmin,
        ymax=ymax,
        resolution_m=resolution_m,
        max_gap_m=max_gap_m,
        ground_points=int(
            getattr(model, "measured_ground_point_count", 0)
        ),
        reconstructed_points=int(
            getattr(model, "synthetic_fill_point_count", 0)
        ),
    )


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
            raise ValueError("R20.7 MDT requires declared EPSG:3763")

    fast_preview = _model_surface_mdt_preview(
        source,
        model,
        resolution_m=resolution_m,
        max_gap_m=max_gap_m,
        max_cells=max_cells,
    )
    if fast_preview is not None:
        if progress is not None:
            progress(100, "R20.7 MDT: preview criado do modelo Ground")
        return fast_preview

    with laspy.open(source) as reader:
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
                    f"R20.7 MDT: Ground {processed:,}/{total:,}",
                )

    measured_count = count.reshape(height, width)
    measured = measured_count > 0
    measured_sum = z_sum.reshape(height, width)

    synthetic_sum = np.zeros(cells, dtype=np.float64)
    synthetic_count = np.zeros(cells, dtype=np.uint32)
    reconstructed_points = 0
    iterator = getattr(model, "iter_synthetic_fill_xyz", None)
    if iterator is not None:
        for sx, sy, sz in iterator():
            sx = np.asarray(sx, dtype=np.float64)
            sy = np.asarray(sy, dtype=np.float64)
            sz = np.asarray(sz, dtype=np.float64)
            valid = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(sz)
            if not np.any(valid):
                continue
            col = np.floor((sx[valid] - xmin) / resolution_m).astype(np.int64)
            row = np.floor((ymax - sy[valid]) / resolution_m).astype(np.int64)
            inside = (
                (col >= 0) & (col < width)
                & (row >= 0) & (row < height)
            )
            if not np.any(inside):
                continue
            col = col[inside]
            row = row[inside]
            zz = sz[valid][inside]
            flat = row * width + col
            synthetic_sum += np.bincount(flat, weights=zz, minlength=cells)
            synthetic_count += np.bincount(flat, minlength=cells).astype(np.uint32)
            reconstructed_points += int(flat.size)

    reconstructed_count = synthetic_count.reshape(height, width)
    reconstructed = (reconstructed_count > 0) & ~measured
    reconstructed_sum = synthetic_sum.reshape(height, width)
    support = measured | reconstructed
    if not np.any(support):
        raise ValueError("Current Ground model produced no MDT support cells")

    elevation = np.full((height, width), np.nan, dtype=np.float32)
    elevation[measured] = (
        measured_sum[measured] / measured_count[measured]
    ).astype(np.float32)
    elevation[reconstructed] = (
        reconstructed_sum[reconstructed] / reconstructed_count[reconstructed]
    ).astype(np.float32)

    state = np.full((height, width), NO_GROUND_OBSERVATION, dtype=np.uint8)
    state[measured] = OBSERVED_GROUND
    state[reconstructed] = RECONSTRUCTED_GROUND
    elevation, state = fill_small_mdt_gaps(
        elevation,
        support,
        resolution_m=resolution_m,
        max_gap_m=max_gap_m,
        base_state=state,
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
        reconstructed_points=reconstructed_points,
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
        raise RuntimeError("R20.7 MDT export requires rasterio") from exc

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
        progress(50, "R20.7 MDT: escrever GeoTIFF")
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(
            np.where(np.isfinite(elevation), elevation, nodata).astype(np.float32),
            1,
        )
        dst.set_band_description(1, "MDT elevation metres")

    if progress is not None:
        progress(75, "R20.7 MDT: escrever mapa Observado/Interpolado")
    state_profile = dict(profile, dtype="uint8", nodata=255)
    with rasterio.open(state_path, "w", **state_profile) as dst:
        dst.write(state.astype(np.uint8), 1)
        dst.set_band_description(
            1,
            "0=NO_GROUND_OBSERVATION;1=MEASURED_GROUND;"
            "2=RECONSTRUCTED_GROUND;3=INTERPOLATED_MDT",
        )

    result = {
        "algorithm": "LAS_CAFIISICA_MDT_R20_7",
        "source": str(preview.source),
        "mdt": str(output),
        "observation_state": str(state_path),
        "crs": "EPSG:3763",
        "resolution_m": float(preview.resolution_m),
        "max_gap_m": float(preview.max_gap_m),
        "ground_points": int(preview.ground_points),
        "reconstructed_points": int(preview.reconstructed_points),
        "observed_cells": int(np.count_nonzero(state == OBSERVED_GROUND)),
        "reconstructed_cells": int(np.count_nonzero(state == RECONSTRUCTED_GROUND)),
        "interpolated_cells": int(np.count_nonzero(state == INTERPOLATED_MDT)),
        "no_ground_observation_cells": int(np.count_nonzero(state == NO_GROUND_OBSERVATION)),
        "reconstructed_is_measured_ground": False,
        "interpolated_is_measured_ground": False,
    }
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("R20_7_MDT=%s", output)
    if progress is not None:
        progress(100, "R20.7 MDT exportado")
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
