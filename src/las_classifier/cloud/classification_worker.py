from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ..classifiers.adaptive_ptd import run_adaptive_ptd
from ..classifiers.csf_engine import run_csf
from ..classifiers.hybrid_ground import run_hybrid_ground
from ..classifiers.l3_dense_ground import run_l3_dense_ground
from ..classifiers.l3_inverted_ground import run_l3_inverted_ground
from ..classifiers.l3_mantle_veto import run_l3_mantle_veto
from ..classifiers.l3_ground_continuity import run_l3_ground_continuity
from ..classifiers.universal_ground import (
    run_universal_ground,
    run_universal_ground_r2051,
    run_universal_ground_r204,
)
from ..terrain.mantle_export import export_mantle_diagnostic
from ..terrain.mdt_export import export_ground_mdt, build_ground_mdt_preview
from ..classifiers.l3_ground_lab import run_l3_ground_lab
from ..classifiers.smrf import SMRFParams, SMRFResult, run_smrf
from ..ground.ground_export import export_ground_only
from ..ground.types import GroundEngineParams
from .exporter import export_classified
from .model import CloudModel


LOGGER = logging.getLogger(
    "las_cafiisica.cloud.classification_worker"
)


class SMRFWorker(QThread):
    """Compatibility worker retained for the legacy engine."""

    completed = Signal(object)
    failed = Signal(str)
    progress_changed = Signal(int, str)

    def __init__(
        self,
        cloud: CloudModel,
        params: SMRFParams,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.cloud = cloud
        self.params = params

    def run(self) -> None:
        try:
            result = run_smrf(
                self.cloud,
                self.params,
                lambda percent, message: self.progress_changed.emit(
                    percent,
                    message,
                ),
            )
        except Exception as exc:
            LOGGER.exception("SMRF classification failed")
            self.failed.emit(str(exc))
            return

        self.completed.emit(result)


class GroundEngineWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)
    progress_changed = Signal(int, str)

    def __init__(
        self,
        cloud: CloudModel,
        engine_name: str,
        params: GroundEngineParams,
        smrf_params: SMRFParams | None = None,
        source_override: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.cloud = cloud
        self.engine_name = engine_name
        self.params = params
        self.smrf_params = smrf_params
        self.source_override = source_override

    def run(self) -> None:
        callback = (
            lambda percent, message: self.progress_changed.emit(
                percent,
                message,
            )
        )
        try:
            if self.engine_name == "Universal Ground R20.6.2":
                result = run_universal_ground(self.cloud, self.params, callback)
            elif self.engine_name == "Universal Ground R20.5.1":
                result = run_universal_ground_r2051(
                    self.cloud, self.params, callback
                )
            elif self.engine_name == "Universal Ground R20.4":
                result = run_universal_ground_r204(
                    self.cloud, self.params, callback
                )
            elif self.engine_name == "L3 Ground Continuity R20.3":
                result = run_l3_ground_continuity(
                    self.cloud, self.params, callback,
                    source_override=self.source_override,
                )
            elif self.engine_name == "L3 Inverted Ground R20.1":
                result = run_l3_mantle_veto(
                    self.cloud, self.params, callback,
                    source_override=self.source_override,
                )
            elif self.engine_name == "L3 Inverted Ground R20":
                result = run_l3_inverted_ground(
                    self.cloud, self.params, callback,
                    source_override=self.source_override,
                )
            elif self.engine_name == "L3 Dense Ground R19":
                result = run_l3_dense_ground(
                    self.cloud,
                    self.params,
                    callback,
                    source_override=self.source_override,
                )
            elif self.engine_name == "L3 Ground Lab":
                result = run_l3_ground_lab(
                    self.cloud,
                    self.params,
                    callback,
                    source_override=self.source_override,
                )
            elif self.engine_name == "Adaptive PTD":
                result = run_adaptive_ptd(
                    self.cloud,
                    self.params,
                    callback,
                )
            elif self.engine_name == "CSF":
                result = run_csf(
                    self.cloud,
                    self.params,
                    callback,
                )
            elif self.engine_name == "SMRF Legacy":
                params = self.smrf_params or SMRFParams()
                result = run_smrf(
                    self.cloud,
                    params,
                    callback,
                )
            else:
                result = run_hybrid_ground(
                    self.cloud,
                    self.params,
                    callback,
                    source_override=self.source_override,
                )
        except Exception as exc:
            LOGGER.exception(
                "Ground engine failed: %s",
                self.engine_name,
            )
            self.failed.emit(str(exc))
            return

        self.completed.emit(result)


class ClassifiedExportWorker(QThread):
    """Compatibility export of the full classified cloud."""

    completed = Signal(str)
    failed = Signal(str)
    progress_changed = Signal(int, str)

    def __init__(
        self,
        source_path: Path,
        output_path: Path,
        result: SMRFResult,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.source_path = source_path
        self.output_path = output_path
        self.result = result

    def run(self) -> None:
        try:
            path = export_classified(
                self.source_path,
                self.output_path,
                self.result.model,
                lambda percent, message: self.progress_changed.emit(
                    percent,
                    message,
                ),
            )
        except Exception as exc:
            LOGGER.exception("Classified export failed")
            self.failed.emit(str(exc))
            return

        self.completed.emit(str(path))


class GroundExportWorker(QThread):
    completed = Signal(str)
    failed = Signal(str)
    progress_changed = Signal(int, str)

    def __init__(
        self,
        source_path: Path,
        output_path: Path,
        result,
        include_synthetic: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.source_path = source_path
        self.output_path = output_path
        self.result = result
        self.include_synthetic = include_synthetic

    def run(self) -> None:
        try:
            path = export_ground_only(
                self.source_path,
                self.output_path,
                self.result.model,
                lambda percent, message: self.progress_changed.emit(
                    percent,
                    message,
                ),
                include_synthetic=self.include_synthetic,
            )
        except Exception as exc:
            LOGGER.exception("Ground-only export failed")
            self.failed.emit(str(exc))
            return

        self.completed.emit(str(path))


class MantleExportWorker(QThread):
    """Write the R20 mantle separately from real measured Ground."""

    completed = Signal(str)
    failed = Signal(str)
    progress_changed = Signal(int, str)

    def __init__(self, output_path: Path, result, parent=None) -> None:
        super().__init__(parent)
        self.output_path = Path(output_path)
        self.result = result

    def run(self) -> None:
        try:
            mantle = getattr(self.result.model, "mantle", None)
            if mantle is None:
                raise ValueError("R20 inverted mantle is not available")
            path = export_mantle_diagnostic(
                mantle, self.output_path,
                lambda p, message: self.progress_changed.emit(p, message),
            )
        except Exception as exc:
            LOGGER.exception("R20 mantle export failed")
            self.failed.emit(str(exc))
            return
        self.completed.emit(str(path))


class MDTPreviewWorker(QThread):
    """Build measured/interpolated terrain in RAM; never export before review."""

    completed = Signal(object)
    failed = Signal(str)
    progress_changed = Signal(int, str)

    def __init__(
        self,
        source_path: Path,
        result,
        resolution_m: float = 0.25,
        max_gap_m: float = 0.75,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.source_path = Path(source_path)
        self.result = result
        self.resolution_m = float(resolution_m)
        self.max_gap_m = float(max_gap_m)

    def run(self) -> None:
        try:
            preview = build_ground_mdt_preview(
                self.source_path,
                self.result.model,
                lambda p, m: self.progress_changed.emit(p, m),
                resolution_m=self.resolution_m,
                max_gap_m=self.max_gap_m,
            )
        except Exception as exc:
            LOGGER.exception("R20.6.3 MDT preview failed")
            self.failed.emit(str(exc))
            return
        self.completed.emit(preview)


class MDTExportWorker(QThread):
    """Export reviewed R20.6.3 MDT + observation-state raster."""

    completed = Signal(object)
    failed = Signal(str)
    progress_changed = Signal(int, str)

    def __init__(
        self,
        source_path: Path,
        output_path: Path,
        result,
        resolution_m: float = 0.25,
        max_gap_m: float = 0.75,
        parent=None,
        preview=None,
    ) -> None:
        super().__init__(parent)
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.result = result
        self.resolution_m = float(resolution_m)
        self.max_gap_m = float(max_gap_m)
        self.preview = preview

    def run(self) -> None:
        try:
            info = export_ground_mdt(
                self.source_path,
                self.output_path,
                self.result.model,
                lambda percent, message: self.progress_changed.emit(percent, message),
                resolution_m=self.resolution_m,
                max_gap_m=self.max_gap_m,
                preview=self.preview,
            )
        except Exception as exc:
            LOGGER.exception("R20.6.3 MDT export failed")
            self.failed.emit(str(exc))
            return
        self.completed.emit(info)
