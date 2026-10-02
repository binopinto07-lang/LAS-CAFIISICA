from __future__ import annotations

import gc
import logging
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .classifiers.smrf import SMRFParams
from .cloud.classification_worker import (
    GroundEngineWorker,
    GroundExportWorker,
    MantleExportWorker,
)
from .cloud.worker import CloudLoadWorker
from .ground.types import GroundEngineParams
from .viewer.widget import PointCloudViewer
from .viewer.workers import ViewerPrepareWorker


LOGGER = logging.getLogger("las_cafiisica.ui")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"LAS-CAFIISICA {__version__}")
        self.resize(1540, 920)
        self.setMinimumSize(1200, 740)

        self._cloud = None
        self._ground_result = None
        self._loader: CloudLoadWorker | None = None
        self._ground_worker: GroundEngineWorker | None = None
        self._export_worker: GroundExportWorker | None = None
        self._mantle_worker: MantleExportWorker | None = None
        self._viewer_worker: ViewerPrepareWorker | None = None
        self._pending_filename: str | None = None
        self._viewer_loaded: set[str] = set()

        self.open_button = QPushButton("OPEN LAS / LAZ")
        self.open_button.clicked.connect(self.open_cloud)

        self.file_label = QLabel("No cloud loaded")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.file_label.setWordWrap(True)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)

        engine_box = QGroupBox("GROUND ENGINE")
        form = QFormLayout()

        self.engine_combo = QComboBox()
        self.engine_combo.addItems(
            [
                "L3 Inverted Ground R20",
                "L3 Dense Ground R19",
                "L3 Ground Lab",
                "Hybrid",
                "Adaptive PTD",
                "CSF",
                "SMRF Legacy",
            ]
        )
        self.engine_combo.setCurrentText("L3 Inverted Ground R20")

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(
            ["Fast", "Balanced", "High", "Extreme"]
        )
        self.quality_combo.setCurrentText("Balanced")

        self.profile_combo = QComboBox()
        self.profile_combo.addItems(
            [
                "General",
                "Mountain / Talude",
                "Forest",
                "Photogrammetry",
                "LiDAR",
                "High Density",
            ]
        )
        self.profile_combo.setCurrentText("Mountain / Talude")

        self.source_combo = QComboBox()
        self.source_combo.addItems(
            [
                "Auto detect",
                "DJI Zenmuse L3 / LiDAR",
                "DJI Zenmuse P1 / Photogrammetry",
                "Unknown",
            ]
        )
        self.source_combo.setCurrentText("Auto detect")

        self.fill_spacing_spin = self._spin(
            0.0,
            0.0,
            1.00,
            0.05,
            2,
        )
        self.fill_spacing_spin.setSpecialValueText("Auto")

        self.include_synthetic = QCheckBox(
            "Include reconstructed ground"
        )
        self.include_synthetic.setChecked(True)
        self.engine_combo.currentTextChanged.connect(
            self._engine_changed
        )
        self._engine_changed(
            self.engine_combo.currentText()
        )

        form.addRow("Engine", self.engine_combo)
        form.addRow("Ground quality", self.quality_combo)
        form.addRow("Profile", self.profile_combo)
        form.addRow("Source", self.source_combo)
        form.addRow(
            "Synthetic spacing (m)",
            self.fill_spacing_spin,
        )
        form.addRow("", self.include_synthetic)

        legacy_box = QGroupBox("SMRF LEGACY / ADVANCED")
        legacy_form = QFormLayout()
        self.cell_spin = self._spin(1.0, 0.10, 20.0, 0.10, 2)
        self.slope_spin = self._spin(0.15, 0.0, 5.0, 0.01, 3)
        self.window_spin = self._spin(18.0, 1.0, 200.0, 1.0, 1)
        self.threshold_spin = self._spin(0.50, 0.0, 10.0, 0.05, 2)
        self.scalar_spin = self._spin(1.25, 0.10, 10.0, 0.05, 2)
        legacy_form.addRow("Cell (m)", self.cell_spin)
        legacy_form.addRow("Slope", self.slope_spin)
        legacy_form.addRow("Window (m)", self.window_spin)
        legacy_form.addRow("Threshold (m)", self.threshold_spin)
        legacy_form.addRow("Scalar", self.scalar_spin)
        legacy_box.setLayout(legacy_form)

        buttons = QHBoxLayout()
        self.ground_button = QPushButton("RUN GROUND ENGINE")
        self.ground_button.setEnabled(False)
        self.ground_button.clicked.connect(self.run_ground_engine)

        self.export_button = QPushButton("EXPORT GROUND ONLY")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_ground_only)
        self.mantle_export_button = QPushButton("EXPORT MANTO (LAZ)")
        self.mantle_export_button.setToolTip(
            "Separate R20 diagnostic surface: inferred cells are NOT measured Ground."
        )
        self.mantle_export_button.setEnabled(False)
        self.mantle_export_button.clicked.connect(self.export_inverted_mantle)

        buttons.addWidget(self.ground_button)
        buttons.addWidget(self.export_button)
        form.addRow(buttons)
        form.addRow(self.mantle_export_button)
        engine_box.setLayout(form)

        self.statistics_view = QPlainTextEdit()
        self.statistics_view.setReadOnly(True)
        self.statistics_view.setPlaceholderText(
            "Open a LAS/LAZ file to analyse the cloud."
        )

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.open_button)
        left_layout.addWidget(self.file_label)
        left_layout.addWidget(self.progress)
        left_layout.addWidget(engine_box)
        left_layout.addWidget(legacy_box)
        left_layout.addWidget(self.statistics_view, 1)
        left.setMinimumWidth(410)

        self.viewer = PointCloudViewer(self)
        self.viewer.mantle_requested.connect(self._request_mantle_view)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 1110])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Ready")

    @staticmethod
    def _spin(
        value: float,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    def _busy(
        self,
        active: bool,
        message: str = "",
    ) -> None:
        self.progress.setVisible(active)
        self.open_button.setEnabled(not active)
        self.ground_button.setEnabled(
            not active and self._cloud is not None
        )
        self.export_button.setEnabled(
            not active and self._ground_result is not None
        )
        self.mantle_export_button.setEnabled(
            not active
            and getattr(
                getattr(self._ground_result, "model", None), "mantle", None
            ) is not None
        )
        if active:
            self.statusBar().showMessage(message)

    def _set_progress(
        self,
        percent: int,
        message: str,
    ) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(percent)
        self.statusBar().showMessage(message)

    def open_cloud(self) -> None:
        if any(
            worker is not None
            for worker in (
                self._loader,
                self._ground_worker,
                self._export_worker,
                self._mantle_worker,
                self._viewer_worker,
            )
        ):
            return

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open LAS/LAZ",
            "",
            "LAS/LAZ point clouds (*.las *.laz)",
        )
        if not filename:
            return

        self._cloud = None
        self._ground_result = None
        self._viewer_loaded.clear()
        self.viewer.clear()
        gc.collect()

        self._pending_filename = filename
        self.progress.setRange(0, 0)
        self._busy(True, "Loading point cloud in background...")

        worker = CloudLoadWorker(filename, self)
        worker.loaded.connect(self._load_succeeded)
        worker.failed.connect(self._load_failed)
        worker.finished.connect(self._load_finished)
        self._loader = worker
        worker.start()

    def _load_succeeded(self, cloud, stats) -> None:
        self._cloud = cloud
        filename = self._pending_filename or str(cloud.path)
        self.file_label.setText(str(Path(filename)))
        self.statistics_view.setPlainText(
            "\n".join(stats.as_display_lines())
        )
        self.statusBar().showMessage(
            f"Loaded {cloud.point_count:,} points"
        )

    def _load_failed(self, message: str) -> None:
        LOGGER.error("Failed to load cloud: %s", message)
        self.statusBar().showMessage("Load failed")
        QMessageBox.critical(
            self,
            "LAS-CAFIISICA",
            f"Could not open file:\n{message}",
        )

    def _load_finished(self) -> None:
        worker = self._loader
        self._loader = None
        self._pending_filename = None
        self.progress.setRange(0, 100)
        self._busy(False)
        if worker is not None:
            worker.deleteLater()
        if self._cloud is not None:
            self._prepare_viewer("original")

    def _request_mantle_view(self) -> None:
        if (
            self._ground_result is None
            or getattr(self._ground_result.model, "mantle", None) is None
        ):
            return
        self._prepare_viewer("mantle")

    def _prepare_viewer(self, kind: str) -> None:
        if self._cloud is None or self._viewer_worker is not None:
            return
        if kind in {"classified", "mantle"} and self._ground_result is None:
            return

        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        label = {
            "original": "original",
            "classified": "final ground",
            "mantle": "inverted Ground mantle (diagnostic)",
        }.get(kind, kind)
        self._busy(True, f"Preparing {label} 3D viewport...")

        worker = ViewerPrepareWorker(
            kind,
            self._cloud.path,
            self._ground_result if kind != "original" else None,
            self,
        )
        worker.progress_changed.connect(self._set_progress)
        worker.ready.connect(self._viewer_ready)
        worker.failed.connect(self._viewer_failed)
        worker.finished.connect(self._viewer_finished)
        self._viewer_worker = worker
        worker.start()

    def _viewer_ready(
        self,
        kind: str,
        dataset: str,
        title: str,
        classified: bool,
    ) -> None:
        self._viewer_loaded.add(kind)
        self.viewer.load_cloud(
            kind,
            dataset,
            title,
            classified,
            activate=True,
        )
        self.viewer.set_view_mode(kind)
        self.statusBar().showMessage(f"3D viewport ready: {title}")

    def _viewer_failed(
        self,
        kind: str,
        message: str,
    ) -> None:
        LOGGER.error(
            "Viewer preparation failed: %s | %s",
            kind,
            message,
        )
        self.statusBar().showMessage(
            f"3D viewport failed ({kind})"
        )
        QMessageBox.warning(
            self,
            "LAS-CAFIISICA — VIEWPORT 3D",
            f"Could not prepare {kind} viewport:\n{message}",
        )

    def _viewer_finished(self) -> None:
        worker = self._viewer_worker
        self._viewer_worker = None
        self._busy(False)
        if worker is not None:
            worker.deleteLater()

    def _engine_params(self) -> GroundEngineParams:
        quality = self.quality_combo.currentText().lower()
        params = GroundEngineParams.preset(quality)
        profile = self.profile_combo.currentText()

        if profile == "Mountain / Talude":
            params = replace(
                params,
                max_iteration_angle_deg=18.0,
                max_iteration_distance=0.18,
                confidence_threshold=0.66,
                gap_max_size=7.0,
            )
        elif profile == "Forest":
            params = replace(
                params,
                max_iteration_angle_deg=14.0,
                max_iteration_distance=0.16,
                confidence_threshold=0.70,
                gap_max_size=6.0,
            )
        elif profile == "Photogrammetry":
            params = replace(
                params,
                max_iteration_angle_deg=16.0,
                max_iteration_distance=0.20,
                confidence_threshold=0.65,
            )
        elif profile == "LiDAR":
            params = replace(
                params,
                max_iteration_angle_deg=15.0,
                max_iteration_distance=0.18,
                confidence_threshold=0.66,
            )
        elif profile == "High Density":
            params = replace(
                params,
                sample_target=max(params.sample_target, 4_000_000),
                candidate_spacing=0.30,
                confidence_threshold=0.67,
            )

        spacing = self.fill_spacing_spin.value()
        if spacing > 0:
            params = replace(
                params,
                synthetic_spacing=spacing,
            )
        return params

    def _engine_changed(
        self,
        engine: str,
    ) -> None:
        measured_l3 = engine in {
            "L3 Inverted Ground R20",
            "L3 Dense Ground R19",
            "L3 Ground Lab",
        }
        if measured_l3:
            self.include_synthetic.setChecked(False)
            self.fill_spacing_spin.setValue(0.0)
        self.include_synthetic.setEnabled(
            not measured_l3
        )
        self.fill_spacing_spin.setEnabled(
            not measured_l3
        )

    def _source_override(self) -> str | None:
        text = self.source_combo.currentText()
        if text == "DJI Zenmuse L3 / LiDAR":
            return "L3_LIDAR"
        if text == "DJI Zenmuse P1 / Photogrammetry":
            return "P1_PHOTOGRAMMETRY"
        if text == "Unknown":
            return "UNKNOWN"
        return None

    def _legacy_params(self) -> SMRFParams:
        fill_spacing = self.fill_spacing_spin.value()
        return SMRFParams(
            cell=self.cell_spin.value(),
            slope=self.slope_spin.value(),
            window=self.window_spin.value(),
            threshold=self.threshold_spin.value(),
            scalar=self.scalar_spin.value(),
            fill_spacing=fill_spacing if fill_spacing > 0 else 0.25,
            terrain3d_enabled=False,
        )

    def run_ground_engine(self) -> None:
        if self._cloud is None or self._ground_worker is not None:
            return

        self._ground_result = None
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        engine = self.engine_combo.currentText()
        self._busy(True, f"Starting {engine}...")

        worker = GroundEngineWorker(
            self._cloud,
            engine,
            self._engine_params(),
            self._legacy_params(),
            source_override=self._source_override(),
            parent=self,
        )
        worker.progress_changed.connect(self._set_progress)
        worker.completed.connect(self._ground_succeeded)
        worker.failed.connect(self._ground_failed)
        worker.finished.connect(self._ground_finished)
        self._ground_worker = worker
        worker.start()

    def _ground_succeeded(self, result) -> None:
        self._ground_result = result
        self.viewer.set_mantle_available(
            getattr(result.model, "mantle", None) is not None
        )
        engine = getattr(result, "engine_name", "SMRF Legacy")
        lines = [
            "",
            f"GROUND ENGINE RESULT — {engine}",
            f"Real ground: {getattr(result, 'ground_count', 0):,}",
            f"Rejected: {getattr(result, 'non_ground_count', 0):,}",
        ]

        analysis = getattr(result, "analysis", None)
        if analysis is not None:
            lines.extend(
                [
                    f"Median spacing: {analysis.median_spacing:.3f} m",
                    f"Local/global density: {analysis.xy_density:.2f} pts/m²",
                ]
            )

        for label, attr in (
            ("Ground seeds", "seed_count"),
            ("PTD candidates", "candidate_count"),
            ("PTD iterations", "ptd_iterations"),
            ("Detected gaps", "detected_gap_count"),
            ("Supported gaps", "supported_gap_count"),
            ("Occluded vegetation gaps", "occluded_gap_count"),
            ("Rejected gaps", "rejected_gap_count"),
            ("Synthetic ground", "synthetic_fill_point_count"),
            ("Terrain 3D voxels", "terrain3d_voxel_count"),
            ("Terrain 3D seed voxels", "terrain3d_seed_voxel_count"),
            ("L3 class2 validated", "original_validated_count"),
            ("L3 recovered high", "recovered_high_count"),
            ("L3 recovered medium", "recovered_medium_count"),
            ("L3 rejected source class2", "rejected_class2_count"),
            ("L3 vegetation", "non_ground_vegetation_count"),
            ("L3 object", "non_ground_object_count"),
            ("L3 unknown", "unknown_count"),
            ("L3 noise", "noise_count"),
            ("L3 return only", "return_only_count"),
            ("L3 return last multi", "return_last_multi_count"),
            ("L3 return first multi", "return_first_multi_count"),
            ("L3 return intermediate", "return_intermediate_count"),
            ("L3 invalid returns", "return_invalid_count"),
            ("L3 spatial cells", "context_cell_count"),
            ("L3 coarse representatives", "coarse_representative_count"),
            ("PTD Ground votes", "ptd_vote_count"),
            ("SMRF Ground votes", "smrf_vote_count"),
            ("CSF Ground votes", "csf_vote_count"),
            ("Consensus 2/3", "consensus_2of3_count"),
            ("L3 return-supported points", "l3_recovered_count"),
            ("L3 recovery voxels", "l3_recovery_voxel_count"),
        ):
            if hasattr(result, attr):
                lines.append(f"{label}: {getattr(result, attr):,}")

        if engine in {"L3 Dense Ground R19", "L3 Inverted Ground R20"}:
            revision = "R20" if engine == "L3 Inverted Ground R20" else "R19"
            for label, attr in (
                ("R19 no spatial evidence", "no_spatial_evidence_count"),
                ("R19 surface gate fail", "surface_gate_fail_count"),
                ("R19 spatial gate fail", "spatial_gate_fail_count"),
                ("R19 vegetation gate", "vegetation_gate_count"),
                ("R19 object roughness", "object_roughness_count"),
                ("R19 normal mismatch", "normal_mismatch_count"),
                ("R19 invalid gate", "invalid_gate_count"),
                ("R19 score below high", "score_below_high_count"),
                ("R19 score below medium", "score_below_medium_count"),
            ):
                lines.append(
                    f"{label}: {getattr(result, attr, 0):,}"
                )
            reasons = getattr(result, "rejection_reason_counts", ())
            if reasons:
                lines.append(f"{revision} EXCLUSIVE REJECTION REASONS:")
                for reason_name, number in reasons:
                    if number:
                        lines.append(f"  {reason_name}: {number:,}")

        if engine == "L3 Inverted Ground R20":
            lines.extend((
                "R20 INVERTED MANTLE (EXPERIMENTAL; SYNTHETIC=0):",
                f"  Measured returns recovered by mantle: {result.mantle_recovered_count:,}",
                f"  Source class2 recovered by mantle: {result.mantle_recovered_class2_count:,}",
                f"  Observed XY cells: {result.mantle_observed_cells:,}",
                f"  Reliable mantle cells: {result.mantle_reliable_cells:,}",
                f"  Small inferred empty cells (diagnostic only): {result.mantle_inferred_cells:,}",
                f"  Observed ambiguous cells: {result.mantle_ambiguous_cells:,}",
                f"  Possible unobserved Ground under returns: {result.mantle_possible_unobserved_cells:,}",
                "  To inspect the 2.5-D mantle use EXPORT MANTO (LAZ).",
                "  Inferred mantle is never written by EXPORT GROUND ONLY.",
            ))

        if hasattr(result, "source_type"):
            lines.append(
                "Source used: "
                f"{result.source_type} "
                f"({getattr(result, 'source_confidence', 0.0) * 100:.1f}%)"
            )

        if hasattr(result, "mean_confidence"):
            lines.append(
                f"Mean confidence: {result.mean_confidence:.3f}"
            )
        lines.append(
            f"Elapsed: {getattr(result, 'elapsed_seconds', 0.0):.1f} s"
        )

        self.statistics_view.appendPlainText("\n".join(lines))
        self.statusBar().showMessage(
            f"{engine} ground extraction complete"
        )

    def _ground_failed(self, message: str) -> None:
        self.statusBar().showMessage("Ground engine failed")
        QMessageBox.critical(
            self,
            "LAS-CAFIISICA — GROUND ENGINE",
            f"Ground engine failed:\n{message}",
        )

    def _ground_finished(self) -> None:
        worker = self._ground_worker
        self._ground_worker = None
        self._busy(False)
        if worker is not None:
            worker.deleteLater()
        if self._ground_result is not None:
            self._prepare_viewer("classified")

    def export_ground_only(self) -> None:
        if (
            self._cloud is None
            or self._ground_result is None
            or self._export_worker is not None
        ):
            return

        source = self._cloud.path
        include_synthetic = self.include_synthetic.isChecked()
        suffix = (
            "_GROUND_COMPLETE.laz"
            if include_synthetic
            else "_GROUND_ONLY.laz"
        )
        suggested = source.with_name(source.stem + suffix)
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Ground Only",
            str(suggested),
            "LAZ (*.laz);;LAS (*.las)",
        )
        if not filename:
            return

        output = Path(filename)
        if output.suffix.lower() not in {".las", ".laz"}:
            output = output.with_suffix(".laz")

        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._busy(True, "Exporting ground-only cloud...")

        worker = GroundExportWorker(
            source,
            output,
            self._ground_result,
            include_synthetic,
            self,
        )
        worker.progress_changed.connect(self._set_progress)
        worker.completed.connect(self._export_succeeded)
        worker.failed.connect(self._export_failed)
        worker.finished.connect(self._export_finished)
        self._export_worker = worker
        worker.start()

    def export_inverted_mantle(self) -> None:
        if (
            self._cloud is None
            or self._ground_result is None
            or self._mantle_worker is not None
            or getattr(self._ground_result.model, "mantle", None) is None
        ):
            return
        source = self._cloud.path
        suggested = source.with_name(source.stem + "_R20_MANTO_DIAGNOSTICO.laz")
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export R20 mantle diagnostic (NOT observed Ground)",
            str(suggested), "LAZ (*.laz);;LAS (*.las)",
        )
        if not filename:
            return
        output = Path(filename)
        if output.suffix.lower() not in {".las", ".laz"}:
            output = output.with_suffix(".laz")

        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._busy(True, "Exporting the separate R20 diagnostic mantle...")
        worker = MantleExportWorker(output, self._ground_result, self)
        worker.progress_changed.connect(self._set_progress)
        worker.completed.connect(self._mantle_export_succeeded)
        worker.failed.connect(self._mantle_export_failed)
        worker.finished.connect(self._mantle_export_finished)
        self._mantle_worker = worker
        worker.start()

    def _mantle_export_succeeded(self, path: str) -> None:
        self.statusBar().showMessage(f"Diagnostic mantle exported: {path}")
        QMessageBox.information(
            self, "LAS-CAFIISICA — R20 Mantle",
            "Independent inferred mantle exported:\n"
            + path
            + "\n\nAll points have LAS class 0 and synthetic=1 (diagnostics ONLY)."
            + "\nMantleState: 1 observed/reliable, 2 unobserved gap candidate,"
            + "\n3 observed/ambiguous, 4 possible unobserved Ground under returns.",
        )

    def _mantle_export_failed(self, message: str) -> None:
        self.statusBar().showMessage("Diagnostic mantle export failed")
        QMessageBox.critical(self, "LAS-CAFIISICA — R20 Mantle", message)

    def _mantle_export_finished(self) -> None:
        worker = self._mantle_worker
        self._mantle_worker = None
        self._busy(False)
        if worker is not None:
            worker.deleteLater()

    def _export_succeeded(self, path: str) -> None:
        self.statusBar().showMessage(f"Ground export complete: {path}")
        QMessageBox.information(
            self,
            "LAS-CAFIISICA",
            f"Ground-only cloud exported:\n{path}",
        )

    def _export_failed(self, message: str) -> None:
        self.statusBar().showMessage("Ground export failed")
        QMessageBox.critical(
            self,
            "LAS-CAFIISICA",
            f"Ground export failed:\n{message}",
        )

    def _export_finished(self) -> None:
        worker = self._export_worker
        self._export_worker = None
        self._busy(False)
        if worker is not None:
            worker.deleteLater()

    def closeEvent(self, event) -> None:
        self.viewer.shutdown()
        super().closeEvent(event)
