from __future__ import annotations

import gc
import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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
    ClassifiedExportWorker,
    SMRFWorker,
)
from .cloud.worker import CloudLoadWorker
from .viewer.widget import PointCloudViewer
from .viewer.workers import ViewerPrepareWorker


LOGGER = logging.getLogger("las_cafiisica.ui")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"LAS-CAFIISICA {__version__}")
        self.resize(1500, 900)
        self.setMinimumSize(1180, 720)

        self._cloud = None
        self._smrf_result = None
        self._loader: CloudLoadWorker | None = None
        self._smrf_worker: SMRFWorker | None = None
        self._export_worker: ClassifiedExportWorker | None = None
        self._viewer_worker: ViewerPrepareWorker | None = None
        self._pending_filename: str | None = None
        self._viewer_loaded: set[str] = set()

        self.open_button = QPushButton("OPEN LAS / LAZ")
        self.open_button.clicked.connect(self.open_cloud)

        self.file_label = QLabel("No cloud loaded")
        self.file_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        self.file_label.setWordWrap(True)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)

        self.statistics_view = QPlainTextEdit()
        self.statistics_view.setReadOnly(True)
        self.statistics_view.setPlaceholderText(
            "Open a LAS/LAZ file to calculate real cloud statistics."
        )

        smrf_box = QGroupBox(
            "GROUND CLASSIFICATION — SMRF"
        )
        form = QFormLayout()
        self.cell_spin = self._spin(
            1.0, 0.10, 20.0, 0.10, 2
        )
        self.slope_spin = self._spin(
            0.15, 0.0, 5.0, 0.01, 3
        )
        self.window_spin = self._spin(
            18.0, 1.0, 200.0, 1.0, 1
        )
        self.threshold_spin = self._spin(
            0.50, 0.0, 10.0, 0.05, 2
        )
        self.scalar_spin = self._spin(
            1.25, 0.10, 10.0, 0.05, 2
        )
        form.addRow("Cell (m)", self.cell_spin)
        form.addRow("Slope", self.slope_spin)
        form.addRow("Window (m)", self.window_spin)
        form.addRow("Threshold (m)", self.threshold_spin)
        form.addRow("Scalar", self.scalar_spin)

        buttons = QHBoxLayout()
        self.smrf_button = QPushButton(
            "CLASSIFY GROUND — SMRF"
        )
        self.smrf_button.setEnabled(False)
        self.smrf_button.clicked.connect(self.run_smrf)
        self.export_button = QPushButton(
            "EXPORT CLASSIFIED LAS / LAZ"
        )
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(
            self.export_classified
        )
        buttons.addWidget(self.smrf_button)
        buttons.addWidget(self.export_button)
        form.addRow(buttons)
        smrf_box.setLayout(form)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.open_button)
        left_layout.addWidget(self.file_label)
        left_layout.addWidget(self.progress)
        left_layout.addWidget(smrf_box)
        left_layout.addWidget(self.statistics_view, 1)
        left.setMinimumWidth(390)

        self.viewer = PointCloudViewer(self)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1080])
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
        self.smrf_button.setEnabled(
            not active and self._cloud is not None
        )
        self.export_button.setEnabled(
            not active and self._smrf_result is not None
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
                self._smrf_worker,
                self._export_worker,
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
        self._smrf_result = None
        self._viewer_loaded.clear()
        self.viewer.clear()
        gc.collect()

        self._pending_filename = filename
        self.progress.setRange(0, 0)
        self._busy(
            True,
            "Loading point cloud in background...",
        )

        worker = CloudLoadWorker(filename, self)
        worker.loaded.connect(self._load_succeeded)
        worker.failed.connect(self._load_failed)
        worker.finished.connect(self._load_finished)
        self._loader = worker
        worker.start()

    def _load_succeeded(self, cloud, stats) -> None:
        self._cloud = cloud
        filename = (
            self._pending_filename or str(cloud.path)
        )
        self.file_label.setText(str(Path(filename)))
        self.statistics_view.setPlainText(
            "\n".join(stats.as_display_lines())
        )
        self.statusBar().showMessage(
            f"Loaded {cloud.point_count:,} points"
        )

    def _load_failed(self, message: str) -> None:
        filename = (
            self._pending_filename or "<unknown>"
        )
        LOGGER.error(
            "Failed to load cloud: %s | %s",
            filename,
            message,
        )
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

    def _prepare_viewer(self, kind: str) -> None:
        if (
            self._cloud is None
            or self._viewer_worker is not None
        ):
            return
        if (
            kind == "classified"
            and self._smrf_result is None
        ):
            return

        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        label = (
            "original"
            if kind == "original"
            else "classified"
        )
        self._busy(
            True,
            f"Preparing {label} 3D viewport...",
        )
        worker = ViewerPrepareWorker(
            kind,
            self._cloud.path,
            (
                self._smrf_result
                if kind == "classified"
                else None
            ),
            self,
        )
        worker.progress_changed.connect(
            self._set_progress
        )
        worker.ready.connect(self._viewer_ready)
        worker.failed.connect(self._viewer_failed)
        worker.finished.connect(
            self._viewer_finished
        )
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
        self.statusBar().showMessage(
            f"3D viewport ready: {title}"
        )

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
            (
                f"Could not prepare {kind} "
                f"viewport:\n{message}"
            ),
        )

    def _viewer_finished(self) -> None:
        worker = self._viewer_worker
        self._viewer_worker = None
        self._busy(False)
        if self._cloud is not None:
            self.statusBar().showMessage(
                (
                    f"Loaded "
                    f"{self._cloud.point_count:,} points"
                )
            )
        if worker is not None:
            worker.deleteLater()

    def _smrf_params(self) -> SMRFParams:
        return SMRFParams(
            cell=self.cell_spin.value(),
            slope=self.slope_spin.value(),
            window=self.window_spin.value(),
            threshold=self.threshold_spin.value(),
            scalar=self.scalar_spin.value(),
        )

    def run_smrf(self) -> None:
        if (
            self._cloud is None
            or self._smrf_worker is not None
        ):
            return

        self._smrf_result = None
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._busy(
            True,
            "Starting SMRF classification...",
        )

        worker = SMRFWorker(
            self._cloud,
            self._smrf_params(),
            self,
        )
        worker.progress_changed.connect(
            self._set_progress
        )
        worker.completed.connect(
            self._smrf_succeeded
        )
        worker.failed.connect(self._smrf_failed)
        worker.finished.connect(
            self._smrf_finished
        )
        self._smrf_worker = worker
        worker.start()

    def _smrf_succeeded(self, result) -> None:
        self._smrf_result = result
        p = result.model.params
        self.statistics_view.appendPlainText(
            "\n".join(
                [
                    "",
                    "SMRF RESULT",
                    (
                        "Ground (class 2): "
                        f"{result.ground_count:,}"
                    ),
                    (
                        "Non-ground (class 1): "
                        f"{result.non_ground_count:,}"
                    ),
                    f"Cell: {p.cell:.2f} m",
                    f"Slope: {p.slope:.3f}",
                    f"Window: {p.window:.1f} m",
                    f"Threshold: {p.threshold:.2f} m",
                    f"Scalar: {p.scalar:.2f}",
                    (
                        "Elapsed: "
                        f"{result.elapsed_seconds:.1f} s"
                    ),
                ]
            )
        )
        self.statusBar().showMessage(
            "SMRF classification complete"
        )

    def _smrf_failed(self, message: str) -> None:
        self.statusBar().showMessage("SMRF failed")
        QMessageBox.critical(
            self,
            "LAS-CAFIISICA — SMRF",
            f"SMRF failed:\n{message}",
        )

    def _smrf_finished(self) -> None:
        worker = self._smrf_worker
        self._smrf_worker = None
        self._busy(False)
        if worker is not None:
            worker.deleteLater()
        if self._smrf_result is not None:
            self._prepare_viewer("classified")

    def export_classified(self) -> None:
        if (
            self._cloud is None
            or self._smrf_result is None
            or self._export_worker is not None
        ):
            return

        source = self._cloud.path
        suggested = source.with_name(
            f"{source.stem}_SMRF.laz"
        )
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export classified cloud",
            str(suggested),
            "LAZ (*.laz);;LAS (*.las)",
        )
        if not filename:
            return

        output = Path(filename)
        if output.suffix.lower() not in {
            ".las",
            ".laz",
        }:
            output = output.with_suffix(".laz")

        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._busy(
            True,
            "Exporting classified cloud...",
        )

        worker = ClassifiedExportWorker(
            source,
            output,
            self._smrf_result,
            self,
        )
        worker.progress_changed.connect(
            self._set_progress
        )
        worker.completed.connect(
            self._export_succeeded
        )
        worker.failed.connect(self._export_failed)
        worker.finished.connect(
            self._export_finished
        )
        self._export_worker = worker
        worker.start()

    def _export_succeeded(self, path: str) -> None:
        self.statusBar().showMessage(
            f"Export complete: {path}"
        )
        QMessageBox.information(
            self,
            "LAS-CAFIISICA",
            (
                "Classified cloud exported:\n"
                f"{path}"
            ),
        )

    def _export_failed(self, message: str) -> None:
        self.statusBar().showMessage(
            "Export failed"
        )
        QMessageBox.critical(
            self,
            "LAS-CAFIISICA",
            f"Export failed:\n{message}",
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
