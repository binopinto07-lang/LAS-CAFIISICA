from __future__ import annotations

import gc
import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .cloud.worker import CloudLoadWorker


LOGGER = logging.getLogger("las_cafiisica.ui")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"LAS-CAFIISICA {__version__}")
        self.resize(860, 600)

        self._cloud = None
        self._loader: CloudLoadWorker | None = None
        self._pending_filename: str | None = None

        self.open_button = QPushButton("OPEN LAS / LAZ")
        self.open_button.clicked.connect(self.open_cloud)

        self.file_label = QLabel("No cloud loaded")
        self.file_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)

        self.statistics_view = QPlainTextEdit()
        self.statistics_view.setReadOnly(True)
        self.statistics_view.setPlaceholderText(
            "Open a LAS/LAZ file to calculate real cloud statistics."
        )

        layout = QVBoxLayout()
        layout.addWidget(self.open_button)
        layout.addWidget(self.file_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.statistics_view, 1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.statusBar().showMessage("Ready")

    def open_cloud(self) -> None:
        if self._loader is not None and self._loader.isRunning():
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
        gc.collect()

        self._pending_filename = filename
        self.open_button.setEnabled(False)
        self.open_button.setText("LOADING...")
        self.progress.setVisible(True)
        self.statusBar().showMessage(
            "Loading point cloud in background..."
        )

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
        filename = self._pending_filename or "<unknown>"
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
        self.progress.setVisible(False)
        self.open_button.setEnabled(True)
        self.open_button.setText("OPEN LAS / LAZ")
        worker = self._loader
        self._loader = None
        self._pending_filename = None
        if worker is not None:
            worker.deleteLater()
