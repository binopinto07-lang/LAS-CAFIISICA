from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from .cloud.loader import load_cloud
from .cloud.statistics import calculate_statistics


LOGGER = logging.getLogger("las_cafiisica.ui")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LAS-CAFIISICA 0.1.0-dev")
        self.resize(860, 600)

        self.open_button = QPushButton("OPEN LAS / LAZ")
        self.open_button.clicked.connect(self.open_cloud)

        self.file_label = QLabel("No cloud loaded")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.statistics_view = QPlainTextEdit()
        self.statistics_view.setReadOnly(True)
        self.statistics_view.setPlaceholderText(
            "Open a LAS/LAZ file to calculate real cloud statistics."
        )

        layout = QVBoxLayout()
        layout.addWidget(self.open_button)
        layout.addWidget(self.file_label)
        layout.addWidget(self.statistics_view, 1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.statusBar().showMessage("Ready")

    def open_cloud(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open LAS/LAZ",
            "",
            "LAS/LAZ point clouds (*.las *.laz)",
        )
        if not filename:
            return

        self.statusBar().showMessage("Loading point cloud...")
        try:
            cloud = load_cloud(filename)
            stats = calculate_statistics(cloud)
        except Exception as exc:
            LOGGER.exception("Failed to load cloud: %s", filename)
            self.statusBar().showMessage("Load failed")
            QMessageBox.critical(
                self, "LAS-CAFIISICA", f"Could not open file:\n{exc}"
            )
            return

        self.file_label.setText(str(Path(filename)))
        self.statistics_view.setPlainText("\n".join(stats.as_display_lines()))
        self.statusBar().showMessage(f"Loaded {cloud.point_count:,} points")
