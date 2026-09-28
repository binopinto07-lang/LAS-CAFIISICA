from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal

from .loader import load_cloud
from .statistics import calculate_statistics


LOGGER = logging.getLogger("las_cafiisica.cloud.worker")


class CloudLoadWorker(QThread):
    loaded = Signal(object, object)
    failed = Signal(str)

    def __init__(self, filename: str, parent=None) -> None:
        super().__init__(parent)
        self.filename = filename

    def run(self) -> None:
        try:
            cloud = load_cloud(self.filename)
            stats = calculate_statistics(cloud)
        except Exception as exc:
            LOGGER.exception(
                "Failed to load cloud: %s",
                self.filename,
            )
            self.failed.emit(str(exc))
            return

        self.loaded.emit(cloud, stats)
