from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ..classifiers.smrf import SMRFParams, SMRFResult, run_smrf
from .exporter import export_classified
from .model import CloudModel


LOGGER = logging.getLogger(
    "las_cafiisica.cloud.classification_worker"
)


class SMRFWorker(QThread):
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


class ClassifiedExportWorker(QThread):
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
