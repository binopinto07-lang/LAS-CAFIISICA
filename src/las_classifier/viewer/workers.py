from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .converter import prepare_classified, prepare_original


class ViewerPrepareWorker(QThread):
    ready = Signal(str, str, str, bool)
    failed = Signal(str, str)
    progress_changed = Signal(int, str)

    def __init__(
        self,
        kind: str,
        source_path: Path,
        result=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if kind not in {"original", "classified"}:
            raise ValueError(f"Unsupported viewer kind: {kind}")
        self.kind = kind
        self.source_path = source_path
        self.result = result

    def run(self) -> None:
        try:
            callback = (
                lambda p, m: self.progress_changed.emit(p, m)
            )
            if self.kind == "original":
                dataset = prepare_original(
                    self.source_path,
                    callback,
                )
                title = self.source_path.name + " — Original"
                classified = False
            else:
                if self.result is None:
                    raise RuntimeError(
                        "Final ground viewer requires a ground result"
                    )
                dataset = prepare_classified(
                    self.source_path,
                    self.result,
                    callback,
                )
                engine = getattr(
                    self.result,
                    "engine_name",
                    "SMRF",
                )
                title = (
                    self.source_path.name
                    + f" — {engine} — Final Ground"
                )
                classified = True
        except Exception as exc:
            self.failed.emit(self.kind, str(exc))
            return

        self.ready.emit(
            self.kind,
            str(dataset),
            title,
            classified,
        )
