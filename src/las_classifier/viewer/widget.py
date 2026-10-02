from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .server import ViewerServer


LOGGER = logging.getLogger("las_cafiisica.viewer.widget")


class LoggingWebPage(QWebEnginePage):
    def javaScriptConsoleMessage(
        self,
        level,
        message: str,
        line_number: int,
        source_id: str,
    ) -> None:
        LOGGER.info(
            "JS_CONSOLE level=%s source=%s line=%s message=%s",
            level,
            source_id,
            line_number,
            message,
        )


class PointCloudViewer(QWidget):
    mantle_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.server = ViewerServer()
        self.server.start()
        self._page_ready = False
        self._pending_scripts: list[str] = []
        self._loaded = {
            "original": False,
            "classified": False,
            "mantle": False,
        }
        self._mantle_available = False

        self.original_button = QPushButton("ORIGINAL")
        self.classified_button = QPushButton("FINAL GROUND")
        self.mantle_button = QPushButton("MANTO R20")
        self.both_button = QPushButton("COMPARAR")
        self.fit_button = QPushButton("ENQUADRAR")
        self.rgb_button = QPushButton("RGB")
        self.elevation_button = QPushButton("ELEVAÇÃO")
        self.class_mode_button = QPushButton("CLASSIFICAÇÃO")

        self.original_button.setEnabled(False)
        self.classified_button.setEnabled(False)
        self.both_button.setEnabled(False)
        self.mantle_button.setEnabled(False)

        self.original_button.clicked.connect(
            lambda: self.set_view_mode("original")
        )
        self.classified_button.clicked.connect(
            lambda: self.set_view_mode("classified")
        )
        self.both_button.clicked.connect(
            lambda: self.set_view_mode("both")
        )
        self.mantle_button.clicked.connect(self._request_mantle_view)
        self.fit_button.clicked.connect(
            lambda: self._run_js("window.LASViewer.fit();")
        )
        self.rgb_button.clicked.connect(
            lambda: self.set_material_mode("rgb")
        )
        self.elevation_button.clicked.connect(
            lambda: self.set_material_mode("elevation")
        )
        self.class_mode_button.clicked.connect(
            lambda: self.set_material_mode("classification")
        )

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("VIEWPORT 3D"))
        toolbar.addWidget(self.original_button)
        toolbar.addWidget(self.classified_button)
        toolbar.addWidget(self.both_button)
        toolbar.addWidget(self.mantle_button)
        toolbar.addStretch(1)
        toolbar.addWidget(self.fit_button)
        toolbar.addWidget(self.rgb_button)
        toolbar.addWidget(self.elevation_button)
        toolbar.addWidget(self.class_mode_button)

        self.web = QWebEngineView(self)
        self.web.setPage(LoggingWebPage(self.web))
        self.web.loadFinished.connect(self._on_load_finished)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(toolbar)
        layout.addWidget(self.web, 1)

        self.web.load(QUrl(self.server.app_url))

    def _on_load_finished(self, ok: bool) -> None:
        self._page_ready = bool(ok)
        LOGGER.info("VIEWER_PAGE_READY=%s", ok)
        if not ok:
            LOGGER.error("Potree viewer HTML failed to load")
            return
        scripts = self._pending_scripts
        self._pending_scripts = []
        for script in scripts:
            self.web.page().runJavaScript(script)

    def _run_js(self, script: str) -> None:
        if self._page_ready:
            self.web.page().runJavaScript(script)
        else:
            self._pending_scripts.append(script)

    def set_mantle_available(self, available: bool) -> None:
        self._mantle_available = bool(available)
        self.mantle_button.setEnabled(self._mantle_available)

    def _request_mantle_view(self) -> None:
        if not self._mantle_available:
            return
        if self._loaded["mantle"]:
            self.set_view_mode("mantle")
        else:
            self.mantle_requested.emit()

    def load_cloud(
        self,
        key: str,
        dataset_dir: str | Path,
        title: str,
        classified: bool,
        activate: bool = True,
    ) -> None:
        url = self.server.register_cloud(key, dataset_dir)
        LOGGER.info(
            "VIEWER_LOAD key=%s url=%s classified=%s",
            key,
            url,
            classified,
        )
        self._loaded[key] = True
        self.original_button.setEnabled(self._loaded["original"])
        self.classified_button.setEnabled(self._loaded["classified"])
        self.both_button.setEnabled(
            self._loaded["original"]
            and self._loaded["classified"]
        )
        self.mantle_button.setEnabled(self._mantle_available)
        script = (
            "window.LASViewer.loadCloud("
            + json.dumps(key)
            + ","
            + json.dumps(url)
            + ","
            + json.dumps(title)
            + ","
            + ("true" if classified else "false")
            + ","
            + ("true" if activate else "false")
            + ");"
        )
        self._run_js(script)

    def set_view_mode(self, mode: str) -> None:
        if mode not in {"original", "classified", "mantle", "both"}:
            return
        self._run_js(
            "window.LASViewer.setViewMode("
            + json.dumps(mode)
            + ");"
        )

    def set_material_mode(self, mode: str) -> None:
        if mode not in {"rgb", "elevation", "classification"}:
            return
        self._run_js(
            "window.LASViewer.setMaterialMode("
            + json.dumps(mode)
            + ");"
        )

    def clear(self) -> None:
        self._loaded = {
            "original": False,
            "classified": False,
            "mantle": False,
        }
        self._mantle_available = False
        self.mantle_button.setEnabled(False)
        self.original_button.setEnabled(False)
        self.classified_button.setEnabled(False)
        self.both_button.setEnabled(False)
        self._run_js("window.LASViewer.clear();")

    def shutdown(self) -> None:
        self.server.stop()
