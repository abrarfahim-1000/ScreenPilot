"""
ui.py — PyQt6 Control Panel for UI Navigator Desktop Client.

Layout (horizontal split)
─────────────────────────
Left pane
  ┌─ Screen Preview ───────────────────────────────┐
  │  Live capture, refreshed every 2–3 s           │
  └────────────────────────────────────────────────┘
  ┌─ Session Setup ─────────────────────────────────┐
  │  Task Goal  : [                               ] │
  │  Server URL : [http://localhost:8080          ] │
  └─────────────────────────────────────────────────┘
  ┌─ Controls ──────────────────────────────────────┐
  │  [▶ Start Session]   [■ Stop Session]           │
  │  Status : Disconnected    Step : 0              │
  └─────────────────────────────────────────────────┘

Right pane
  ┌─ Action Log ────────────────────────────────────┐
  │  [12:34:01] Session abc123 starting …           │
  │  [12:34:01] Connected to http://…               │
  │  [12:34:04] Step 1 — captured frame 1280×720    │
  │  …                                              │
  └─────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import (
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QFont, QPixmap, QImage, QColor, QPalette, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QGroupBox,
    QStatusBar,
)

logger = logging.getLogger(__name__)

_CLR_CONNECTED    = "#27ae60"   # green
_CLR_DISCONNECTED = "#e74c3c"   # red
_CLR_WORKING      = "#f39c12"   # amber
_CLR_BG           = "#1e1e1e"   # dark bg
_CLR_FG           = "#d4d4d4"   # light text
_CLR_PANEL        = "#252526"   # slightly lighter panel
_CLR_BORDER       = "#3c3c3c"
_CLR_LOG_BG       = "#0d0d0d"
_CLR_LOG_FG       = "#c8c8c8"
_CLR_ACCENT       = "#569cd6"   # blue accent

_SCREEN_W = 640
_SCREEN_H = 360

class ApiKeyDialog(QDialog):
    """
    Modal dialog that prompts the user for their Gemini API key.

    The key field is masked (password mode).  A hint below explains where
    the key is stored so users know it is never sent to logs or third parties.
    """

    def __init__(self, parent=None, first_time: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gemini API Key")
        self.setMinimumWidth(460)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # Heading
        if first_time:
            heading = QLabel("A Gemini API key is required to start a session.")
        else:
            heading = QLabel("Enter a new Gemini API key (replaces the stored one).")
        heading.setWordWrap(True)
        heading.setStyleSheet(f"color: {_CLR_FG}; font-size: 10pt;")
        layout.addWidget(heading)

        # Key input
        self._key_input = QLineEdit()
        self._key_input.setPlaceholderText("AIza…")
        self._key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_input.setMinimumHeight(32)
        layout.addWidget(self._key_input)

        # Show/hide toggle
        show_row = QHBoxLayout()
        self._show_btn = QPushButton("Show")
        self._show_btn.setFixedWidth(60)
        self._show_btn.setCheckable(True)
        self._show_btn.setStyleSheet(_btn_style(_CLR_BORDER, "#505050", text_color=_CLR_FG))
        self._show_btn.toggled.connect(self._toggle_visibility)
        show_row.addStretch()
        show_row.addWidget(self._show_btn)
        layout.addLayout(show_row)

        # Storage hint
        hint = QLabel(
            "⛔  Your key is saved in the <b>OS credential store</b> — "
            "Windows Credential Manager / macOS Keychain / Linux Secret Service. "
            "It is never written to disk in plaintext."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: #888; font-size: 8pt;")
        layout.addWidget(hint)

        # OK / Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        # Style the OK button green
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn:
            ok_btn.setStyleSheet(_btn_style("#2ecc71", "#27ae60"))
        layout.addWidget(buttons)

        self.setStyleSheet(f"""
            QDialog {{
                background: {_CLR_PANEL};
                color: {_CLR_FG};
            }}
            QLineEdit {{
                background: #3c3c3c;
                color: {_CLR_FG};
                border: 1px solid {_CLR_BORDER};
                border-radius: 3px;
                padding: 4px 6px;
            }}
            QLineEdit:focus {{ border-color: {_CLR_ACCENT}; }}
            QDialogButtonBox {{ background: transparent; }}
        """)

    def _toggle_visibility(self, checked: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        self._key_input.setEchoMode(mode)
        self._show_btn.setText("Hide" if checked else "Show")

    def _on_accept(self) -> None:
        key = self._key_input.text().strip()
        if not key:
            QMessageBox.warning(self, "No key entered", "Please enter an API key before clicking OK.")
            return
        # Basic format check: Gemini keys start with "AIza" and are 39 chars
        if not key.startswith("AIza") or len(key) < 35:
            QMessageBox.warning(
                self,
                "Invalid key format",
                "Gemini API keys start with \"AIza\" and are at least 35 characters long.\n"
                "Please check your key at console.cloud.google.com.",
            )
            return
        self.accept()

    def key_value(self) -> str:
        return self._key_input.text().strip()

class _ScreenRefreshThread(QThread):
    """
    Captures a frame every ~2.5 s (when no session is active) and emits
    the raw JPEG bytes.  The main window replaces this feed with the
    session's own frame emissions while a session is running.
    """
    frame_ready = pyqtSignal(bytes)

    _INTERVAL_MS = 2500

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._active = True

    def stop(self) -> None:
        self._active = False
        self.requestInterruption()

    def run(self) -> None:
        # Import here so the thread owns its own mss context
        from capture import FrameCapturer
        capturer = FrameCapturer(on_frame=lambda _f: None)
        while self._active and not self.isInterruptionRequested():
            try:
                frame = capturer.capture_once()
                self.frame_ready.emit(frame.jpeg_bytes)
            except Exception as exc:
                logger.debug("Screen refresh error: %s", exc)
            self.msleep(self._INTERVAL_MS)

class ControlPanel(QMainWindow):
    """
    Minimal PyQt6 control panel.

    Instantiate and call ``show()`` to display.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("UI Navigator — Control Panel")
        self.setMinimumSize(1100, 600)
        self.resize(1280, 720)

        self._session: Optional[object] = None   # SessionManager | None
        self._session_active = False

        from keystore import KeyStore
        self._keystore = KeyStore()

        self._setup_ui()
        self._apply_dark_theme()
        self._setup_screen_refresher()

    def _setup_ui(self) -> None:
        # ── Central widget ─────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        root.addWidget(splitter)

        # ── Left pane ──────────────────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Screen preview
        screen_group = _make_group("Screen Preview")
        screen_vbox = QVBoxLayout(screen_group)
        screen_vbox.setContentsMargins(4, 4, 4, 4)

        self.screen_label = QLabel()
        self.screen_label.setFixedSize(_SCREEN_W, _SCREEN_H)
        self.screen_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.screen_label.setStyleSheet(
            f"background: #0a0a0a; border: 1px solid {_CLR_BORDER};"
        )
        self.screen_label.setText("⬛  No capture yet")
        screen_vbox.addWidget(self.screen_label)
        left_layout.addWidget(screen_group)

        # Session setup
        setup_group = _make_group("Session Setup")
        setup_layout = _make_form_layout()
        setup_group.setLayout(setup_layout)

        lbl_goal = QLabel("Task Goal:")
        lbl_goal.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.task_input = QLineEdit()
        self.task_input.setPlaceholderText("e.g. Open Chrome and navigate to github.com")
        setup_layout.addRow(lbl_goal, self.task_input)

        lbl_server = QLabel("Server URL:")
        lbl_server.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.server_input = QLineEdit("http://localhost:8080")
        setup_layout.addRow(lbl_server, self.server_input)

        # API key row — shown as masked dots if a key is stored
        lbl_key = QLabel("Gemini Key:")
        lbl_key.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        key_row = QHBoxLayout()
        self.key_status_label = QLabel()
        self._refresh_key_status_label()
        key_row.addWidget(self.key_status_label)
        key_row.addStretch()
        change_key_btn = QPushButton("Change…")
        change_key_btn.setFixedWidth(72)
        change_key_btn.setStyleSheet(_btn_style("#3c3c3c", "#505050", text_color=_CLR_FG))
        change_key_btn.clicked.connect(self._on_change_key)
        key_row.addWidget(change_key_btn)

        remove_key_btn = QPushButton("Remove")
        remove_key_btn.setFixedWidth(66)
        remove_key_btn.setStyleSheet(_btn_style("#5a1a1a", "#7a2020", text_color="#ff8080"))
        remove_key_btn.clicked.connect(self._on_remove_key)
        key_row.addWidget(remove_key_btn)

        key_container = QWidget()
        key_container.setLayout(key_row)
        setup_layout.addRow(lbl_key, key_container)

        left_layout.addWidget(setup_group)

        # Controls
        ctrl_group = _make_group("Controls")
        ctrl_vbox = QVBoxLayout(ctrl_group)
        ctrl_vbox.setSpacing(6)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start Session")
        self.start_btn.setFixedHeight(36)
        self.start_btn.setStyleSheet(_btn_style("#2ecc71", "#27ae60"))
        self.start_btn.clicked.connect(self._on_start)

        self.stop_btn = QPushButton("■  Stop Session")
        self.stop_btn.setFixedHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(_btn_style("#e74c3c", "#c0392b"))
        self.stop_btn.clicked.connect(self._on_stop)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        ctrl_vbox.addLayout(btn_row)

        status_row = QHBoxLayout()
        self.status_dot = QLabel("●")
        self.status_dot.setFixedWidth(16)
        self.status_dot.setStyleSheet(f"color: {_CLR_DISCONNECTED};")

        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet(f"color: {_CLR_FG}; font-weight: bold;")

        self.step_label = QLabel("Step: —")
        self.step_label.setStyleSheet(f"color: {_CLR_FG};")
        self.step_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        status_row.addWidget(self.step_label)
        ctrl_vbox.addLayout(status_row)

        left_layout.addWidget(ctrl_group)
        left_layout.addStretch()

        # ── Right pane: log feed ────────────────────────────────────────
        log_group = _make_group("Action Log")
        log_vbox = QVBoxLayout(log_group)
        log_vbox.setContentsMargins(4, 4, 4, 4)

        self.log_feed = QTextEdit()
        self.log_feed.setReadOnly(True)
        self.log_feed.setFont(QFont("Consolas", 9))
        self.log_feed.setStyleSheet(
            f"background: {_CLR_LOG_BG}; color: {_CLR_LOG_FG}; "
            f"border: 1px solid {_CLR_BORDER};"
        )
        self.log_feed.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        log_vbox.addWidget(self.log_feed)

        clear_row = QHBoxLayout()
        clear_row.addStretch()
        clear_btn = QPushButton("Clear Log")
        clear_btn.setFixedWidth(90)
        clear_btn.setStyleSheet(_btn_style(_CLR_BORDER, _CLR_BORDER, text_color=_CLR_FG))
        clear_btn.clicked.connect(self.log_feed.clear)
        clear_row.addWidget(clear_btn)
        log_vbox.addLayout(clear_row)

        # ── Splitter sizing ─────────────────────────────────────────────
        splitter.addWidget(left)
        splitter.addWidget(log_group)
        splitter.setStretchFactor(0, 0)   # left: fixed-ish
        splitter.setStretchFactor(1, 1)   # right: stretches

        # ── Status bar ─────────────────────────────────────────────────
        sb = QStatusBar()
        sb.setStyleSheet(f"background: {_CLR_PANEL}; color: {_CLR_FG};")
        self.setStatusBar(sb)
        sb.showMessage("Ready — configure task goal and start a session")

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {_CLR_BG};
                color: {_CLR_FG};
            }}
            QGroupBox {{
                background-color: {_CLR_PANEL};
                border: 1px solid {_CLR_BORDER};
                border-radius: 4px;
                margin-top: 18px;
                padding: 6px;
                font-weight: bold;
                color: {_CLR_ACCENT};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}
            QLineEdit {{
                background: #3c3c3c;
                color: {_CLR_FG};
                border: 1px solid {_CLR_BORDER};
                border-radius: 3px;
                padding: 4px 6px;
            }}
            QLineEdit:focus {{
                border-color: {_CLR_ACCENT};
            }}
            QSplitter::handle {{
                background: {_CLR_BORDER};
            }}
            QLabel {{
                color: {_CLR_FG};
            }}
            QScrollBar:vertical {{
                background: {_CLR_PANEL};
                width: 10px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {_CLR_BORDER};
                border-radius: 3px;
                min-height: 20px;
            }}
        """)

    def _setup_screen_refresher(self) -> None:
        self._refresher = _ScreenRefreshThread(self)
        self._refresher.frame_ready.connect(self._update_frame_display)
        self._refresher.start()

    def _update_frame_display(self, jpeg_bytes: bytes) -> None:
        """Convert JPEG bytes to a QPixmap and display it."""
        try:
            img = QImage.fromData(jpeg_bytes, "JPEG")
            if img.isNull():
                return
            pixmap = QPixmap.fromImage(img).scaled(
                _SCREEN_W,
                _SCREEN_H,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.screen_label.setPixmap(pixmap)
            self.screen_label.setText("")
        except Exception as exc:
            logger.debug("Frame display error: %s", exc)

    def _on_start(self) -> None:
        from session import SessionManager

        # ── Ensure an API key is available ──────────────────────────
        api_key = self._keystore.load()
        if not api_key:
            api_key = self._prompt_for_key(first_time=True)
            if not api_key:
                return

        task = self.task_input.text().strip()
        server = self.server_input.text().strip() or "http://localhost:8080"

        # ── HTTPS guard for non-local servers ─────────────────────────
        _local = ("localhost", "127.0.0.1", "::1")
        is_local = any(h in server for h in _local)
        if server.startswith("http://") and not is_local:
            reply = QMessageBox.warning(
                self,
                "Insecure connection",
                "The server URL uses plain HTTP on a non-local host.\n"
                "Your API key will be transmitted in cleartext.\n\n"
                "Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._session = SessionManager(
            server_url=server,
            task_goal=task,
            api_key=api_key,
        )
        self._session.action_logged.connect(self._append_log)
        self._session.status_changed.connect(self._update_status)
        self._session.frame_ready.connect(self._update_frame_display)
        self._session.session_ended.connect(self._on_session_ended)
        self._session.auth_error.connect(self._on_auth_error)

        self._set_session_active(True)
        self._append_log(f"[UI] Starting session — server: {server}")
        self._session.start()

    def _on_change_key(self) -> None:
        """Let the user update the stored API key."""
        new_key = self._prompt_for_key(first_time=False)
        if new_key is not None:
            self._refresh_key_status_label()

    def _on_remove_key(self) -> None:
        """Delete the stored API key from the OS credential store."""
        if not self._keystore.exists():
            return
        reply = QMessageBox.question(
            self,
            "Remove API key",
            "Remove the stored Gemini API key?\n"
            "You will be asked to enter it again on the next session start.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._keystore.delete()
            self._refresh_key_status_label()
            self._append_log("[UI] API key removed from credential store")

    def _on_auth_error(self, detail: str) -> None:
        """
        Called when the server returns 401/403.  The session has already
        stopped itself.  Show the key dialog so the user can correct the key.
        """
        self._set_session_active(False)
        self._append_log(f"[UI] ⚠ Auth error: {detail}")
        QMessageBox.warning(
            self,
            "Authentication failed",
            f"The server rejected the API key:\n{detail}\n\n"
            "Please enter a valid key.",
        )
        self._on_change_key()

    def _prompt_for_key(self, *, first_time: bool) -> Optional[str]:
        """
        Show the API key dialog.  Returns the accepted key string,
        or None if the user cancelled.
        On success the key is written to .env via KeyStore.
        """
        dlg = ApiKeyDialog(parent=self, first_time=first_time)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            key = dlg.key_value()
            if key:
                self._keystore.save(key)
                self._refresh_key_status_label()
                self._append_log("[UI] API key saved to OS credential store")
                return key
        return None

    def _refresh_key_status_label(self) -> None:
        """Update the key status display without revealing the key."""
        if self._keystore.exists():
            raw = self._keystore.load()
            masked = raw[:4] + "*" * max(0, len(raw) - 8) + raw[-4:] if len(raw) > 8 else "****"
            self.key_status_label.setText(f"{masked}  ✓ stored")
            self.key_status_label.setStyleSheet(f"color: {_CLR_CONNECTED};")
        else:
            self.key_status_label.setText("Not set — will be asked on Start")
            self.key_status_label.setStyleSheet(f"color: {_CLR_DISCONNECTED};")

    def _on_stop(self) -> None:
        if self._session is not None:
            self._append_log("[UI] Stop requested…")
            self._session.request_stop()

    def _on_session_ended(self, reason: str) -> None:
        self._set_session_active(False)
        self._append_log(f"[UI] Session ended: {reason}")
        self.statusBar().showMessage(f"Session ended: {reason}")

    def _set_session_active(self, active: bool) -> None:
        self._session_active = active
        self.start_btn.setEnabled(not active)
        self.stop_btn.setEnabled(active)
        self.task_input.setEnabled(not active)
        self.server_input.setEnabled(not active)
        if not active:
            self._update_status("Disconnected", 0)

    def _update_status(self, status: str, step: int) -> None:
        # Dot colour
        colour_map = {
            "Connected":   _CLR_CONNECTED,
            "Connecting…": _CLR_WORKING,
            "Error":       _CLR_WORKING,
            "Disconnected": _CLR_DISCONNECTED,
        }
        dot_color = colour_map.get(status, _CLR_WORKING)
        self.status_dot.setStyleSheet(f"color: {dot_color};")
        self.status_label.setText(status)
        if step > 0:
            self.step_label.setText(f"Step: {step}")
        else:
            self.step_label.setText("Step: —")
        self.statusBar().showMessage(f"Status: {status}  |  Step: {step}")

    def _append_log(self, line: str) -> None:
        # Colour-code lines based on prefix
        html_line = _colorize_log_line(line)
        self.log_feed.moveCursor(QTextCursor.MoveOperation.End)
        self.log_feed.insertHtml(html_line + "<br>")
        self.log_feed.moveCursor(QTextCursor.MoveOperation.End)

    def closeEvent(self, event) -> None:
        if self._session is not None:
            self._session.request_stop()
            self._session.wait(3000)
        self._refresher.stop()
        self._refresher.wait(3000)
        super().closeEvent(event)


def _make_group(title: str) -> QGroupBox:
    g = QGroupBox(title)
    return g


def _make_form_layout():
    from PyQt6.QtWidgets import QFormLayout
    fl = QFormLayout()
    fl.setContentsMargins(6, 6, 6, 6)
    fl.setSpacing(6)
    fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return fl


def _btn_style(bg: str, bg_hover: str, text_color: str = "#ffffff") -> str:
    return (
        f"QPushButton {{"
        f"  background: {bg}; color: {text_color};"
        f"  border: none; border-radius: 4px; padding: 4px 12px;"
        f"  font-weight: bold;"
        f"}}"
        f"QPushButton:hover {{ background: {bg_hover}; }}"
        f"QPushButton:disabled {{ background: #555; color: #888; }}"
    )

def _colorize_log_line(line: str) -> str:
    """Return an HTML-coloured span for a log line."""
    line_html = (
        line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )
    if "✓" in line or "Connected" in line:
        color = "#4ec994"
    elif "✗" in line or "Fatal" in line or "ABORT" in line or "error" in line.lower():
        color = "#f47174"
    elif "⚠" in line or "Modal" in line or "Warn" in line.lower():
        color = "#f0c060"
    elif "Step" in line and "captured" in line:
        color = "#9cdcfe"
    elif "[UI]" in line:
        color = "#888888"
    else:
        color = _CLR_LOG_FG

    return f'<span style="color:{color}; font-family:Consolas,monospace; font-size:9pt;">{line_html}</span>'
