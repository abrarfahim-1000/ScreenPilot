"""
session.py — Session manager for UI Navigator Desktop Client.

Manages the full capture → transmit → execute loop as a QThread so the
PyQt6 UI never blocks.  Emits Qt signals to update the control panel.

Lifecycle
---------
    mgr = SessionManager(server_url="http://localhost:8080", task_goal="...", session_id="abc")
    mgr.action_logged.connect(log_widget.append)
    mgr.status_changed.connect(on_status)
    mgr.frame_ready.connect(on_frame)       # latest JPEG bytes for display
    mgr.start()          # begins QThread
    ...
    mgr.request_stop()   # graceful shutdown
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Optional

import requests
from PyQt6.QtCore import QThread, pyqtSignal

from capture import CapturedFrame, FrameCapturer
from executor import ActionExecutor, ExecutionResult

logger = logging.getLogger(__name__)

_FRAME_ENDPOINT = "/session/frame"
_HEALTH_ENDPOINT = "/health"

# How long to pause between full loop iterations (after executing all actions).
_LOOP_INTERVAL_S: float = 2.5

# Connection / read timeout for requests.
# The server makes two sequential Gemini API calls (perceive + plan).
# Each can take 15–30 s; use a generous read timeout so we never drop a step.
_REQUEST_TIMEOUT_S: tuple[float, float] = (5.0, 90.0)


def _ts() -> str:
    """Return a short HH:MM:SS timestamp string."""
    return time.strftime("%H:%M:%S")

class SessionManager(QThread):
    """
    Background QThread that drives the perception-action loop.

    Signals
    -------
    action_logged(str)
        A human-readable log line (timestamp + message) to append to the feed.
    status_changed(str, int)
        (status_text, step_id) — e.g. ("Connected", 3) or ("Disconnected", 0).
    frame_ready(bytes)
        Raw JPEG bytes of the most recently captured frame, for live display.
    session_ended(str)
        Reason the session stopped (e.g. "User requested stop", "ABORT action").
    """

    action_logged: pyqtSignal = pyqtSignal(str)
    status_changed: pyqtSignal = pyqtSignal(str, int)
    frame_ready: pyqtSignal = pyqtSignal(bytes)
    session_ended: pyqtSignal = pyqtSignal(str)
    auth_error: pyqtSignal = pyqtSignal(str)       # emitted on HTTP 401/403
    task_completed: pyqtSignal = pyqtSignal(str)   # emitted when model declares goal achieved
    hand_off_requested: pyqtSignal = pyqtSignal(str)  # emitted when HAND_OFF_TO_USER received

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        task_goal: str = "",
        session_id: Optional[str] = None,
        api_key: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.server_url = server_url.rstrip("/")
        self.task_goal = task_goal
        self.session_id: str = session_id or uuid.uuid4().hex[:12]
        self._api_key: str = api_key
        self._executor = ActionExecutor()
        self._stop_requested = False
        self._step_id: int = 0
        self._capturer = FrameCapturer(
            on_frame=lambda f: None,    # we use capture_once(), not the callback
            interval=_LOOP_INTERVAL_S,
        )

    def request_stop(self) -> None:
        """Signal the loop to exit cleanly after the current step."""
        self._stop_requested = True

    def run(self) -> None:
        self._stop_requested = False
        self._step_id = 0

        self._log(f"Session {self.session_id} starting — goal: {self.task_goal or '(none)'}")
        self._emit_status("Connecting…", 0)

        if not self._health_check():
            self._log("✗ Server unreachable — is it running?")
            self._emit_status("Disconnected", 0)
            self.session_ended.emit("Server unreachable")
            return

        self._log(f"✓ Connected to {self.server_url}")
        self._emit_status("Connected", 0)

        try:
            while not self._stop_requested:
                ended_with_verify = self._run_one_step()
                if not self._stop_requested:
                    # After a VERIFY-terminated plan, re-check quickly (500 ms)
                    # so the agent sees the result of its actions sooner.
                    # Otherwise, use the normal polling interval.
                    wait_ms = 500 if ended_with_verify else int(_LOOP_INTERVAL_S * 1000)
                    self.msleep(wait_ms)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error in session loop")
            self._log(f"✗ Fatal error: {exc}")
        finally:
            self._emit_status("Disconnected", self._step_id)
            reason = "User requested stop" if self._stop_requested else "Loop ended"
            self._log(f"Session ended: {reason}")
            self.session_ended.emit(reason)

    def _run_one_step(self) -> bool:
        """
        Run one full perception → plan → execute cycle.

        Returns
        -------
        bool
            True if the plan ended with a VERIFY action (signals that the agent
            wants a quick re-check rather than the normal 2.5 s wait).
        """
        # 1. Capture frame
        try:
            frame: CapturedFrame = self._capturer.capture_once()
        except Exception as exc:
            self._log(f"✗ Capture failed: {exc}")
            return False

        # Emit latest JPEG for display
        self.frame_ready.emit(frame.jpeg_bytes)

        self._step_id += 1
        self._emit_status("Connected", self._step_id)
        self._log(f"Step {self._step_id} — captured frame ({frame.width}×{frame.height})")

        # 2. Send frame to server
        response_data = self._send_frame(frame)
        if response_data is None:
            self._emit_status("Error", self._step_id)
            return False

        # 3. Log perception summary
        perception = response_data.get("perception", {})
        summary = perception.get("screen_summary", "")
        if summary:
            self._log(f"  Perception: {summary}")
        modal = perception.get("unexpected_modal")
        if modal:
            self._log(f"  ⚠ Modal detected: {modal}")

        # 4. Execute actions
        actions: list[dict] = response_data.get("actions", [])
        self._log(f"  {len(actions)} action(s) to execute")

        # ── Task-completion detection ────────────────────────────────
        # Per planning rule 10: "If the task goal is already complete,
        # output only a VERIFY action."  A single-action [VERIFY] plan
        # is the model's explicit signal that the goal has been reached.
        if len(actions) == 1 and actions[0].get("type", "").upper() == "VERIFY":
            completion_note = (
                perception.get("screen_summary")
                or "Task complete as confirmed by the agent."
            )
            self._log(f"  ✓ Task completed — agent reports goal achieved")
            self._log(f"  Screen: {completion_note}")
            self._stop_requested = True
            self.task_completed.emit(completion_note)
            return True

        ended_with_verify = False
        for action in actions:
            if self._stop_requested:
                break
            result: ExecutionResult = self._executor.execute(action)
            self._log(f"  {result}")

            # Emit frame update after each action so the UI stays live
            try:
                live_frame = self._capturer.capture_once()
                self.frame_ready.emit(live_frame.jpeg_bytes)
            except Exception:
                pass

            action_type = action.get("type", "").upper()

            # Stop on ABORT ───────────────────────────────────────────
            if action_type == "ABORT":
                self._log("  ⛔ ABORT action received — stopping session")
                self._stop_requested = True
                break

            # Pause and request human help on HAND_OFF_TO_USER ────────
            if action_type == "HAND_OFF_TO_USER":
                message = (
                    action.get("summary")
                    or action.get("reason")
                    or "The agent cannot proceed — human action required."
                )
                self._log(f"  🤚 HAND_OFF_TO_USER — pausing session: {message}")
                self._stop_requested = True
                self.hand_off_requested.emit(message)
                break

            # Track whether the plan ended with a VERIFY ──────────────
            if action_type == "VERIFY":
                ended_with_verify = True
            else:
                ended_with_verify = False

            # Pause and let UI settle between physical actions
            if not result.skipped:
                self.msleep(300)

        return ended_with_verify

    def _api_headers(self) -> dict:
        """HTTP headers injected into every request to the server."""
        headers = {}
        if self._api_key:
            headers["X-Gemini-Api-Key"] = self._api_key
        return headers

    def _health_check(self) -> bool:
        try:
            r = requests.get(
                f"{self.server_url}{_HEALTH_ENDPOINT}",
                headers=self._api_headers(),
                timeout=_REQUEST_TIMEOUT_S[0],
            )
            return r.status_code == 200
        except Exception:
            return False

    def _send_frame(self, frame: CapturedFrame) -> Optional[dict]:
        metadata = {
            "session_id": self.session_id,
            "step_id": self._step_id,
            "task_goal": self.task_goal,
            "frame_hash": frame.frame_hash,
            "timestamp": frame.timestamp,
            "width": frame.width,
            "height": frame.height,
            "monitor_index": frame.monitor_index,
        }
        try:
            resp = requests.post(
                f"{self.server_url}{_FRAME_ENDPOINT}",
                headers=self._api_headers(),
                files={"frame": ("frame.jpg", frame.jpeg_bytes, "image/jpeg")},
                data={"metadata": json.dumps(metadata)},
                timeout=_REQUEST_TIMEOUT_S,
            )
            if resp.status_code in (401, 403):
                detail = resp.text[:200] or f"HTTP {resp.status_code}"
                self._log(f"  ⚠ Auth error ({resp.status_code}) — invalid or missing API key")
                self.auth_error.emit(detail)
                self._stop_requested = True
                return None
            if resp.status_code != 200:
                self._log(f"  ✗ Server error {resp.status_code}: {resp.text[:200]}")
                return None
            return resp.json()
        except requests.exceptions.ConnectionError:
            self._log("  ✗ Connection lost — server unreachable")
            self._emit_status("Disconnected", self._step_id)
            return None
        except requests.exceptions.Timeout:
            self._log("  ✗ Request timed out")
            return None
        except Exception as exc:  # noqa: BLE001
            self._log(f"  ✗ Request failed: {exc}")
            return None

    def _log(self, message: str) -> None:
        self.action_logged.emit(f"[{_ts()}] {message}")

    def _emit_status(self, status: str, step: int) -> None:
        self.status_changed.emit(status, step)
