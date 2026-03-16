"""
session.py — Session manager for UI Navigator Desktop Client.

Manages the full capture → transmit → execute loop as a QThread so the
PyQt6 UI never blocks.  Emits Qt signals to update the control panel.

Lifecycle
---------
    mgr = SessionManager(server_url="https://ui-navigator-314272999720.asia-southeast1.run.app", task_goal="...", session_id="abc")
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
import threading
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
    confirmation_required: pyqtSignal = pyqtSignal(str)  # emitted when CONFIRM action received

    def __init__(
        self,
        server_url: str = "https://ui-navigator-314272999720.asia-southeast1.run.app",
        task_goal: str = "",
        session_id: Optional[str] = None,
        api_key: str = "",
        plan_first: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.server_url = server_url.rstrip("/")
        self.task_goal = task_goal
        self.session_id: str = session_id or uuid.uuid4().hex[:12]
        self._api_key: str = api_key
        self._plan_first: bool = plan_first       # fetch /session/plan before acting
        self.session_plan: Optional[dict] = None  # populated by _fetch_session_plan()
        self._stop_requested = False
        self._step_id: int = 0
        # Artifact URLs captured during execution — injected into WRITE_REPORT
        self._last_gcs_url: str = ""
        self._last_cloud_run_url: str = ""
        # Confirmation gate — threading.Event so the UI thread can unblock this thread
        self._confirmation_event = threading.Event()
        self._confirmation_result: bool = False
        self._capturer = FrameCapturer(
            on_frame=lambda f: None,    # we use capture_once(), not the callback
            interval=_LOOP_INTERVAL_S,
        )
        # Wire the capturer + server context into the executor
        self._executor = ActionExecutor(
            capture_fn=self._capturer.capture_once,
            server_url=self.server_url,
            session_id=self.session_id,
        )

    def request_stop(self) -> None:
        """Signal the loop to exit cleanly after the current step."""
        self._stop_requested = True

    def confirm_action(self, confirmed: bool) -> None:
        """
        Called by the UI thread when the user responds to a CONFIRM dialog.

        Sets the result and unblocks the session loop which is waiting on
        ``_confirmation_event``.  Thread-safe: uses a ``threading.Event``.
        """
        self._confirmation_result = confirmed
        self._confirmation_event.set()

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

        # ── Plan-first mode ──────────────────────────────────────────────
        # If plan_first=True, ask the server to decompose the task into steps
        # before starting the action loop.  The plan is stored in
        # self.session_plan but the loop still drives each step via task_goal
        # (future work: step the plan instead of a monolithic goal).
        if self._plan_first and self.task_goal.strip():
            self._log("Plan-first mode — fetching session plan…")
            self.session_plan = self._fetch_session_plan()
            if self.session_plan:
                step_count = len(self.session_plan.get("steps", []))
                self._log(f"  ✓ Session plan ready: {step_count} step(s)")
                for i, step in enumerate(self.session_plan.get("steps", []), 1):
                    self._log(f"  Step {i}: {step.get('description', '')}")
            else:
                self._log("  ⚠ Plan fetch failed — proceeding with monolithic goal")

        try:
            while not self._stop_requested:
                ended_with_verify = self._run_one_step()
                if self._stop_requested:
                    break
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
        if self._stop_requested:
            return False
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
        expected: dict = response_data.get("expected", {})
        max_retries: int = int(expected.get("max_retries", 3))
        on_failure: str = expected.get("on_failure", "HAND_OFF_TO_USER").upper()
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

        # Fallback elements from perception for click-with-verify recovery
        fallback_elements: list[dict] = [
            e for e in perception.get("elements", [])
            if e.get("priority", 1) > 1
        ]

        ended_with_verify = False
        last_major_action_type: str = ""
        # Track whether the user has confirmed in this step (for DEPLOY_CLOUD_RUN)
        _user_confirmed: bool = False

        for action in actions:
            if self._stop_requested:
                break

            action_type = action.get("type", "").upper()

            # ── CONFIRM: pause loop and ask the user ──────────────────
            if action_type == "CONFIRM":
                message = (
                    action.get("message")
                    or "Do you want to proceed with this action?"
                )
                action_ref = action.get("action_ref", "")
                self._log(
                    f"  ⚠ Confirmation required"
                    + (f" [{action_ref}]" if action_ref else "")
                    + f": {message}"
                )
                # Reset event for a fresh wait
                self._confirmation_event.clear()
                self._confirmation_result = False
                # Ask UI to show a dialog (emits to main thread)
                self.confirmation_required.emit(message)
                # Block the session thread for up to 120 s
                responded = self._confirmation_event.wait(timeout=120.0)
                if not responded or not self._confirmation_result:
                    self._log("  ✗ Action cancelled — user declined (or timed out)")
                    self._stop_requested = True
                    break
                _user_confirmed = True
                self._log("  ✓ Confirmation received — proceeding")
                continue  # CONFIRM itself has no physical execution

            # Inject confirmed=True into the immediately following DEPLOY action
            if action_type == "DEPLOY_CLOUD_RUN" and _user_confirmed:
                action = dict(action)
                action["confirmed"] = True

            result: ExecutionResult = self._executor.execute(action)
            self._log(f"  {result}")

            # ── Capture artifact URLs for downstream WRITE_REPORT ─────
            if action_type == "UPLOAD_GCS" and result.success:
                gcs_url = result.extra.get("gcs_url", "")
                if gcs_url:
                    self._last_gcs_url = gcs_url
                    self._log(f"  ↳ GCS URL captured: {gcs_url}")

            if action_type == "DEPLOY_CLOUD_RUN" and result.success:
                service_url = result.extra.get("service_url", "")
                if service_url:
                    self._last_cloud_run_url = service_url
                    self._log(f"  ↳ Cloud Run URL captured: {service_url}")

            # Inject captured URLs into WRITE_REPORT before executing
            if action_type == "WRITE_REPORT":
                action = dict(action)
                if not action.get("gcs_log_url") and self._last_gcs_url:
                    action["gcs_log_url"] = self._last_gcs_url
                if not action.get("cloud_run_url") and self._last_cloud_run_url:
                    action["cloud_run_url"] = self._last_cloud_run_url
                result = self._executor.execute(action)
                self._log(f"  {result}")
                if result.success:
                    self._log("  ✓ Workflow complete — stopping session after WRITE_REPORT")
                    self._stop_requested = True
                    self.task_completed.emit(result.message)
                    break

            # Emit frame update after each action so the UI stays live
            try:
                live_frame = self._capturer.capture_once()
                self.frame_ready.emit(live_frame.jpeg_bytes)
            except Exception:
                pass

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

            # ── Bounded recovery on VERIFY failure ───────────────────
            if action_type == "VERIFY" and not result.success and not result.skipped:
                recovered = self._run_recovery(
                    failed_action=action,
                    last_major_action_type=last_major_action_type,
                    fallback_elements=fallback_elements,
                    max_retries=max_retries,
                    on_failure=on_failure,
                )
                if not recovered and on_failure == "HAND_OFF_TO_USER":
                    break

            # Track whether the plan ended with a VERIFY ──────────────
            if action_type == "VERIFY":
                ended_with_verify = True
            else:
                ended_with_verify = False
                last_major_action_type = action_type

            # Pause and let UI settle between physical actions
            if not result.skipped:
                self.msleep(300)

        return ended_with_verify

    def _run_recovery(
        self,
        failed_action: dict,
        last_major_action_type: str,
        fallback_elements: list[dict],
        max_retries: int,
        on_failure: str,
    ) -> bool:
        """
        **Bounded recovery playbook** (context-aware visual recovery).

        Called when a VERIFY action returns ``success=False``.  Applies
        deterministic recovery strategies in priority order, up to
        ``max_retries`` attempts.  After exhausting retries, emits
        HAND_OFF_TO_USER and sets ``_stop_requested``.

        Recovery strategies (deterministic-first, no extra Gemini call):
        1. Try fallback elements (priority=2+) from the perception output.
        2. Scroll down to reveal hidden content.
        3. Scroll up to reset position.
        4. Zoom out (CTRL -) to expose off-screen elements.
        5. Switch tab (CTRL+TAB).
        6. In-page search (CTRL+F).
        7. Refocus address bar (CTRL+L) — useful when browser navigation fails.

        Returns True if a recovery attempt was made (regardless of outcome).
        """
        description = failed_action.get("description") or failed_action.get("reason") or "unknown"
        self._log(f"  ⚠ VERIFY failed — starting context-aware visual recovery: {description}")

        # ── Strategy 1: try fallback element clicks ─────────────────
        if fallback_elements and last_major_action_type == "CLICK":
            for fb in fallback_elements[:max_retries]:
                bbox = fb.get("bbox")
                if bbox and len(bbox) == 4:
                    cx = bbox[0] + bbox[2] // 2
                    cy = bbox[1] + bbox[3] // 2
                    label = fb.get("label", f"fallback@{cx},{cy}")
                    self._log(f"  ↩ Recovery: clicking fallback element '{label}' at ({cx},{cy})")
                    result = self._executor.execute({"type": "CLICK", "x": cx, "y": cy, "reason": f"recovery: fallback element '{label}'"})
                    self._log(f"  {result}")
                    return True

        # ── Deterministic recovery sequence ─────────────────────────
        recovery_steps = [
            (1, "Scroll down to reveal content",
             {"type": "SCROLL", "dx": 0, "dy": -3, "reason": "recovery: scroll down"}),
            (2, "Scroll up to reset position",
             {"type": "SCROLL", "dx": 0, "dy": 3, "reason": "recovery: scroll up"}),
            (3, "Zoom out to expose off-screen elements",
             {"type": "HOTKEY", "keys": ["ctrl", "-"], "reason": "recovery: zoom out"}),
            (4, "Switch tab",
             {"type": "HOTKEY", "keys": ["ctrl", "tab"], "reason": "recovery: switch tab"}),
            (5, "In-page search",
             {"type": "HOTKEY", "keys": ["ctrl", "f"], "reason": "recovery: open search"}),
            (6, "Refocus address bar",
             {"type": "HOTKEY", "keys": ["ctrl", "l"], "reason": "recovery: focus address bar"}),
        ]

        for attempt, (strategy_num, strategy_name, recovery_action) in enumerate(
            recovery_steps[:max_retries], start=1
        ):
            self._log(
                f"  ↩ Recovery attempt {attempt}/{max_retries}: {strategy_name}"
            )
            result = self._executor.execute(recovery_action)
            self._log(f"  {result}")
            self.msleep(500)  # let UI settle after recovery action

        # ── Exhausted retries ────────────────────────────────────────
        summary = (
            f"VERIFY failed after {max_retries} recovery attempt(s) for: {description}. "
            f"Last major action: {last_major_action_type}. Human intervention required."
        )
        self._log(f"  🤚 Recovery exhausted — {summary}")
        if on_failure == "HAND_OFF_TO_USER":
            self._stop_requested = True
            self.hand_off_requested.emit(summary)
        elif on_failure == "ABORT":
            self._stop_requested = True
            self.session_ended.emit(f"ABORT after recovery failure: {summary}")
        # on_failure == "RETRY" → let the outer loop continue to the next frame
        return True

    def _fetch_session_plan(self) -> Optional[dict]:
        """
        Call POST /session/plan to decompose self.task_goal into numbered steps.

        Returns the parsed JSON response dict (``SessionPlan`` shape) or None
        on error.  Non-fatal — the session loop continues even if planning fails.
        """
        try:
            resp = requests.post(
                f"{self.server_url}/session/plan",
                headers=self._api_headers(),
                json={
                    "session_id": self.session_id,
                    "task_goal": self.task_goal,
                    "context": "",
                },
                timeout=_REQUEST_TIMEOUT_S,
            )
            if resp.status_code != 200:
                self._log(f"  ✗ /session/plan returned {resp.status_code}: {resp.text[:200]}")
                return None
            return resp.json()
        except Exception as exc:
            self._log(f"  ✗ /session/plan request failed: {exc}")
            return None

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
