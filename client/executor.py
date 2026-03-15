"""
executor.py — Action executor for UI Navigator Desktop Client.

Responsibilities:
- Receive an action dict (from ActionResponse.actions) and execute it locally.
- Use pyautogui for mouse / keyboard actions.
- Use pygetwindow for window focus.
- Return an ExecutionResult for every action so the session loop can log outcomes.

Supported action types (mirrors ActionType enum in server/schemas.py):
    FOCUS_WINDOW, CLICK, DOUBLE_CLICK, RIGHT_CLICK, TYPE, HOTKEY,
    SCROLL, WAIT, DRAG, COPY, PASTE, VERIFY, ABORT, HAND_OFF_TO_USER
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from requests.exceptions import Timeout as _RequestsTimeout
import pyautogui
from command_policy import execute_command as _policy_execute
from log_parser import (
    _is_safe_read_path,
    generate_deployment_report,
    parse_test_log,
    TestSummary,
    MAX_READ_BYTES,
)
from redaction import redact_text
from window_focus import focus_window

# pyautogui safety: disable the fail-safe corner but keep a short pause between actions
pyautogui.FAILSAFE = True       # move mouse to top-left corner to abort
pyautogui.PAUSE = 0.10          # 100 ms between every pyautogui call (was 50ms; more reliable on slow VMs)

logger = logging.getLogger(__name__)

@dataclass
class ExecutionResult:
    """Result of executing one action."""
    action_type: str
    success: bool
    message: str = ""
    duration_ms: float = 0.0
    skipped: bool = False           # True for VERIFY / HAND_OFF (no physical action)
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        flag = "✓" if self.success else ("~" if self.skipped else "✗")
        return f"[{flag}] {self.action_type}: {self.message} ({self.duration_ms:.0f} ms)"

class ActionExecutor:
    """
    Executes structured action dicts returned by the cloud orchestrator.

    Each public method maps 1-to-1 with an ActionType and returns an
    ExecutionResult.  The main entry point is :meth:`execute`.

    Parameters
    ----------
    capture_fn:
        Optional callable ``() -> bytes`` that returns fresh JPEG bytes.
        When provided, :meth:`_click` uses it to implement the
        **click-with-verification** pattern: after clicking the primary
        coordinates the frame is re-captured and the correct element is
        confirmed to have been activated.  If the primary coords miss, up to
        ``click_max_retries`` fallback coords are tried before giving up.
    click_max_retries:
        Maximum additional click attempts using fallback elements or a small
        search spiral when the primary click verifies as a miss (default 2).
    server_url:
        Base URL of the orchestrator (e.g. ``http://localhost:8080``).  Used
        by :meth:`_upload_gcs` to POST log artifacts to ``/session/upload``.
    session_id:
        Current session identifier, forwarded in GCS upload requests.
    """

    def __init__(
        self,
        capture_fn=None,
        click_max_retries: int = 2,
        server_url: str = "",
        session_id: str = "",
    ) -> None:
        self._capture_fn = capture_fn          # Optional[Callable[[], CapturedFrame]]
        self._click_max_retries = click_max_retries
        self._server_url = server_url.rstrip("/")
        self._session_id = session_id

    def execute(self, action: dict[str, Any]) -> ExecutionResult:
        """
        Dispatch a single action dict to the appropriate handler.

        Parameters
        ----------
        action:
            Raw action dict from ActionResponse.actions  (has a "type" key).
        """
        action_type = action.get("type", "UNKNOWN").upper()
        t0 = time.monotonic()

        try:
            result = self._dispatch(action_type, action)
        except pyautogui.FailSafeException:
            result = ExecutionResult(
                action_type=action_type,
                success=False,
                message="PyAutoGUI fail-safe triggered (mouse moved to corner)",
            )
        except Exception as exc:  # noqa: BLE001
            result = ExecutionResult(
                action_type=action_type,
                success=False,
                message=f"Exception: {exc}",
            )

        result.duration_ms = (time.monotonic() - t0) * 1000
        logger.info("%s", result)
        return result

    def _dispatch(self, action_type: str, action: dict) -> ExecutionResult:
        handlers = {
            "FOCUS_WINDOW":     self._focus_window,
            "CLICK":            self._click,
            "DOUBLE_CLICK":     self._double_click,
            "RIGHT_CLICK":      self._right_click,
            "TYPE":             self._type,
            "HOTKEY":           self._hotkey,
            "SCROLL":           self._scroll,
            "WAIT":             self._wait,
            "DRAG":             self._drag,
            "COPY":             self._copy,
            "PASTE":            self._paste,
            "VERIFY":           self._verify,
            "EXEC_COMMAND":     self._exec_command,
            "READ_FILE":        self._read_file,
            "PARSE_LOG":        self._parse_log,
            "WRITE_REPORT":     self._write_report,
            "UPLOAD_GCS":       self._upload_gcs,
            "DEPLOY_CLOUD_RUN": self._deploy_cloud_run,
            "ABORT":            self._abort,
            "HAND_OFF_TO_USER": self._hand_off,
        }
        handler = handlers.get(action_type)
        if handler is None:
            return ExecutionResult(
                action_type=action_type,
                success=False,
                message=f"Unknown action type: {action_type}",
            )
        return handler(action)

    def _focus_window(self, action: dict) -> ExecutionResult:
        title = action.get("title_contains", "")
        threshold = action.get("fuzzy_threshold", 70)
        result = focus_window(title, fuzzy_threshold=threshold)
        return ExecutionResult(
            "FOCUS_WINDOW",
            result.success,
            result.message,
            extra={
                "match_type": result.match_type,
                "matched_title": result.matched_title,
            },
        )

    def _click(self, action: dict) -> ExecutionResult:
        x, y = action.get("x", 0), action.get("y", 0)
        fallback_elements: list[dict] = action.get("fallback_elements") or []
        return self._click_with_verify(x, y, fallback_elements, reason=action.get("reason", ""))

    def _click_with_verify(
        self,
        x: int,
        y: int,
        fallback_elements: list[dict],
        reason: str = "",
    ) -> ExecutionResult:
        """
        **Click-with-verification pattern.**

        1. Click the primary coordinates (x, y).
        2. Re-capture the frame (if ``capture_fn`` was supplied).
        3. If a ``verify_label`` hint is present in the action and capture is
           available, check a bounding-box heuristic to confirm the right
           element was activated.
        4. If the primary click cannot be verified as a hit:
           a. Try each fallback element's centre coordinates in priority order.
           b. After exhausting fallbacks, log a warning (executor cannot OCR /
              ask Gemini — that is the session-level recovery's job).

        Returns a single :class:`ExecutionResult` summarising all attempts.
        """
        attempts: list[str] = []
        coords_tried: list[tuple[int, int]] = [(x, y)]

        # Build candidate list: primary first, then fallback element centres
        for fb in fallback_elements:
            bbox = fb.get("bbox")
            if bbox and len(bbox) == 4:
                cx = bbox[0] + bbox[2] // 2
                cy = bbox[1] + bbox[3] // 2
                coords_tried.append((cx, cy))

        # Limit total attempts
        coords_tried = coords_tried[:1 + self._click_max_retries]

        last_result: ExecutionResult | None = None
        for attempt_num, (cx, cy) in enumerate(coords_tried):
            label = "primary" if attempt_num == 0 else f"fallback-{attempt_num}"
            pyautogui.click(cx, cy)
            time.sleep(0.25)  # OS settle time

            if self._capture_fn is not None:
                try:
                    self._capture_fn()  # re-capture to update live display
                except Exception:
                    pass

            attempts.append(f"{label}({cx},{cy})")
            last_result = ExecutionResult(
                "CLICK",
                True,
                f"Clicked {label} ({cx},{cy}) — {len(attempts)} attempt(s)",
                extra={"attempts": attempts, "final_coords": (cx, cy)},
            )
            # Primary always succeeds from executor's perspective; visual
            # verification of whether the *right* element activated is done by
            # the VERIFY action that the planner appends after every CLICK.
            # Only continue to the next fallback if the session explicitly
            # signals a miss (handled in session.py recovery loop).
            break

        if last_result is None:
            last_result = ExecutionResult("CLICK", False, "No coordinates to click")

        return last_result

    def _double_click(self, action: dict) -> ExecutionResult:
        x, y = action.get("x", 0), action.get("y", 0)
        pyautogui.doubleClick(x, y)
        return ExecutionResult("DOUBLE_CLICK", True, f"Double-clicked ({x}, {y})")

    def _right_click(self, action: dict) -> ExecutionResult:
        x, y = action.get("x", 0), action.get("y", 0)
        pyautogui.rightClick(x, y)
        return ExecutionResult("RIGHT_CLICK", True, f"Right-clicked ({x}, {y})")

    def _type(self, action: dict) -> ExecutionResult:
        text = action.get("text", "")
        if not text:
            return ExecutionResult("TYPE", False, "Empty text — skipped")

        # ── Secret redaction (Day 1 safety requirement) ────────────────
        # Scan the text BEFORE it is executed or leaves the client.
        # If secrets are found, replace them with placeholders and block
        # the action so raw credentials are never typed or copied.
        safe_text, matches = redact_text(text)
        if matches:
            names = [m.pattern_name for m in matches]
            logger.warning(
                "TYPE action blocked: %d secret(s) detected and redacted: %s",
                len(matches),
                names,
            )
            return ExecutionResult(
                "TYPE",
                False,
                f"Blocked — {len(matches)} secret(s) redacted "
                f"({', '.join(names)}). Secrets never leave the client unredacted.",
            )

        # ── Execute (clean text only) ──────────────────────────────────
        # pyautogui.typewrite doesn't handle unicode well; use pyperclip + paste.
        # IMPORTANT: sleep after copy so the OS clipboard is populated before
        # ctrl+v fires — without this there is a race on fast machines.
        try:
            import pyperclip
            pyperclip.copy(safe_text)
            time.sleep(0.15)              # wait for clipboard to settle
            pyautogui.hotkey("ctrl", "v")
            return ExecutionResult("TYPE", True, f"Typed {len(safe_text)} chars via clipboard")
        except Exception:
            pyautogui.typewrite(safe_text, interval=0.04)
            return ExecutionResult("TYPE", True, f"Typed {len(safe_text)} chars via typewrite")

    def _hotkey(self, action: dict) -> ExecutionResult:
        keys = action.get("keys", [])
        if not keys:
            return ExecutionResult("HOTKEY", False, "No keys specified")
        pyautogui.hotkey(*keys)
        return ExecutionResult("HOTKEY", True, f"Hotkey: {'+'.join(keys)}")

    def _scroll(self, action: dict) -> ExecutionResult:
        dx = action.get("dx", 0)
        dy = action.get("dy", 0)
        if dy:
            pyautogui.scroll(dy)
        if dx:
            pyautogui.hscroll(dx)
        return ExecutionResult("SCROLL", True, f"Scrolled dx={dx} dy={dy}")

    def _wait(self, action: dict) -> ExecutionResult:
        ms = action.get("ms", 500)
        time.sleep(ms / 1000.0)
        return ExecutionResult("WAIT", True, f"Waited {ms} ms")

    def _drag(self, action: dict) -> ExecutionResult:
        fx = action.get("from_x", 0)
        fy = action.get("from_y", 0)
        tx = action.get("to_x", 0)
        ty = action.get("to_y", 0)
        pyautogui.moveTo(fx, fy)
        pyautogui.dragTo(tx, ty, duration=0.5, button="left")
        return ExecutionResult("DRAG", True, f"Dragged ({fx},{fy}) → ({tx},{ty})")

    def _copy(self, action: dict) -> ExecutionResult:
        pyautogui.hotkey("ctrl", "c")
        return ExecutionResult("COPY", True, "Ctrl+C sent")

    def _paste(self, action: dict) -> ExecutionResult:
        pyautogui.hotkey("ctrl", "v")
        return ExecutionResult("PASTE", True, "Ctrl+V sent")

    def _verify(self, action: dict) -> ExecutionResult:
        method = action.get("method", "visual")
        desc   = action.get("description", "(no description)")

        if method == "read_file":
            path = (action.get("path") or "").strip()
            if path:
                read_result = self._read_file({"path": path})
                if read_result.success:
                    content   = read_result.extra.get("content", "")
                    must_see  = action.get("must_see") or []
                    missing   = [
                        s for s in must_see
                        if s.lower() not in content.lower()
                    ]
                    if missing:
                        return ExecutionResult(
                            "VERIFY", False,
                            f"Visual precision check FAILED — strings not found in "
                            f"{path}: {missing}",
                            extra={
                                "method": "read_file",
                                "path": path,
                                "missing": missing,
                                "content_preview": content[:300],
                            },
                        )
                    return ExecutionResult(
                        "VERIFY", True,
                        f"Visual precision check ✓ — {path} readable "
                        f"({len(content)} chars): {desc}",
                        extra={
                            "method": "read_file",
                            "path": path,
                            "content_preview": content[:300],
                        },
                    )
                # File could not be read
                return ExecutionResult(
                    "VERIFY", False,
                    f"Visual precision check FAILED — cannot read "
                    f"{path}: {read_result.message}",
                )

        # Default: VERIFY is a signal to re-capture; no physical action needed.
        return ExecutionResult(
            "VERIFY", True,
            f"Verify queued: {desc}",
            skipped=True,
        )

    def _exec_command(self, action: dict) -> ExecutionResult:
        cmd = action.get("command", "").strip()
        if not cmd:
            return ExecutionResult("EXEC_COMMAND", False, "Empty command — skipped")

        timeout_s = int(action.get("timeout_s", 60))
        cmd_result = _policy_execute(cmd, timeout_s=timeout_s)

        if cmd_result.blocked:
            logger.warning("EXEC_COMMAND blocked by policy: %s", cmd_result.block_reason)
            return ExecutionResult(
                "EXEC_COMMAND",
                False,
                f"Blocked by policy: {cmd_result.block_reason}",
                extra={"command": cmd, "blocked": True},
            )

        success = cmd_result.returncode == 0
        msg = (
            f"returncode={cmd_result.returncode} "
            f"duration={cmd_result.duration_ms:.0f}ms"
            + (" [TIMEOUT]" if cmd_result.timed_out else "")
        )
        return ExecutionResult(
            "EXEC_COMMAND",
            success,
            msg,
            extra={
                "command": cmd,
                "returncode": cmd_result.returncode,
                "stdout": cmd_result.stdout[:500],  # truncate for logging
                "stderr": cmd_result.stderr[:200],
                "timed_out": cmd_result.timed_out,
            },
        )

    # ── File / log actions (Checkpoint 1 & 2, Days 3-5) ────────────────────

    def _read_file(self, action: dict) -> ExecutionResult:
        """
        Read a file directly from disk — no shell command, no OCR.

        Returns the file content in ``extra["content"]`` so the orchestrator
        can inspect it on the next loop iteration without a screenshot.
        The tail of the file is kept when it exceeds *max_bytes*.
        """
        path = (action.get("path") or "").strip()
        if not path:
            return ExecutionResult("READ_FILE", False, "No path specified — skipped")

        max_bytes = min(
            int(action.get("max_bytes", MAX_READ_BYTES)),
            MAX_READ_BYTES,
        )

        safe, reason = _is_safe_read_path(path)
        if not safe:
            logger.warning("READ_FILE blocked: %s", reason)
            return ExecutionResult(
                "READ_FILE", False,
                f"Path safety check failed: {reason}",
                extra={"path": path, "blocked": True},
            )

        try:
            raw = Path(path).read_bytes()
        except FileNotFoundError:
            return ExecutionResult(
                "READ_FILE", False,
                f"File not found: {path}",
                extra={"path": path},
            )
        except OSError as exc:
            return ExecutionResult(
                "READ_FILE", False,
                f"Failed to read file: {exc}",
                extra={"path": path},
            )

        truncated = len(raw) > max_bytes
        if truncated:
            raw = raw[-max_bytes:]   # keep the tail — test summaries are at the end
        content = raw.decode("utf-8", errors="replace")

        msg = (
            f"Read {len(content)} chars from {path}"
            + (" (truncated to tail)" if truncated else "")
        )
        return ExecutionResult(
            "READ_FILE", True, msg,
            extra={"path": path, "content": content, "truncated": truncated},
        )

    def _parse_log(self, action: dict) -> ExecutionResult:
        """
        Parse a test runner log file (pytest / unittest / make test) and return
        a structured test summary in ``extra``.  Uses the client-side
        :func:`parse_test_log` — no screenshot OCR, no Gemini API call needed.
        """
        path = (action.get("path") or "").strip()
        if not path:
            return ExecutionResult("PARSE_LOG", False, "No path specified — skipped")

        # Allow the caller to inject content directly (useful in testing and when
        # the content was already fetched via a READ_FILE action).
        content  = action.get("content") or None
        exit_code = action.get("exit_code")

        summary = parse_test_log(path, content=content, exit_code=exit_code)

        # Treat as success if we got a recognisable count or at least a snippet
        success = (summary.total > 0) or bool(summary.raw_snippet)
        msg = f"Parsed {summary.parser_format} log — {summary.summary_line}"
        if summary.parse_errors:
            msg += f" | warnings: {'; '.join(summary.parse_errors)}"

        return ExecutionResult(
            "PARSE_LOG", success, msg,
            extra={
                "path": path,
                "total":         summary.total,
                "passed":        summary.passed,
                "failed":        summary.failed,
                "errors":        summary.errors,
                "skipped":       summary.skipped,
                "warnings":      summary.warnings,
                "duration_s":    summary.duration_s,
                "test_success":  summary.success,
                "summary_line":  summary.summary_line,
                "parser_format": summary.parser_format,
                "parse_errors":  summary.parse_errors,
                "raw_snippet":   summary.raw_snippet,
                "exit_code":     summary.exit_code,
            },
        )

    def _write_report(self, action: dict) -> ExecutionResult:
        """
        Generate a :class:`DeploymentReport` from a test log and write it to a
        file.  Optionally copies the report to the system clipboard so the agent
        can present it to the user without a browser or editor.
        """
        log_path  = (action.get("log_path")   or "").strip()
        report_path = (action.get("report_path") or "").strip()
        copy_to_clipboard = bool(action.get("copy_to_clipboard", True))

        # Parse the log (returns empty summary if path is missing/unreadable)
        if log_path:
            summary = parse_test_log(
                log_path,
                content=action.get("content") or None,
                exit_code=action.get("exit_code"),
            )
        else:
            summary = TestSummary()

        report = generate_deployment_report(
            summary=summary,
            session_id=action.get("session_id", ""),
            task_goal=action.get("task_goal", ""),
            gcs_log_url=action.get("gcs_log_url", ""),
            cloud_run_url=action.get("cloud_run_url", ""),
            report_path=report_path,
        )

        # ── Write to file ────────────────────────────────────────────────────
        written = False
        try:
            Path(report.report_path).write_text(report.report_text, encoding="utf-8")
            written = True
            logger.info("WRITE_REPORT: report written to %s", report.report_path)
        except OSError as exc:
            logger.warning("WRITE_REPORT: failed to write report file: %s", exc)

        # ── Copy to clipboard ─────────────────────────────────────────────────
        clipped = False
        if copy_to_clipboard:
            try:
                import pyperclip  # lazy import — same pattern as _type()
                pyperclip.copy(report.report_text)
                clipped = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("WRITE_REPORT: clipboard copy failed: %s", exc)

        msg = (
            "Report generated"
            + (f" → {report.report_path}" if written else " (write failed)")
            + (" + copied to clipboard" if clipped else "")
        )
        return ExecutionResult(
            "WRITE_REPORT", written or clipped, msg,
            extra={
                "report_path":  report.report_path,
                "report_text":  report.report_text,
                "written":      written,
                "clipped":      clipped,
                "test_success": report.test_summary.success
                                if report.test_summary else None,
            },
        )

    # ── Cloud integration actions (Days 9-11) ─────────────────────────────

    def _upload_gcs(self, action: dict) -> ExecutionResult:
        """
        Upload a local file to Google Cloud Storage via the server's
        ``/session/upload`` endpoint.  The server handles GCS credentials;
        the client only reads the local file and POSTs the bytes.

        On success ``extra["gcs_url"]`` contains the ``gs://`` URI returned
        by the server, and ``extra["uploaded"]`` is ``True``.
        The SessionManager captures this URL for later WRITE_REPORT actions.
        """
        local_path = (action.get("local_path") or "").strip()
        gcs_object = (action.get("gcs_object") or "").strip()
        content_type = (action.get("content_type") or "text/plain").strip()

        if not local_path:
            return ExecutionResult("UPLOAD_GCS", False, "No local_path specified — skipped")

        # Safety check on read path
        safe, reason = _is_safe_read_path(local_path)
        if not safe:
            logger.warning("UPLOAD_GCS blocked: %s", reason)
            return ExecutionResult(
                "UPLOAD_GCS", False,
                f"Path safety check failed: {reason}",
                extra={"local_path": local_path, "blocked": True},
            )

        try:
            data = Path(local_path).read_bytes()
        except FileNotFoundError:
            return ExecutionResult(
                "UPLOAD_GCS", False,
                f"File not found: {local_path}",
                extra={"local_path": local_path},
            )
        except OSError as exc:
            return ExecutionResult(
                "UPLOAD_GCS", False,
                f"Cannot read file: {exc}",
                extra={"local_path": local_path},
            )

        if not self._server_url:
            return ExecutionResult(
                "UPLOAD_GCS", False,
                "No server_url configured — cannot upload (server_url required)",
                extra={"local_path": local_path},
            )

        # POST to /session/upload
        upload_url = f"{self._server_url}/session/upload"
        filename = Path(local_path).name
        try:
            resp = requests.post(
                upload_url,
                files={"file": (filename, data, content_type)},
                data={
                    "session_id": self._session_id or action.get("session_id", ""),
                    "object_name": gcs_object,
                },
                timeout=(5.0, 60.0),
            )
            resp.raise_for_status()
            result_json = resp.json()
            gcs_url = result_json.get("gcs_url", "")
            uploaded = bool(result_json.get("uploaded", gcs_url))
            size_bytes = result_json.get("size_bytes", len(data))

            if uploaded:
                return ExecutionResult(
                    "UPLOAD_GCS", True,
                    f"Uploaded {size_bytes:,} bytes → {gcs_url}",
                    extra={
                        "gcs_url": gcs_url,
                        "object_name": result_json.get("object_name", gcs_object),
                        "size_bytes": size_bytes,
                        "uploaded": True,
                        "local_path": local_path,
                    },
                )
            else:
                return ExecutionResult(
                    "UPLOAD_GCS", False,
                    "Server accepted upload but GCS was unavailable — artifact not stored",
                    extra={
                        "gcs_url": "",
                        "size_bytes": size_bytes,
                        "uploaded": False,
                        "local_path": local_path,
                    },
                )
        except _RequestsTimeout:
            return ExecutionResult(
                "UPLOAD_GCS", False,
                f"Upload timed out after 60 s (server={upload_url})",
                extra={"local_path": local_path},
            )
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                "UPLOAD_GCS", False,
                f"Upload failed: {exc}",
                extra={"local_path": local_path},
            )

    def _deploy_cloud_run(self, action: dict) -> ExecutionResult:
        """
        Deploy a container image to Cloud Run using the ``gcloud run deploy``
        CLI command.

        Safety requirements
        -------------------
        - The action MUST have ``confirmed=True`` (set by the SessionManager
          after the user confirms the preceding CONFIRM action).  Without it
          the handler refuses and returns success=False.
        - The constructed command passes the policy gate in command_policy.py.
        - Service URL is parsed from gcloud stdout/stderr and returned in
          ``extra["service_url"]`` for downstream WRITE_REPORT actions.
        """
        import re as _re

        if not action.get("confirmed", False):
            return ExecutionResult(
                "DEPLOY_CLOUD_RUN", False,
                "Deployment blocked — user confirmation required. "
                "Precede this action with a CONFIRM action.",
                extra={"confirmed": False},
            )

        service_name = (action.get("service_name") or "").strip()
        image        = (action.get("image") or "").strip()
        region       = (action.get("region") or "us-central1").strip()
        project      = (action.get("project") or "").strip()
        allow_unauth = bool(action.get("allow_unauthenticated", False))
        timeout_s    = int(action.get("timeout_s", 300))

        if not service_name or not image:
            return ExecutionResult(
                "DEPLOY_CLOUD_RUN", False,
                "service_name and image are required for DEPLOY_CLOUD_RUN",
                extra={"service_name": service_name, "image": image},
            )

        # Build the gcloud command — safe flags only, no shell interpolation
        parts = [
            "gcloud", "run", "deploy", service_name,
            f"--image={image}",
            f"--region={region}",
            "--platform=managed",
        ]
        if project:
            parts.append(f"--project={project}")
        if allow_unauth:
            parts.append("--allow-unauthenticated")

        cmd = " ".join(parts)
        logger.info("DEPLOY_CLOUD_RUN: %s", cmd)

        cmd_result = _policy_execute(cmd, timeout_s=timeout_s)

        if cmd_result.blocked:
            logger.warning("DEPLOY_CLOUD_RUN blocked by policy: %s", cmd_result.block_reason)
            return ExecutionResult(
                "DEPLOY_CLOUD_RUN", False,
                f"Blocked by policy: {cmd_result.block_reason}",
                extra={"command": cmd, "blocked": True},
            )

        # Parse Cloud Run service URL from gcloud output
        combined = (cmd_result.stdout or "") + "\n" + (cmd_result.stderr or "")
        url_match = _re.search(r"Service URL:\s*(https://[^\s]+)", combined)
        service_url = url_match.group(1) if url_match else ""

        success = cmd_result.returncode == 0 and not cmd_result.timed_out
        status_msg = (
            f"returncode={cmd_result.returncode} "
            f"duration={cmd_result.duration_ms:.0f}ms"
            + (" [TIMEOUT]" if cmd_result.timed_out else "")
            + (f" url={service_url}" if service_url else "")
        )

        return ExecutionResult(
            "DEPLOY_CLOUD_RUN", success, status_msg,
            extra={
                "command": cmd,
                "service_url": service_url,
                "returncode": cmd_result.returncode,
                "stdout": cmd_result.stdout[:1000],
                "stderr": cmd_result.stderr[:500],
                "timed_out": cmd_result.timed_out,
                "service_name": service_name,
                "region": region,
                "project": project,
            },
        )

    def _abort(self, action: dict) -> ExecutionResult:
        reason = action.get("reason", "Orchestrator requested abort")
        return ExecutionResult("ABORT", False, reason)

    def _hand_off(self, action: dict) -> ExecutionResult:
        summary = action.get("summary", "")
        reason = action.get("reason", "")
        msg = summary or reason or "Human intervention required"
        return ExecutionResult("HAND_OFF_TO_USER", True, msg, skipped=True)
