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
from typing import Any

import pyautogui
from redaction import redact_text

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
    """

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
        try:
            import pygetwindow as gw
            wins = gw.getWindowsWithTitle(title)
            if not wins:
                return ExecutionResult("FOCUS_WINDOW", False, f"No window found matching '{title}'")
            win = wins[0]
            # Use ctypes on Windows for reliable foreground activation.
            # pygetwindow's .activate() is often ignored by the OS when another
            # app has the lock; SetForegroundWindow + ShowWindow bypass that.
            try:
                import ctypes
                hwnd = win._hWnd          # Win32Window exposes _hWnd
                ctypes.windll.user32.ShowWindow(hwnd, 9)   # SW_RESTORE
                ctypes.windll.user32.SetForegroundWindow(hwnd)
            except Exception:
                win.activate()            # fallback for non-Windows / missing _hWnd
            time.sleep(0.6)               # give window time to receive keyboard focus
            return ExecutionResult("FOCUS_WINDOW", True, f"Focused '{win.title}'")
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult("FOCUS_WINDOW", False, f"Could not focus window: {exc}")

    def _click(self, action: dict) -> ExecutionResult:
        x, y = action.get("x", 0), action.get("y", 0)
        pyautogui.click(x, y)
        # Extra settle time: taskbar/app-launcher clicks need the OS to bring
        # the new window to the foreground before keyboard events can land in it.
        time.sleep(0.25)
        return ExecutionResult("CLICK", True, f"Clicked ({x}, {y})")

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
        desc = action.get("description", "(no description)")
        # VERIFY is a signal to re-capture; no physical action needed
        return ExecutionResult(
            "VERIFY", True,
            f"Verify queued: {desc}",
            skipped=True,
        )

    def _abort(self, action: dict) -> ExecutionResult:
        reason = action.get("reason", "Orchestrator requested abort")
        return ExecutionResult("ABORT", False, reason)

    def _hand_off(self, action: dict) -> ExecutionResult:
        summary = action.get("summary", "")
        reason = action.get("reason", "")
        msg = summary or reason or "Human intervention required"
        return ExecutionResult("HAND_OFF_TO_USER", True, msg, skipped=True)
