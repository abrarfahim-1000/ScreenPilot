"""
window_focus.py — Reliable cross-platform window focus helper.

Features
--------
- Exact case-insensitive full-title match
- Partial (substring) case-insensitive match
- Fuzzy similarity match via rapidfuzz (configurable threshold, 0–100)
- Windows: Win32 SetForegroundWindow + SW_RESTORE via ctypes for reliable
  foreground activation (pygetwindow's .activate() is frequently ignored by
  the OS while another application holds the input lock)
- Returns a structured FocusResult for audit logging

Match strategy (in order — first successful match wins)
---------------------------------------------------------
  1. Exact case-insensitive title match.
  2. Partial (substring) case-insensitive match.
  3. Fuzzy similarity match (rapidfuzz WRatio ≥ fuzzy_threshold).

Usage
-----
    from window_focus import focus_window

    result = focus_window("Terminal")
    if result.success:
        print(f"Focused via {result.match_type}: {result.matched_title}")
    else:
        print(f"Could not focus: {result.message}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Settle time in seconds after activating a window so the OS fully brings it
# to the foreground before keyboard events land in it.
_FOCUS_SETTLE_S: float = 0.6

# Default rapidfuzz similarity score threshold (0–100).
DEFAULT_FUZZY_THRESHOLD: int = 70


@dataclass
class FocusResult:
    """Structured outcome of a focus_window() call."""
    success: bool
    matched_title: str = ""
    match_type: str = ""   # "exact" | "partial" | "fuzzy" | ""
    message: str = ""


def focus_window(
    title_contains: str,
    fuzzy_threshold: int = DEFAULT_FUZZY_THRESHOLD,
) -> FocusResult:
    """
    Bring the window whose title best matches *title_contains* to the
    foreground.

    Parameters
    ----------
    title_contains:
        String to look for in window titles.
    fuzzy_threshold:
        Minimum rapidfuzz WRatio similarity score (0–100) for a fuzzy match
        to be accepted.  Lower values widen the match; 70 is a safe default.

    Returns
    -------
    FocusResult with success=True/False and audit information.
    """
    query = title_contains.strip()
    if not query:
        return FocusResult(success=False, message="Empty title query")

    try:
        import pygetwindow as gw
    except ImportError:
        return FocusResult(
            success=False,
            message="pygetwindow is not installed — cannot enumerate windows",
        )

    try:
        all_windows = gw.getAllWindows()
    except Exception as exc:  # noqa: BLE001
        return FocusResult(success=False, message=f"Cannot enumerate windows: {exc}")

    # Keep only windows that have a non-blank title.
    candidates = [w for w in all_windows if w.title and w.title.strip()]
    if not candidates:
        return FocusResult(
            success=False,
            message=f"No open windows found while searching for '{title_contains}'",
        )

    query_lower = query.lower()

    # ── Pass 1: exact case-insensitive match ─────────────────────────────────
    for win in candidates:
        if win.title.strip().lower() == query_lower:
            return _activate(win, "exact")

    # ── Pass 2: partial (substring) case-insensitive match ───────────────────
    for win in candidates:
        if query_lower in win.title.lower():
            return _activate(win, "partial")

    # ── Pass 3: fuzzy similarity match via rapidfuzz ─────────────────────────
    try:
        from rapidfuzz import fuzz, process as fz_process  # type: ignore

        titles_lower = [w.title.lower() for w in candidates]
        match = fz_process.extractOne(
            query_lower,
            titles_lower,
            scorer=fuzz.WRatio,
            score_cutoff=fuzzy_threshold,
        )
        if match is not None:
            _matched_str, _score, idx = match
            logger.debug(
                "Fuzzy window match: '%s' → '%s' (score=%d)",
                title_contains,
                candidates[idx].title,
                _score,
            )
            return _activate(candidates[idx], "fuzzy")
    except ImportError:
        logger.debug("rapidfuzz not available — fuzzy window matching skipped")

    return FocusResult(
        success=False,
        message=(
            f"No window found matching '{title_contains}' "
            f"(tried exact / partial / fuzzy threshold={fuzzy_threshold})"
        ),
    )


def _activate(win, match_type: str) -> FocusResult:
    """
    Activate *win* using Win32 if available, otherwise pygetwindow.activate().

    Returns a FocusResult.
    """
    title = win.title
    try:
        if not _try_win32_activate(win):
            win.activate()
        time.sleep(_FOCUS_SETTLE_S)
        msg = f"Focused via {match_type}: '{title}'"
        logger.info(msg)
        return FocusResult(
            success=True,
            matched_title=title,
            match_type=match_type,
            message=msg,
        )
    except Exception as exc:  # noqa: BLE001
        return FocusResult(
            success=False,
            matched_title=title,
            match_type=match_type,
            message=f"Activation failed for '{title}': {exc}",
        )


def _try_win32_activate(win) -> bool:
    """
    Use Win32 API (ctypes) for reliable foreground activation on Windows.

    Returns True if the Win32 path succeeded; False tells the caller to fall
    back to pygetwindow's .activate().
    """
    try:
        import ctypes
        hwnd = win._hWnd                               # Win32Window exposes _hWnd
        ctypes.windll.user32.ShowWindow(hwnd, 9)       # SW_RESTORE (un-minimise)
        ctypes.windll.user32.SetForegroundWindow(hwnd)
        return True
    except Exception:  # noqa: BLE001
        return False
