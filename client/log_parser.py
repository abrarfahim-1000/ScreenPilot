"""
log_parser.py — Test log file parser for UI Navigator Desktop Client.

Parses pytest / unittest / make test output from a captured log file
and produces a structured TestSummary + DeploymentReport.

Deliberately reads the log file directly — no screenshot OCR — for reliable
metric extraction independent of terminal font rendering or screen resolution.

Usage
-----
    from log_parser import parse_test_log, generate_deployment_report

    summary = parse_test_log("/tmp/test_out.log")
    if summary.success:
        print(summary.summary_line)   # "42 passed in 3.14s"
    else:
        print(f"Tests failed: {summary.failed} failures, {summary.errors} errors")

    report = generate_deployment_report(
        summary=summary,
        session_id="abc123",
        task_goal="Run tests and deploy to Cloud Run",
    )
    print(report.report_text)
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TestSummary:
    """Structured result extracted from a test runner log file."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    warnings: int = 0
    duration_s: Optional[float] = None
    exit_code: Optional[int] = None       # from EXEC_COMMAND returncode, if provided
    log_path: str = ""
    raw_snippet: str = ""                 # last 20 lines for quick reference
    parser_format: str = "unknown"        # "pytest" | "unittest" | "unknown"
    parse_errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True only when at least one test ran and no failures or errors found."""
        return self.total > 0 and self.failed == 0 and self.errors == 0

    @property
    def summary_line(self) -> str:
        """Compact one-liner suitable for logs and display."""
        parts: list[str] = []
        if self.passed:
            parts.append(f"{self.passed} passed")
        if self.failed:
            parts.append(f"{self.failed} failed")
        if self.errors:
            parts.append(f"{self.errors} error{'s' if self.errors != 1 else ''}")
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        if self.warnings:
            parts.append(f"{self.warnings} warning{'s' if self.warnings != 1 else ''}")
        if not parts:
            return "no tests found"
        line = ", ".join(parts)
        if self.duration_s is not None:
            line += f" in {self.duration_s:.2f}s"
        return line

    @property
    def status_emoji(self) -> str:
        if self.total == 0:
            return "⚠"
        return "✓" if self.success else "✗"


@dataclass
class DeploymentReport:
    """
    End-to-end report generated after the test + deploy workflow.

    Populated incrementally: test_summary is filled after tests run;
    gcs_log_url and cloud_run_url are filled by later steps (Days 9-11).
    """
    generated_at: str = ""              # ISO 8601 UTC timestamp
    session_id: str = ""
    task_goal: str = ""
    test_summary: Optional[TestSummary] = None
    gcs_log_url: str = ""               # populated after GCS upload (Days 9-11)
    cloud_run_url: str = ""             # populated after Cloud Run deploy (Days 9-11)
    report_path: str = ""               # path where this report was / will be written
    report_text: str = ""               # formatted report text ready for display/clipboard


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

# Patterns that are always blocked regardless of allow-list hits.
_BLOCKED_PATH_RES: list[tuple[str, re.Pattern]] = [
    ("path_traversal",   re.compile(r"\.\.(?:[/\\]|$)")),
    ("unix_sys",         re.compile(r"^/(?:etc|sys|proc|root|boot|dev)/", re.I)),
    ("windows_system",   re.compile(r"^[A-Za-z]:[/\\](?:Windows|Program Files)", re.I)),
]


def _is_safe_read_path(path: str) -> tuple[bool, str]:
    """
    Return ``(True, "")`` if *path* is safe to read, or ``(False, reason)`` if not.

    Allowed roots
    -------------
    - Unix ``/tmp/`` (always allowed — standard demo log location)
    - OS temp directory (``tempfile.gettempdir()``, handles Windows ``%TEMP%``)
    - Current working directory (project repository)

    Always blocked
    --------------
    - Path traversal (``..``)
    - Unix system dirs: ``/etc/``, ``/sys/``, ``/proc/``, ``/root/``, ``/boot/``, ``/dev/``
    - Windows system dirs: ``C:\\Windows\\``, ``C:\\Program Files``
    """
    # ── Path traversal: block immediately, before resolution ────────────────
    if ".." in path:
        return False, "Path traversal not allowed ('..'' detected)"

    # ── Blocked pattern check on raw path ───────────────────────────────────
    for rule_name, pattern in _BLOCKED_PATH_RES:
        if pattern.search(path):
            return False, f"Path blocked by rule '{rule_name}'"

    # ── /tmp/ is always safe (Unix demo convention) ──────────────────────────
    normalized = path.replace("\\", "/")
    if normalized.startswith("/tmp/"):
        return True, ""

    # ── Resolve and compare to safe roots ────────────────────────────────────
    try:
        resolved = str(Path(path).resolve())
    except (ValueError, OSError) as exc:
        return False, f"Cannot resolve path: {exc}"

    # Re-check blocked patterns on resolved path (catches symlink escapes)
    for rule_name, pattern in _BLOCKED_PATH_RES:
        if pattern.search(resolved):
            return False, f"Resolved path blocked by rule '{rule_name}'"

    tmpdir = str(Path(tempfile.gettempdir()).resolve())
    cwd = str(Path.cwd().resolve())

    if resolved.startswith(tmpdir) or resolved.startswith(cwd):
        return True, ""

    return False, (
        f"Path '{resolved}' is outside safe roots "
        f"(tmp: {tmpdir}, cwd: {cwd})"
    )


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

# pytest final summary line:
#   "=== 5 passed in 2.34s ==="
#   "=== 1 failed, 4 passed in 3.45s ==="
#   "=== 2 errors, 3 warnings in 1.00s ==="
_PYTEST_SUMMARY_RE = re.compile(
    r"={3,}\s*(.+?)\s+in\s+([\d.]+)s\s*={3,}",
    re.IGNORECASE,
)
_PYTEST_PASSED_RE   = re.compile(r"(\d+)\s+passed",           re.IGNORECASE)
_PYTEST_FAILED_RE   = re.compile(r"(\d+)\s+failed",           re.IGNORECASE)
_PYTEST_ERROR_RE    = re.compile(r"(\d+)\s+error(?:s)?",      re.IGNORECASE)
_PYTEST_SKIPPED_RE  = re.compile(r"(\d+)\s+skip(?:ped)?",     re.IGNORECASE)
_PYTEST_WARN_RE     = re.compile(r"(\d+)\s+warning(?:s)?",    re.IGNORECASE)

# unittest:
#   "Ran 6 tests in 2.345s"  then  "OK"  or  "FAILED (failures=2, errors=1)"
_UNITTEST_RAN_RE        = re.compile(r"Ran\s+(\d+)\s+tests?\s+in\s+([\d.]+)s", re.IGNORECASE)
_UNITTEST_OK_RE         = re.compile(r"^\s*OK\s*(?:\(.*\))?\s*$",              re.MULTILINE)
_UNITTEST_FAILED_RE     = re.compile(r"^\s*FAILED\s*\(",                        re.MULTILINE)
_UNITTEST_FAIL_COUNT_RE = re.compile(r"failures=(\d+)",                         re.IGNORECASE)
_UNITTEST_ERR_COUNT_RE  = re.compile(r"errors=(\d+)",                           re.IGNORECASE)


def _parse_pytest(content: str) -> Optional[TestSummary]:
    """
    Try to parse *content* as pytest output.

    Returns a :class:`TestSummary` if a pytest summary line was found,
    else ``None``.  When multiple summary lines exist (e.g., re-runs), the
    **last** one is used because that reflects the final result.
    """
    matches = list(_PYTEST_SUMMARY_RE.finditer(content))
    if not matches:
        return None

    m = matches[-1]   # last summary line = final result
    summary_text = m.group(1)
    duration_s = float(m.group(2))

    def _int(regex: re.Pattern, text: str) -> int:
        hit = regex.search(text)
        return int(hit.group(1)) if hit else 0

    passed   = _int(_PYTEST_PASSED_RE,  summary_text)
    failed   = _int(_PYTEST_FAILED_RE,  summary_text)
    errors   = _int(_PYTEST_ERROR_RE,   summary_text)
    skipped  = _int(_PYTEST_SKIPPED_RE, summary_text)
    warnings = _int(_PYTEST_WARN_RE,    summary_text)
    total    = passed + failed + errors

    return TestSummary(
        total=total,
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        warnings=warnings,
        duration_s=duration_s,
        parser_format="pytest",
    )


def _parse_unittest(content: str) -> Optional[TestSummary]:
    """
    Try to parse *content* as unittest output.

    Returns a :class:`TestSummary` if a ``Ran N tests in Xs`` line was found,
    else ``None``.
    """
    m = _UNITTEST_RAN_RE.search(content)
    if not m:
        return None

    total      = int(m.group(1))
    duration_s = float(m.group(2))

    failed = errors = 0
    if _UNITTEST_FAILED_RE.search(content):
        fm = _UNITTEST_FAIL_COUNT_RE.search(content)
        em = _UNITTEST_ERR_COUNT_RE.search(content)
        failed = int(fm.group(1)) if fm else 0
        errors = int(em.group(1)) if em else 0

    passed = max(0, total - failed - errors)

    return TestSummary(
        total=total,
        passed=passed,
        failed=failed,
        errors=errors,
        duration_s=duration_s,
        parser_format="unittest",
    )


def _extract_raw_snippet(content: str, tail_lines: int = 20) -> str:
    """Return the last *tail_lines* non-empty lines from *content*."""
    lines = [ln for ln in content.splitlines() if ln.strip()]
    snippet = lines[-tail_lines:] if len(lines) > tail_lines else lines
    return "\n".join(snippet)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MAX_READ_BYTES: int = 65_536   # 64 KB safety cap on log file reads


def parse_test_log(
    path: str,
    content: Optional[str] = None,
    exit_code: Optional[int] = None,
) -> TestSummary:
    """
    Parse a test runner log file and return a structured :class:`TestSummary`.

    Parameters
    ----------
    path:
        Absolute or relative path to the log file.  Used as ``log_path`` in
        the returned summary.  If *content* is provided, no disk I/O occurs.
    content:
        Pre-loaded text content (e.g. from an EXEC_COMMAND stdout).  When
        provided, the file at *path* is not read.
    exit_code:
        Optional exit code from the command that produced the log (e.g. from
        an EXEC_COMMAND ExecutionResult), stored verbatim in the summary.

    Returns
    -------
    TestSummary — always returns an object; ``parse_errors`` captures any
    read or parse failures so callers never need to catch exceptions.

    Notes
    -----
    - Parsers tried in order: pytest → unittest.
    - If the log is larger than :data:`MAX_READ_BYTES`, the **tail** is kept
      (test summaries always appear at the end of the output).
    - Path safety is enforced unless *content* is supplied directly.
    """
    parse_errs: list[str] = []

    # ── Read file if content not supplied ────────────────────────────────────
    if content is None:
        safe, reason = _is_safe_read_path(path)
        if not safe:
            logger.warning("parse_test_log: path safety check failed: %s", reason)
            return TestSummary(
                log_path=path,
                parse_errors=[f"Path safety check failed: {reason}"],
            )
        try:
            raw = Path(path).read_bytes()
            if len(raw) > MAX_READ_BYTES:
                logger.warning(
                    "parse_test_log: %s is %d bytes → truncating to last %d",
                    path, len(raw), MAX_READ_BYTES,
                )
                parse_errs.append(
                    f"Log truncated from {len(raw)} to {MAX_READ_BYTES} bytes"
                )
                raw = raw[-MAX_READ_BYTES:]   # keep the tail — summary is at end
            content = raw.decode("utf-8", errors="replace")
        except FileNotFoundError:
            return TestSummary(
                log_path=path,
                parse_errors=[f"Log file not found: {path}"],
            )
        except OSError as exc:
            return TestSummary(
                log_path=path,
                parse_errors=[f"Failed to read log file: {exc}"],
            )

    raw_snippet = _extract_raw_snippet(content)

    # ── Try parsers in priority order ────────────────────────────────────────
    summary = _parse_pytest(content)
    if summary is None:
        summary = _parse_unittest(content)
    if summary is None:
        parse_errs.append(
            "No recognizable test summary found in log (tried pytest, unittest)"
        )
        logger.warning("parse_test_log: no recognizable summary in %s", path)
        summary = TestSummary(parser_format="unknown")

    summary.log_path      = path
    summary.raw_snippet   = raw_snippet
    summary.exit_code     = exit_code
    summary.parse_errors.extend(parse_errs)
    return summary


def generate_deployment_report(
    summary: TestSummary,
    session_id: str = "",
    task_goal: str = "",
    gcs_log_url: str = "",
    cloud_run_url: str = "",
    report_path: str = "",
) -> DeploymentReport:
    """
    Build a :class:`DeploymentReport` from a :class:`TestSummary` and
    optional cloud artifact URLs.

    Parameters
    ----------
    summary:
        Parsed test results from :func:`parse_test_log`.
    session_id:
        Agent session identifier (for traceability in the report header).
    task_goal:
        The original high-level instruction given to the agent.
    gcs_log_url:
        GCS URL of the uploaded log file (filled in Days 9-11 of the plan).
    cloud_run_url:
        Cloud Run service URL after deploy (filled in Days 9-11 of the plan).
    report_path:
        Where to write the report file.  If empty, an auto-generated path
        under the OS temp directory is used.

    Returns
    -------
    DeploymentReport — ``report_text`` contains the full formatted text;
    ``report_path`` indicates where it should be written.
    """
    now = datetime.now(timezone.utc).isoformat()
    sep = "=" * 60

    if summary.total == 0:
        status_label = "UNKNOWN"
    elif summary.success:
        status_label = "PASS"
    else:
        status_label = "FAIL"

    lines = [
        sep,
        "DEPLOYMENT REPORT",
        f"Generated : {now}",
        f"Session   : {session_id or '(none)'}",
        f"Goal      : {task_goal or '(none)'}",
        sep,
        "",
        "TEST RESULTS",
        f"  Status   : {status_label}  {summary.status_emoji}",
        f"  Summary  : {summary.summary_line}",
        f"  Format   : {summary.parser_format}",
        f"  Log file : {summary.log_path or '(none)'}",
    ]

    if summary.exit_code is not None:
        lines.append(f"  Exit code: {summary.exit_code}")

    if summary.parse_errors:
        lines += ["", "  ⚠ Parser notes:"]
        for err in summary.parse_errors:
            lines.append(f"    - {err}")

    lines += [
        "",
        "ARTIFACTS",
        f"  GCS log  : {gcs_log_url or '(pending — upload in Days 9-11)'}",
        f"  Cloud Run: {cloud_run_url or '(pending — deploy in Days 9-11)'}",
    ]

    if summary.raw_snippet:
        lines += ["", "LOG TAIL (last 20 lines)", "---", summary.raw_snippet]

    lines += ["", sep]
    report_text = "\n".join(lines)

    if not report_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = str(Path(tempfile.gettempdir()) / f"deploy_report_{ts}.txt")

    return DeploymentReport(
        generated_at=now,
        session_id=session_id,
        task_goal=task_goal,
        test_summary=summary,
        gcs_log_url=gcs_log_url,
        cloud_run_url=cloud_run_url,
        report_path=report_path,
        report_text=report_text,
    )
