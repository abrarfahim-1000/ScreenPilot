"""
tests/test_log_parser.py — Tests for client/log_parser.py and the new
executor READ_FILE / PARSE_LOG / WRITE_REPORT / VERIFY(read_file) actions.

Covers
------
  1.  TestSummary dataclass properties (success, summary_line, status_emoji)
  2.  _parse_pytest() — all output variants
  3.  _parse_unittest() — OK and FAILED variants
  4.  parse_test_log() — from content string (no disk I/O)
  5.  parse_test_log() — from file (disk I/O, safety checks, truncation)
  6.  generate_deployment_report() — all fields + edge cases
  7.  _is_safe_read_path() — allowed and blocked patterns
  8.  ActionExecutor.READ_FILE  — success, not-found, blocked, truncation
  9.  ActionExecutor.PARSE_LOG  — success, no summary, content injection
  10. ActionExecutor.WRITE_REPORT — write + clipboard, write-fail, no log_path
  11. ActionExecutor.VERIFY method="read_file" — must_see pass/fail, fallback

Run with:  pytest tests/test_log_parser.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

# ── import path setup ─────────────────────────────────────────────────────────
CLIENT_DIR = os.path.join(os.path.dirname(__file__), "..", "client")
sys.path.insert(0, CLIENT_DIR)

from log_parser import (
    DeploymentReport,
    TestSummary,
    MAX_READ_BYTES,
    _extract_raw_snippet,
    _is_safe_read_path,
    _parse_pytest,
    _parse_unittest,
    generate_deployment_report,
    parse_test_log,
)
from executor import ActionExecutor, ExecutionResult


# ─────────────────────────────────────────────────────────────────────────────
# Shared pytest/unittest log fixtures
# ─────────────────────────────────────────────────────────────────────────────

PYTEST_PASS_SIMPLE = """\
================================ test session starts =================================
platform linux -- Python 3.11.0, pytest-7.4.0
collected 5 items

tests/test_capture.py .....                                                     [100%]

================================== 5 passed in 2.34s ==================================
"""

PYTEST_MIXED = """\
================================ test session starts =================================
collected 10 items

tests/test_foo.py ....F.E..                                                     [100%]

================================== FAILURES ==========================================
...
============================== short test summary info ================================
FAILED tests/test_foo.py::test_alpha - AssertionError
========= 7 passed, 1 failed, 1 error in 4.56s =========
"""

PYTEST_SKIPPED_WARNINGS = """\
=============================== test session starts ====================================
collected 15 items

tests/test_bar.py ....ssss.....                                                  [100%]

======================== 11 passed, 4 skipped, 2 warnings in 1.23s ========================
"""

PYTEST_ALL_FAILED = """\
================================ test session starts =================================
collected 3 items

tests/test_bad.py FFF                                                            [100%]

============================= 3 failed in 0.87s ==============================
"""

PYTEST_ERRORS_ONLY = """\
================================ test session starts =================================
collected 2 items

tests/test_broken.py EE                                                          [100%]

========================= 2 errors in 0.12s ==========================
"""

PYTEST_MULTIPLE_SUMMARIES = """\
======= 3 failed in 1.10s =======

-- rerunning failures --

======= 3 passed in 1.50s =======
"""

UNITTEST_OK = """\
......
----------------------------------------------------------------------
Ran 6 tests in 2.345s

OK
"""

UNITTEST_FAILED = """\
...F..
======================================================================
FAIL: test_method (tests.test_foo.MyTests)
----------------------------------------------------------------------
AssertionError: False is not true
----------------------------------------------------------------------
Ran 6 tests in 1.200s

FAILED (failures=1)
"""

UNITTEST_MULTI_FAIL = """\
FEF
======================================================================
Ran 3 tests in 0.500s

FAILED (failures=2, errors=1)
"""

EMPTY_LOG = ""

GARBAGE_LOG = "This is some random output that isn't a test runner.\nNo tests here."


# ─────────────────────────────────────────────────────────────────────────────
# 1 · TestSummary properties
# ─────────────────────────────────────────────────────────────────────────────

class TestTestSummaryProperties:

    def test_success_true_when_passed_only(self):
        s = TestSummary(total=5, passed=5)
        assert s.success is True

    def test_success_false_when_failed(self):
        s = TestSummary(total=5, passed=4, failed=1)
        assert s.success is False

    def test_success_false_when_errors(self):
        s = TestSummary(total=5, passed=4, errors=1)
        assert s.success is False

    def test_success_false_when_zero_total(self):
        """No tests found → success must be False (not vacuously true)."""
        s = TestSummary(total=0, passed=0)
        assert s.success is False

    def test_summary_line_passed_only(self):
        s = TestSummary(total=5, passed=5, duration_s=2.34)
        assert s.summary_line == "5 passed in 2.34s"

    def test_summary_line_mixed(self):
        s = TestSummary(total=10, passed=7, failed=1, errors=1, skipped=1)
        line = s.summary_line
        assert "7 passed" in line
        assert "1 failed" in line
        assert "1 error" in line
        assert "1 skipped" in line

    def test_summary_line_no_duration(self):
        s = TestSummary(total=3, passed=3)
        assert "in" not in s.summary_line

    def test_summary_line_no_tests(self):
        s = TestSummary()
        assert s.summary_line == "no tests found"

    def test_status_emoji_pass(self):
        s = TestSummary(total=3, passed=3)
        assert s.status_emoji == "✓"

    def test_status_emoji_fail(self):
        s = TestSummary(total=3, passed=2, failed=1)
        assert s.status_emoji == "✗"

    def test_status_emoji_unknown(self):
        s = TestSummary(total=0)
        assert s.status_emoji == "⚠"

    def test_warnings_singular_plural(self):
        s1 = TestSummary(total=1, passed=1, warnings=1)
        s2 = TestSummary(total=1, passed=1, warnings=3)
        assert "1 warning" in s1.summary_line
        assert "3 warnings" in s2.summary_line

    def test_errors_singular_plural(self):
        s1 = TestSummary(total=1, errors=1)
        s2 = TestSummary(total=2, errors=2)
        assert "1 error" in s1.summary_line
        assert "2 errors" in s2.summary_line


# ─────────────────────────────────────────────────────────────────────────────
# 2 · _parse_pytest — output variants
# ─────────────────────────────────────────────────────────────────────────────

class TestParsePytest:

    def test_simple_pass(self):
        s = _parse_pytest(PYTEST_PASS_SIMPLE)
        assert s is not None
        assert s.passed == 5
        assert s.failed == 0
        assert s.errors == 0
        assert s.total == 5
        assert abs(s.duration_s - 2.34) < 0.01
        assert s.parser_format == "pytest"

    def test_mixed_failures_errors(self):
        s = _parse_pytest(PYTEST_MIXED)
        assert s is not None
        assert s.passed == 7
        assert s.failed == 1
        assert s.errors == 1
        assert s.total == 9

    def test_skipped_and_warnings(self):
        s = _parse_pytest(PYTEST_SKIPPED_WARNINGS)
        assert s is not None
        assert s.passed == 11
        assert s.skipped == 4
        assert s.warnings == 2

    def test_all_failed(self):
        s = _parse_pytest(PYTEST_ALL_FAILED)
        assert s is not None
        assert s.failed == 3
        assert s.passed == 0
        assert s.total == 3

    def test_errors_only(self):
        s = _parse_pytest(PYTEST_ERRORS_ONLY)
        assert s is not None
        assert s.errors == 2
        assert s.total == 2

    def test_multiple_summaries_uses_last(self):
        """Re-run scenario: final summary must win."""
        s = _parse_pytest(PYTEST_MULTIPLE_SUMMARIES)
        assert s is not None
        assert s.passed == 3
        assert s.failed == 0

    def test_returns_none_for_non_pytest(self):
        assert _parse_pytest(UNITTEST_OK) is None

    def test_returns_none_for_empty(self):
        assert _parse_pytest(EMPTY_LOG) is None

    def test_returns_none_for_garbage(self):
        assert _parse_pytest(GARBAGE_LOG) is None


# ─────────────────────────────────────────────────────────────────────────────
# 3 · _parse_unittest — OK and FAILED variants
# ─────────────────────────────────────────────────────────────────────────────

class TestParseUnittest:

    def test_ok(self):
        s = _parse_unittest(UNITTEST_OK)
        assert s is not None
        assert s.total == 6
        assert s.passed == 6
        assert s.failed == 0
        assert s.errors == 0
        assert abs(s.duration_s - 2.345) < 0.001
        assert s.parser_format == "unittest"

    def test_failed_one(self):
        s = _parse_unittest(UNITTEST_FAILED)
        assert s is not None
        assert s.total == 6
        assert s.failed == 1
        assert s.errors == 0
        assert s.passed == 5

    def test_failed_multi(self):
        s = _parse_unittest(UNITTEST_MULTI_FAIL)
        assert s is not None
        assert s.total == 3
        assert s.failed == 2
        assert s.errors == 1
        assert s.passed == 0

    def test_returns_none_for_pytest(self):
        assert _parse_unittest(PYTEST_PASS_SIMPLE) is None

    def test_returns_none_for_empty(self):
        assert _parse_unittest(EMPTY_LOG) is None

    def test_returns_none_for_garbage(self):
        assert _parse_unittest(GARBAGE_LOG) is None


# ─────────────────────────────────────────────────────────────────────────────
# 4 · parse_test_log() — content injection (no disk I/O)
# ─────────────────────────────────────────────────────────────────────────────

class TestParseTestLogFromContent:
    """All these tests supply content= so no file system is touched."""

    def test_pytest_pass_summary(self):
        s = parse_test_log("/tmp/fake.log", content=PYTEST_PASS_SIMPLE)
        assert s.passed == 5
        assert s.log_path == "/tmp/fake.log"
        assert s.parser_format == "pytest"
        assert s.parse_errors == []

    def test_pytest_failure_sets_success_false(self):
        s = parse_test_log("/tmp/fake.log", content=PYTEST_MIXED)
        assert s.success is False

    def test_unittest_ok_content(self):
        s = parse_test_log("/tmp/fake.log", content=UNITTEST_OK)
        assert s.total == 6
        assert s.success is True
        assert s.parser_format == "unittest"

    def test_empty_content_returns_unknown(self):
        s = parse_test_log("/tmp/fake.log", content=EMPTY_LOG)
        assert s.parser_format == "unknown"
        assert len(s.parse_errors) > 0

    def test_garbage_content_parse_error_recorded(self):
        s = parse_test_log("/tmp/fake.log", content=GARBAGE_LOG)
        assert s.parser_format == "unknown"
        assert any("No recognizable" in e for e in s.parse_errors)

    def test_exit_code_stored(self):
        s = parse_test_log("/tmp/fake.log", content=PYTEST_PASS_SIMPLE, exit_code=0)
        assert s.exit_code == 0

    def test_exit_code_nonzero_stored(self):
        s = parse_test_log("/tmp/fake.log", content=PYTEST_ALL_FAILED, exit_code=1)
        assert s.exit_code == 1

    def test_exit_code_none_when_not_provided(self):
        s = parse_test_log("/tmp/fake.log", content=PYTEST_PASS_SIMPLE)
        assert s.exit_code is None

    def test_raw_snippet_populated(self):
        s = parse_test_log("/tmp/fake.log", content=PYTEST_PASS_SIMPLE)
        assert len(s.raw_snippet) > 0

    def test_log_path_attribute(self):
        s = parse_test_log("/some/path/test.log", content=PYTEST_PASS_SIMPLE)
        assert s.log_path == "/some/path/test.log"

    def test_pytest_wins_over_unittest_when_both_present(self):
        """When log contains both formats (unlikely but possible), pytest is parsed first."""
        combined = PYTEST_PASS_SIMPLE + "\n" + UNITTEST_OK
        s = parse_test_log("/tmp/fake.log", content=combined)
        assert s.parser_format == "pytest"

    def test_unicode_content_handled(self):
        content = PYTEST_PASS_SIMPLE + "\n日本語テスト output here\n"
        s = parse_test_log("/tmp/fake.log", content=content)
        assert s.total == 5  # parser still works


# ─────────────────────────────────────────────────────────────────────────────
# 5 · parse_test_log() — file I/O paths
# ─────────────────────────────────────────────────────────────────────────────

class TestParseTestLogFile:

    def test_reads_real_temp_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False, encoding="utf-8"
        ) as f:
            f.write(PYTEST_PASS_SIMPLE)
            fpath = f.name
        try:
            s = parse_test_log(fpath)
            assert s.passed == 5
            assert s.parse_errors == []
        finally:
            os.unlink(fpath)

    def test_file_not_found_returns_parse_error(self):
        s = parse_test_log("/tmp/_no_such_file_xyz_test.log")
        assert s.total == 0
        assert any("not found" in e.lower() for e in s.parse_errors)

    def test_truncation_when_over_max_bytes(self):
        # Build a log that exceeds MAX_READ_BYTES (64 KB)
        padding = "x" * (MAX_READ_BYTES + 1000)
        content = padding + "\n" + PYTEST_PASS_SIMPLE
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            fpath = f.name
        try:
            s = parse_test_log(fpath)
            # Truncation warning MUST be recorded
            assert any("truncated" in e.lower() for e in s.parse_errors)
            # Parser should still find the summary (it's at the tail)
            assert s.passed == 5
        finally:
            os.unlink(fpath)

    def test_path_traversal_blocked(self):
        s = parse_test_log("/tmp/../etc/passwd")
        assert any("traversal" in e.lower() or "safety" in e.lower() for e in s.parse_errors)

    def test_unix_sys_path_blocked(self):
        s = parse_test_log("/etc/shadow")
        assert any("safety" in e.lower() for e in s.parse_errors)

    def test_safe_cwd_path_allowed(self):
        """A file inside the project directory (cwd) must be readable."""
        tmp = Path(tempfile.mktemp(dir=str(Path.cwd()), suffix=".log"))
        tmp.write_text(UNITTEST_OK, encoding="utf-8")
        try:
            s = parse_test_log(str(tmp))
            assert s.total == 6
        finally:
            tmp.unlink()

    def test_safe_tmpdir_path_allowed(self):
        tmpdir = Path(tempfile.gettempdir())
        tmp = tmpdir / "_test_safe_parse.log"
        tmp.write_text(PYTEST_PASS_SIMPLE, encoding="utf-8")
        try:
            s = parse_test_log(str(tmp))
            assert s.passed == 5
        finally:
            tmp.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# 6 · _is_safe_read_path()
# ─────────────────────────────────────────────────────────────────────────────

class TestIsafeReadPath:

    def test_tmp_slash_always_allowed(self):
        safe, reason = _is_safe_read_path("/tmp/test_out.log")
        assert safe is True, reason

    def test_os_temp_dir_allowed(self):
        tmpdir = tempfile.gettempdir()
        safe, reason = _is_safe_read_path(os.path.join(tmpdir, "output.log"))
        assert safe is True, reason

    def test_cwd_path_allowed(self):
        path = str(Path.cwd() / "tests" / "test_capture.py")
        safe, reason = _is_safe_read_path(path)
        assert safe is True, reason

    def test_path_traversal_blocked(self):
        safe, reason = _is_safe_read_path("/tmp/../etc/passwd")
        assert safe is False
        assert "traversal" in reason.lower()

    def test_unix_etc_blocked(self):
        safe, reason = _is_safe_read_path("/etc/passwd")
        assert safe is False

    def test_unix_proc_blocked(self):
        safe, reason = _is_safe_read_path("/proc/1/environ")
        assert safe is False

    def test_unix_sys_blocked(self):
        safe, reason = _is_safe_read_path("/sys/kernel/debug/tracing")
        assert safe is False

    def test_arbitrary_outside_path_blocked(self):
        safe, reason = _is_safe_read_path("/home/otheruser/secrets.txt")
        assert safe is False

    @pytest.mark.parametrize("path", [
        r"C:\Windows\System32\config\SAM",
        r"C:\Windows\win.ini",
    ])
    def test_windows_system_path_blocked(self, path):
        safe, reason = _is_safe_read_path(path)
        assert safe is False

    def test_empty_path_returns_safe_or_error_not_crash(self):
        """Empty path should not raise — either blocked or safe."""
        safe, reason = _is_safe_read_path("")
        # An empty path resolves to cwd — that IS safe, so we just check no crash.
        assert isinstance(safe, bool)


# ─────────────────────────────────────────────────────────────────────────────
# 7 · generate_deployment_report()
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateDeploymentReport:

    def _summary_pass(self) -> TestSummary:
        return TestSummary(total=5, passed=5, duration_s=2.34, log_path="/tmp/test.log",
                           parser_format="pytest")

    def _summary_fail(self) -> TestSummary:
        return TestSummary(total=5, passed=3, failed=2, duration_s=3.0,
                           log_path="/tmp/test.log", parser_format="pytest")

    def test_report_status_pass(self):
        r = generate_deployment_report(self._summary_pass())
        assert "PASS" in r.report_text
        assert "FAIL" not in r.report_text.split("PASS")[0]

    def test_report_status_fail(self):
        r = generate_deployment_report(self._summary_fail())
        assert "FAIL" in r.report_text

    def test_report_status_unknown_when_no_tests(self):
        s = TestSummary()
        r = generate_deployment_report(s)
        assert "UNKNOWN" in r.report_text

    def test_session_id_in_report(self):
        r = generate_deployment_report(self._summary_pass(), session_id="abc123")
        assert "abc123" in r.report_text

    def test_task_goal_in_report(self):
        r = generate_deployment_report(self._summary_pass(), task_goal="Run all tests")
        assert "Run all tests" in r.report_text

    def test_gcs_url_in_report(self):
        r = generate_deployment_report(
            self._summary_pass(), gcs_log_url="gs://bucket/logs/test.log"
        )
        assert "gs://bucket/logs/test.log" in r.report_text

    def test_cloud_run_url_in_report(self):
        r = generate_deployment_report(
            self._summary_pass(), cloud_run_url="https://myapp-abc.run.app"
        )
        assert "https://myapp-abc.run.app" in r.report_text

    def test_pending_placeholders_when_no_urls(self):
        r = generate_deployment_report(self._summary_pass())
        assert "pending" in r.report_text.lower()

    def test_log_tail_included(self):
        s = self._summary_pass()
        s.raw_snippet = "PASSED tests/test_foo.py::test_something"
        r = generate_deployment_report(s)
        assert "LOG TAIL" in r.report_text
        assert "test_something" in r.report_text

    def test_auto_report_path_under_tmpdir(self):
        r = generate_deployment_report(self._summary_pass())
        tmpdir = str(Path(tempfile.gettempdir()).resolve())
        assert Path(r.report_path).resolve().parts[0:len(Path(tmpdir).parts)] == Path(tmpdir).parts

    def test_custom_report_path_honoured(self):
        r = generate_deployment_report(self._summary_pass(), report_path="/tmp/myreport.txt")
        assert r.report_path == "/tmp/myreport.txt"

    def test_generated_at_is_iso8601(self):
        r = generate_deployment_report(self._summary_pass())
        from datetime import datetime
        # Should parse without error
        datetime.fromisoformat(r.generated_at.replace("Z", "+00:00"))

    def test_summary_line_in_report(self):
        r = generate_deployment_report(self._summary_pass())
        assert "5 passed" in r.report_text

    def test_parse_errors_mentioned_in_report(self):
        s = self._summary_pass()
        s.parse_errors = ["Log truncated from 90000 to 65536 bytes"]
        r = generate_deployment_report(s)
        assert "truncated" in r.report_text.lower()

    def test_exit_code_in_report_when_provided(self):
        s = self._summary_pass()
        s.exit_code = 0
        r = generate_deployment_report(s)
        assert "Exit code" in r.report_text


# ─────────────────────────────────────────────────────────────────────────────
# 8 · ActionExecutor.READ_FILE
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutorReadFile:

    def _executor(self) -> ActionExecutor:
        return ActionExecutor()

    def test_reads_existing_tmp_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", dir=tempfile.gettempdir(),
            delete=False, encoding="utf-8",
        ) as f:
            f.write("hello test content\n5 passed in 1.00s\n")
            fpath = f.name
        try:
            result = self._executor().execute({"type": "READ_FILE", "path": fpath})
            assert result.success is True
            assert result.action_type == "READ_FILE"
            assert "hello test content" in result.extra["content"]
            assert result.extra["truncated"] is False
        finally:
            os.unlink(fpath)

    def test_file_not_found_returns_failure(self):
        result = self._executor().execute({
            "type": "READ_FILE", "path": "/tmp/_no_such_file_abc.log"
        })
        assert result.success is False
        assert "not found" in result.message.lower()

    def test_blocked_path_returns_failure(self):
        result = self._executor().execute({
            "type": "READ_FILE", "path": "/etc/passwd"
        })
        assert result.success is False
        assert result.extra.get("blocked") is True

    def test_no_path_returns_failure(self):
        result = self._executor().execute({"type": "READ_FILE", "path": ""})
        assert result.success is False
        assert result.action_type == "READ_FILE"

    def test_content_in_extra(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", dir=tempfile.gettempdir(),
            delete=False, encoding="utf-8",
        ) as f:
            f.write("pytest output\n5 passed in 2.00s\n")
            fpath = f.name
        try:
            result = self._executor().execute({"type": "READ_FILE", "path": fpath})
            assert "content" in result.extra
            assert "5 passed" in result.extra["content"]
        finally:
            os.unlink(fpath)

    def test_truncation_flagged_in_extra(self):
        big_content = "a" * (MAX_READ_BYTES + 500) + "\n5 passed in 1.23s\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", dir=tempfile.gettempdir(),
            delete=False, encoding="utf-8",
        ) as f:
            f.write(big_content)
            fpath = f.name
        try:
            result = self._executor().execute({"type": "READ_FILE", "path": fpath})
            assert result.success is True
            assert result.extra["truncated"] is True
            assert "truncated" in result.message.lower()
        finally:
            os.unlink(fpath)

    def test_duration_recorded(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", dir=tempfile.gettempdir(),
            delete=False, encoding="utf-8",
        ) as f:
            f.write("5 passed in 1.00s\n")
            fpath = f.name
        try:
            result = self._executor().execute({"type": "READ_FILE", "path": fpath})
            assert result.duration_ms >= 0.0
        finally:
            os.unlink(fpath)

    def test_path_traversal_blocked(self):
        result = self._executor().execute({
            "type": "READ_FILE", "path": "/tmp/../etc/shadow"
        })
        assert result.success is False


# ─────────────────────────────────────────────────────────────────────────────
# 9 · ActionExecutor.PARSE_LOG
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutorParseLog:

    def _executor(self) -> ActionExecutor:
        return ActionExecutor()

    def test_parse_pytest_from_tmp_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", dir=tempfile.gettempdir(),
            delete=False, encoding="utf-8",
        ) as f:
            f.write(PYTEST_PASS_SIMPLE)
            fpath = f.name
        try:
            result = self._executor().execute({"type": "PARSE_LOG", "path": fpath})
            assert result.success is True
            assert result.action_type == "PARSE_LOG"
            assert result.extra["passed"] == 5
            assert result.extra["failed"] == 0
            assert result.extra["test_success"] is True
            assert result.extra["parser_format"] == "pytest"
        finally:
            os.unlink(fpath)

    def test_parse_unittest_from_tmp_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", dir=tempfile.gettempdir(),
            delete=False, encoding="utf-8",
        ) as f:
            f.write(UNITTEST_FAILED)
            fpath = f.name
        try:
            result = self._executor().execute({"type": "PARSE_LOG", "path": fpath})
            assert result.extra["failed"] == 1
            assert result.extra["test_success"] is False
        finally:
            os.unlink(fpath)

    def test_parse_with_content_injection(self):
        """Content injection bypasses disk I/O — useful when content came from READ_FILE."""
        result = self._executor().execute({
            "type": "PARSE_LOG",
            "path": "/tmp/injected.log",
            "content": PYTEST_PASS_SIMPLE,
        })
        assert result.success is True
        assert result.extra["passed"] == 5

    def test_no_path_returns_failure(self):
        result = self._executor().execute({"type": "PARSE_LOG", "path": ""})
        assert result.success is False

    def test_extra_contains_summary_fields(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", dir=tempfile.gettempdir(),
            delete=False, encoding="utf-8",
        ) as f:
            f.write(PYTEST_SKIPPED_WARNINGS)
            fpath = f.name
        try:
            result = self._executor().execute({"type": "PARSE_LOG", "path": fpath})
            extra = result.extra
            assert "passed" in extra
            assert "failed" in extra
            assert "errors" in extra
            assert "skipped" in extra
            assert "warnings" in extra
            assert "duration_s" in extra
            assert "summary_line" in extra
            assert "raw_snippet" in extra
        finally:
            os.unlink(fpath)

    def test_summary_line_in_message(self):
        result = self._executor().execute({
            "type": "PARSE_LOG",
            "path": "/tmp/injected.log",
            "content": PYTEST_PASS_SIMPLE,
        })
        assert "5 passed" in result.message

    def test_exit_code_forwarded(self):
        result = self._executor().execute({
            "type": "PARSE_LOG",
            "path": "/tmp/injected.log",
            "content": PYTEST_ALL_FAILED,
            "exit_code": 1,
        })
        assert result.extra["exit_code"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 10 · ActionExecutor.WRITE_REPORT
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutorWriteReport:

    def _executor(self) -> ActionExecutor:
        return ActionExecutor()

    def _write_to_temp(self, log_content: str) -> tuple[str, str]:
        """Write log_content to a temp file, return (log_path, report_dest)."""
        tmpdir = tempfile.gettempdir()
        log_f = tempfile.NamedTemporaryFile(
            mode="w", suffix="_test.log", dir=tmpdir, delete=False, encoding="utf-8"
        )
        log_f.write(log_content)
        log_f.close()
        report_dest = os.path.join(tmpdir, "_test_report_out.txt")
        return log_f.name, report_dest

    def test_write_report_creates_file(self):
        log_path, report_dest = self._write_to_temp(PYTEST_PASS_SIMPLE)
        try:
            result = self._executor().execute({
                "type": "WRITE_REPORT",
                "log_path": log_path,
                "report_path": report_dest,
                "copy_to_clipboard": False,
                "task_goal": "run tests",
            })
            assert result.success is True
            assert Path(report_dest).exists()
            txt = Path(report_dest).read_text(encoding="utf-8")
            assert "PASS" in txt
            assert "5 passed" in txt
        finally:
            os.unlink(log_path)
            Path(report_dest).unlink(missing_ok=True)

    def test_report_text_in_extra(self):
        _, report_dest = self._write_to_temp(PYTEST_PASS_SIMPLE)
        log_path = _
        try:
            result = self._executor().execute({
                "type": "WRITE_REPORT",
                "log_path": log_path,
                "report_path": report_dest,
                "copy_to_clipboard": False,
            })
            assert "report_text" in result.extra
            assert len(result.extra["report_text"]) > 0
        finally:
            os.unlink(log_path)
            Path(report_dest).unlink(missing_ok=True)

    def test_no_log_path_still_produces_report(self):
        tmpdir = tempfile.gettempdir()
        report_dest = os.path.join(tmpdir, "_test_no_log_report.txt")
        result = self._executor().execute({
            "type": "WRITE_REPORT",
            "log_path": "",
            "report_path": report_dest,
            "copy_to_clipboard": False,
        })
        # Should still succeed (empty summary)
        assert result.action_type == "WRITE_REPORT"
        Path(report_dest).unlink(missing_ok=True)

    def test_clipboard_copy_attempted(self):
        log_path, report_dest = self._write_to_temp(PYTEST_PASS_SIMPLE)
        try:
            mock_clip = MagicMock()
            with patch.dict("sys.modules", {"pyperclip": mock_clip}):
                result = self._executor().execute({
                    "type": "WRITE_REPORT",
                    "log_path": log_path,
                    "report_path": report_dest,
                    "copy_to_clipboard": True,
                })
            # pyperclip.copy should be called with the report text
            mock_clip.copy.assert_called_once()
        finally:
            os.unlink(log_path)
            Path(report_dest).unlink(missing_ok=True)

    def test_clipboard_failure_does_not_crash(self):
        log_path, report_dest = self._write_to_temp(PYTEST_PASS_SIMPLE)
        try:
            mock_clip = MagicMock()
            mock_clip.copy.side_effect = Exception("no clipboard")
            with patch.dict("sys.modules", {"pyperclip": mock_clip}):
                result = self._executor().execute({
                    "type": "WRITE_REPORT",
                    "log_path": log_path,
                    "report_path": report_dest,
                    "copy_to_clipboard": True,
                })
            # write succeeded even though clipboard failed
            assert result.extra["written"] is True
            assert result.extra["clipped"] is False
        finally:
            os.unlink(log_path)
            Path(report_dest).unlink(missing_ok=True)

    def test_write_failure_handled_gracefully(self):
        log_path, _ = self._write_to_temp(PYTEST_PASS_SIMPLE)
        try:
            # Use an invalid path to trigger write failure
            result = self._executor().execute({
                "type": "WRITE_REPORT",
                "log_path": log_path,
                "report_path": "/no/such/directory/report.txt",
                "copy_to_clipboard": False,
            })
            assert result.extra["written"] is False
        finally:
            os.unlink(log_path)

    def test_test_success_flag_in_extra(self):
        log_path, report_dest = self._write_to_temp(PYTEST_PASS_SIMPLE)
        try:
            result = self._executor().execute({
                "type": "WRITE_REPORT",
                "log_path": log_path,
                "report_path": report_dest,
                "copy_to_clipboard": False,
            })
            assert result.extra["test_success"] is True
        finally:
            os.unlink(log_path)
            Path(report_dest).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 11 · ActionExecutor.VERIFY method="read_file"
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutorVerifyReadFile:

    def _executor(self) -> ActionExecutor:
        return ActionExecutor()

    def _write_tmp(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", dir=tempfile.gettempdir(),
            delete=False, encoding="utf-8",
        )
        f.write(content)
        f.close()
        return f.name

    def test_verify_file_readable(self):
        fpath = self._write_tmp("5 passed in 1.00s\nPASS\n")
        try:
            result = self._executor().execute({
                "type": "VERIFY",
                "method": "read_file",
                "path": fpath,
                "description": "confirm test output readable",
            })
            assert result.success is True
            assert result.action_type == "VERIFY"
            assert "✓" in result.message or "readable" in result.message.lower()
        finally:
            os.unlink(fpath)

    def test_verify_must_see_all_present(self):
        fpath = self._write_tmp("5 passed in 1.00s\nSummary: OK\nexit code 0\n")
        try:
            result = self._executor().execute({
                "type": "VERIFY",
                "method": "read_file",
                "path": fpath,
                "must_see": ["passed", "Summary", "exit code 0"],
            })
            assert result.success is True
        finally:
            os.unlink(fpath)

    def test_verify_must_see_missing_string_fails(self):
        fpath = self._write_tmp("3 failed in 2.00s\n")
        try:
            result = self._executor().execute({
                "type": "VERIFY",
                "method": "read_file",
                "path": fpath,
                "must_see": ["passed", "exit code 0"],
                "description": "confirm pass",
            })
            assert result.success is False
            assert "not found" in result.message.lower() or "fail" in result.message.lower()
            assert "passed" in str(result.extra.get("missing", []))
        finally:
            os.unlink(fpath)

    def test_verify_must_see_case_insensitive(self):
        fpath = self._write_tmp("ALL TESTS PASSED\n")
        try:
            result = self._executor().execute({
                "type": "VERIFY",
                "method": "read_file",
                "path": fpath,
                "must_see": ["passed"],  # lowercase should still match
            })
            assert result.success is True
        finally:
            os.unlink(fpath)

    def test_verify_file_not_found_fails(self):
        result = self._executor().execute({
            "type": "VERIFY",
            "method": "read_file",
            "path": "/tmp/_verify_no_such_file.log",
        })
        assert result.success is False
        assert "fail" in result.message.lower() or "not found" in result.message.lower()

    def test_verify_visual_fallback_skipped(self):
        """method='visual' (default) must still return skipped=True."""
        result = self._executor().execute({
            "type": "VERIFY",
            "method": "visual",
            "description": "check button clicked",
        })
        assert result.success is True
        assert result.skipped is True

    def test_verify_no_method_defaults_to_visual(self):
        result = self._executor().execute({
            "type": "VERIFY",
            "description": "check default method",
        })
        assert result.skipped is True

    def test_verify_blocked_path_fails(self):
        result = self._executor().execute({
            "type": "VERIFY",
            "method": "read_file",
            "path": "/etc/passwd",
        })
        assert result.success is False


# ─────────────────────────────────────────────────────────────────────────────
# 12 · _extract_raw_snippet helper
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractRawSnippet:

    def test_short_content_returned_intact(self):
        content = "line1\nline2\nline3\n"
        snippet = _extract_raw_snippet(content, tail_lines=20)
        assert "line1" in snippet
        assert "line3" in snippet

    def test_long_content_truncated_to_tail(self):
        lines = [f"line {i}" for i in range(100)]
        content = "\n".join(lines)
        snippet = _extract_raw_snippet(content, tail_lines=20)
        assert "line 99" in snippet
        assert "line 0" not in snippet

    def test_blank_lines_filtered(self):
        content = "\n\n\nline1\n\n\nline2\n\n"
        snippet = _extract_raw_snippet(content)
        assert "line1" in snippet
        assert "line2" in snippet
