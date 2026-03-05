"""
command_policy.py — Safe command execution policy for UI Navigator.

Implements a two-tier allowlist / blocklist gate for terminal commands
before any shell process is spawned.

Architecture
------------
  1. **Blocklist** (hard stop): dangerous patterns that are ALWAYS rejected,
     regardless of any allowlist entry.
  2. **Allowlist** (explicit permit): command prefixes or patterns that are
     known-safe for the demo workflow.
  3. **Default deny**: commands matching neither list are blocked.
     Add patterns to ALLOWED_PATTERNS when a new safe command is needed.

Explicitly allowlisted (per project plan)
-----------------------------------------
  - make test / pytest  (test runners)
  - gcloud run deploy   (deploy to Cloud Run)
  - gsutil cp           (GCS uploads)
  - echo, ls, pwd, env  (read-only shell builtins)
  - git status/log/diff (read-only git)
  - python client|server|tests/*.py  (in-repo scripts only)

Usage
-----
    from command_policy import check_command, execute_command

    check = check_command("make test 2>&1 | tee /tmp/test_out.log")
    if not check.allowed:
        print(f"Blocked: {check.reason}")
    else:
        result = execute_command("make test 2>&1 | tee /tmp/test_out.log")
        print(result.stdout)
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Blocklist — patterns matched against the full command string.
# A match here is a hard stop; the command will never be executed.
# ---------------------------------------------------------------------------

_BLOCKED_RAW: list[tuple[str, str]] = [
    # Unix recursive or forced deletion (-r/-R/-rf/-fr/--recursive)
    (
        "rm_recursive",
        r"\brm\b.*\s-[a-zA-Z]*[rR]|\brm\b.*--recursive\b",
    ),
    # Windows recursive deletion
    (
        "del_recursive",
        r"\bdel\b.*/[sS]\b|deltree\b|\brmdir\b.*/[sS]\b",
    ),
    # System shutdown / reboot / halt
    (
        "shutdown",
        r"\bshutdown\b|\breboot\b|\binit\s+[06]\b"
        r"|\bsystemctl\s+(poweroff|reboot|halt)\b",
    ),
    # Broad process kills
    (
        "kill_all",
        r"\bkillall\b|\bpkill\s+(-9|-SIGKILL)\b|\bkill\s+-9\s+-1\b",
    ),
    # Disk overwrite / formatting
    (
        "disk_clobber",
        r"\bdd\b.+of=/dev/|\bmkfs\b|\bformat\s+[a-zA-Z]:\\\B",
    ),
    # Credential-extraction tooling
    (
        "credential_dump",
        r"\bsecretsdump\b|\bmimikatz\b|\bprocdump\b",
    ),
    # Cloud resource deletion (gcloud, kubectl)
    (
        "cloud_delete",
        r"\bgcloud\b.+\b(delete|destroy|remove)\b"
        r"|\bkubectl\b.+\b(delete|destroy)\b",
    ),
    # Pipe from web to shell (common RCE vector)
    (
        "curl_pipe_sh",
        r"\bcurl\b.+\|\s*(ba)?sh\b|\bwget\b.+\|\s*(ba)?sh\b",
    ),
]

BLOCKED_PATTERNS: list[tuple[str, re.Pattern]] = [
    (name, re.compile(pattern, re.IGNORECASE | re.DOTALL))
    for name, pattern in _BLOCKED_RAW
]


# ---------------------------------------------------------------------------
# Allowlist — patterns for commands that are explicitly permitted.
# The first matching rule wins; order reflects descending specificity.
# ---------------------------------------------------------------------------

_ALLOWED_RAW: list[tuple[str, str]] = [
    # ── Test runners ─────────────────────────────────────────────────────────
    ("make_test",        r"^\s*make\s+test\b"),
    ("pytest",           r"^\s*pytest\b"),
    ("python_m_pytest",  r"^\s*python\s+-m\s+pytest\b"),
    # ── Cloud operations ─────────────────────────────────────────────────────
    ("gcloud_run_deploy", r"^\s*gcloud\s+run\s+deploy\b"),
    ("gsutil_cp",         r"^\s*gsutil\s+cp\b"),
    # ── Log inspection ───────────────────────────────────────────────────────
    ("cat_tmp",          r"^\s*cat\s+/tmp/"),
    ("tail_log",         r"^\s*tail\b.*/tmp/"),
    # ── Safe shell builtins / read-only introspection ────────────────────────
    ("echo",             r"^\s*echo\b"),
    ("pwd",              r"^\s*pwd\b"),
    ("ls_safe",          r"^\s*ls(\s|$)"),
    ("env_list",         r"^\s*env\b|\s*printenv\b|\s*set\b"),
    # ── In-repo Python scripts only (not arbitrary paths) ────────────────────
    ("python_repo_script",
     r"^\s*python\s+(client|server|tests)/\S+\.py\b"),
    # ── Read-only git ────────────────────────────────────────────────────────
    ("git_readonly",
     r"^\s*git\s+(status|log|diff|show|branch|remote)\b"),
]

ALLOWED_PATTERNS: list[tuple[str, re.Pattern]] = [
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in _ALLOWED_RAW
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CommandCheckResult:
    """Result of the two-tier policy check for a terminal command."""
    allowed: bool
    reason: str
    matched_rule: str = ""


@dataclass
class CommandResult:
    """Result of executing a shell command via :func:`execute_command`."""
    command: str
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    blocked: bool = False
    block_reason: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Policy check
# ---------------------------------------------------------------------------

def check_command(command: str) -> CommandCheckResult:
    """
    Run *command* through the two-tier policy gate and return whether it
    is allowed.

    Evaluation order
    ----------------
    1. **Blocklist** — if any blocked pattern matches, the command is denied
       immediately regardless of the allowlist.
    2. **Allowlist** — if an explicit allow rule matches, the command is
       permitted.
    3. **Default deny** — commands not matching any rule are blocked.

    Parameters
    ----------
    command:
        Full shell command string to evaluate.

    Returns
    -------
    CommandCheckResult with ``allowed`` set and a human-readable ``reason``.
    """
    stripped = command.strip()
    if not stripped:
        return CommandCheckResult(
            allowed=False,
            reason="Empty command rejected",
            matched_rule="empty",
        )

    # ── Pass 1: blocklist (hard stop) ────────────────────────────────────────
    for rule_name, pattern in BLOCKED_PATTERNS:
        if pattern.search(stripped):
            logger.warning(
                "Command BLOCKED by rule '%s': %r",
                rule_name,
                stripped[:200],
            )
            return CommandCheckResult(
                allowed=False,
                reason=f"Command blocked by safety rule '{rule_name}'",
                matched_rule=rule_name,
            )

    # ── Pass 2: allowlist ────────────────────────────────────────────────────
    for rule_name, pattern in ALLOWED_PATTERNS:
        if pattern.search(stripped):
            logger.info(
                "Command ALLOWED by rule '%s': %r",
                rule_name,
                stripped[:200],
            )
            return CommandCheckResult(
                allowed=True,
                reason=f"Command permitted by allowlist rule '{rule_name}'",
                matched_rule=rule_name,
            )

    # ── Pass 3: default deny ─────────────────────────────────────────────────
    logger.warning("Command BLOCKED (not on allowlist): %r", stripped[:200])
    return CommandCheckResult(
        allowed=False,
        reason=(
            "Command is not on the allowlist. "
            "Add it to ALLOWED_PATTERNS in command_policy.py if it is safe."
        ),
        matched_rule="default_deny",
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_command(
    command: str,
    timeout_s: int = 60,
    shell: bool = True,
) -> CommandResult:
    """
    Execute *command* as a subprocess after passing the policy check.

    Blocked commands are returned immediately as a ``CommandResult`` with
    ``blocked=True`` — no subprocess is ever spawned for them.

    Parameters
    ----------
    command:
        Shell command string.
    timeout_s:
        Maximum seconds to wait before killing the subprocess (1–300).
    shell:
        Passed to ``subprocess.run``.  ``True`` is required for pipes,
        redirects, and compound commands (e.g. ``make test 2>&1 | tee …``).

    Returns
    -------
    CommandResult with all execution details.
    """
    check = check_command(command)
    if not check.allowed:
        return CommandResult(
            command=command,
            returncode=-1,
            stdout="",
            stderr="",
            duration_ms=0.0,
            blocked=True,
            block_reason=check.reason,
        )

    t0 = time.monotonic()
    timed_out = False
    returncode = -1
    stdout = ""
    stderr = ""

    try:
        proc = subprocess.run(
            command,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired:
        returncode = -1
        stderr = f"Command timed out after {timeout_s}s"
        timed_out = True
    except Exception as exc:  # noqa: BLE001
        returncode = -1
        stderr = f"Execution error: {exc}"

    duration_ms = (time.monotonic() - t0) * 1000

    logger.info(
        "EXEC_COMMAND returncode=%d duration=%.0fms timed_out=%s: %r",
        returncode,
        duration_ms,
        timed_out,
        command[:200],
    )

    return CommandResult(
        command=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_ms=duration_ms,
        timed_out=timed_out,
    )
