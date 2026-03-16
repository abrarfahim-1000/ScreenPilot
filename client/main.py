"""
main.py — Entry point for UI Navigator Desktop Client.

Usage
-----
    python client/main.py

Or from repo root:
    python -m client.main

Environment variables (optional):
    NAVIGATOR_SERVER   Server URL to pre-populate (default: https://ui-navigator-314272999720.asia-southeast1.run.app)
    NAVIGATOR_GOAL     Task goal to pre-populate
    GEMINI_API_KEY     If set, silently seeds the OS keystore so the API-key
                       dialog is skipped on first launch.  The same variable
                       used to start the server is therefore sufficient for
                       the client as well.
"""

from __future__ import annotations

import logging
import os
import sys

# Ensure the client directory is on sys.path so sibling imports work when
# called as  `python client/main.py`  from the repo root.
_CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CLIENT_DIR not in sys.path:
    sys.path.insert(0, _CLIENT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from ui import ControlPanel


def main() -> int:
    # High-DPI support
    app = QApplication(sys.argv)
    app.setApplicationName("UI Navigator")
    app.setOrganizationName("Local-to-Cloud")

    window = ControlPanel()

    # Pre-populate from environment variables if set
    server_env = os.environ.get("NAVIGATOR_SERVER", "")
    goal_env   = os.environ.get("NAVIGATOR_GOAL", "")
    if server_env:
        window.server_input.setText(server_env)
    if goal_env:
        window.task_input.setText(goal_env)

    # If GEMINI_API_KEY is present in the environment (e.g. the same shell
    # session used to launch the server), seed the OS keystore silently so
    # the API-key dialog is never shown.  An already-stored key is left
    # untouched — the env var only fills in the gap on a fresh machine.
    key_env = os.environ.get("GEMINI_API_KEY", "").strip()
    if key_env and not window._keystore.exists():
        window._keystore.save(key_env)
        window._refresh_key_status_label()
        logging.getLogger(__name__).info(
            "API key seeded from GEMINI_API_KEY environment variable"
        )

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
