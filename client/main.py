"""
main.py — Entry point for UI Navigator Desktop Client.

Usage
-----
    python client/main.py

Or from repo root:
    python -m client.main

Environment variables (optional):
    NAVIGATOR_SERVER   Server URL to pre-populate (default: http://localhost:8080)
    NAVIGATOR_GOAL     Task goal to pre-populate
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

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
