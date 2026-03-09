"""
firestore_session.py — Firestore-backed session state and audit log.

Responsibilities
----------------
- Persist session creation, step results, VERIFY outcomes, and recovery events
  to Google Cloud Firestore so every agent action is auditable.
- Large artifacts (screenshots, raw model output) are stored in GCS; only
  metadata and summaries are written here.
- All writes are best-effort: a Firestore failure never crashes the API server.
  The caller should log the exception and continue.

Collection layout
-----------------
  ui_navigator_sessions/{session_id}/
      session document  — task_goal, started_at, status, step_count
      steps/{step_id}   — perception summary, actions list, verify results
      recovery/{step_id}_{attempt} — recovery strategy + outcome
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lazily imported so that the module can be imported even if the
# google-cloud-firestore package is absent (e.g. during unit tests).
_firestore_mod = None


def _get_firestore():
    """Lazy-import google.cloud.firestore and return the module."""
    global _firestore_mod
    if _firestore_mod is None:
        try:
            from google.cloud import firestore as _fs
            _firestore_mod = _fs
        except ImportError:
            logger.warning(
                "google-cloud-firestore not installed — Firestore logging disabled. "
                "Run: pip install google-cloud-firestore"
            )
            _firestore_mod = False  # sentinel: import attempted but failed
    return _firestore_mod if _firestore_mod else None


class FirestoreSessionStore:
    """
    Thin wrapper around the Firestore SDK for UI Navigator session state.

    Parameters
    ----------
    project_id:
        GCP project ID.  If None, the SDK infers it from the environment
        (``GOOGLE_CLOUD_PROJECT`` env var or ADC).
    collection:
        Top-level Firestore collection name (default ``ui_navigator_sessions``).
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        collection: str = "ui_navigator_sessions",
    ) -> None:
        self._collection = collection
        fs = _get_firestore()
        if fs is None:
            self._db = None
            return
        try:
            self._db = fs.Client(project=project_id)
            logger.info(
                "FirestoreSessionStore initialised (project=%s collection=%s)",
                project_id or "inferred",
                collection,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Firestore init failed — audit logging disabled: %s", exc)
            self._db = None

    @property
    def available(self) -> bool:
        """True if the Firestore client is ready to accept writes."""
        return self._db is not None

    # ── Session lifecycle ────────────────────────────────────────────────────

    def create_session(
        self,
        session_id: str,
        task_goal: str,
        server_version: str = "0.1.0",
    ) -> None:
        """Write the initial session document."""
        if not self.available:
            return
        try:
            doc = self._db.collection(self._collection).document(session_id)
            doc.set(
                {
                    "session_id": session_id,
                    "task_goal": task_goal,
                    "started_at": time.time(),
                    "status": "active",
                    "step_count": 0,
                    "server_version": server_version,
                },
                merge=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Firestore create_session failed: %s", exc)

    def close_session(self, session_id: str, reason: str = "completed") -> None:
        """Mark the session as closed with a reason."""
        if not self.available:
            return
        try:
            doc = self._db.collection(self._collection).document(session_id)
            fs = _get_firestore()
            doc.update(
                {
                    "status": "closed",
                    "ended_at": time.time(),
                    "close_reason": reason,
                    "step_count": fs.transforms.Increment(0),  # don't reset
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Firestore close_session failed: %s", exc)

    # ── Step logging ─────────────────────────────────────────────────────────

    def log_step(
        self,
        session_id: str,
        step_id: int,
        perception_summary: str,
        actions: list[dict],
        expected: dict,
        raw_model_output: Optional[str] = None,
    ) -> None:
        """
        Write a step document under ``steps/{step_id}``.

        Keeps the perception summary, the action list (as plain dicts),
        and the expected policy block.  The full raw_model_output is stored
        for audit purposes but truncated to 4 KB to stay within Firestore limits.
        """
        if not self.available:
            return
        try:
            step_doc = (
                self._db.collection(self._collection)
                .document(session_id)
                .collection("steps")
                .document(str(step_id))
            )
            # Truncate raw output — Firestore field size limit is 1 MiB but
            # we keep it much smaller for cost efficiency.
            raw_trunc = (raw_model_output or "")[:4096]
            step_doc.set(
                {
                    "session_id": session_id,
                    "step_id": step_id,
                    "timestamp": time.time(),
                    "perception_summary": perception_summary,
                    "action_count": len(actions),
                    "action_types": [a.get("type", "UNKNOWN") for a in actions],
                    "expected": expected,
                    "raw_model_output": raw_trunc,
                }
            )
            # Increment step counter on the session document
            fs = _get_firestore()
            self._db.collection(self._collection).document(session_id).update(
                {"step_count": fs.transforms.Increment(1)}
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Firestore log_step failed (step=%d): %s", step_id, exc)

    def log_verify(
        self,
        session_id: str,
        step_id: int,
        method: str,
        success: bool,
        description: str,
        extra: Optional[dict] = None,
    ) -> None:
        """
        Append a VERIFY result to the step document as a sub-field.

        Uses ``set(merge=True)`` so it can be called after ``log_step``
        without overwriting the step document.
        """
        if not self.available:
            return
        try:
            step_doc = (
                self._db.collection(self._collection)
                .document(session_id)
                .collection("steps")
                .document(str(step_id))
            )
            verify_record: dict[str, Any] = {
                "verify_method": method,
                "verify_success": success,
                "verify_description": description,
                "verify_timestamp": time.time(),
            }
            if extra:
                # Only store safe scalar / small string values
                verify_record["verify_extra"] = {
                    k: str(v)[:500] for k, v in extra.items() if isinstance(v, (str, int, float, bool))
                }
            step_doc.set(verify_record, merge=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Firestore log_verify failed (session=%s step=%d): %s",
                session_id, step_id, exc,
            )

    def log_recovery(
        self,
        session_id: str,
        step_id: int,
        attempt: int,
        strategy: str,
        result: str,
    ) -> None:
        """
        Write a recovery event document under ``recovery/{step_id}_{attempt}``.

        Used to build the "context-aware visual recovery" audit trail shown in
        the submission and demo.
        """
        if not self.available:
            return
        try:
            doc_id = f"{step_id}_{attempt}"
            recovery_doc = (
                self._db.collection(self._collection)
                .document(session_id)
                .collection("recovery")
                .document(doc_id)
            )
            recovery_doc.set(
                {
                    "session_id": session_id,
                    "step_id": step_id,
                    "attempt": attempt,
                    "strategy": strategy,
                    "result": result,
                    "timestamp": time.time(),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Firestore log_recovery failed (session=%s step=%d attempt=%d): %s",
                session_id, step_id, attempt, exc,
            )

    def log_session_plan(
        self,
        session_id: str,
        task_goal: str,
        steps: list[dict],
    ) -> None:
        """
        Persist the session plan steps on the session document.

        Called by the /session/plan endpoint so the plan is queryable
        alongside the live step logs.
        """
        if not self.available:
            return
        try:
            doc = self._db.collection(self._collection).document(session_id)
            doc.set(
                {
                    "session_id": session_id,
                    "task_goal": task_goal,
                    "plan_steps": steps,
                    "plan_created_at": time.time(),
                },
                merge=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Firestore log_session_plan failed: %s", exc)
