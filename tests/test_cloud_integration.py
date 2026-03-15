"""
tests/test_cloud_integration.py — Tests for Days 9-11 Cloud Integration features.

Covers
------
1. GCSArtifactStore
     - available=False when bucket_name is empty
     - available=False when google-cloud-storage not installed
     - upload_bytes() returns gs:// URL on success, "" on failure
     - upload_bytes() returns "" when unavailable
     - upload_file() returns "" when file not found
     - public_url() and gs_url() format correctly

2. server/schemas — new action types
     - ActionType.UPLOAD_GCS, DEPLOY_CLOUD_RUN, CONFIRM present in enum
     - UploadGCSAction, DeployCloudRunAction, ConfirmAction serialise / validate
     - ActionItem union accepts all three new types
     - DeployCloudRunAction.confirmed defaults to False

3. /session/upload endpoint
     - Responds 200 with uploaded=False when GCS store unavailable
     - Responds 200 with uploaded=True when GCS store available (mocked)
     - Rejects non-file uploads (field missing)
     - Returns correct size_bytes

4. executor._upload_gcs
     - Skips on empty local_path
     - Blocks path traversal
     - Returns success=False when server_url not set
     - Posts to /session/upload and returns gcs_url on success (mocked requests)
     - Returns success=False on HTTP error

5. executor._deploy_cloud_run
     - Returns success=False without confirmed=True
     - Returns success=False on missing service_name or image
     - Constructs correct gcloud command flags
     - Parses Service URL from stdout
     - Passes policy gate (gcloud run deploy is allowlisted)
     - Returns blocked=True for gcloud delete (blocklisted)

6. executor CONFIRM (handled in session, not executor directly)
     - CONFIRM action type is unknown to executor → unknown action result

7. session._run_one_step — CONFIRM gate
     - CONFIRM emits confirmation_required signal, pauses, continues on confirm
     - CONFIRM + cancel → session stops
     - UPLOAD_GCS result → _last_gcs_url captured
     - DEPLOY_CLOUD_RUN result → _last_cloud_run_url captured
     - WRITE_REPORT injected with captured URLs

8. command_policy — new allowlisted patterns
     - gcloud run services describe allowed
     - gcloud run services list allowed
     - gsutil ls allowed
     - gcloud run deploy with flags allowed

Run with:
    pytest tests/test_cloud_integration.py -v
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from PIL import Image

# ── path setup ────────────────────────────────────────────────────────────────
SERVER_DIR = os.path.join(os.path.dirname(__file__), "..", "server")
CLIENT_DIR = os.path.join(os.path.dirname(__file__), "..", "client")
sys.path.insert(0, SERVER_DIR)
sys.path.insert(0, CLIENT_DIR)

# ── server imports ────────────────────────────────────────────────────────────
from gcs_storage import GCSArtifactStore
from schemas import (
    ActionType,
    ConfirmAction,
    DeployCloudRunAction,
    UploadGCSAction,
)

# ── client imports ────────────────────────────────────────────────────────────
from command_policy import check_command
from executor import ActionExecutor, ExecutionResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_jpeg() -> bytes:
    img = Image.new("RGB", (64, 48), color=(100, 149, 237))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def _make_frame_metadata(**overrides) -> dict:
    base = {
        "session_id": "test-ci-001",
        "step_id": 1,
        "task_goal": "Run tests, upload logs, deploy to Cloud Run",
        "frame_hash": "c" * 32,
        "timestamp": time.time(),
        "width": 1280,
        "height": 720,
        "monitor_index": 1,
    }
    base.update(overrides)
    return base


@pytest.fixture
def tmp_log(tmp_path: Path) -> Path:
    """Write a passing test log to a temp file and return its path."""
    log = tmp_path / "test_out.log"
    log.write_text(
        "collected 5 items\n"
        ".....\n"
        "========== 5 passed in 1.23s ==========\n"
    )
    return log


# ─────────────────────────────────────────────────────────────────────────────
# 1 · GCSArtifactStore
# ─────────────────────────────────────────────────────────────────────────────

class TestGCSArtifactStore:

    def test_unavailable_no_bucket(self):
        store = GCSArtifactStore(bucket_name=None)
        assert not store.available

    def test_unavailable_empty_bucket(self):
        store = GCSArtifactStore(bucket_name="")
        assert not store.available

    def test_unavailable_when_sdk_missing(self):
        """If google-cloud-storage is importable but init fails, available=False."""
        with patch("gcs_storage._get_storage") as mock_get:
            with patch("gcs_storage._gcs_mod", None, create=True):
                mock_storage = MagicMock()
                mock_storage.Client.side_effect = RuntimeError("no credentials")
                mock_get.return_value = mock_storage
                store = GCSArtifactStore(bucket_name="my-bucket")
                assert not store.available

    def test_upload_bytes_returns_gs_url(self):
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        with patch("gcs_storage._get_storage") as mock_get:
            mock_storage = MagicMock()
            mock_storage.Client.return_value = mock_client
            mock_get.return_value = mock_storage
            store = GCSArtifactStore(bucket_name="my-bucket")

        # Must be available to proceed
        assert store.available
        url = store.upload_bytes(b"hello world", "sessions/abc/logs/test.log", "text/plain")
        assert url == "gs://my-bucket/sessions/abc/logs/test.log"
        mock_blob.upload_from_string.assert_called_once_with(b"hello world", content_type="text/plain")

    def test_upload_bytes_returns_empty_on_failure(self):
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.upload_from_string.side_effect = RuntimeError("GCS error")
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        with patch("gcs_storage._get_storage") as mock_get:
            mock_storage = MagicMock()
            mock_storage.Client.return_value = mock_client
            mock_get.return_value = mock_storage
            store = GCSArtifactStore(bucket_name="my-bucket")

        url = store.upload_bytes(b"data", "object.txt")
        assert url == ""

    def test_upload_bytes_noop_when_unavailable(self):
        store = GCSArtifactStore(bucket_name="")
        url = store.upload_bytes(b"data", "object.txt")
        assert url == ""

    def test_upload_file_not_found(self):
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        with patch("gcs_storage._get_storage") as mock_get:
            mock_storage = MagicMock()
            mock_storage.Client.return_value = mock_client
            mock_get.return_value = mock_storage
            store = GCSArtifactStore(bucket_name="my-bucket")

        url = store.upload_file("/nonexistent/file.log", "obj/path.log")
        assert url == ""

    def test_public_url(self):
        store = GCSArtifactStore.__new__(GCSArtifactStore)
        store._bucket_name = "my-bucket"
        assert store.public_url("sessions/abc/log.txt") == (
            "https://storage.googleapis.com/my-bucket/sessions/abc/log.txt"
        )

    def test_gs_url(self):
        store = GCSArtifactStore.__new__(GCSArtifactStore)
        store._bucket_name = "my-bucket"
        assert store.gs_url("sessions/abc/log.txt") == "gs://my-bucket/sessions/abc/log.txt"


# ─────────────────────────────────────────────────────────────────────────────
# 2 · Schemas — new action types
# ─────────────────────────────────────────────────────────────────────────────

class TestNewActionTypes:

    def test_upload_gcs_in_enum(self):
        assert ActionType.UPLOAD_GCS == "UPLOAD_GCS"

    def test_deploy_cloud_run_in_enum(self):
        assert ActionType.DEPLOY_CLOUD_RUN == "DEPLOY_CLOUD_RUN"

    def test_confirm_in_enum(self):
        assert ActionType.CONFIRM == "CONFIRM"

    def test_upload_gcs_action_validates(self):
        action = UploadGCSAction(
            local_path="/tmp/test_out.log",
            gcs_object="sessions/abc/logs/test_out.log",
            reason="Upload test log",
        )
        assert action.type == ActionType.UPLOAD_GCS
        assert action.local_path == "/tmp/test_out.log"
        assert action.content_type == "text/plain"

    def test_deploy_cloud_run_action_defaults(self):
        action = DeployCloudRunAction(
            service_name="ui-navigator",
            image="gcr.io/proj/image:v1",
        )
        assert action.type == ActionType.DEPLOY_CLOUD_RUN
        assert action.region == "us-central1"
        assert action.confirmed is False      # must start unconfirmed

    def test_deploy_cloud_run_action_confirmed_param(self):
        action = DeployCloudRunAction(
            service_name="ui-navigator",
            image="gcr.io/proj/image:v1",
            confirmed=True,
        )
        assert action.confirmed is True

    def test_confirm_action_validates(self):
        action = ConfirmAction(
            message="Deploy to Cloud Run in us-central1?",
            action_ref="DEPLOY_CLOUD_RUN",
        )
        assert action.type == ActionType.CONFIRM
        assert action.message == "Deploy to Cloud Run in us-central1?"

    def test_confirm_action_default_message(self):
        action = ConfirmAction()
        assert "proceed" in action.message.lower()

    def test_upload_gcs_serialises(self):
        action = UploadGCSAction(local_path="/tmp/log.txt")
        d = action.model_dump()
        assert d["type"] == "UPLOAD_GCS"
        assert d["local_path"] == "/tmp/log.txt"

    def test_deploy_cloud_run_serialises(self):
        action = DeployCloudRunAction(
            service_name="svc",
            image="gcr.io/p/img:latest",
            allow_unauthenticated=True,
        )
        d = action.model_dump()
        assert d["type"] == "DEPLOY_CLOUD_RUN"
        assert d["allow_unauthenticated"] is True

    def test_confirm_serialises(self):
        action = ConfirmAction(message="Are you sure?", action_ref="DEPLOY_CLOUD_RUN")
        d = action.model_dump()
        assert d["type"] == "CONFIRM"
        assert d["action_ref"] == "DEPLOY_CLOUD_RUN"


# ─────────────────────────────────────────────────────────────────────────────
# 3 · /session/upload endpoint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def test_client_no_gcs():
    """FastAPI TestClient with GCS store unavailable."""
    from fastapi.testclient import TestClient
    with patch("app.GeminiPerceptionClient"):
        with patch("app.FirestoreSessionStore") as MockStore:
            with patch("app.GCSArtifactStore") as MockGCS:
                MockStore.return_value.available = False
                MockGCS.return_value.available = False
                with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                    import app as app_module
                    with TestClient(app_module.app) as client:
                        yield client


@pytest.fixture
def test_client_with_gcs():
    """FastAPI TestClient with GCS store mocked as available."""
    from fastapi.testclient import TestClient
    with patch("app.GeminiPerceptionClient"):
        with patch("app.FirestoreSessionStore") as MockStore:
            with patch("app.GCSArtifactStore") as MockGCS:
                MockStore.return_value.available = False
                mock_gcs = MockGCS.return_value
                mock_gcs.available = True
                mock_gcs.bucket_name = "test-bucket"
                mock_gcs.upload_bytes.return_value = "gs://test-bucket/sessions/s1/artifacts/test.log"
                with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                    import app as app_module
                    with TestClient(app_module.app) as client:
                        yield client, mock_gcs


class TestUploadEndpoint:

    def test_upload_no_gcs_returns_not_uploaded(self, test_client_no_gcs):
        client = test_client_no_gcs
        resp = client.post(
            "/session/upload",
            files={"file": ("test.log", b"5 passed in 1.23s", "text/plain")},
            data={"session_id": "s1"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["uploaded"] is False
        assert body["gcs_url"] == ""
        assert body["size_bytes"] == len(b"5 passed in 1.23s")

    def test_upload_with_gcs_returns_gcs_url(self, test_client_with_gcs):
        client, mock_gcs = test_client_with_gcs
        data = b"5 passed in 1.23s"
        resp = client.post(
            "/session/upload",
            files={"file": ("test.log", data, "text/plain")},
            data={"session_id": "s1"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["uploaded"] is True
        assert body["gcs_url"].startswith("gs://")
        assert body["size_bytes"] == len(data)

    def test_upload_with_explicit_object_name(self, test_client_with_gcs):
        client, mock_gcs = test_client_with_gcs
        resp = client.post(
            "/session/upload",
            files={"file": ("report.txt", b"content", "text/plain")},
            data={"session_id": "s1", "object_name": "custom/path/report.txt"},
        )
        assert resp.status_code == 200
        # object_name passed through
        args = mock_gcs.upload_bytes.call_args
        assert args[0][1] == "custom/path/report.txt"

    def test_upload_default_object_path_includes_session(self, test_client_with_gcs):
        client, mock_gcs = test_client_with_gcs
        resp = client.post(
            "/session/upload",
            files={"file": ("my.log", b"data", "text/plain")},
            data={"session_id": "mysession"},
        )
        assert resp.status_code == 200
        args = mock_gcs.upload_bytes.call_args
        obj_name = args[0][1]
        assert "mysession" in obj_name
        assert "my.log" in obj_name


# ─────────────────────────────────────────────────────────────────────────────
# 4 · executor._upload_gcs
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutorUploadGCS:

    def _make_executor(self, server_url="http://localhost:8080", session_id="test-session"):
        with patch("pyautogui.FAILSAFE", False):
            with patch("pyautogui.PAUSE", 0):
                return ActionExecutor(server_url=server_url, session_id=session_id)

    def test_empty_path_skipped(self):
        exc = self._make_executor()
        result = exc.execute({"type": "UPLOAD_GCS", "local_path": ""})
        assert not result.success
        assert "No local_path" in result.message

    def test_path_traversal_blocked(self):
        exc = self._make_executor()
        result = exc.execute({"type": "UPLOAD_GCS", "local_path": "/tmp/../etc/passwd"})
        assert not result.success
        assert "safety" in result.message.lower() or "blocked" in result.message.lower()

    def test_no_server_url(self, tmp_log):
        exc = self._make_executor(server_url="")
        result = exc.execute({"type": "UPLOAD_GCS", "local_path": str(tmp_log)})
        assert not result.success
        assert "server_url" in result.message

    def test_file_not_found(self):
        exc = self._make_executor()
        result = exc.execute({
            "type": "UPLOAD_GCS",
            "local_path": "/tmp/definitely_does_not_exist_xyz.log",
        })
        assert not result.success
        assert "not found" in result.message.lower()

    def test_success_with_mocked_server(self, tmp_log):
        exc = self._make_executor()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "uploaded": True,
            "gcs_url": "gs://test-bucket/sessions/test-session/artifacts/test_out.log",
            "object_name": "sessions/test-session/artifacts/test_out.log",
            "size_bytes": 64,
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("executor.requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            result = exc.execute({
                "type": "UPLOAD_GCS",
                "local_path": str(tmp_log),
            })

        assert result.success
        assert "gs://" in result.extra["gcs_url"]
        assert result.extra["uploaded"] is True

    def test_http_error_returns_failure(self, tmp_log):
        import requests as _req
        exc = self._make_executor()
        with patch("executor.requests") as mock_requests:
            mock_requests.post.side_effect = _req.exceptions.ConnectionError("refused")
            result = exc.execute({
                "type": "UPLOAD_GCS",
                "local_path": str(tmp_log),
            })
        assert not result.success
        assert "Upload failed" in result.message

    def test_gcs_unavailable_returns_not_uploaded(self, tmp_log):
        exc = self._make_executor()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "uploaded": False,
            "gcs_url": "",
            "object_name": "sessions/test-session/artifacts/test_out.log",
            "size_bytes": 64,
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("executor.requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            result = exc.execute({
                "type": "UPLOAD_GCS",
                "local_path": str(tmp_log),
            })

        assert not result.success
        assert result.extra["uploaded"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 5 · executor._deploy_cloud_run
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutorDeployCloudRun:

    def _make_executor(self):
        with patch("pyautogui.FAILSAFE", False):
            return ActionExecutor(server_url="http://localhost:8080", session_id="sess")

    def test_blocked_without_confirmed(self):
        exc = self._make_executor()
        result = exc.execute({
            "type": "DEPLOY_CLOUD_RUN",
            "service_name": "ui-navigator",
            "image": "gcr.io/proj/img:v1",
        })
        assert not result.success
        assert "confirmation" in result.message.lower()
        assert result.extra.get("confirmed") is False

    def test_blocked_missing_service_name(self):
        exc = self._make_executor()
        result = exc.execute({
            "type": "DEPLOY_CLOUD_RUN",
            "service_name": "",
            "image": "gcr.io/proj/img:v1",
            "confirmed": True,
        })
        assert not result.success
        assert "service_name" in result.message

    def test_blocked_missing_image(self):
        exc = self._make_executor()
        result = exc.execute({
            "type": "DEPLOY_CLOUD_RUN",
            "service_name": "svc",
            "image": "",
            "confirmed": True,
        })
        assert not result.success
        assert "image" in result.message

    def test_gcloud_command_constructed_correctly(self):
        from command_policy import CommandResult
        exc = self._make_executor()
        fake_result = CommandResult(
            command="gcloud run deploy svc --image=gcr.io/proj/img:v1 --region=us-central1 --platform=managed --project=my-proj",
            returncode=0,
            stdout="Service URL: https://svc-xxxxx-uc.a.run.app\n",
            stderr="",
            duration_ms=3500.0,
        )
        with patch("executor._policy_execute", return_value=fake_result) as mock_exec:
            result = exc.execute({
                "type": "DEPLOY_CLOUD_RUN",
                "service_name": "svc",
                "image": "gcr.io/proj/img:v1",
                "region": "us-central1",
                "project": "my-proj",
                "confirmed": True,
            })
        assert result.success
        cmd_used = mock_exec.call_args[0][0]
        assert "gcloud run deploy svc" in cmd_used
        assert "--image=gcr.io/proj/img:v1" in cmd_used
        assert "--region=us-central1" in cmd_used
        assert "--project=my-proj" in cmd_used
        assert "--platform=managed" in cmd_used

    def test_service_url_parsed_from_stdout(self):
        from command_policy import CommandResult
        exc = self._make_executor()
        fake_result = CommandResult(
            command="gcloud run deploy",
            returncode=0,
            stdout="Deploying container...\nService URL: https://my-svc-abc123-uc.a.run.app\nDone.\n",
            stderr="",
            duration_ms=4200.0,
        )
        with patch("executor._policy_execute", return_value=fake_result):
            result = exc.execute({
                "type": "DEPLOY_CLOUD_RUN",
                "service_name": "my-svc",
                "image": "gcr.io/proj/img:v1",
                "confirmed": True,
            })
        assert result.success
        assert result.extra["service_url"] == "https://my-svc-abc123-uc.a.run.app"

    def test_service_url_empty_when_not_in_output(self):
        from command_policy import CommandResult
        exc = self._make_executor()
        fake_result = CommandResult(
            command="gcloud run deploy",
            returncode=0,
            stdout="Deployed successfully\n",
            stderr="",
            duration_ms=2000.0,
        )
        with patch("executor._policy_execute", return_value=fake_result):
            result = exc.execute({
                "type": "DEPLOY_CLOUD_RUN",
                "service_name": "svc",
                "image": "gcr.io/proj/img:v1",
                "confirmed": True,
            })
        assert result.success
        assert result.extra["service_url"] == ""

    def test_nonzero_returncode_failure(self):
        from command_policy import CommandResult
        exc = self._make_executor()
        fake_result = CommandResult(
            command="gcloud run deploy",
            returncode=1,
            stdout="",
            stderr="ERROR: (gcloud.run.deploy) Image not found\n",
            duration_ms=1000.0,
        )
        with patch("executor._policy_execute", return_value=fake_result):
            result = exc.execute({
                "type": "DEPLOY_CLOUD_RUN",
                "service_name": "svc",
                "image": "gcr.io/proj/img:v1",
                "confirmed": True,
            })
        assert not result.success
        assert result.extra["returncode"] == 1

    def test_allow_unauthenticated_flag(self):
        from command_policy import CommandResult
        exc = self._make_executor()
        fake_result = CommandResult(
            command="gcloud run deploy",
            returncode=0,
            stdout="Service URL: https://svc.run.app\n",
            stderr="",
            duration_ms=3000.0,
        )
        with patch("executor._policy_execute", return_value=fake_result) as mock_exec:
            exc.execute({
                "type": "DEPLOY_CLOUD_RUN",
                "service_name": "svc",
                "image": "gcr.io/proj/img:v1",
                "allow_unauthenticated": True,
                "confirmed": True,
            })
        cmd_used = mock_exec.call_args[0][0]
        assert "--allow-unauthenticated" in cmd_used

    def test_no_allow_unauthenticated_by_default(self):
        from command_policy import CommandResult
        exc = self._make_executor()
        fake_result = CommandResult(
            command="gcloud run deploy",
            returncode=0,
            stdout="Service URL: https://svc.run.app\n",
            stderr="",
            duration_ms=3000.0,
        )
        with patch("executor._policy_execute", return_value=fake_result) as mock_exec:
            exc.execute({
                "type": "DEPLOY_CLOUD_RUN",
                "service_name": "svc",
                "image": "gcr.io/proj/img:v1",
                "confirmed": True,
            })
        cmd_used = mock_exec.call_args[0][0]
        assert "--allow-unauthenticated" not in cmd_used


# ─────────────────────────────────────────────────────────────────────────────
# 6 · executor — CONFIRM is not directly handled (dispatched to session layer)
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutorConfirmUnknown:

    def test_confirm_is_unknown_action_to_executor(self):
        exc = ActionExecutor(server_url="http://localhost:8080")
        result = exc.execute({"type": "CONFIRM", "message": "Deploy?"})
        # CONFIRM is NOT in the executor dispatch table — it's intercepted in session.py
        assert result.action_type == "CONFIRM"
        assert not result.success
        assert "Unknown action type" in result.message


# ─────────────────────────────────────────────────────────────────────────────
# 7 · session — CONFIRM flow + URL capture
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionConfirmFlow:
    """
    Test the session manager's CONFIRM gate and artifact URL tracking using
    a minimal stub that exercises _run_one_step without a real Qt event loop.
    """

    def _make_session(self):
        """Return a SessionManager with Qt signals stubbed out."""
        from session import SessionManager

        with patch("session.FrameCapturer") as MockCapturer:
            mock_frame = MagicMock()
            mock_frame.jpeg_bytes = b"fake_jpeg"
            mock_frame.width = 1280
            mock_frame.height = 720
            MockCapturer.return_value.capture_once.return_value = mock_frame

            mgr = SessionManager.__new__(SessionManager)
            mgr.server_url = "http://localhost:8080"
            mgr.task_goal = "test task"
            mgr.session_id = "test-session"
            mgr._api_key = "test-key"
            mgr._plan_first = False
            mgr.session_plan = None
            mgr._stop_requested = False
            mgr._step_id = 0
            mgr._last_gcs_url = ""
            mgr._last_cloud_run_url = ""

            import threading
            mgr._confirmation_event = threading.Event()
            mgr._confirmation_result = False

            # Stub signals so they don't need a real Qt app
            mgr.action_logged = MagicMock()
            mgr.action_logged.emit = MagicMock()
            mgr.status_changed = MagicMock()
            mgr.status_changed.emit = MagicMock()
            mgr.frame_ready = MagicMock()
            mgr.frame_ready.emit = MagicMock()
            mgr.session_ended = MagicMock()
            mgr.session_ended.emit = MagicMock()
            mgr.hand_off_requested = MagicMock()
            mgr.hand_off_requested.emit = MagicMock()
            mgr.task_completed = MagicMock()
            mgr.task_completed.emit = MagicMock()
            mgr.confirmation_required = MagicMock()
            mgr.confirmation_required.emit = MagicMock()

            # Stub capturer
            mgr._capturer = MockCapturer.return_value

            # Stub executor
            mgr._executor = MagicMock()
            mgr._executor.execute.return_value = ExecutionResult(
                action_type="WAIT", success=True, message="ok"
            )

            # Stub msleep
            mgr.msleep = MagicMock()

            return mgr

    def _call_run_one_step_with_actions(self, mgr, actions, perception=None):
        """Patch _send_frame to return a fixed actions list, then call _run_one_step."""
        if perception is None:
            perception = {
                "screen_summary": "Terminal open",
                "unexpected_modal": None,
                "elements": [],
            }
        response_data = {
            "perception": perception,
            "actions": actions,
            "expected": {"max_retries": 3, "on_failure": "HAND_OFF_TO_USER"},
        }
        with patch.object(mgr, "_send_frame", return_value=response_data):
            return mgr._run_one_step()

    def test_gcs_url_captured_from_upload_result(self):
        mgr = self._make_session()
        gcs_result = ExecutionResult(
            action_type="UPLOAD_GCS",
            success=True,
            message="Uploaded 100 bytes → gs://bucket/log.txt",
            extra={"gcs_url": "gs://bucket/log.txt", "uploaded": True},
        )
        mgr._executor.execute.side_effect = [
            gcs_result,
            ExecutionResult("VERIFY", True, "ok", skipped=True),
        ]
        actions = [
            {"type": "UPLOAD_GCS", "local_path": "/tmp/test_out.log"},
            {"type": "VERIFY", "method": "visual"},
        ]
        self._call_run_one_step_with_actions(mgr, actions)
        assert mgr._last_gcs_url == "gs://bucket/log.txt"

    def test_cloud_run_url_captured_from_deploy_result(self):
        mgr = self._make_session()
        deploy_result = ExecutionResult(
            action_type="DEPLOY_CLOUD_RUN",
            success=True,
            message="returncode=0",
            extra={"service_url": "https://svc-abc.run.app", "returncode": 0},
        )
        mgr._executor.execute.side_effect = [
            deploy_result,
            ExecutionResult("VERIFY", True, "ok", skipped=True),
        ]
        actions = [
            {"type": "DEPLOY_CLOUD_RUN", "service_name": "svc",
             "image": "gcr.io/proj/img:v1", "confirmed": True},
            {"type": "VERIFY", "method": "visual"},
        ]
        self._call_run_one_step_with_actions(mgr, actions)
        assert mgr._last_cloud_run_url == "https://svc-abc.run.app"

    def test_write_report_injected_with_captured_urls(self):
        mgr = self._make_session()
        mgr._last_gcs_url = "gs://bucket/log.txt"
        mgr._last_cloud_run_url = "https://svc.run.app"

        captured_actions = []

        def capture_execute(action):
            captured_actions.append(dict(action))
            return ExecutionResult(
                action_type=action.get("type", ""), success=True, message="ok"
            )

        mgr._executor.execute.side_effect = capture_execute
        actions = [
            {"type": "WRITE_REPORT", "log_path": "/tmp/test_out.log",
             "copy_to_clipboard": False},
        ]
        self._call_run_one_step_with_actions(mgr, actions)

        # The WRITE_REPORT action should have been called twice (once from the
        # loop injection, once directly) OR the captured call should have URLs
        write_calls = [a for a in captured_actions if a.get("type") == "WRITE_REPORT"]
        assert write_calls, "WRITE_REPORT should have been executed"
        last_write = write_calls[-1]
        assert last_write.get("gcs_log_url") == "gs://bucket/log.txt"
        assert last_write.get("cloud_run_url") == "https://svc.run.app"

    def test_confirm_approved_continues_execution(self):
        mgr = self._make_session()

        # Simulate user approving confirmation after a short delay
        def approve_after_delay():
            time.sleep(0.05)
            mgr.confirm_action(True)

        threading.Thread(target=approve_after_delay, daemon=True).start()

        wait_result = ExecutionResult("WAIT", True, "ok")
        mgr._executor.execute.return_value = wait_result

        actions = [
            {"type": "CONFIRM", "message": "Deploy to Cloud Run?",
             "action_ref": "DEPLOY_CLOUD_RUN"},
            {"type": "WAIT", "ms": 100},
        ]
        self._call_run_one_step_with_actions(mgr, actions)

        # Session should NOT have been stopped
        assert not mgr._stop_requested
        # confirmation_required signal should have been emitted
        mgr.confirmation_required.emit.assert_called_once_with("Deploy to Cloud Run?")

    def test_confirm_declined_stops_session(self):
        mgr = self._make_session()

        def decline_after_delay():
            time.sleep(0.05)
            mgr.confirm_action(False)

        threading.Thread(target=decline_after_delay, daemon=True).start()

        actions = [
            {"type": "CONFIRM", "message": "Deploy?"},
            {"type": "WAIT", "ms": 100},
        ]
        self._call_run_one_step_with_actions(mgr, actions)

        assert mgr._stop_requested
        # WAIT should never have been executed
        wait_calls = [
            call_args for call_args in mgr._executor.execute.call_args_list
            if call_args[0][0].get("type") == "WAIT"
        ]
        assert not wait_calls, "WAIT should not execute after a declined CONFIRM"


# ─────────────────────────────────────────────────────────────────────────────
# 8 · command_policy — new allowlisted patterns
# ─────────────────────────────────────────────────────────────────────────────

class TestCommandPolicyCloud:

    def test_gcloud_run_deploy_with_flags(self):
        cmd = (
            "gcloud run deploy ui-navigator "
            "--image=gcr.io/my-proj/ui-navigator:latest "
            "--region=us-central1 --project=my-proj --platform=managed"
        )
        result = check_command(cmd)
        assert result.allowed, f"Expected allowed but got blocked: {result.reason}"

    def test_gcloud_run_services_describe(self):
        result = check_command("gcloud run services describe ui-navigator --region=us-central1")
        assert result.allowed, f"Expected allowed but got: {result.reason}"

    def test_gcloud_run_services_list(self):
        result = check_command("gcloud run services list --region=us-central1")
        assert result.allowed, f"Expected allowed but got: {result.reason}"

    def test_gsutil_ls_allowed(self):
        result = check_command("gsutil ls gs://my-bucket/sessions/")
        assert result.allowed, f"Expected allowed but got: {result.reason}"

    def test_gcloud_delete_still_blocked(self):
        result = check_command("gcloud run services delete ui-navigator")
        assert not result.allowed, "gcloud delete must remain blocked"

    def test_gcloud_destroy_still_blocked(self):
        result = check_command("gcloud run deploy --destroy")
        assert not result.allowed, "gcloud deploy --destroy should be blocked (destroy is in blocklist)"
