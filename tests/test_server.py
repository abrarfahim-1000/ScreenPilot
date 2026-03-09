"""
tests/test_server.py — Tests for server/schemas.py, server/gemini.py, server/app.py

Run with:  pytest tests/test_server.py -v
"""

from __future__ import annotations

import io
import json
import sys
import os
import time
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from fastapi.testclient import TestClient
from pydantic import ValidationError

# ── path setup ────────────────────────────────────────────────────────────────
SERVER_DIR = os.path.join(os.path.dirname(__file__), "..", "server")
sys.path.insert(0, SERVER_DIR)

from schemas import (
    ActionExpected,
    ActionResponse,
    ActionType,
    ClickAction,
    FrameMetadata,
    HandOffAction,
    HealthResponse,
    HotkeyAction,
    OnFailure,
    PerceptionOutput,
    TypeAction,
    UIElement,
    VerifyAction,
    WaitAction,
)
from gemini import (
    GeminiPerceptionClient,
    _fallback_perception,
    build_perception_prompt,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_jpeg(width: int = 160, height: int = 120,
               color: tuple = (100, 149, 237)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def _make_frame_metadata(**overrides) -> dict:
    base = {
        "session_id": "test-session-001",
        "step_id": 1,
        "task_goal": "Run tests and deploy to Cloud Run",
        "frame_hash": "a" * 32,
        "timestamp": time.time(),
        "width": 1280,
        "height": 720,
        "monitor_index": 1,
    }
    base.update(overrides)
    return base


def _make_good_perception(with_modal: bool = False) -> PerceptionOutput:
    return PerceptionOutput(
        screen_summary="VS Code terminal open, tests passing.",
        unexpected_modal="Disk full warning dialog" if with_modal else None,
        elements=[
            UIElement(
                label="Run Tests button",
                hint="top-right",
                bbox=[1200, 50, 120, 40],
                confidence=0.88,
                priority=1,
            ),
            UIElement(
                label="Run Tests button (alt)",
                hint="toolbar",
                bbox=[1100, 50, 80, 30],
                confidence=0.55,
                priority=2,
            ),
        ],
        risks=[],
        next_best_action="Click the Run Tests button",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def good_perception() -> PerceptionOutput:
    return _make_good_perception()


@pytest.fixture
def modal_perception() -> PerceptionOutput:
    return _make_good_perception(with_modal=True)


@pytest.fixture
def test_client(good_perception):
    """
    FastAPI TestClient with GeminiPerceptionClient fully mocked.
    Yields (TestClient, mock_gemini_instance) so tests can override
    perceive() / plan() return values.

    Sets a dummy GEMINI_API_KEY env var so the lifespan initialises
    _gemini_client with the mock instance (avoids 503 in tests that
    don't supply an X-Gemini-Api-Key header).
    """
    default_plan = [
        {
            "type": "WAIT",
            "ms": 500,
            "reason": "Test default plan: wait",
        },
        {
            "type": "VERIFY",
            "method": "visual",
            "description": "Visual precision check: default test plan",
            "reason": "Mandatory VERIFY",
        },
    ]
    with patch("app.GeminiPerceptionClient") as MockClass:
        with patch("app.FirestoreSessionStore") as MockStore:
            with patch.dict(os.environ, {"GEMINI_API_KEY": "test-fake-api-key-for-tests"}):
                MockStore.return_value.available = False
                mock_instance = MockClass.return_value
                mock_instance.model = "gemini-2.5-flash"
                mock_instance.perceive.return_value = (good_perception, "raw_model_output_text")
                mock_instance.plan.return_value = (default_plan, "plan_raw_output_text")
                # Import app here so the patched class is in effect at import time
                import app as app_module
                with TestClient(app_module.app) as client:
                    yield client, mock_instance


# ─────────────────────────────────────────────────────────────────────────────
# TestFrameMetadata
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameMetadata:
    def test_valid_metadata(self):
        m = FrameMetadata(**_make_frame_metadata())
        assert m.session_id == "test-session-001"
        assert m.step_id == 1
        assert m.width == 1280

    def test_defaults_applied(self):
        m = FrameMetadata(
            session_id="s1",
            frame_hash="a" * 32,
            timestamp=time.time(),
            width=1280,
            height=720,
        )
        assert m.step_id == 0
        assert m.task_goal == ""
        assert m.monitor_index == 1

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            FrameMetadata(session_id="s1", timestamp=1.0, width=1280, height=720)
            # missing frame_hash

    def test_negative_width_rejected(self):
        with pytest.raises(ValidationError):
            FrameMetadata(**_make_frame_metadata(width=-1))

    def test_negative_step_id_rejected(self):
        with pytest.raises(ValidationError):
            FrameMetadata(**_make_frame_metadata(step_id=-1))


# ─────────────────────────────────────────────────────────────────────────────
# TestPerceptionOutput
# ─────────────────────────────────────────────────────────────────────────────

class TestPerceptionOutput:
    def test_valid_perception(self, good_perception):
        assert good_perception.unexpected_modal is None
        assert len(good_perception.elements) == 2
        assert good_perception.elements[0].priority == 1
        assert good_perception.elements[1].priority == 2

    def test_elements_default_empty(self):
        p = PerceptionOutput(screen_summary="blank screen")
        assert p.elements == []
        assert p.risks == []
        assert p.unexpected_modal is None

    def test_ui_element_confidence_bounds(self):
        with pytest.raises(ValidationError):
            UIElement(label="x", confidence=1.5, priority=1)
        with pytest.raises(ValidationError):
            UIElement(label="x", confidence=-0.1, priority=1)

    def test_ui_element_priority_minimum(self):
        with pytest.raises(ValidationError):
            UIElement(label="x", confidence=0.9, priority=0)


# ─────────────────────────────────────────────────────────────────────────────
# TestActionSchemas
# ─────────────────────────────────────────────────────────────────────────────

class TestActionSchemas:
    def test_click_action(self):
        a = ClickAction(x=100, y=200, reason="click the button")
        assert a.type == ActionType.CLICK
        assert a.x == 100

    def test_type_action(self):
        a = TypeAction(text="make test\n", reason="run tests")
        assert a.type == ActionType.TYPE
        assert a.text == "make test\n"

    def test_hotkey_action(self):
        a = HotkeyAction(keys=["ctrl", "c"])
        assert a.keys == ["ctrl", "c"]

    def test_verify_action_defaults(self):
        v = VerifyAction(reason="check state")
        assert v.type == ActionType.VERIFY
        assert v.method == "visual"
        assert v.path is None

    def test_verify_action_read_file(self):
        v = VerifyAction(method="read_file", path="/tmp/test_out.log", reason="check log")
        assert v.method == "read_file"
        assert v.path == "/tmp/test_out.log"

    def test_hand_off_action(self):
        h = HandOffAction(summary="Tried 3 times, still failing", reason="max retries")
        assert h.type == ActionType.HAND_OFF_TO_USER
        assert "3 times" in h.summary

    def test_wait_action_non_negative(self):
        with pytest.raises(ValidationError):
            WaitAction(ms=-1)


# ─────────────────────────────────────────────────────────────────────────────
# TestActionExpected
# ─────────────────────────────────────────────────────────────────────────────

class TestActionExpected:
    def test_defaults(self):
        e = ActionExpected()
        assert e.timeout_ms == 15000
        assert e.max_retries == 3
        assert e.on_failure == OnFailure.HAND_OFF_TO_USER
        assert e.must_see == []

    def test_custom_values(self):
        e = ActionExpected(
            must_see=["PASS", "exit code 0"],
            timeout_ms=5000,
            max_retries=1,
            on_failure=OnFailure.ABORT,
        )
        assert e.must_see == ["PASS", "exit code 0"]
        assert e.on_failure == OnFailure.ABORT


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildPerceptionPrompt
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPerceptionPrompt:
    def test_returns_string(self):
        prompt = build_perception_prompt("s1", 3, "Deploy to Cloud Run")
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_session_id(self):
        prompt = build_perception_prompt("my-session-xyz", 0, "test goal")
        assert "my-session-xyz" in prompt

    def test_contains_step_id(self):
        prompt = build_perception_prompt("s1", 42, "test goal")
        assert "42" in prompt

    def test_contains_task_goal(self):
        prompt = build_perception_prompt("s1", 0, "Run tests and upload logs")
        assert "Run tests and upload logs" in prompt

    def test_contains_modal_precheck_instruction(self):
        prompt = build_perception_prompt("s1", 0, "any goal")
        # The mandatory modal pre-check instruction must be present every time
        lower = prompt.lower()
        assert "modal" in lower or "dialog" in lower or "unexpected" in lower

    def test_empty_goal_replaced_with_fallback(self):
        prompt = build_perception_prompt("s1", 0, "")
        assert "No specific goal" in prompt

    def test_max_elements_appears_in_prompt(self):
        prompt = build_perception_prompt("s1", 0, "goal", max_elements=5)
        assert "5" in prompt

    def test_prompt_is_deterministic(self):
        p1 = build_perception_prompt("s1", 1, "goal A")
        p2 = build_perception_prompt("s1", 1, "goal A")
        assert p1 == p2


# ─────────────────────────────────────────────────────────────────────────────
# TestGeminiParseResponse
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiParseResponse:
    def _valid_json(self, **overrides) -> str:
        base = {
            "screen_summary": "Terminal open with test results.",
            "unexpected_modal": None,
            "elements": [
                {
                    "label": "Deploy button",
                    "hint": "top-right",
                    "bbox": [1510, 110, 140, 55],
                    "confidence": 0.84,
                    "priority": 1,
                }
            ],
            "risks": ["Potential destructive action: deployment"],
            "next_best_action": "Click Deploy button",
        }
        base.update(overrides)
        return json.dumps(base)

    def test_parses_valid_json(self):
        result = GeminiPerceptionClient._parse_response(self._valid_json())
        assert isinstance(result, PerceptionOutput)
        assert result.screen_summary == "Terminal open with test results."
        assert len(result.elements) == 1
        assert result.elements[0].confidence == 0.84

    def test_parses_null_modal(self):
        result = GeminiPerceptionClient._parse_response(self._valid_json())
        assert result.unexpected_modal is None

    def test_parses_string_modal(self):
        result = GeminiPerceptionClient._parse_response(
            self._valid_json(unexpected_modal="Disk full alert")
        )
        assert result.unexpected_modal == "Disk full alert"

    def test_strips_markdown_fences(self):
        raw = "```json\n" + self._valid_json() + "\n```"
        result = GeminiPerceptionClient._parse_response(raw)
        assert isinstance(result, PerceptionOutput)

    def test_strips_plain_fences(self):
        raw = "```\n" + self._valid_json() + "\n```"
        result = GeminiPerceptionClient._parse_response(raw)
        assert isinstance(result, PerceptionOutput)

    def test_raises_on_invalid_json(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            GeminiPerceptionClient._parse_response("not json at all")

    def test_raises_on_missing_required_field(self):
        data = {"unexpected_modal": None, "elements": []}  # missing screen_summary
        with pytest.raises((ValidationError, Exception)):
            GeminiPerceptionClient._parse_response(json.dumps(data))

    def test_empty_elements_list(self):
        result = GeminiPerceptionClient._parse_response(
            self._valid_json(elements=[])
        )
        assert result.elements == []

    def test_multiple_elements_parsed(self):
        elements = [
            {"label": "A", "hint": "top", "bbox": None, "confidence": 0.9, "priority": 1},
            {"label": "B", "hint": "bottom", "bbox": None, "confidence": 0.6, "priority": 2},
        ]
        result = GeminiPerceptionClient._parse_response(
            self._valid_json(elements=elements)
        )
        assert len(result.elements) == 2
        assert result.elements[1].label == "B"


# ─────────────────────────────────────────────────────────────────────────────
# TestGeminiPerceive (mocked _call_gemini)
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiPerceive:
    """Test the full perceive() flow by patching only _call_gemini."""

    def _make_client(self) -> GeminiPerceptionClient:
        # api_key irrelevant — we patch _call_gemini
        with patch("gemini.genai.Client"):
            return GeminiPerceptionClient(api_key="fake-key")

    def _good_raw_json(self) -> str:
        return json.dumps({
            "screen_summary": "Browser showing Cloud Run console.",
            "unexpected_modal": None,
            "elements": [
                {"label": "Deploy", "hint": "top", "bbox": [10, 20, 100, 40],
                 "confidence": 0.9, "priority": 1}
            ],
            "risks": [],
            "next_best_action": "Click Deploy",
        })

    def test_perceive_returns_perception_and_raw_text(self):
        client = self._make_client()
        raw = self._good_raw_json()
        with patch.object(client, "_call_gemini", return_value=raw):
            perception, raw_text = client.perceive(
                b"\xff\xd8dummy", "s1", 1, "deploy"
            )
        assert isinstance(perception, PerceptionOutput)
        assert raw_text == raw

    def test_perceive_calls_gemini_once(self):
        client = self._make_client()
        with patch.object(client, "_call_gemini", return_value=self._good_raw_json()) as mock_call:
            client.perceive(b"\xff\xd8dummy", "s1", 1, "deploy")
        mock_call.assert_called_once()

    def test_perceive_passes_correct_jpeg_bytes(self):
        client = self._make_client()
        jpeg = b"\xff\xd8some_jpeg_data"
        captured = {}
        def fake_call(jpeg_bytes, prompt):
            captured["jpeg"] = jpeg_bytes
            return self._good_raw_json()
        with patch.object(client, "_call_gemini", side_effect=fake_call):
            client.perceive(jpeg, "s1", 1, "goal")
        assert captured["jpeg"] == jpeg

    def test_perceive_includes_task_goal_in_prompt(self):
        client = self._make_client()
        captured_prompt = {}
        def fake_call(jpeg_bytes, prompt):
            captured_prompt["p"] = prompt
            return self._good_raw_json()
        with patch.object(client, "_call_gemini", side_effect=fake_call):
            client.perceive(b"\xff\xd8dummy", "s1", 1, "unique-task-goal-xyz")
        assert "unique-task-goal-xyz" in captured_prompt["p"]

    def test_perceive_returns_fallback_on_api_error(self):
        client = self._make_client()
        with patch.object(client, "_call_gemini", side_effect=RuntimeError("API down")):
            perception, raw_text = client.perceive(b"\xff\xd8dummy", "s1", 1, "goal")
        assert "unavailable" in perception.screen_summary.lower()
        assert "API error" in raw_text or "API down" in raw_text

    def test_perceive_returns_fallback_on_bad_json(self):
        client = self._make_client()
        with patch.object(client, "_call_gemini", return_value="not valid json"):
            perception, raw_text = client.perceive(b"\xff\xd8dummy", "s1", 1, "goal")
        assert "unavailable" in perception.screen_summary.lower()

    def test_perceive_returns_fallback_on_empty_response(self):
        client = self._make_client()
        with patch.object(client, "_call_gemini", return_value=""):
            perception, raw_text = client.perceive(b"\xff\xd8dummy", "s1", 1, "goal")
        assert "unavailable" in perception.screen_summary.lower()

    def test_perceive_detects_modal(self):
        client = self._make_client()
        raw = json.dumps({
            "screen_summary": "Terminal visible with a modal overlay.",
            "unexpected_modal": "Unsaved changes dialog",
            "elements": [],
            "risks": [],
            "next_best_action": "Dismiss modal",
        })
        with patch.object(client, "_call_gemini", return_value=raw):
            perception, _ = client.perceive(b"\xff\xd8dummy", "s1", 2, "goal")
        assert perception.unexpected_modal == "Unsaved changes dialog"


# ─────────────────────────────────────────────────────────────────────────────
# TestFallbackPerception
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackPerception:
    def test_returns_perception_output(self):
        result = _fallback_perception("test reason")
        assert isinstance(result, PerceptionOutput)

    def test_summary_contains_reason(self):
        result = _fallback_perception("network error")
        assert "network error" in result.screen_summary

    def test_next_best_action_suggests_hand_off(self):
        result = _fallback_perception("timeout")
        assert "HAND_OFF" in result.next_best_action.upper()

    def test_risks_populated(self):
        result = _fallback_perception("some failure")
        assert len(result.risks) >= 1

    def test_no_elements(self):
        result = _fallback_perception("any")
        assert result.elements == []


# ─────────────────────────────────────────────────────────────────────────────
# TestHealthEndpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, test_client):
        client, _ = test_client
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_ok_status(self, test_client):
        client, _ = test_client
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_returns_version(self, test_client):
        client, _ = test_client
        data = client.get("/health").json()
        assert "version" in data


# ─────────────────────────────────────────────────────────────────────────────
# TestProcessFrameEndpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessFrameEndpoint:
    """Integration tests for POST /session/frame."""

    def _post_frame(self, client, jpeg: bytes, meta: dict):
        return client.post(
            "/session/frame",
            data={"metadata": json.dumps(meta)},
            files={"frame": ("frame.jpg", io.BytesIO(jpeg), "image/jpeg")},
        )

    def test_returns_200_with_valid_payload(self, test_client):
        client, _ = test_client
        resp = self._post_frame(client, _make_jpeg(), _make_frame_metadata())
        assert resp.status_code == 200

    def test_response_contains_session_id(self, test_client):
        client, _ = test_client
        meta = _make_frame_metadata(session_id="my-session-42")
        resp = self._post_frame(client, _make_jpeg(), meta)
        assert resp.json()["session_id"] == "my-session-42"

    def test_response_contains_step_id(self, test_client):
        client, _ = test_client
        meta = _make_frame_metadata(step_id=7)
        resp = self._post_frame(client, _make_jpeg(), meta)
        assert resp.json()["step_id"] == 7

    def test_response_contains_actions(self, test_client):
        client, _ = test_client
        resp = self._post_frame(client, _make_jpeg(), _make_frame_metadata())
        data = resp.json()
        assert "actions" in data
        assert isinstance(data["actions"], list)

    def test_response_contains_perception(self, test_client):
        client, _ = test_client
        resp = self._post_frame(client, _make_jpeg(), _make_frame_metadata())
        data = resp.json()
        assert "perception" in data
        assert "screen_summary" in data["perception"]

    def test_nominal_path_includes_verify_action(self, test_client):
        """No modal → response must contain a VERIFY action (visual precision check)."""
        client, _ = test_client
        resp = self._post_frame(client, _make_jpeg(), _make_frame_metadata())
        action_types = [a["type"] for a in resp.json()["actions"]]
        assert "VERIFY" in action_types

    def test_response_contains_expected_block(self, test_client):
        client, _ = test_client
        resp = self._post_frame(client, _make_jpeg(), _make_frame_metadata())
        data = resp.json()
        assert "expected" in data
        assert "max_retries" in data["expected"]
        assert "on_failure" in data["expected"]

    def test_modal_detection_triggers_handoff(self, test_client):
        """When Gemini reports an unexpected modal, response must contain HAND_OFF_TO_USER."""
        client, mock_gemini = test_client
        mock_gemini.perceive.return_value = (
            _make_good_perception(with_modal=True),
            "raw_output_with_modal",
        )
        resp = self._post_frame(client, _make_jpeg(), _make_frame_metadata())
        action_types = [a["type"] for a in resp.json()["actions"]]
        assert "HAND_OFF_TO_USER" in action_types

    def test_gemini_called_with_correct_session_id(self, test_client):
        client, mock_gemini = test_client
        meta = _make_frame_metadata(session_id="verify-session-id")
        self._post_frame(client, _make_jpeg(), meta)
        call_kwargs = mock_gemini.perceive.call_args
        assert call_kwargs.kwargs.get("session_id") == "verify-session-id" or \
               call_kwargs.args[1] == "verify-session-id"

    def test_gemini_called_with_correct_step_id(self, test_client):
        client, mock_gemini = test_client
        meta = _make_frame_metadata(step_id=99)
        self._post_frame(client, _make_jpeg(), meta)
        call_kwargs = mock_gemini.perceive.call_args
        assert 99 in call_kwargs.args or call_kwargs.kwargs.get("step_id") == 99

    def test_missing_frame_returns_422(self, test_client):
        client, _ = test_client
        resp = client.post(
            "/session/frame",
            data={"metadata": json.dumps(_make_frame_metadata())},
            # no 'frame' file
        )
        assert resp.status_code == 422

    def test_missing_metadata_returns_422(self, test_client):
        client, _ = test_client
        resp = client.post(
            "/session/frame",
            files={"frame": ("frame.jpg", io.BytesIO(_make_jpeg()), "image/jpeg")},
            # no metadata
        )
        assert resp.status_code == 422

    def test_invalid_metadata_json_returns_422(self, test_client):
        client, _ = test_client
        resp = client.post(
            "/session/frame",
            data={"metadata": "not-valid-json"},
            files={"frame": ("frame.jpg", io.BytesIO(_make_jpeg()), "image/jpeg")},
        )
        assert resp.status_code == 422

    def test_non_jpeg_file_returns_415(self, test_client):
        client, _ = test_client
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # PNG magic bytes
        resp = self._post_frame(client, png_bytes, _make_frame_metadata())
        assert resp.status_code == 415

    def test_raw_model_output_included_in_response(self, test_client):
        client, mock_gemini = test_client
        mock_gemini.perceive.return_value = (
            _make_good_perception(),
            "special_raw_output_marker",
        )
        resp = self._post_frame(client, _make_jpeg(), _make_frame_metadata())
        assert resp.json()["raw_model_output"] == "special_raw_output_marker"

    def test_perception_fallback_triggers_handoff(self, test_client):
        """When Gemini perception is unavailable, return HAND_OFF_TO_USER."""
        client, mock_gemini = test_client
        mock_gemini.perceive.return_value = (
            _fallback_perception("simulated API failure"),
            "error text",
        )
        resp = self._post_frame(client, _make_jpeg(), _make_frame_metadata())
        action_types = [a["type"] for a in resp.json()["actions"]]
        assert "HAND_OFF_TO_USER" in action_types
