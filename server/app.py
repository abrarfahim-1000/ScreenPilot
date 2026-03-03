"""
app.py — Cloud Run FastAPI orchestrator for UI Navigator.

Endpoints:
    GET  /health               Liveness / readiness check
    POST /session/frame        Main loop: receive frame + metadata → return actions

The /session/frame endpoint:
  - Accepts a multipart form upload with:
      • frame    : JPEG image file (UploadFile)
      • metadata : JSON string of FrameMetadata fields
  - Calls GeminiPerceptionClient to get structured perception
  - Packages perception into an ActionResponse (action list + expected block)
  - Returns ActionResponse as JSON

Run locally:
    uvicorn app:app --reload --port 8080
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

import re as _re

from gemini import GeminiPerceptionClient
from schemas import (
    ActionExpected,
    ActionResponse,
    HandOffAction,
    HealthResponse,
    FrameMetadata,
    OnFailure,
    VerifyAction,
    WaitAction,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_gemini_client: GeminiPerceptionClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gemini_client
    startup_key = os.environ.get("GEMINI_API_KEY", "")
    if startup_key:
        _gemini_client = GeminiPerceptionClient(
            api_key=startup_key,
            model=os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"),
        )
        logger.info("GeminiPerceptionClient initialised (model=%s)", _gemini_client.model)
    else:
        _gemini_client = None
        logger.info(
            "No GEMINI_API_KEY env var — shared client skipped. "
            "Keys must be supplied per-request via X-Gemini-Api-Key header."
        )
    yield
    _gemini_client = None


app = FastAPI(
    title="UI Navigator Orchestrator",
    description=(
        "Cloud Run backend for the Local-to-Cloud UI Navigator agent. "
        "Receives desktop screen frames, calls Gemini for visual perception, "
        "and returns structured OS-level actions."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

def _build_action_response(
    metadata: FrameMetadata,
    perception,
    raw_model_output: str,
    planned_actions: list[dict] | None = None,
) -> ActionResponse:
    """
    Translate a PerceptionOutput and a planned action list into an
    executable ActionResponse.

    Priority
    --------
    1. Unexpected modal → always HAND_OFF_TO_USER regardless of plan.
    2. Perception failed → HAND_OFF_TO_USER.
    3. planned_actions provided by the Gemini planner → use them directly.
    4. Fallback (no planner result) → WAIT + VERIFY skeleton.
    """
    if perception.unexpected_modal:
        logger.warning(
            "Unexpected modal detected (session=%s step=%d): %s",
            metadata.session_id,
            metadata.step_id,
            perception.unexpected_modal,
        )
        return ActionResponse(
            session_id=metadata.session_id,
            step_id=metadata.step_id,
            perception=perception,
            actions=[
                HandOffAction(
                    summary=(
                        f"Unexpected modal/dialog detected: {perception.unexpected_modal}. "
                        "Human action required."
                    ),
                    reason="Modal pre-check triggered",
                ).model_dump()
            ],
            expected=ActionExpected(
                must_see=[],
                timeout_ms=0,
                max_retries=0,
                on_failure=OnFailure.HAND_OFF_TO_USER,
            ),
            raw_model_output=raw_model_output,
        )

    if not perception.elements and "unavailable" in perception.screen_summary:
        return ActionResponse(
            session_id=metadata.session_id,
            step_id=metadata.step_id,
            perception=perception,
            actions=[
                HandOffAction(
                    summary=perception.screen_summary,
                    reason="Perception unavailable — cannot plan actions",
                ).model_dump()
            ],
            expected=ActionExpected(
                must_see=[],
                timeout_ms=0,
                max_retries=0,
                on_failure=OnFailure.HAND_OFF_TO_USER,
            ),
            raw_model_output=raw_model_output,
        )

    if planned_actions:
        actions_out = planned_actions
    else:
        # Fallback: no planner result — re-observe on next cycle
        actions_out = [
            WaitAction(ms=500, reason="Allow UI to settle before next observation").model_dump(),
            VerifyAction(
                method="visual",
                description=f"Visual precision check: confirm expected state after step {metadata.step_id}",
                reason="Mandatory VERIFY after every major step",
            ).model_dump(),
        ]

    return ActionResponse(
        session_id=metadata.session_id,
        step_id=metadata.step_id,
        perception=perception,
        actions=actions_out,
        expected=ActionExpected(
            must_see=[],
            timeout_ms=15000,
            max_retries=3,
            on_failure=OnFailure.HAND_OFF_TO_USER,
        ),
        raw_model_output=raw_model_output,
    )

_URL_PATTERN = _re.compile(r'https?://[^\s\'"]+', _re.IGNORECASE)


def _extract_urls(text: str) -> list[str]:
    """Return all HTTP/HTTPS URLs found in *text*."""
    return _URL_PATTERN.findall(text)


def _actions_already_navigate(actions: list[dict], url: str) -> bool:
    """Return True if the action list already types *url* into a browser."""
    url_bare = url.rstrip('\n').rstrip('/')
    for a in actions:
        if a.get('type') == 'TYPE':
            typed = a.get('text', '').rstrip('\n').rstrip('/')
            if typed.lower() == url_bare.lower():
                return True
    return False


def _inject_browser_navigation(actions: list[dict], url: str) -> list[dict]:
    """
    Prepend a guaranteed browser-navigation sequence before the model's actions
    when the model failed to include one itself.

    Sequence injected:
        FOCUS_WINDOW(Chrome) → WAIT(800) → HOTKEY(ctrl+l) → WAIT(300)
        → TYPE(url\n) → WAIT(1500) → <remaining model actions>
    The trailing VERIFY from the model is preserved at the end.
    """
    nav = [
        {"type": "FOCUS_WINDOW", "title_contains": "Chrome",
         "reason": "Bring Chrome to foreground before navigation [injected]"},
        {"type": "WAIT", "ms": 800,
         "reason": "Wait for Chrome to receive keyboard focus [injected]"},
        {"type": "HOTKEY", "keys": ["ctrl", "l"],
         "reason": "Focus address bar [injected]"},
        {"type": "WAIT", "ms": 300,
         "reason": "Wait for address bar to be active [injected]"},
        {"type": "TYPE", "text": url.rstrip('\n') + '\n',
         "reason": f"Navigate to {url} [injected]"},
        {"type": "WAIT", "ms": 1500,
         "reason": "Wait for page to load [injected]"},
    ]
    # Keep any non-TYPE/HOTKEY/WAIT/FOCUS actions the model generated,
    # and always keep the trailing VERIFY.
    passthrough_types = {"VERIFY", "CLICK", "SCROLL", "HAND_OFF_TO_USER", "ABORT"}
    remainder = [a for a in actions if a.get('type') in passthrough_types]
    # Guarantee VERIFY at the end
    if not remainder or remainder[-1].get('type') != 'VERIFY':
        remainder.append({
            "type": "VERIFY", "method": "visual",
            "description": f"Visual precision check: confirm {url} loaded",
            "reason": "Mandatory VERIFY [injected]",
        })
    return nav + remainder


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health_check() -> HealthResponse:
    """Liveness / readiness probe for Cloud Run."""
    return HealthResponse(status="ok", version="0.1.0")


@app.post(
    "/session/frame",
    response_model=ActionResponse,
    status_code=status.HTTP_200_OK,
    tags=["session"],
    summary="Process a captured screen frame and return the next actions",
)
async def process_frame(
    request: Request,
    frame: UploadFile,
    metadata: Annotated[str, Form()],
) -> JSONResponse:
    """
    Receive a JPEG frame + JSON metadata from the desktop client.

    - **frame**: JPEG image (multipart file upload)
    - **metadata**: JSON string matching the ``FrameMetadata`` schema

    Returns an ``ActionResponse`` containing the perception result and an
    ordered list of actions the client should execute.
    """
    try:
        meta_dict = json.loads(metadata)
        frame_meta = FrameMetadata.model_validate(meta_dict)
    except (json.JSONDecodeError, Exception) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid metadata JSON: {exc}",
        ) from exc

    jpeg_bytes = await frame.read()
    if not jpeg_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Frame upload is empty",
        )
    if jpeg_bytes[:2] != b"\xff\xd8":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Frame must be a JPEG image (magic bytes FF D8 expected)",
        )

    logger.info(
        "Received frame: session=%s step=%d size=%d bytes hash=%s",
        frame_meta.session_id,
        frame_meta.step_id,
        len(jpeg_bytes),
        frame_meta.frame_hash,
    )

    # ── Resolve Gemini client ────────────────────────────────────────────
    # The desktop client forwards the user's API key as X-Gemini-Api-Key.
    # If present, create a lightweight per-request client so the server
    # never needs the key baked into its own environment.
    override_key = request.headers.get("X-Gemini-Api-Key", "").strip()
    if override_key:
        active_client = GeminiPerceptionClient(
            api_key=override_key,
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"),
        )
        logger.info("Using per-request API key (session=%s)", frame_meta.session_id)
    elif _gemini_client is not None:
        active_client = _gemini_client
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini client not initialised and no X-Gemini-Api-Key header provided",
        )

    perception, raw_output = active_client.perceive(
        jpeg_bytes=jpeg_bytes,
        session_id=frame_meta.session_id,
        step_id=frame_meta.step_id,
        task_goal=frame_meta.task_goal,
    )

    # ── Planning: ask Gemini for a concrete action list ─────────────────────
    # Skip if perception failed or modal detected (handled downstream).
    planned_actions: list[dict] | None = None
    perception_ok = (
        perception.unexpected_modal is None
        and not (not perception.elements and "unavailable" in perception.screen_summary)
    )
    if perception_ok and frame_meta.task_goal.strip():
        planned_actions, plan_raw = active_client.plan(
            perception=perception,
            session_id=frame_meta.session_id,
            step_id=frame_meta.step_id,
            task_goal=frame_meta.task_goal,
        )
        logger.info(
            "Planner returned %d action(s) for session=%s step=%d",
            len(planned_actions),
            frame_meta.session_id,
            frame_meta.step_id,
        )

        # ── Navigation injection guard ─────────────────────────────────
        # If the task contains a URL but the model did not emit a proper
        # navigation sequence, inject one unconditionally so the browser
        # always ends up at the right address regardless of model behaviour.
        urls_in_goal = _extract_urls(frame_meta.task_goal)
        if urls_in_goal:
            target_url = urls_in_goal[0]
            if not _actions_already_navigate(planned_actions, target_url):
                logger.info(
                    "Navigation injection: model missed URL %s — prepending sequence",
                    target_url,
                )
                planned_actions = _inject_browser_navigation(planned_actions, target_url)

    # ── Build and return action response ────────────────────────────────
    response = _build_action_response(frame_meta, perception, raw_output, planned_actions)

    logger.info(
        "Returning %d action(s) for session=%s step=%d",
        len(response.actions),
        frame_meta.session_id,
        frame_meta.step_id,
    )

    return JSONResponse(content=response.model_dump())
