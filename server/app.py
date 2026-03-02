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
            model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
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
) -> ActionResponse:
    """
    Translate a PerceptionOutput into an executable ActionResponse.

    Current strategy (Day 1–2 skeleton):
    - If an unexpected modal is detected → ABORT and hand off to user.
    - Otherwise → issue a WAIT + VERIFY so the client re-captures and sends
      a fresh frame, giving the full planning loop a chance to run on stable
      state.  (Real planning / action selection lives in `planner.py` — Day 6.)
    """
    actions: list = []

    if perception.unexpected_modal:
        # Unexpected dialog present — hand off immediately
        logger.warning(
            "Unexpected modal detected (session=%s step=%d): %s",
            metadata.session_id,
            metadata.step_id,
            perception.unexpected_modal,
        )
        actions.append(
            HandOffAction(
                summary=(
                    f"Unexpected modal/dialog detected: {perception.unexpected_modal}. "
                    "Human action required."
                ),
                reason="Modal pre-check triggered",
            )
        )
        expected = ActionExpected(
            must_see=[],
            timeout_ms=0,
            max_retries=0,
            on_failure=OnFailure.HAND_OFF_TO_USER,
        )
    elif not perception.elements and "unavailable" in perception.screen_summary:
        # Perception failed (fallback response from GeminiPerceptionClient)
        actions.append(
            HandOffAction(
                summary=perception.screen_summary,
                reason="Perception unavailable — cannot plan actions",
            )
        )
        expected = ActionExpected(
            must_see=[],
            timeout_ms=0,
            max_retries=0,
            on_failure=OnFailure.HAND_OFF_TO_USER,
        )
    else:
        # Nominal path: brief wait then a visual verification re-capture
        actions.append(
            WaitAction(ms=500, reason="Allow UI to settle before next observation")
        )
        actions.append(
            VerifyAction(
                method="visual",
                description=f"Visual precision check: confirm expected state after step {metadata.step_id}",
                reason="Mandatory VERIFY after every major step",
            )
        )
        expected = ActionExpected(
            must_see=[],
            timeout_ms=15000,
            max_retries=3,
            on_failure=OnFailure.HAND_OFF_TO_USER,
        )

    return ActionResponse(
        session_id=metadata.session_id,
        step_id=metadata.step_id,
        perception=perception,
        actions=[a.model_dump() for a in actions],
        expected=expected,
        raw_model_output=raw_model_output,
    )

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
    # ── Validate metadata ────────────────────────────────────────────────
    try:
        meta_dict = json.loads(metadata)
        frame_meta = FrameMetadata.model_validate(meta_dict)
    except (json.JSONDecodeError, Exception) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid metadata JSON: {exc}",
        ) from exc

    # ── Read JPEG bytes ─────────────────────────────────────────────────
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
            model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
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

    # ── Build and return action response ────────────────────────────────
    response = _build_action_response(frame_meta, perception, raw_output)

    logger.info(
        "Returning %d action(s) for session=%s step=%d",
        len(response.actions),
        frame_meta.session_id,
        frame_meta.step_id,
    )

    return JSONResponse(content=response.model_dump())
