"""
schemas.py — Shared Pydantic models for UI Navigator Cloud Orchestrator.

Covers:
- Inbound frame payload (client → server)
- Perception output (Gemini → server)
- Action schema (server → client), including VERIFY as a first-class action type
- Action response envelope with max_retries / on_failure fields
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

class FrameMetadata(BaseModel):
    """Metadata accompanying each captured frame upload."""
    session_id: str = Field(..., description="Unique session identifier")
    step_id: int = Field(0, ge=0, description="Monotonically increasing step counter")
    task_goal: str = Field(
        "",
        description="High-level instruction the agent is trying to accomplish",
    )
    frame_hash: str = Field(..., description="MD5 hex of the JPEG bytes (diff key)")
    timestamp: float = Field(..., description="time.time() on the client when captured")
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    monitor_index: int = Field(1, ge=0)

class UIElement(BaseModel):
    """A detected UI element in the current frame."""
    label: str
    hint: str = Field("", description="Human-readable location hint, e.g. 'top-right'")
    bbox: Optional[list[int]] = Field(
        None,
        description="[x, y, w, h] bounding box in pixels, if detectable",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    priority: int = Field(1, ge=1, description="1 = primary target, 2+ = fallback")


class PerceptionOutput(BaseModel):
    """Structured perception result returned by the Gemini multimodal call."""
    screen_summary: str = Field(
        ...,
        description="Concise natural-language description of the current screen state",
    )
    unexpected_modal: Optional[str] = Field(
        None,
        description="Description of any unexpected dialog/modal, or null if none",
    )
    elements: list[UIElement] = Field(
        default_factory=list,
        description="Priority-ranked list of actionable UI elements (primary + fallbacks)",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Risk notes (e.g. 'Potential destructive action: deployment')",
    )
    next_best_action: str = Field(
        "",
        description="Model's suggested next action as a short free-text hint",
    )

class ActionType(str, Enum):
    FOCUS_WINDOW = "FOCUS_WINDOW"
    CLICK = "CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    TYPE = "TYPE"
    HOTKEY = "HOTKEY"
    SCROLL = "SCROLL"
    WAIT = "WAIT"
    DRAG = "DRAG"
    COPY = "COPY"
    PASTE = "PASTE"
    VERIFY = "VERIFY"   # ← first-class action, not a side effect
    ABORT = "ABORT"
    HAND_OFF_TO_USER = "HAND_OFF_TO_USER"


# ── Individual action union ──────────────────────────────────────────────────

class FocusWindowAction(BaseModel):
    type: Literal[ActionType.FOCUS_WINDOW] = ActionType.FOCUS_WINDOW
    title_contains: str
    reason: str = ""


class ClickAction(BaseModel):
    type: Literal[ActionType.CLICK] = ActionType.CLICK
    x: int
    y: int
    reason: str = ""


class DoubleClickAction(BaseModel):
    type: Literal[ActionType.DOUBLE_CLICK] = ActionType.DOUBLE_CLICK
    x: int
    y: int
    reason: str = ""


class RightClickAction(BaseModel):
    type: Literal[ActionType.RIGHT_CLICK] = ActionType.RIGHT_CLICK
    x: int
    y: int
    reason: str = ""


class TypeAction(BaseModel):
    type: Literal[ActionType.TYPE] = ActionType.TYPE
    text: str
    reason: str = ""


class HotkeyAction(BaseModel):
    type: Literal[ActionType.HOTKEY] = ActionType.HOTKEY
    keys: list[str]
    reason: str = ""


class ScrollAction(BaseModel):
    type: Literal[ActionType.SCROLL] = ActionType.SCROLL
    dx: int = 0
    dy: int = 0
    reason: str = ""


class WaitAction(BaseModel):
    type: Literal[ActionType.WAIT] = ActionType.WAIT
    ms: int = Field(..., ge=0)
    reason: str = ""


class DragAction(BaseModel):
    type: Literal[ActionType.DRAG] = ActionType.DRAG
    from_x: int = Field(..., alias="from_x")
    from_y: int = Field(..., alias="from_y")
    to_x: int = Field(..., alias="to_x")
    to_y: int = Field(..., alias="to_y")
    reason: str = ""

    model_config = {"populate_by_name": True}


class CopyAction(BaseModel):
    type: Literal[ActionType.COPY] = ActionType.COPY
    reason: str = ""


class PasteAction(BaseModel):
    type: Literal[ActionType.PASTE] = ActionType.PASTE
    reason: str = ""


class VerifyAction(BaseModel):
    """
    VERIFY is a first-class action type.
    It is logged, auditable, and appears explicitly in every action list
    after major steps.
    """
    type: Literal[ActionType.VERIFY] = ActionType.VERIFY
    method: Literal["visual", "read_file", "check_url"] = "visual"
    path: Optional[str] = Field(None, description="File path for read_file method")
    url: Optional[str] = Field(None, description="URL for check_url method")
    description: str = Field(
        "",
        description="Human-readable description of what to verify (visual precision check)",
    )
    reason: str = ""


class AbortAction(BaseModel):
    type: Literal[ActionType.ABORT] = ActionType.ABORT
    reason: str = ""


class HandOffAction(BaseModel):
    type: Literal[ActionType.HAND_OFF_TO_USER] = ActionType.HAND_OFF_TO_USER
    summary: str = Field("", description="Summary of what was attempted before handing off")
    reason: str = ""


# Discriminated union of all action types
ActionItem = Union[
    FocusWindowAction,
    ClickAction,
    DoubleClickAction,
    RightClickAction,
    TypeAction,
    HotkeyAction,
    ScrollAction,
    WaitAction,
    DragAction,
    CopyAction,
    PasteAction,
    VerifyAction,
    AbortAction,
    HandOffAction,
]


class OnFailure(str, Enum):
    RETRY = "RETRY"
    HAND_OFF_TO_USER = "HAND_OFF_TO_USER"
    ABORT = "ABORT"


class ActionExpected(BaseModel):
    """Expected post-step state + retry policy."""
    must_see: list[str] = Field(
        default_factory=list,
        description="Strings/phrases that should be visible/present after the step",
    )
    timeout_ms: int = Field(15000, ge=0)
    max_retries: int = Field(3, ge=0)
    on_failure: OnFailure = OnFailure.HAND_OFF_TO_USER

class ActionResponse(BaseModel):
    """
    Complete response sent from the Cloud Run orchestrator back to the
    desktop client after processing a frame.
    """
    session_id: str
    step_id: int
    perception: PerceptionOutput
    actions: list[Any] = Field(
        default_factory=list,
        description="Ordered list of actions for the client to execute (≤ 6 per step)",
        max_length=6,
    )
    expected: ActionExpected = Field(default_factory=ActionExpected)
    raw_model_output: Optional[str] = Field(
        None,
        description="Raw Gemini response text stored for audit log",
    )

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
