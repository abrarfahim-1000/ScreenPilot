"""
gemini.py — Gemini multimodal perception client for UI Navigator.

Responsibilities:
- Build the structured perception prompt (including mandatory modal pre-check)
- Call Gemini via the Google GenAI SDK with JPEG frame + prompt
- Parse and validate the structured JSON response into a PerceptionOutput
- Return a safe fallback PerceptionOutput on any error so the orchestrator
  never crashes due to a model failure
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from google import genai
from google.genai import types
from pydantic import ValidationError

from schemas import ActionType, PerceptionOutput, UIElement

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_TEMPERATURE = 0.2   # low temp for deterministic action selection

# How many UI element fallbacks to request.
_MAX_ELEMENTS = 3

_PLANNING_PROMPT_TEMPLATE = """\
You are a UI Navigator action planner. Based on the current screen perception \
and task goal, decide the NEXT 1–6 actions to execute.

## CURRENT SCREEN PERCEPTION
{perception_json}

## TASK GOAL
{task_goal}

## SESSION
Session: {session_id}  |  Step: {step_id}

## RULES
1. Return a JSON array of action objects — nothing else (no markdown, no prose).
2. Maximum 6 actions (not counting the mandatory terminal VERIFY).
3. Always end the array with a VERIFY action (method="visual").
4. Use ONLY these action types:
   FOCUS_WINDOW, CLICK, DOUBLE_CLICK, RIGHT_CLICK, TYPE, HOTKEY,
   SCROLL, WAIT, VERIFY, ABORT, HAND_OFF_TO_USER
5. Field reference:
   {{"type":"FOCUS_WINDOW","title_contains":"...","reason":"..."}}
   {{"type":"CLICK","x":<int>,"y":<int>,"reason":"..."}}
   {{"type":"TYPE","text":"...","reason":"..."}}
   {{"type":"HOTKEY","keys":["ctrl","l"],"reason":"..."}}
   {{"type":"SCROLL","dx":0,"dy":-3,"reason":"..."}}
   {{"type":"WAIT","ms":1000,"reason":"..."}}
   {{"type":"VERIFY","method":"visual","description":"...","reason":"..."}}
   {{"type":"ABORT","reason":"..."}}
   {{"type":"HAND_OFF_TO_USER","summary":"...","reason":"..."}}
6. To click a UI element from perception: compute centre as
   x = bbox[0] + bbox[2]//2,  y = bbox[1] + bbox[3]//2.
7. **Opening / focusing a browser or app — ALWAYS follow this exact sequence:**
   a. FOCUS_WINDOW {{"title_contains":"Chrome"}} (or "Firefox", "Edge", etc.)
   b. WAIT {{"ms":800}}   ← mandatory: gives the OS time to grant keyboard focus
   c. HOTKEY {{"keys":["ctrl","l"]}}
   d. WAIT {{"ms":300}}   ← mandatory: address bar needs to be ready
   e. TYPE {{"text":"https://...\n"}}
   NEVER try to open a browser by clicking taskbar coordinates — use FOCUS_WINDOW.
   If the browser is not yet open, use HOTKEY to launch it (e.g. Win key search).
8. After ANY CLICK that opens or switches application windows (taskbar, dock,
   app icons), ALWAYS add WAIT(ms=800) immediately after the CLICK before any
   keyboard action.
9. Never include API keys, passwords, tokens, or any secrets in TYPE text.
10. If the task goal is already complete, output only a VERIFY action.
11. If you cannot safely proceed, output HAND_OFF_TO_USER.
"""

_PERCEPTION_PROMPT_TEMPLATE = """\
You are a UI Navigator agent. You are looking at a screenshot of a desktop UI.

## YOUR TASK
Analyse the current screen and return a JSON object with the following fields:

```json
{{
  "screen_summary": "<one-sentence description of what is visible>",
  "unexpected_modal": <null or "description of any unexpected dialog, popup, or overlay">,
  "elements": [
    {{
      "label": "<human-readable name of the UI element>",
      "hint": "<location hint, e.g. 'top-right', 'centre', 'bottom-left taskbar'>",
      "bbox": [x, y, width, height],
      "confidence": 0.0-1.0,
      "priority": 1
    }}
  ],
  "risks": ["<risk note if any>"],
  "next_best_action": "<short description of the most sensible next action given the task goal>"
}}
```

## RULES
1. **Modal pre-check (mandatory):** Before anything else, check whether an unexpected
   modal, dialog, alert, or overlay is visible. Set `unexpected_modal` to a short
   description if one is present, or `null` if the screen is clear.
2. Return at most {max_elements} elements, sorted by `priority` (1 = primary target).
3. `bbox` format is [x, y, width, height] in pixels. Use `null` if not detectable.
4. Keep `screen_summary` to one sentence.
5. Return **only** the JSON object — no markdown fences, no prose.

## CURRENT TASK GOAL
{task_goal}

## STEP
Session: {session_id}  |  Step: {step_id}
"""


def build_perception_prompt(
    session_id: str,
    step_id: int,
    task_goal: str,
    max_elements: int = _MAX_ELEMENTS,
) -> str:
    """
    Build the full text prompt for the perception call.

    Separated from the Gemini client so it can be unit-tested without any
    network dependency.
    """
    task_description = task_goal.strip() or "No specific goal provided — describe what you see."
    return _PERCEPTION_PROMPT_TEMPLATE.format(
        max_elements=max_elements,
        task_goal=task_description,
        session_id=session_id,
        step_id=step_id,
    )


def build_planning_prompt(
    session_id: str,
    step_id: int,
    task_goal: str,
    perception: "PerceptionOutput",
) -> str:
    """
    Build the planning prompt from a completed perception output.

    Text-only (no image needed); Gemini derives click coordinates from the
    bounding boxes already extracted during perception.
    """
    perception_json = json.dumps(perception.model_dump(), indent=2)
    task_description = task_goal.strip() or "No specific goal provided."
    return _PLANNING_PROMPT_TEMPLATE.format(
        perception_json=perception_json,
        task_goal=task_description,
        session_id=session_id,
        step_id=step_id,
    )

def _fallback_perception(reason: str) -> PerceptionOutput:
    """Return a safe PerceptionOutput that signals the orchestrator to pause."""
    return PerceptionOutput(
        screen_summary=f"[Perception unavailable: {reason}]",
        unexpected_modal=None,
        elements=[],
        risks=[f"Perception failed: {reason}"],
        next_best_action="HAND_OFF_TO_USER — cannot proceed without valid perception",
    )


def _fallback_plan(reason: str) -> list[dict]:
    """Return a safe action list when planning fails."""
    return [
        {
            "type": "HAND_OFF_TO_USER",
            "summary": f"Planning failed: {reason}",
            "reason": "Planning could not produce a valid action list",
        },
        {
            "type": "VERIFY",
            "method": "visual",
            "description": "Visual precision check: confirm screen state after planning failure",
            "reason": "Mandatory VERIFY",
        },
    ]


class GeminiPerceptionClient:
    """
    Wraps the Google GenAI SDK to provide structured UI perception.

    Parameters
    ----------
    api_key   : Gemini API key. Defaults to the ``GEMINI_API_KEY`` env var.
    model     : Gemini model name (default ``gemini-3-flash-preview``).
    temperature : Sampling temperature (low = more deterministic, default 0.2).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not resolved_key:
            logger.warning(
                "GEMINI_API_KEY not set — Gemini calls will fail until a key is supplied."
            )
        self._client = genai.Client(api_key=resolved_key)
        self.model = model
        self.temperature = temperature

    def perceive(
        self,
        jpeg_bytes: bytes,
        session_id: str,
        step_id: int,
        task_goal: str = "",
    ) -> tuple[PerceptionOutput, str]:
        """
        Send *jpeg_bytes* to Gemini and return structured perception output.

        Returns
        -------
        (perception_output, raw_model_text)
            ``raw_model_text`` is the unmodified string from the model (for
            audit logging).  On error it contains the error message.
        """
        prompt = build_perception_prompt(
            session_id=session_id,
            step_id=step_id,
            task_goal=task_goal,
        )

        try:
            raw_text = self._call_gemini(jpeg_bytes, prompt)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Gemini API call failed (session=%s step=%d): %s",
                session_id, step_id, exc,
            )
            return _fallback_perception(f"API error: {exc}"), str(exc)

        try:
            perception = self._parse_response(raw_text)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to parse Gemini response (session=%s step=%d): %s\nRaw: %s",
                session_id, step_id, exc, raw_text[:500],
            )
            return _fallback_perception(f"Parse error: {exc}"), raw_text

        return perception, raw_text

    def plan(
        self,
        perception: PerceptionOutput,
        session_id: str,
        step_id: int,
        task_goal: str = "",
    ) -> tuple[list[dict], str]:
        """
        Generate a concrete action list from a completed perception output.

        This is a text-only call (no image) — Gemini derives coordinates from
        the bounding boxes already extracted during perception.

        Returns
        -------
        (actions, raw_model_text)
            ``actions`` is a list of action dicts ready to be returned to the
            client.  On error it returns a safe HAND_OFF_TO_USER + VERIFY pair.
        """
        prompt = build_planning_prompt(
            session_id=session_id,
            step_id=step_id,
            task_goal=task_goal,
            perception=perception,
        )
        try:
            raw_text = self._call_text_gemini(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Gemini planning call failed (session=%s step=%d): %s",
                session_id, step_id, exc,
            )
            return _fallback_plan(f"API error: {exc}"), str(exc)

        try:
            actions = self._parse_plan_response(raw_text)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to parse plan response (session=%s step=%d): %s\nRaw: %s",
                session_id, step_id, exc, raw_text[:500],
            )
            return _fallback_plan(f"Parse error: {exc}"), raw_text

        return actions, raw_text

    def _call_gemini(self, jpeg_bytes: bytes, prompt: str) -> str:
        """
        Make the actual Gemini API call.

        Kept as a separate method so tests can patch just this one method
        without needing to mock the full SDK internals.
        """
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/jpeg",
                            data=jpeg_bytes,
                        )
                    ),
                    types.Part(text=prompt),
                ],
            )
        ]
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
        )
        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text or ""

    def _call_text_gemini(self, prompt: str) -> str:
        """
        Text-only Gemini call (no image). Used for the planning step.

        Kept as a separate method so tests can patch it independently.
        """
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
        )
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    @staticmethod
    def _parse_response(raw_text: str) -> PerceptionOutput:
        """
        Parse raw model text into a validated :class:`PerceptionOutput`.

        Strips accidental markdown fences before JSON parsing.
        Raises ``ValueError`` or ``ValidationError`` on bad input.
        """
        text = raw_text.strip()
        # Strip potential markdown code fences
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        data = json.loads(text)
        return PerceptionOutput.model_validate(data)

    # Allowlisted action types the planner may return.  Any type not in this
    # set is stripped before the response reaches the client.
    _ALLOWED_PLAN_TYPES: frozenset[str] = frozenset({
        "FOCUS_WINDOW", "CLICK", "DOUBLE_CLICK", "RIGHT_CLICK",
        "TYPE", "HOTKEY", "SCROLL", "WAIT", "VERIFY",
        "ABORT", "HAND_OFF_TO_USER",
    })

    @classmethod
    def _parse_plan_response(cls, raw_text: str) -> list[dict]:
        """
        Parse the planning model output into a validated list of action dicts.

        - Strips markdown fences.
        - Ensures the result is a non-empty JSON array.
        - Filters out any action whose ``type`` is not in the allowlist.
        - Guarantees at least one VERIFY action is present (appends one if absent).
        """
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")
        if not data:
            raise ValueError("Planning returned an empty action list")

        # Filter to allowlisted types only
        safe: list[dict] = [
            a for a in data
            if isinstance(a, dict)
            and str(a.get("type", "")).upper() in cls._ALLOWED_PLAN_TYPES
        ]
        if not safe:
            raise ValueError("All proposed actions were rejected by the allowlist")

        # Normalise type field to uppercase
        for a in safe:
            a["type"] = str(a["type"]).upper()

        # Ensure the last action is a VERIFY
        if safe[-1].get("type") != "VERIFY":
            safe.append({
                "type": "VERIFY",
                "method": "visual",
                "description": "Visual precision check: confirm expected state",
                "reason": "Mandatory VERIFY appended by orchestrator",
            })

        return safe
