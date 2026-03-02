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

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.2   # low temp for deterministic action selection

# How many UI element fallbacks to request.
_MAX_ELEMENTS = 3

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

def _fallback_perception(reason: str) -> PerceptionOutput:
    """Return a safe PerceptionOutput that signals the orchestrator to pause."""
    return PerceptionOutput(
        screen_summary=f"[Perception unavailable: {reason}]",
        unexpected_modal=None,
        elements=[],
        risks=[f"Perception failed: {reason}"],
        next_best_action="HAND_OFF_TO_USER — cannot proceed without valid perception",
    )


class GeminiPerceptionClient:
    """
    Wraps the Google GenAI SDK to provide structured UI perception.

    Parameters
    ----------
    api_key   : Gemini API key. Defaults to the ``GEMINI_API_KEY`` env var.
    model     : Gemini model name (default ``gemini-2.0-flash``).
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
