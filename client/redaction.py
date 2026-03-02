"""
redaction.py — Secret redaction for UI Navigator Desktop Client.

Responsibilities:
- Scan TYPE action text for secrets before transmission or execution.
- Replace detected secrets with safe placeholders so they never leave the client
  unredacted.
- Blur / mask rectangular regions of a captured JPEG frame before it is sent to
  the cloud (e.g. regions identified by the policy engine as containing
  credentials).

Usage
-----
    from redaction import redact_text, is_safe_to_type, mask_frame_regions

    # Text path (called before every TYPE action)
    safe_text, matches = redact_text("export API_KEY=ABCDE12345FGHIJ67890")
    if matches:
        logging.warning("Secrets redacted: %s", [m.pattern_name for m in matches])
    # → safe_text == "export API_KEY=[REDACTED:long_uppercase_token]"

    # Frame path (called before transmitting a JPEG frame to the cloud)
    masked_jpeg = mask_frame_regions(jpeg_bytes, regions=[(100, 200, 300, 40)])
"""

import io
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Secret patterns
# ---------------------------------------------------------------------------

# Each entry: (pattern_name, compiled_regex)
# Order matters — more specific patterns first so they are labelled accurately.
_RAW_PATTERNS: list[tuple[str, str]] = [
    # PEM / private key blocks
    ("pem_block",       r"-----BEGIN [A-Z ]+-----[\s\S]*?-----END [A-Z ]+-----"),
    # JWT: three base64url segments separated by dots (header.payload.sig)
    ("jwt_token",       r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    # Bearer / Authorization header values
    ("bearer_token",    r"(?:Bearer|Token)\s+[A-Za-z0-9._~+/=-]{16,}"),
    # AWS access key IDs
    ("aws_access_key",  r"AKIA[0-9A-Z]{16}"),
    # GCP / generic service-account JSON key fields (private_key_id / client_id style)
    ("gcp_key_field",   r'"(?:private_key_id|client_id|client_email)"\s*:\s*"[^"]+"'),
    # Generic long-uppercase-alphanumeric tokens (API keys, GCP API keys, etc.)
    # Must be at least 20 chars of [A-Z0-9] to avoid matching normal words.
    ("long_uppercase_token", r"[A-Z0-9_-]{20,}"),
    # Hex secrets (MD5 / SHA digests used as tokens, 32+ hex chars)
    ("hex_secret",      r"\b[0-9a-f]{32,}\b"),
    # Password / secret assignments in shell / env files
    ("password_assignment",
     r"(?i)(?:password|passwd|secret|token|apikey|api_key)\s*[=:]\s*\S{6,}"),
]

SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    (name, re.compile(pattern, re.MULTILINE))
    for name, pattern in _RAW_PATTERNS
]

# Placeholder template; keeps the pattern name so logs are informative.
_PLACEHOLDER = "[REDACTED:{name}]"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RedactionMatch:
    """A single detected secret within a piece of text."""
    pattern_name: str      # which pattern triggered
    start: int             # start index in original text
    end: int               # end index in original text
    original_value: str    # the matched text (NOT logged externally; keep in memory only)
    placeholder: str       # the replacement string used in redacted output


# ---------------------------------------------------------------------------
# Text redaction
# ---------------------------------------------------------------------------

def scan_text(text: str) -> list[RedactionMatch]:
    """
    Scan *text* for all known secret patterns.

    Returns a list of :class:`RedactionMatch` objects sorted by start position.
    Overlapping matches are deduplicated (first match wins by start position).
    """
    matches: list[RedactionMatch] = []
    seen_ranges: list[tuple[int, int]] = []

    for pattern_name, regex in SECRET_PATTERNS:
        for m in regex.finditer(text):
            start, end = m.start(), m.end()
            # Skip if this range overlaps a previously recorded match
            if any(s <= start < e or s < end <= e for s, e in seen_ranges):
                continue
            placeholder = _PLACEHOLDER.format(name=pattern_name)
            matches.append(RedactionMatch(
                pattern_name=pattern_name,
                start=start,
                end=end,
                original_value=m.group(),
                placeholder=placeholder,
            ))
            seen_ranges.append((start, end))

    matches.sort(key=lambda m: m.start)
    return matches


def redact_text(text: str) -> tuple[str, list[RedactionMatch]]:
    """
    Replace every detected secret in *text* with a safe placeholder.

    Returns:
        (redacted_text, list_of_matches)

    If no secrets are found the returned string equals the input and the list
    is empty.
    """
    matches = scan_text(text)
    if not matches:
        return text, []

    # Rebuild the string by splicing in placeholders (work right-to-left to
    # preserve earlier indices while modifying later positions).
    result = list(text)
    for match in reversed(matches):
        result[match.start:match.end] = list(match.placeholder)

    return "".join(result), matches


def is_safe_to_type(text: str) -> bool:
    """
    Return True if *text* contains no detected secrets.

    This is a quick guard used by the action executor before every TYPE action.
    """
    return len(scan_text(text)) == 0


# ---------------------------------------------------------------------------
# Frame masking
# ---------------------------------------------------------------------------

# Blur radius applied to masked regions. Higher = more opaque.
_BLUR_RADIUS: int = 20


def mask_frame_regions(
    jpeg_bytes: bytes,
    regions: list[tuple[int, int, int, int]],
    blur_radius: int = _BLUR_RADIUS,
    fill_color: Optional[tuple[int, int, int]] = None,
) -> bytes:
    """
    Apply a heavy Gaussian blur (or solid fill) to rectangular *regions* of a
    JPEG frame, then re-encode as JPEG.

    Parameters
    ----------
    jpeg_bytes  : raw JPEG bytes of the captured frame
    regions     : list of (x, y, width, height) rectangles to mask, in pixels
    blur_radius : Gaussian blur radius (higher = more obscured, default 20)
    fill_color  : if provided, fill the region with this solid RGB colour
                  instead of blurring (e.g. ``(0, 0, 0)`` for black)

    Returns
    -------
    JPEG bytes of the masked frame (same quality as input, RGB).
    """
    if not regions:
        return jpeg_bytes

    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    img_w, img_h = img.size

    for (rx, ry, rw, rh) in regions:
        # Clamp to image bounds
        x1 = max(0, rx)
        y1 = max(0, ry)
        x2 = min(img_w, rx + rw)
        y2 = min(img_h, ry + rh)
        if x2 <= x1 or y2 <= y1:
            continue  # degenerate / out-of-bounds region

        if fill_color is not None:
            # Solid colour fill
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], fill=fill_color)
        else:
            # Gaussian blur of the region
            region_crop = img.crop((x1, y1, x2, y2))
            # Apply box blur multiple times for a strong pixelation effect, then
            # finish with Gaussian blur so edges blend smoothly.
            blurred = region_crop.filter(ImageFilter.BoxBlur(blur_radius))
            blurred = blurred.filter(ImageFilter.GaussianBlur(radius=blur_radius // 2))
            img.paste(blurred, (x1, y1))

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=70, optimize=True)
    return out.getvalue()


def redact_frame(
    jpeg_bytes: bytes,
    regions: list[tuple[int, int, int, int]],
    *,
    blur_radius: int = _BLUR_RADIUS,
    fill_color: Optional[tuple[int, int, int]] = None,
) -> tuple[bytes, bool]:
    """
    Convenience wrapper: applies :func:`mask_frame_regions` only when *regions*
    is non-empty.

    Returns
    -------
    (masked_jpeg_bytes, was_modified)
    """
    if not regions:
        return jpeg_bytes, False
    return mask_frame_regions(jpeg_bytes, regions, blur_radius=blur_radius,
                              fill_color=fill_color), True
