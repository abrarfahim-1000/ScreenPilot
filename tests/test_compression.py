"""
tests/test_compression.py — Dedicated tests for the frame-compression feature.

Verifies:
  1. Output is a valid JPEG.
  2. JPEG quality is 70 (not a higher quality — bytes must be smaller than q=100).
  3. Resolution cap: any frame larger than 1280×720 is downscaled to fit.
  4. Resolution floor: images already within 1280×720 are NOT upscaled.
  5. Aspect ratio is preserved after downscaling.
  6. Compressed size is meaningfully smaller than the raw uncompressed size.
  7. CapturedFrame objects produced by FrameCapturer.capture_once() honour
     the same quality and resolution constraints end-to-end.

Run with:  pytest tests/test_compression.py -v
"""

import io
import struct
import sys
import os
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# ── resolve import path ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "client"))

from capture import (
    CapturedFrame,
    FrameCapturer,
    _compress_frame,
    JPEG_QUALITY,
    MAX_HEIGHT,
    MAX_WIDTH,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _solid_image(width: int, height: int, color=(120, 80, 200)) -> Image.Image:
    """Return a solid-colour RGB image (avoids JPEG artefact variance)."""
    return Image.new("RGB", (width, height), color=color)


def _noise_image(width: int, height: int) -> Image.Image:
    """Return a high-entropy image (random-ish pixels) for size comparisons."""
    import random
    random.seed(42)
    pixels = bytes(random.randint(0, 255) for _ in range(width * height * 3))
    return Image.frombytes("RGB", (width, height), pixels)


def _jpeg_encode(img: Image.Image, quality: int) -> bytes:
    """Encode a PIL image to JPEG bytes at the given quality."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _make_mock_sct(width: int = 1920, height: int = 1080):
    """Return a mock mss screen-capture context for a given resolution."""
    pil = _solid_image(width, height, color=(0, 128, 255))
    pil_rgba = pil.convert("RGBA")
    raw_bytes = pil_rgba.tobytes()

    mock_grab = MagicMock()
    mock_grab.size = (width, height)
    mock_grab.bgra = raw_bytes

    mock_sct = MagicMock()
    mock_sct.__enter__ = MagicMock(return_value=mock_sct)
    mock_sct.__exit__ = MagicMock(return_value=False)
    mock_sct.monitors = [
        {},
        {"left": 0, "top": 0, "width": width, "height": height},
    ]
    mock_sct.grab = MagicMock(return_value=mock_grab)
    return mock_sct


# ─────────────────────────────────────────────────────────────────────────────
# 1 · JPEG validity
# ─────────────────────────────────────────────────────────────────────────────

class TestJpegValidity:
    """Output bytes must be a well-formed JPEG regardless of input size."""

    def test_starts_with_jpeg_magic_bytes(self):
        img = _solid_image(800, 600)
        jpeg_bytes, _, _ = _compress_frame(img)
        # JPEG SOI marker
        assert jpeg_bytes[:2] == b"\xff\xd8", "First two bytes must be JPEG SOI marker"

    def test_ends_with_jpeg_eoi_marker(self):
        img = _solid_image(800, 600)
        jpeg_bytes, _, _ = _compress_frame(img)
        # JPEG EOI marker
        assert jpeg_bytes[-2:] == b"\xff\xd9", "Last two bytes must be JPEG EOI marker"

    def test_pil_can_decode_output(self):
        """PIL must be able to open the compressed bytes as a JPEG."""
        img = _solid_image(1920, 1080)
        jpeg_bytes, _, _ = _compress_frame(img)
        decoded = Image.open(io.BytesIO(jpeg_bytes))
        assert decoded.format == "JPEG"

    def test_decoded_image_has_correct_dimensions(self):
        """Dimensions reported by _compress_frame must match the decoded JPEG."""
        img = _solid_image(1920, 1080)
        jpeg_bytes, reported_w, reported_h = _compress_frame(img)
        decoded = Image.open(io.BytesIO(jpeg_bytes))
        assert decoded.width == reported_w
        assert decoded.height == reported_h


# ─────────────────────────────────────────────────────────────────────────────
# 2 · JPEG quality == 70
# ─────────────────────────────────────────────────────────────────────────────

class TestJpegQuality:
    """
    JPEG quality affects file size directly.  A q=70 encode of the same image
    must produce fewer bytes than a q=95 encode and more bytes than a q=20
    encode.  We use a noise image to amplify size differences.
    """

    def test_quality_constant_is_70(self):
        assert JPEG_QUALITY == 70, f"JPEG_QUALITY should be 70, got {JPEG_QUALITY}"

    def test_output_smaller_than_high_quality(self):
        """q=70 output must be smaller than q=95 output of the same image."""
        img = _noise_image(640, 480)
        compressed_q70, _, _ = _compress_frame(img)
        high_quality_bytes = _jpeg_encode(img, quality=95)
        assert len(compressed_q70) < len(high_quality_bytes), (
            f"q=70 ({len(compressed_q70)} B) should be smaller than "
            f"q=95 ({len(high_quality_bytes)} B)"
        )

    def test_output_larger_than_low_quality(self):
        """q=70 output must be larger than q=20 output of the same image."""
        img = _noise_image(640, 480)
        compressed_q70, _, _ = _compress_frame(img)
        low_quality_bytes = _jpeg_encode(img, quality=20)
        assert len(compressed_q70) > len(low_quality_bytes), (
            f"q=70 ({len(compressed_q70)} B) should be larger than "
            f"q=20 ({len(low_quality_bytes)} B)"
        )

    def test_output_size_close_to_reference_q70(self):
        """
        The output size must be within ±15% of an independent q=70 encode of
        the same (already-within-bounds) image, confirming the quality setting.
        """
        img = _noise_image(640, 480)           # smaller than MAX — no resize
        compressed, _, _ = _compress_frame(img)
        reference = _jpeg_encode(img, quality=70)
        ratio = len(compressed) / len(reference)
        assert 0.85 <= ratio <= 1.15, (
            f"Size ratio {ratio:.3f} is outside the ±15% band around a "
            f"reference q=70 encode (compressed={len(compressed)} B, "
            f"reference={len(reference)} B)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3 · Resolution cap (max 1280 × 720)
# ─────────────────────────────────────────────────────────────────────────────

class TestResolutionCap:
    """Any frame exceeding 1280×720 must be downscaled to fit."""

    def test_constants_are_correct(self):
        assert MAX_WIDTH == 1280
        assert MAX_HEIGHT == 720

    def test_full_hd_downscaled(self):
        img = _solid_image(1920, 1080)
        _, w, h = _compress_frame(img)
        assert w <= MAX_WIDTH and h <= MAX_HEIGHT

    def test_4k_downscaled(self):
        img = _solid_image(3840, 2160)
        _, w, h = _compress_frame(img)
        assert w <= MAX_WIDTH and h <= MAX_HEIGHT

    def test_ultra_wide_downscaled(self):
        img = _solid_image(2560, 1080)
        _, w, h = _compress_frame(img)
        assert w <= MAX_WIDTH and h <= MAX_HEIGHT

    def test_tall_portrait_downscaled(self):
        img = _solid_image(720, 1440)
        _, w, h = _compress_frame(img)
        assert h <= MAX_HEIGHT

    def test_exact_boundary_not_resized(self):
        """An image exactly 1280×720 must not be altered."""
        img = _solid_image(MAX_WIDTH, MAX_HEIGHT)
        _, w, h = _compress_frame(img)
        assert w == MAX_WIDTH and h == MAX_HEIGHT

    def test_just_over_width_only(self):
        """1281×100 — only width exceeds the cap."""
        img = _solid_image(MAX_WIDTH + 1, 100)
        _, w, h = _compress_frame(img)
        assert w <= MAX_WIDTH

    def test_just_over_height_only(self):
        """100×721 — only height exceeds the cap."""
        img = _solid_image(100, MAX_HEIGHT + 1)
        _, w, h = _compress_frame(img)
        assert h <= MAX_HEIGHT


# ─────────────────────────────────────────────────────────────────────────────
# 4 · Resolution floor — small images must NOT be upscaled
# ─────────────────────────────────────────────────────────────────────────────

class TestResolutionFloor:
    """Images already within the cap must keep their original dimensions."""

    def test_small_image_unchanged(self):
        img = _solid_image(320, 240)
        _, w, h = _compress_frame(img)
        assert w == 320 and h == 240

    def test_medium_image_unchanged(self):
        img = _solid_image(1024, 600)
        _, w, h = _compress_frame(img)
        assert w == 1024 and h == 600

    def test_single_pixel_unchanged(self):
        img = _solid_image(1, 1)
        _, w, h = _compress_frame(img)
        assert w == 1 and h == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5 · Aspect ratio preservation
# ─────────────────────────────────────────────────────────────────────────────

class TestAspectRatio:
    """Downscaling must preserve the original aspect ratio within 2%."""

    @pytest.mark.parametrize("orig_w,orig_h", [
        (1920, 1080),   # 16:9
        (2560, 1600),   # 16:10
        (3840, 2160),   # 4K 16:9
        (2560, 1080),   # ultra-wide 64:27
        (1280, 1024),   # 5:4 — height-limited
    ])
    def test_aspect_ratio_preserved(self, orig_w, orig_h):
        img = _solid_image(orig_w, orig_h)
        _, w, h = _compress_frame(img)
        expected_ratio = orig_w / orig_h
        actual_ratio = w / h
        assert abs(expected_ratio - actual_ratio) / expected_ratio < 0.02, (
            f"Aspect ratio changed: original={expected_ratio:.4f}, "
            f"compressed={actual_ratio:.4f} for input {orig_w}×{orig_h}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6 · File-size reduction
# ─────────────────────────────────────────────────────────────────────────────

class TestFileSizeReduction:
    """Compressed JPEG must be smaller than the raw uncompressed pixels."""

    def test_compressed_smaller_than_raw_rgb(self):
        """JPEG output must be smaller than the raw RGB byte count."""
        img = _solid_image(1280, 720)
        jpeg_bytes, w, h = _compress_frame(img)
        raw_size = w * h * 3  # 3 bytes per RGB pixel
        assert len(jpeg_bytes) < raw_size, (
            f"JPEG ({len(jpeg_bytes)} B) should be smaller than "
            f"raw RGB ({raw_size} B)"
        )

    def test_compressed_smaller_than_png(self):
        """JPEG output must be smaller than an equivalent PNG encode."""
        img = _noise_image(640, 480)
        jpeg_bytes, _, _ = _compress_frame(img)

        png_buf = io.BytesIO()
        img.save(png_buf, format="PNG")
        png_size = len(png_buf.getvalue())

        assert len(jpeg_bytes) < png_size, (
            f"JPEG ({len(jpeg_bytes)} B) should be smaller than "
            f"PNG ({png_size} B)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7 · End-to-end: CapturedFrame via FrameCapturer.capture_once()
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndCompression:
    """
    FrameCapturer.capture_once() must produce CapturedFrame objects whose
    jpeg_bytes satisfy both the quality and resolution constraints.
    """

    @patch("capture.mss.mss")
    def test_captured_frame_is_valid_jpeg(self, mock_mss_cls):
        mock_mss_cls.return_value = _make_mock_sct(1920, 1080)
        capturer = FrameCapturer(on_frame=lambda f: None)
        frame = capturer.capture_once()

        assert frame.jpeg_bytes[:2] == b"\xff\xd8"
        assert frame.jpeg_bytes[-2:] == b"\xff\xd9"

    @patch("capture.mss.mss")
    def test_captured_frame_dimensions_within_cap(self, mock_mss_cls):
        mock_mss_cls.return_value = _make_mock_sct(1920, 1080)
        capturer = FrameCapturer(on_frame=lambda f: None)
        frame = capturer.capture_once()

        assert frame.width <= MAX_WIDTH, (
            f"Frame width {frame.width} exceeds MAX_WIDTH {MAX_WIDTH}"
        )
        assert frame.height <= MAX_HEIGHT, (
            f"Frame height {frame.height} exceeds MAX_HEIGHT {MAX_HEIGHT}"
        )

    @patch("capture.mss.mss")
    def test_captured_frame_reported_dims_match_jpeg(self, mock_mss_cls):
        """CapturedFrame.width/height must match the actual decoded JPEG size."""
        mock_mss_cls.return_value = _make_mock_sct(1920, 1080)
        capturer = FrameCapturer(on_frame=lambda f: None)
        frame = capturer.capture_once()

        decoded = Image.open(io.BytesIO(frame.jpeg_bytes))
        assert decoded.width == frame.width
        assert decoded.height == frame.height

    @patch("capture.mss.mss")
    def test_4k_source_downscaled_end_to_end(self, mock_mss_cls):
        """A 3840×2160 source must be downscaled by the full capture pipeline."""
        mock_mss_cls.return_value = _make_mock_sct(3840, 2160)
        capturer = FrameCapturer(on_frame=lambda f: None)
        frame = capturer.capture_once()

        assert frame.width <= MAX_WIDTH
        assert frame.height <= MAX_HEIGHT

    @patch("capture.mss.mss")
    def test_small_source_not_upscaled_end_to_end(self, mock_mss_cls):
        """A 640×480 source must keep its original dimensions."""
        mock_mss_cls.return_value = _make_mock_sct(640, 480)
        capturer = FrameCapturer(on_frame=lambda f: None)
        frame = capturer.capture_once()

        assert frame.width == 640
        assert frame.height == 480
