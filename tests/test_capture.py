"""
tests/test_capture.py — Tests for client/capture.py

Run with:  pytest tests/test_capture.py -v
"""

import io
import time
import threading
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# ── import the module under test ──────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "client"))

from capture import (
    CapturedFrame,
    FrameCapturer,
    _compress_frame,
    _frame_hash,
    JPEG_QUALITY,
    MAX_HEIGHT,
    MAX_WIDTH,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_image(width: int = 1920, height: int = 1080, color: tuple = (100, 149, 237)) -> Image.Image:
    """Return a solid-color PIL image."""
    return Image.new("RGB", (width, height), color=color)


def _image_to_jpeg_bytes(img: Image.Image, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _make_mock_mss_frame(width: int = 1920, height: int = 1080):
    """Return a fake mss grab() result (BGRA bytes)."""
    pil = _make_image(width, height, color=(0, 128, 255))
    # mss returns BGRA; PIL RGB image needs to be converted
    pil_bgra = pil.convert("RGBA")
    raw_bytes = pil_bgra.tobytes()

    mock_grab = MagicMock()
    mock_grab.size = (width, height)
    mock_grab.bgra = raw_bytes
    return mock_grab


def _make_mock_sct(width: int = 1920, height: int = 1080):
    """Return a mock mss context manager."""
    mock_sct = MagicMock()
    mock_sct.__enter__ = MagicMock(return_value=mock_sct)
    mock_sct.__exit__ = MagicMock(return_value=False)
    mock_sct.monitors = [
        {},                          # index 0 = all monitors (unused)
        {"left": 0, "top": 0, "width": width, "height": height},
    ]
    mock_sct.grab = MagicMock(return_value=_make_mock_mss_frame(width, height))
    return mock_sct


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: _compress_frame
# ─────────────────────────────────────────────────────────────────────────────

class TestCompressFrame:
    def test_large_image_is_downscaled(self):
        img = _make_image(3840, 2160)
        jpeg_bytes, w, h = _compress_frame(img)
        assert w <= MAX_WIDTH
        assert h <= MAX_HEIGHT
        assert len(jpeg_bytes) > 0

    def test_small_image_is_not_upscaled(self):
        img = _make_image(640, 480)
        jpeg_bytes, w, h = _compress_frame(img)
        assert w == 640
        assert h == 480

    def test_aspect_ratio_preserved(self):
        img = _make_image(1920, 1080)  # 16:9
        _, w, h = _compress_frame(img)
        aspect_original = 1920 / 1080
        aspect_result = w / h
        assert abs(aspect_original - aspect_result) < 0.02  # within 2%

    def test_returns_valid_jpeg(self):
        img = _make_image(800, 600)
        jpeg_bytes, _, _ = _compress_frame(img)
        # JPEG magic bytes: FF D8
        assert jpeg_bytes[:2] == b"\xff\xd8"

    def test_exact_max_dimensions_unchanged(self):
        img = _make_image(MAX_WIDTH, MAX_HEIGHT)
        _, w, h = _compress_frame(img)
        assert w == MAX_WIDTH
        assert h == MAX_HEIGHT

    def test_width_limited_landscape(self):
        """Image wider than MAX_WIDTH but height OK → only width constrains."""
        img = _make_image(2560, 600)
        _, w, h = _compress_frame(img)
        assert w <= MAX_WIDTH

    def test_height_limited_portrait(self):
        """Image taller than MAX_HEIGHT but width OK → only height constrains."""
        img = _make_image(640, 1440)
        _, w, h = _compress_frame(img)
        assert h <= MAX_HEIGHT


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: _frame_hash
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameHash:
    def test_returns_hex_string(self):
        h = _frame_hash(b"hello")
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_input_same_hash(self):
        assert _frame_hash(b"data") == _frame_hash(b"data")

    def test_different_input_different_hash(self):
        assert _frame_hash(b"data1") != _frame_hash(b"data2")

    def test_hash_length_is_32(self):
        # MD5 hex digest is always 32 characters
        assert len(_frame_hash(b"anything")) == 32


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: CapturedFrame dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestCapturedFrame:
    def test_defaults(self):
        f = CapturedFrame(
            timestamp=1.0,
            jpeg_bytes=b"data",
            width=640,
            height=480,
            frame_hash="abc",
        )
        assert f.changed is True
        assert f.monitor_index == 1
        assert f.metadata == {}


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: FrameCapturer (with mocked mss)
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameCapturerCaptureOnce:
    @patch("capture.mss.mss")
    def test_capture_once_returns_frame(self, mock_mss_cls):
        mock_mss_cls.return_value = _make_mock_sct()
        capturer = FrameCapturer(on_frame=lambda f: None)
        frame = capturer.capture_once()

        assert isinstance(frame, CapturedFrame)
        assert frame.jpeg_bytes[:2] == b"\xff\xd8"   # valid JPEG
        assert frame.width <= MAX_WIDTH
        assert frame.height <= MAX_HEIGHT
        assert frame.changed is True  # force=True bypasses diff

    @patch("capture.mss.mss")
    def test_capture_once_increments_counter(self, mock_mss_cls):
        mock_mss_cls.return_value = _make_mock_sct()
        capturer = FrameCapturer(on_frame=lambda f: None)
        capturer.capture_once()
        assert capturer.frames_captured == 1

    @patch("capture.mss.mss")
    def test_capture_once_sets_hash(self, mock_mss_cls):
        mock_mss_cls.return_value = _make_mock_sct()
        capturer = FrameCapturer(on_frame=lambda f: None)
        frame = capturer.capture_once()
        assert frame.frame_hash is not None and len(frame.frame_hash) == 32


class TestFrameDiff:
    @patch("capture.mss.mss")
    def test_identical_frames_skipped_when_threshold_positive(self, mock_mss_cls):
        """
        If the screen doesn't change at all, all successive captures should be
        marked changed=False when diff_threshold > 0.
        """
        # Same mss mock returns the same pixels every call → same JPEG → same hash
        mock_mss_cls.return_value = _make_mock_sct()
        capturer = FrameCapturer(on_frame=lambda f: None, diff_threshold=1)

        # First capture: no previous hash → always changed
        f1 = capturer._capture()
        assert f1.changed is True

        # Second capture: same frame → should be skipped
        mock_mss_cls.return_value = _make_mock_sct()
        f2 = capturer._capture()
        assert f2.changed is False

    @patch("capture.mss.mss")
    def test_zero_threshold_always_transmits(self, mock_mss_cls):
        """diff_threshold=0 means every frame is transmitted regardless."""
        mock_mss_cls.return_value = _make_mock_sct()
        capturer = FrameCapturer(on_frame=lambda f: None, diff_threshold=0)

        capturer._capture()                         # prime last_hash
        mock_mss_cls.return_value = _make_mock_sct()
        f2 = capturer._capture()
        assert f2.changed is True                   # 0 threshold → always changed

    @patch("capture.mss.mss")
    def test_different_frames_always_transmitted(self, mock_mss_cls):
        """Different pixel content → different hash → changed=True even with threshold."""
        mock_mss_cls.return_value = _make_mock_sct(1920, 1080)
        capturer = FrameCapturer(on_frame=lambda f: None, diff_threshold=4)
        capturer._capture()  # prime

        # Return a very different image
        mock_mss_cls.return_value = _make_mock_sct(800, 600)
        f2 = capturer._capture()
        assert f2.changed is True


class TestFrameCapturerLifecycle:
    @patch("capture.mss.mss")
    def test_start_stop(self, mock_mss_cls):
        mock_mss_cls.return_value = _make_mock_sct()
        received = []
        capturer = FrameCapturer(on_frame=received.append, interval=0.3)
        capturer.start()
        assert capturer.is_running

        time.sleep(0.8)  # allow 2–3 captures
        capturer.stop()
        assert not capturer.is_running

    @patch("capture.mss.mss")
    def test_callback_called_with_changed_frames(self, mock_mss_cls):
        """
        With diff_threshold=0, every frame fires the callback.
        """
        # Return a fresh mock for every mss() call so each frame has same content
        mock_mss_cls.return_value = _make_mock_sct()
        received = []
        capturer = FrameCapturer(
            on_frame=received.append,
            interval=0.2,
            diff_threshold=0,
        )
        capturer.start()
        time.sleep(0.9)
        capturer.stop()

        assert len(received) >= 2, f"Expected ≥2 frames, got {len(received)}"

    @patch("capture.mss.mss")
    def test_double_start_safe(self, mock_mss_cls):
        mock_mss_cls.return_value = _make_mock_sct()
        capturer = FrameCapturer(on_frame=lambda f: None, interval=0.5)
        capturer.start()
        capturer.start()  # second start should be a no-op
        assert capturer.is_running
        capturer.stop()

    @patch("capture.mss.mss")
    def test_stats_accumulate(self, mock_mss_cls):
        mock_mss_cls.return_value = _make_mock_sct()
        capturer = FrameCapturer(on_frame=lambda f: None, interval=0.2, diff_threshold=0)
        capturer.start()
        time.sleep(0.7)
        capturer.stop()
        assert capturer.frames_captured > 0
        assert capturer.frames_transmitted + capturer.frames_skipped == capturer.frames_captured

    @patch("capture.mss.mss")
    def test_frame_timestamp_is_recent(self, mock_mss_cls):
        mock_mss_cls.return_value = _make_mock_sct()
        capturer = FrameCapturer(on_frame=lambda f: None)
        before = time.time()
        frame = capturer.capture_once()
        after = time.time()
        assert before <= frame.timestamp <= after


class TestFrameCapturerInterval:
    @patch("capture.mss.mss")
    def test_respects_interval_approximately(self, mock_mss_cls):
        """
        With interval=0.5 s, running for 2 s should yield ~4 frames (±1).
        """
        mock_mss_cls.return_value = _make_mock_sct()
        received = []
        capturer = FrameCapturer(
            on_frame=received.append,
            interval=0.5,
            diff_threshold=0,
        )
        capturer.start()
        time.sleep(2.2)
        capturer.stop()

        # Expect 3–5 frames (generous range due to CI scheduling jitter)
        assert 3 <= len(received) <= 6, f"Expected 3–6 frames, got {len(received)}"
