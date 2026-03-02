"""
capture.py — Screen capture module for UI Navigator Desktop Client.

Responsibilities:
- Capture screenshots every 2–3 seconds (configurable interval)
- Local frame-diff check: skip transmission if pixel hash delta is below threshold
- Frame compression: JPEG quality=70, max resolution 1280×720
- Provide a FrameCapturer class with start/stop/callback interface
"""

import hashlib
import io
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import mss
import mss.tools
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS: float = 2.5  
DEFAULT_DIFF_THRESHOLD: int = 0        
JPEG_QUALITY: int = 70
MAX_WIDTH: int = 1280
MAX_HEIGHT: int = 720

@dataclass
class CapturedFrame:
    timestamp: float                  
    jpeg_bytes: bytes            
    width: int                       
    height: int                   
    frame_hash: str                   # MD5 hex of the JPEG bytes (used for diff)
    changed: bool = True              # False if frame was skipped (not transmitted)
    monitor_index: int = 1            # mss monitor index (1 = primary)
    metadata: dict = field(default_factory=dict)


def _compress_frame(pil_image: Image.Image) -> tuple[bytes, int, int]:

    w, h = pil_image.size
    if w > MAX_WIDTH or h > MAX_HEIGHT:
        ratio = min(MAX_WIDTH / w, MAX_HEIGHT / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    jpeg_bytes = buf.getvalue()
    return jpeg_bytes, pil_image.width, pil_image.height

def _frame_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

class FrameCapturer:
    """
    Continuously captures the screen and calls `on_frame` whenever a
    sufficiently different frame is ready.

    Usage
    -----
        def handle(frame: CapturedFrame):
            print(f"New frame at {frame.timestamp:.2f}, size={len(frame.jpeg_bytes)} bytes")

        capturer = FrameCapturer(on_frame=handle)
        capturer.start()
        time.sleep(10)
        capturer.stop()

    Parameters
    ----------
    on_frame        : callback invoked for every *changed* frame
    interval        : seconds between capture attempts (default 2.5)
    diff_threshold  : minimum number of differing hex characters between
                      successive frame hashes to count as a real change.
                      0 = always treat as changed (transmit every frame).
                      Increase to skip cosmetic UI flickers (suggest 4–8).
    monitor_index   : which monitor to capture (1 = primary, 0 = all monitors)
    """

    def __init__(
        self,
        on_frame: Callable[[CapturedFrame], None],
        interval: float = DEFAULT_INTERVAL_SECONDS,
        diff_threshold: int = DEFAULT_DIFF_THRESHOLD,
        monitor_index: int = 1,
    ):
        self.on_frame = on_frame
        self.interval = interval
        self.diff_threshold = diff_threshold
        self.monitor_index = monitor_index

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_hash: Optional[str] = None

        # Stats
        self.frames_captured: int = 0
        self.frames_transmitted: int = 0
        self.frames_skipped: int = 0

    def start(self) -> None:
        """Start the background capture loop."""
        if self._thread and self._thread.is_alive():
            logger.warning("FrameCapturer already running — ignoring start()")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="FrameCapturer")
        self._thread.start()
        logger.info(
            "FrameCapturer started (interval=%.1fs, diff_threshold=%d, monitor=%d)",
            self.interval,
            self.diff_threshold,
            self.monitor_index,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background capture loop and wait for it to finish."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info(
            "FrameCapturer stopped. captured=%d transmitted=%d skipped=%d",
            self.frames_captured,
            self.frames_transmitted,
            self.frames_skipped,
        )

    def capture_once(self) -> CapturedFrame:
        """
        Capture a single frame *right now*, bypass the diff check, and
        return it (does NOT call on_frame). Useful for on-demand snapshots.
        """
        return self._capture(force=True)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        """Background thread: captures frames at the configured interval."""
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            try:
                frame = self._capture()
                if frame.changed:
                    self.frames_transmitted += 1
                    try:
                        self.on_frame(frame)
                    except Exception:  # noqa: BLE001
                        logger.exception("on_frame callback raised an exception")
                else:
                    self.frames_skipped += 1
            except Exception:  # noqa: BLE001
                logger.exception("Error during screen capture")

            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.0, self.interval - elapsed)
            self._stop_event.wait(timeout=sleep_time)

    def _capture(self, force: bool = False) -> CapturedFrame:
        """Capture one frame, apply diff check, and return a CapturedFrame."""
        with mss.mss() as sct:
            monitor = sct.monitors[self.monitor_index]
            raw = sct.grab(monitor)
            pil_img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

        jpeg_bytes, w, h = _compress_frame(pil_img)
        current_hash = _frame_hash(jpeg_bytes)
        self.frames_captured += 1

        # Diff check
        changed = True
        if not force and self._last_hash is not None and self.diff_threshold > 0:
            diff = sum(a != b for a, b in zip(current_hash, self._last_hash))
            if diff <= self.diff_threshold:
                changed = False

        if changed:
            self._last_hash = current_hash

        return CapturedFrame(
            timestamp=time.time(),
            jpeg_bytes=jpeg_bytes,
            width=w,
            height=h,
            frame_hash=current_hash,
            changed=changed,
            monitor_index=self.monitor_index,
        )
