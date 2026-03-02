"""
tests/test_redaction.py — Tests for client/redaction.py

Run with:  pytest tests/test_redaction.py -v
"""

import io
import sys
import os

import pytest
from PIL import Image

# ── import the module under test ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "client"))

from redaction import (
    RedactionMatch,
    SECRET_PATTERNS,
    is_safe_to_type,
    mask_frame_regions,
    redact_frame,
    redact_text,
    scan_text,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_jpeg(width: int = 320, height: int = 240,
               color: tuple = (100, 149, 237)) -> bytes:
    """Return a minimal solid-colour JPEG as raw bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def _jpeg_to_pil(jpeg_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")


def _pixel_at(jpeg_bytes: bytes, x: int, y: int) -> tuple:
    return _jpeg_to_pil(jpeg_bytes).getpixel((x, y))


# ─────────────────────────────────────────────────────────────────────────────
# TestScanText — pattern detection
# ─────────────────────────────────────────────────────────────────────────────

class TestScanText:
    def test_clean_text_returns_empty_list(self):
        assert scan_text("echo hello world") == []

    def test_empty_string_returns_empty_list(self):
        assert scan_text("") == []

    def test_detects_jwt_token(self):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        matches = scan_text(f"Authorization: Bearer {jwt}")
        # Should detect jwt_token or bearer_token
        names = [m.pattern_name for m in matches]
        assert any(n in ("jwt_token", "bearer_token") for n in names)

    def test_detects_bearer_token(self):
        text = "Authorization: Bearer ya29.A0ARrdaM-verylongtokenvalue1234567890"
        matches = scan_text(text)
        names = [m.pattern_name for m in matches]
        assert "bearer_token" in names

    def test_detects_pem_block(self):
        pem = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA0Z3VS5JJcds3xHn/ygWep4PAtEsHAAAAAA==\n"
            "-----END RSA PRIVATE KEY-----"
        )
        matches = scan_text(pem)
        names = [m.pattern_name for m in matches]
        assert "pem_block" in names

    def test_detects_aws_access_key(self):
        text = "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        matches = scan_text(text)
        names = [m.pattern_name for m in matches]
        assert "aws_access_key" in names

    def test_detects_long_uppercase_token(self):
        text = "APIKEY=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234"
        matches = scan_text(text)
        names = [m.pattern_name for m in matches]
        assert any(n in ("long_uppercase_token", "password_assignment") for n in names)

    def test_detects_hex_secret(self):
        text = "session_token=a3f5b1c2d4e6789012345678901234ab"
        matches = scan_text(text)
        names = [m.pattern_name for m in matches]
        assert any(n in ("hex_secret", "password_assignment") for n in names)

    def test_detects_password_assignment(self):
        matches = scan_text("DB_PASSWORD=supersecret123")
        names = [m.pattern_name for m in matches]
        assert "password_assignment" in names

    def test_detects_secret_assignment(self):
        matches = scan_text("SECRET=myverysecretvalue")
        names = [m.pattern_name for m in matches]
        assert "password_assignment" in names

    def test_matches_sorted_by_start_position(self):
        text = "token=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234 and password=hunter2"
        matches = scan_text(text)
        starts = [m.start for m in matches]
        assert starts == sorted(starts)

    def test_short_uppercase_not_flagged(self):
        """Short all-caps like 'PASS' or 'ERROR' should not trigger long_uppercase_token."""
        assert scan_text("PASS") == []
        assert scan_text("ERROR") == []
        assert scan_text("OK") == []

    def test_normal_make_command_is_clean(self):
        assert scan_text("make test 2>&1 | tee /tmp/test_out.log") == []

    def test_gcloud_deploy_command_is_clean(self):
        assert scan_text("gcloud run deploy my-service --region=us-central1") == []

    def test_match_contains_correct_offsets(self):
        text = "key=ABCDEFGHIJKLMNOPQRSTUVWXYZ0"
        matches = scan_text(text)
        assert len(matches) >= 1
        m = matches[0]
        assert text[m.start:m.end] == m.original_value

    def test_overlapping_matches_deduped(self):
        """A region matched by an earlier pattern should not also appear for a later one."""
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        matches = scan_text(jwt)
        # All match ranges must be non-overlapping
        ranges = [(m.start, m.end) for m in matches]
        for i, (s1, e1) in enumerate(ranges):
            for j, (s2, e2) in enumerate(ranges):
                if i != j:
                    assert not (s1 < e2 and s2 < e1), (
                        f"Overlapping matches at indices {i} and {j}: "
                        f"[{s1}:{e1}] and [{s2}:{e2}]"
                    )


# ─────────────────────────────────────────────────────────────────────────────
# TestRedactText
# ─────────────────────────────────────────────────────────────────────────────

class TestRedactText:
    def test_clean_text_returned_unchanged(self):
        text = "echo hello world"
        redacted, matches = redact_text(text)
        assert redacted == text
        assert matches == []

    def test_returns_tuple_of_str_and_list(self):
        result = redact_text("some text")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], list)

    def test_secret_replaced_with_placeholder(self):
        text = "DB_PASSWORD=supersecret123"
        redacted, matches = redact_text(text)
        assert len(matches) >= 1
        assert "supersecret123" not in redacted
        assert "[REDACTED:" in redacted

    def test_placeholder_contains_pattern_name(self):
        text = "DB_PASSWORD=supersecret123"
        redacted, matches = redact_text(text)
        for m in matches:
            assert m.pattern_name in redacted or m.placeholder in redacted

    def test_multiple_secrets_all_replaced(self):
        text = "USER_TOKEN=ABCDEFGHIJKLMNOPQRSTUVWXYZ\nDB_PASSWORD=verysecret99"
        redacted, matches = redact_text(text)
        assert len(matches) >= 2
        assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in redacted
        assert "verysecret99" not in redacted

    def test_surrounding_text_preserved(self):
        text = "before SECRET=abc123xyz after"
        redacted, _ = redact_text(text)
        assert redacted.startswith("before ")
        assert redacted.endswith(" after")

    def test_redacted_text_length_changes(self):
        """Placeholder is a different length from original — check no IndexError."""
        text = "API_KEY=AAABBBCCCDDDEEEFFFGGG"
        redacted, matches = redact_text(text)
        # Should not raise; result is a valid string
        assert isinstance(redacted, str)
        assert len(redacted) > 0

    def test_empty_string(self):
        redacted, matches = redact_text("")
        assert redacted == ""
        assert matches == []

    def test_pem_block_replaced(self):
        pem = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA0Z3VS5JJcds3xHn/ygWep4PAtEsHAAAAAA==\n"
            "-----END RSA PRIVATE KEY-----"
        )
        redacted, matches = redact_text(pem)
        assert "BEGIN RSA PRIVATE KEY" not in redacted
        assert "[REDACTED:pem_block]" in redacted

    def test_match_original_value_is_correct(self):
        text = "password=hunter2ok"
        _, matches = redact_text(text)
        assert len(matches) >= 1
        # The matched original value should appear in the original text
        for m in matches:
            assert text[m.start:m.end] == m.original_value


# ─────────────────────────────────────────────────────────────────────────────
# TestIsSafeToType
# ─────────────────────────────────────────────────────────────────────────────

class TestIsSafeToType:
    def test_clean_command_is_safe(self):
        assert is_safe_to_type("make test 2>&1 | tee /tmp/test_out.log") is True

    def test_gcloud_deploy_is_safe(self):
        assert is_safe_to_type("gcloud run deploy my-service --region=us-central1") is True

    def test_empty_string_is_safe(self):
        assert is_safe_to_type("") is True

    def test_plain_text_is_safe(self):
        assert is_safe_to_type("Hello, world!") is True

    def test_password_assignment_is_not_safe(self):
        assert is_safe_to_type("DB_PASSWORD=supersecret123") is False

    def test_bearer_token_is_not_safe(self):
        assert is_safe_to_type("Bearer ya29.A0ARrdaM-verylongtokenvalue1234567890") is False

    def test_aws_key_is_not_safe(self):
        assert is_safe_to_type("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE") is False

    def test_pem_key_is_not_safe(self):
        pem = "-----BEGIN PRIVATE KEY-----\nABCDEFGH\n-----END PRIVATE KEY-----"
        assert is_safe_to_type(pem) is False

    def test_returns_bool(self):
        result = is_safe_to_type("hello")
        assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────────────────────
# TestMaskFrameRegions
# ─────────────────────────────────────────────────────────────────────────────

class TestMaskFrameRegions:
    def test_returns_valid_jpeg(self):
        jpeg = _make_jpeg(320, 240, color=(200, 50, 50))
        result = mask_frame_regions(jpeg, regions=[(0, 0, 100, 100)])
        assert result[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_empty_regions_returns_original_bytes(self):
        jpeg = _make_jpeg(320, 240)
        result = mask_frame_regions(jpeg, regions=[])
        assert result == jpeg

    def test_masked_region_differs_from_solid_original(self):
        """
        After heavy blur, the pixel at the centre of the masked region should
        differ from the original only if the region is not uniformly coloured.
        We use a red–blue split image to guarantee the blur mixes colours.
        """
        # Build half-red, half-blue 100×100 image
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        for y in range(50, 100):
            for x in range(100):
                img.putpixel((x, y), (0, 0, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        original_jpeg = buf.getvalue()

        # Mask the boundary band (rows 40–60) where red meets blue
        masked_jpeg = mask_frame_regions(original_jpeg, regions=[(0, 40, 100, 20)])

        # The original pixel at (50, 50) is pure blue (0, 0, 255).
        # After blurring over the red/blue boundary it should be noticeably different.
        orig_pixel = _pixel_at(original_jpeg, 50, 50)
        masked_pixel = _pixel_at(masked_jpeg, 50, 50)
        total_diff = sum(abs(int(a) - int(b)) for a, b in zip(orig_pixel, masked_pixel))
        assert total_diff > 10, (
            f"Masked pixel ({masked_pixel}) too close to original ({orig_pixel}); "
            "blur did not change the image"
        )

    def test_fill_color_mode_makes_region_uniform(self):
        """fill_color=(0,0,0) should paint the region solid black."""
        jpeg = _make_jpeg(200, 200, color=(200, 200, 200))
        masked = mask_frame_regions(jpeg, regions=[(50, 50, 100, 100)],
                                    fill_color=(0, 0, 0))
        # Centre of filled region should be close to black (JPEG lossy, so ≤10 off)
        r, g, b = _pixel_at(masked, 100, 100)
        assert r < 20 and g < 20 and b < 20, (
            f"Expected near-black pixel, got ({r},{g},{b})"
        )

    def test_out_of_bounds_region_clamped_no_error(self):
        """Regions extending beyond the image boundary should not raise."""
        jpeg = _make_jpeg(100, 100)
        # Region deliberately larger than the image
        result = mask_frame_regions(jpeg, regions=[(-10, -10, 200, 200)])
        assert result[:2] == b"\xff\xd8"

    def test_degenerate_zero_size_region_ignored(self):
        jpeg = _make_jpeg(100, 100)
        result = mask_frame_regions(jpeg, regions=[(10, 10, 0, 0)])
        assert result[:2] == b"\xff\xd8"

    def test_multiple_regions_all_masked(self):
        """Two non-overlapping fill-black regions both become black."""
        jpeg = _make_jpeg(300, 200, color=(200, 200, 200))
        regions = [(10, 10, 50, 50), (200, 100, 50, 50)]
        masked = mask_frame_regions(jpeg, regions=regions, fill_color=(0, 0, 0))
        # Sample centre of each region
        for (rx, ry, rw, rh) in regions:
            cx, cy = rx + rw // 2, ry + rh // 2
            r, g, b = _pixel_at(masked, cx, cy)
            assert r < 20 and g < 20 and b < 20, (
                f"Region ({rx},{ry},{rw},{rh}) centre pixel not black: ({r},{g},{b})"
            )

    def test_unmasked_region_unchanged(self):
        """Pixels outside all masked regions should be unchanged (within JPEG tolerance)."""
        jpeg = _make_jpeg(200, 200, color=(200, 100, 50))
        masked = mask_frame_regions(jpeg, regions=[(0, 0, 50, 50)],
                                    fill_color=(0, 0, 0))
        # Pixel at (150, 150) is well outside the masked region
        orig = _pixel_at(jpeg, 150, 150)
        after = _pixel_at(masked, 150, 150)
        diff = sum(abs(int(a) - int(b)) for a, b in zip(orig, after))
        assert diff < 15, (
            f"Unmasked pixel changed too much: {orig} → {after}"
        )

    def test_custom_blur_radius(self):
        """Passing a custom blur_radius should still return a valid JPEG."""
        jpeg = _make_jpeg(200, 200)
        result = mask_frame_regions(jpeg, regions=[(0, 0, 100, 100)], blur_radius=5)
        assert result[:2] == b"\xff\xd8"


# ─────────────────────────────────────────────────────────────────────────────
# TestRedactFrame
# ─────────────────────────────────────────────────────────────────────────────

class TestRedactFrame:
    def test_empty_regions_returns_original_and_false(self):
        jpeg = _make_jpeg()
        result_bytes, modified = redact_frame(jpeg, regions=[])
        assert result_bytes is jpeg  # same object, no copy
        assert modified is False

    def test_non_empty_regions_returns_modified_and_true(self):
        jpeg = _make_jpeg()
        result_bytes, modified = redact_frame(jpeg, regions=[(0, 0, 50, 50)],
                                              fill_color=(0, 0, 0))
        assert modified is True
        assert result_bytes[:2] == b"\xff\xd8"

    def test_returns_tuple_of_bytes_and_bool(self):
        jpeg = _make_jpeg()
        result = redact_frame(jpeg, regions=[])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bytes)
        assert isinstance(result[1], bool)

    def test_fill_color_forwarded(self):
        jpeg = _make_jpeg(100, 100, color=(200, 200, 200))
        result_bytes, _ = redact_frame(jpeg, regions=[(10, 10, 80, 80)],
                                       fill_color=(0, 0, 0))
        r, g, b = _pixel_at(result_bytes, 50, 50)
        assert r < 20 and g < 20 and b < 20


# ─────────────────────────────────────────────────────────────────────────────
# TestSecretPatternsCompleteness (smoke tests for all registered patterns)
# ─────────────────────────────────────────────────────────────────────────────

class TestSecretPatternsCompleteness:
    """Make sure every named pattern in SECRET_PATTERNS fires on at least one sample."""

    SAMPLES = {
        "pem_block": (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "AAABBBCCCDDDEEE==\n"
            "-----END RSA PRIVATE KEY-----"
        ),
        "jwt_token": (
            "eyJhbGciOiJIUzI1NiJ9"
            ".eyJzdWIiOiJ1c2VyIn0"
            ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        ),
        "bearer_token": "Bearer ya29.A0ARrdaMverylongtokenvalue1234567890",
        "aws_access_key": "AKIAIOSFODNN7EXAMPLE",
        "gcp_key_field": '"private_key_id": "1234abcd5678ef"',
        "long_uppercase_token": "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234",
        "hex_secret": "a3f5b1c2d4e6789012345678901234ab",
        "password_assignment": "DB_PASSWORD=supersecret123",
    }

    @pytest.mark.parametrize("pattern_name,sample", SAMPLES.items())
    def test_pattern_fires(self, pattern_name, sample):
        matches = scan_text(sample)
        names = [m.pattern_name for m in matches]
        assert pattern_name in names, (
            f"Pattern '{pattern_name}' did not match sample: {sample!r}\n"
            f"Detected patterns: {names}"
        )
