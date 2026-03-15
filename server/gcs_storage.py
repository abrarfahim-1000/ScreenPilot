"""
gcs_storage.py — Google Cloud Storage artifact store for UI Navigator.

Responsibilities
----------------
- Upload session artifacts (frames, logs, reports) to a GCS bucket
- Return gs:// and public HTTPS URLs for uploaded objects
- All operations are best-effort: failures are logged but never raise

Usage
-----
    store = GCSArtifactStore(bucket_name="my-bucket", project_id="my-project")
    if store.available:
        url = store.upload_bytes(data, "sessions/abc/frames/step_1.jpg", "image/jpeg")
        print(url)  # gs://my-bucket/sessions/abc/frames/step_1.jpg
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazily imported so that the module can be imported even when the
# google-cloud-storage package is absent (e.g. during unit tests).
_gcs_mod = None


def _get_storage():
    """Lazy-import google.cloud.storage and return the module."""
    global _gcs_mod
    if _gcs_mod is None:
        try:
            from google.cloud import storage as _st
            _gcs_mod = _st
        except ImportError:
            logger.warning(
                "google-cloud-storage not installed — GCS artifact storage disabled. "
                "Run: pip install google-cloud-storage"
            )
            _gcs_mod = False  # sentinel: import attempted but failed
    return _gcs_mod if _gcs_mod else None


class GCSArtifactStore:
    """
    Thin wrapper around the GCS SDK for UI Navigator artifact storage.

    Parameters
    ----------
    bucket_name:
        GCS bucket name.  If None or empty, all operations are no-ops.
    project_id:
        GCP project ID.  If None, inferred from the environment
        (``GOOGLE_CLOUD_PROJECT`` env var or ADC).
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> None:
        self._bucket_name = (bucket_name or "").strip()
        self._project_id = project_id
        self._client = None
        self._bucket = None

        if not self._bucket_name:
            logger.info(
                "GCSArtifactStore: no GCS_BUCKET configured — artifact storage disabled"
            )
            return

        storage = _get_storage()
        if storage is None:
            return

        try:
            self._client = storage.Client(project=project_id)
            self._bucket = self._client.bucket(self._bucket_name)
            logger.info(
                "GCSArtifactStore initialised (bucket=%s project=%s)",
                self._bucket_name,
                project_id or "inferred from ADC",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "GCSArtifactStore init failed — artifact storage disabled: %s", exc
            )
            self._client = None
            self._bucket = None

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True if the GCS client is initialised and ready to accept uploads."""
        return self._bucket is not None

    @property
    def bucket_name(self) -> str:
        """Name of the target GCS bucket."""
        return self._bucket_name

    # ── Upload helpers ─────────────────────────────────────────────────────

    def upload_bytes(
        self,
        data: bytes,
        object_name: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload *data* to ``{bucket}/{object_name}``.

        Returns
        -------
        str
            ``gs://{bucket}/{object_name}`` on success, empty string on failure.
        """
        if not self.available:
            return ""
        try:
            blob = self._bucket.blob(object_name)  # type: ignore[union-attr]
            blob.upload_from_string(data, content_type=content_type)
            url = self.gs_url(object_name)
            logger.info("GCS upload: %s (%d bytes, type=%s)", url, len(data), content_type)
            return url
        except Exception as exc:  # noqa: BLE001
            logger.warning("GCS upload_bytes failed (%s): %s", object_name, exc)
            return ""

    def upload_file(
        self,
        local_path: str,
        object_name: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Read *local_path* and upload its contents to ``{bucket}/{object_name}``.

        Returns
        -------
        str
            ``gs://{bucket}/{object_name}`` on success, empty string on failure.
        """
        if not self.available:
            return ""
        try:
            data = Path(local_path).read_bytes()
        except (OSError, FileNotFoundError) as exc:
            logger.warning("GCS upload_file: cannot read '%s': %s", local_path, exc)
            return ""
        return self.upload_bytes(data, object_name, content_type)

    # ── URL helpers ─────────────────────────────────────────────────────────

    def gs_url(self, object_name: str) -> str:
        """Return the ``gs://`` URI for *object_name* in this bucket."""
        return f"gs://{self._bucket_name}/{object_name}"

    def public_url(self, object_name: str) -> str:
        """
        Return the public HTTPS URL for *object_name*.

        The bucket must have ``allUsers`` read access enabled for this URL
        to work without authentication.  For private buckets use signed URLs.
        """
        return f"https://storage.googleapis.com/{self._bucket_name}/{object_name}"
