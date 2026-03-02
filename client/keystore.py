"""
keystore.py — Secure storage of the Gemini API key for UI Navigator.

Uses the operating-system credential store via the ``keyring`` library:
  • Windows  → Windows Credential Manager
  • macOS    → Keychain
  • Linux    → Secret Service (GNOME Keyring / KWallet)

The key is never written to disk in plaintext.  The ``keyring`` library
handles encryption and access-control at the OS level — the same mechanism
used by password managers, SSH agents, and browser credential stores.

Public API
----------
    from keystore import KeyStore

    ks = KeyStore()
    key = ks.load()          # returns "" if not set
    ks.save("AIza...")       # stores in OS credential vault
    ks.delete()              # removes from vault
    ks.exists()              # True if a non-empty key is stored
"""

from __future__ import annotations

import logging
from typing import Optional

import keyring
import keyring.errors

logger = logging.getLogger(__name__)

_SERVICE  = "ui-navigator"   # keychain service / app name
_USERNAME = "gemini_api_key" # credential username slot


class KeyStore:
    """
    Thin wrapper around the OS credential store.

    Parameters
    ----------
    service:  keychain service name (default ``"ui-navigator"``)
    username: credential slot name  (default ``"gemini_api_key"``)
    """

    def __init__(
        self,
        service: str = _SERVICE,
        username: str = _USERNAME,
    ) -> None:
        self._service  = service
        self._username = username

    def load(self) -> str:
        """
        Retrieve the stored API key from the OS credential store.

        Returns an empty string if no key has been saved yet or if the
        keyring backend is unavailable.
        """
        try:
            value = keyring.get_password(self._service, self._username)
            return value or ""
        except keyring.errors.KeyringError as exc:
            logger.warning("keyring load failed: %s", exc)
            return ""

    def save(self, api_key: str) -> None:
        """
        Persist *api_key* in the OS credential store.

        Raises
        ------
        RuntimeError
            If the keyring backend refuses the write (e.g. no unlock daemon
            available on a headless Linux box).
        """
        api_key = api_key.strip()
        try:
            keyring.set_password(self._service, self._username, api_key)
        except keyring.errors.KeyringError as exc:
            raise RuntimeError(
                f"Could not save API key to the OS credential store: {exc}"
            ) from exc

    def delete(self) -> None:
        """Remove the stored key.  No-op if no key exists."""
        try:
            keyring.delete_password(self._service, self._username)
        except keyring.errors.PasswordDeleteError:
            pass   # already absent
        except keyring.errors.KeyringError as exc:
            logger.warning("keyring delete failed: %s", exc)

    def exists(self) -> bool:
        """Return True if a non-empty key is stored."""
        return bool(self.load())
