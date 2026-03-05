"""system_prompt.py — Read the soul.

One function. Reads soul.md from the identity directory pointed to
by JE_NE_SAIS_QUOI. Returns a string. That's it.
"""

from __future__ import annotations

import os
from pathlib import Path


def read_soul(identity_dir: str | Path | None = None) -> str:
    """Read soul.md from the identity directory.

    Args:
        identity_dir: Path to the identity directory. If None, reads
                      from $JE_NE_SAIS_QUOI environment variable.

    Returns:
        The contents of prompts/system/soul.md as a string.

    Raises:
        FileNotFoundError: If soul.md doesn't exist.
        RuntimeError: If no identity directory is configured.
    """
    if identity_dir is None:
        identity_dir = os.environ.get("JE_NE_SAIS_QUOI")
        if not identity_dir:
            raise RuntimeError(
                "No identity directory configured. "
                "Set JE_NE_SAIS_QUOI or pass identity_dir."
            )

    soul_path = Path(identity_dir) / "prompts" / "system" / "soul.md"

    if not soul_path.exists():
        raise FileNotFoundError(
            f"Soul not found at {soul_path}. "
            f"Expected prompts/system/soul.md inside {identity_dir}."
        )

    return soul_path.read_text()
