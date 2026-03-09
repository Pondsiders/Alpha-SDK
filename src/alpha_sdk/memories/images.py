"""Image processing for visual memory.

Handles thumbnail generation for the Mind's Eye feature:
- Resize images to 768px long edge (464 tokens, sharp enough to read text)
- Convert to JPEG quality 80 (8x smaller than PNG, identical token cost)
- Store thumbnails in a persistent directory for recall injection

Ported from alpha_sdk v0.x.
"""

import base64
import hashlib
from io import BytesIO
from pathlib import Path

# Where thumbnails live — syncs via Syncthing, persists across sessions
THUMBNAIL_DIR = Path("/Pondside/Alpha-Home/images/thumbnails")

# Processing parameters (from the Mind's Eye token experiment, Feb 8 2026)
MAX_LONG_EDGE = 768   # pixels — 464 tokens, readable, good quality
JPEG_QUALITY = 80     # good balance of size vs quality


def _ensure_thumbnail_dir() -> Path:
    """Create the thumbnail directory if it doesn't exist."""
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    return THUMBNAIL_DIR


def _thumbnail_filename(source_path: str) -> str:
    """Generate a deterministic thumbnail filename from the source path."""
    path_hash = hashlib.sha256(source_path.encode()).hexdigest()[:16]
    source_stem = Path(source_path).stem[:40]
    return f"{source_stem}_{path_hash}.jpg"


def create_thumbnail(source_path: str) -> str | None:
    """Create a 768px JPEG thumbnail from a source image.

    Args:
        source_path: Path to the original image file

    Returns:
        Path to the created thumbnail, or None on failure
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    source = Path(source_path)
    if not source.exists():
        return None

    try:
        img = Image.open(source)

        # Convert RGBA/palette to RGB for JPEG
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Resize to fit within MAX_LONG_EDGE on the long side
        width, height = img.size
        if max(width, height) > MAX_LONG_EDGE:
            if width >= height:
                new_width = MAX_LONG_EDGE
                new_height = int(height * (MAX_LONG_EDGE / width))
            else:
                new_height = MAX_LONG_EDGE
                new_width = int(width * (MAX_LONG_EDGE / height))
            img = img.resize((new_width, new_height), Image.LANCZOS)

        # Save thumbnail
        thumb_dir = _ensure_thumbnail_dir()
        thumb_name = _thumbnail_filename(source_path)
        thumb_path = thumb_dir / thumb_name

        img.save(thumb_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)

        return str(thumb_path)

    except Exception:
        return None


def load_thumbnail_base64(thumbnail_path: str) -> str | None:
    """Load a thumbnail and return it as a base64-encoded string.

    Args:
        thumbnail_path: Path to the thumbnail JPEG

    Returns:
        Base64-encoded string, or None if file not found
    """
    path = Path(thumbnail_path)
    if not path.exists():
        return None

    try:
        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return None
