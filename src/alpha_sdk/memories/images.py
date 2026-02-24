"""Image processing for visual memory.

Handles thumbnail generation for the Mind's Eye feature:
- Resize images to 768px long edge (sweet spot: 464 tokens, sharp enough to read text)
- Convert to JPEG quality 80 (8x smaller than PNG, identical token cost)
- Store thumbnails in a persistent directory for recall injection

The thumbnail is what goes into the API. The original is preserved separately.
This is the safety valve: raw images can be 10-20 MB and poison JSONL transcripts;
768px JPEG thumbnails are ~100 KB and always safe under the 32 MB request limit.
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
    """Generate a deterministic thumbnail filename from the source path.

    Uses a hash so the same source always produces the same thumbnail name.
    This means re-storing the same image doesn't create duplicates.
    """
    path_hash = hashlib.sha256(source_path.encode()).hexdigest()[:16]
    source_stem = Path(source_path).stem[:40]  # Keep some readability
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


def process_inline_image(image_data: str, media_type: str = "image/png") -> tuple[str, str] | None:
    """Process an inline base64 image (from Duckpond paste/attach).

    Takes raw base64 image data, creates a 768px JPEG thumbnail,
    saves it to the thumbnails directory, and returns the new base64
    and the file path.

    This is the safety valve for pasted images: raw retina screenshots
    get downsized before they can poison JSONL transcripts.

    Args:
        image_data: Base64-encoded image data
        media_type: MIME type of the image (e.g., "image/png", "image/jpeg")

    Returns:
        Tuple of (new_base64, thumbnail_path) or None on failure
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        # Decode base64 to bytes
        raw_bytes = base64.b64decode(image_data)

        # Open image from bytes
        img = Image.open(BytesIO(raw_bytes))

        # Convert to RGB for JPEG
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if needed
        width, height = img.size
        if max(width, height) > MAX_LONG_EDGE:
            if width >= height:
                new_width = MAX_LONG_EDGE
                new_height = int(height * (MAX_LONG_EDGE / width))
            else:
                new_height = MAX_LONG_EDGE
                new_width = int(width * (MAX_LONG_EDGE / height))
            img = img.resize((new_width, new_height), Image.LANCZOS)

        # Generate deterministic filename from content hash
        content_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
        thumb_name = f"inline_{content_hash}.jpg"
        thumb_dir = _ensure_thumbnail_dir()
        thumb_path = thumb_dir / thumb_name

        # Save thumbnail to disk
        img.save(thumb_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)

        # Also generate base64 of the thumbnail
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        new_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return new_base64, str(thumb_path)

    except Exception:
        return None


def load_thumbnail_base64(thumbnail_path: str) -> str | None:
    """Load a thumbnail and return it as a base64-encoded string.

    This is what gets injected into the API as an image content block.

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
