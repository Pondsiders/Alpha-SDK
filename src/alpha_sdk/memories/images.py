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

import logfire

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
        logfire.error("Pillow not installed — cannot create thumbnail")
        return None

    source = Path(source_path)
    if not source.exists():
        logfire.warning(f"Image not found: {source_path}")
        return None

    with logfire.span("images.create_thumbnail", source=source_path) as span:
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
                span.set_attribute("resized", True)
                span.set_attribute("new_size", f"{new_width}x{new_height}")
            else:
                span.set_attribute("resized", False)
                # Still save as JPEG for consistency, even if not resized

            # Save thumbnail
            thumb_dir = _ensure_thumbnail_dir()
            thumb_name = _thumbnail_filename(source_path)
            thumb_path = thumb_dir / thumb_name

            img.save(thumb_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)

            file_size = thumb_path.stat().st_size
            span.set_attribute("thumbnail_path", str(thumb_path))
            span.set_attribute("file_size_kb", file_size / 1024)

            logfire.info(
                f"Thumbnail created: {thumb_name} ({file_size / 1024:.1f} KB)",
                source=source_path,
                thumbnail=str(thumb_path),
            )
            return str(thumb_path)

        except Exception as e:
            logfire.error(f"Thumbnail creation failed: {e}", source=source_path)
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
        logfire.error("Pillow not installed — cannot process inline image")
        return None

    with logfire.span("images.process_inline", media_type=media_type) as span:
        try:
            # Decode base64 to bytes
            raw_bytes = base64.b64decode(image_data)
            original_size_kb = len(raw_bytes) / 1024
            span.set_attribute("original_size_kb", original_size_kb)

            # Open image from bytes
            img = Image.open(BytesIO(raw_bytes))
            original_dimensions = f"{img.size[0]}x{img.size[1]}"
            span.set_attribute("original_dimensions", original_dimensions)

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
                span.set_attribute("resized", True)
                span.set_attribute("new_dimensions", f"{new_width}x{new_height}")

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

            file_size_kb = thumb_path.stat().st_size / 1024
            span.set_attribute("thumbnail_path", str(thumb_path))
            span.set_attribute("thumbnail_size_kb", file_size_kb)

            logfire.info(
                f"Inline image processed: {original_dimensions} ({original_size_kb:.0f}KB) → "
                f"{img.size[0]}x{img.size[1]} ({file_size_kb:.1f}KB)",
                thumbnail=str(thumb_path),
            )
            return new_base64, str(thumb_path)

        except Exception as e:
            logfire.error(f"Inline image processing failed: {e}")
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
        logfire.warning(f"Thumbnail not found: {thumbnail_path}")
        return None

    try:
        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except Exception as e:
        logfire.error(f"Failed to load thumbnail: {e}")
        return None
