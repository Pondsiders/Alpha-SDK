"""Vision captioning via Ollama multimodal models.

Sends images to Gemma 3 12B for single-sentence descriptions.
Used for image-triggered memory recall — the caption becomes the search query.
"""

import os

import httpx

# Configuration from environment — same pattern as embeddings.py
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://alpha-pi:11434")
VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "gemma3:12b-it-qat")

CAPTION_PROMPT = "Describe this image in 1-2 sentences. Be specific about what you see."


async def caption_image(base64_data: str, timeout: float = 15.0) -> str | None:
    """Caption an image using a vision model via Ollama.

    Sends a base64-encoded JPEG to Gemma 3 12B and gets back a 1-2 sentence
    description. Used as the search query for image-triggered memory recall.

    Args:
        base64_data: Base64-encoded image data (typically 768px JPEG thumbnail)
        timeout: Maximum seconds to wait for captioning

    Returns:
        Caption text, or None if captioning fails (graceful degradation)
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{OLLAMA_URL.rstrip('/')}/api/generate",
                json={
                    "model": VISION_MODEL,
                    "prompt": CAPTION_PROMPT,
                    "images": [base64_data],
                    "stream": False,
                    "keep_alive": -1,  # Keep model loaded
                },
            )
            response.raise_for_status()
            data = response.json()
            caption = data.get("response", "").strip()

            if not caption:
                return None

            return caption

    except httpx.TimeoutException:
        return None
    except httpx.ConnectError:
        return None
    except Exception:
        return None
