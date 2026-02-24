"""Forge imagination tools — native MCP server for alpha_sdk.

Calls Forge's /imagine endpoint over HTTP to generate images on the GPU.
Returns images as base64 content blocks so Alpha sees what she imagined.

Forge URL comes from FORGE_URL env var (default: http://primer:8200).
If Forge is unreachable, the tool fails gracefully with an error message.

Usage:
    from alpha_sdk.tools.forge import create_forge_server

    mcp_servers = {
        "forge": create_forge_server()
    }
"""

import os
from typing import Any

import httpx

from claude_agent_sdk import tool, create_sdk_mcp_server


FORGE_URL = os.getenv("FORGE_URL", "http://primer:8200")


def create_forge_server():
    """Create the Forge MCP server with imagination tools.

    Returns:
        MCP server configuration dict
    """

    @tool(
        "imagine",
        "Generate an image from a text prompt. Returns the image directly so you can see it. "
        "The image is also saved to disk. Use descriptive, detailed prompts for best results. "
        "Optional parameters: negative_prompt (what to avoid), steps (inference steps, default 8), "
        "width/height (default 1152x768, 3:2 widescreen), seed (for reproducibility).",
        {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text description of the image to generate. Be specific and descriptive.",
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "What to avoid in the image (optional).",
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of inference steps. More steps = better quality but slower. Default 8.",
                },
                "width": {
                    "type": "integer",
                    "description": "Image width in pixels. Default 1152.",
                },
                "height": {
                    "type": "integer",
                    "description": "Image height in pixels. Default 768.",
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility (optional).",
                },
            },
            "required": ["prompt"],
        },
    )
    async def imagine(args: dict[str, Any]) -> dict[str, Any]:
        """Generate an image via Forge and return it as a viewable content block."""
        prompt = args["prompt"]

        try:
            # Build the request payload with 3:2 landscape defaults
            payload: dict[str, Any] = {
                "prompt": prompt,
                "width": args.get("width", 1152),
                "height": args.get("height", 768),
            }
            if "negative_prompt" in args:
                payload["negative_prompt"] = args["negative_prompt"]
            if "steps" in args:
                payload["steps"] = args["steps"]
            if "seed" in args:
                payload["seed"] = args["seed"]

            # Call Forge
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{FORGE_URL}/imagine",
                    json=payload,
                    timeout=300.0,  # Image generation can take a while
                )

            if resp.status_code != 200:
                error_text = resp.text[:500]
                return {"content": [{"type": "text", "text": f"Image generation failed (HTTP {resp.status_code}): {error_text}"}]}

            result = resp.json()
            image_b64 = result.get("base64", "")
            image_path = result.get("path", "unknown")
            gen_time = result.get("generation_time", 0)
            model = result.get("model", "unknown")

            # Re-encode to JPEG/80 for context delivery.
            # Forge saves at JPEG/90 — re-encoding at 80 shaves ~30% off
            # the base64 for minimal quality loss. No downscaling: the dream
            # IS the right size already (1152x768 ≈ 464 tokens, ~200 KB).
            import base64 as b64_mod
            from io import BytesIO
            from PIL import Image

            raw_bytes = b64_mod.b64decode(image_b64)
            img = Image.open(BytesIO(raw_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=80, optimize=True)
            delivery_b64 = b64_mod.b64encode(buf.getvalue()).decode("utf-8")

            # MCP spec image format: "type": "image", "data": base64, "mimeType": mime
            return {
                "content": [
                    {
                        "type": "image",
                        "data": delivery_b64,
                        "mimeType": "image/jpeg",
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Image generated and saved to {image_path}\n"
                            f"Model: {model}\n"
                            f"Generation time: {gen_time:.1f}s\n"
                            f"Prompt: {prompt}"
                        ),
                    },
                ]
            }

        except httpx.ConnectError:
            return {"content": [{"type": "text", "text": f"Forge is not running at {FORGE_URL}. Start Forge on primer to enable image generation."}]}
        except httpx.TimeoutException:
            return {"content": [{"type": "text", "text": "Image generation timed out (>5 minutes). The model may still be loading."}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error generating image: {e}"}]}

    # Bundle into MCP server
    return create_sdk_mcp_server(
        name="forge",
        version="1.0.0",
        tools=[imagine],
    )
