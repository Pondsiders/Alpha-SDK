"""Fetch tool — universal URL reader for alpha_sdk.

Fetches any URL and returns content in a format Alpha can work with:
- text/markdown responses pass through directly
- text/html responses are converted to markdown via html2text
- image/* responses are returned as base64 image content blocks
- GitHub URLs are rewritten to fetch raw content (README, source files)
- JSON APIs return formatted inline
- RSS/Atom feeds parsed into clean readable summaries
- Cloudflare Browser Rendering available for JS-heavy pages (render=true)

Three tiers:
1. Accept: text/markdown header (free, instant, Cloudflare sites only)
2. html2text fallback for HTML (free, fast, works on everything)
3. Cloudflare Browser Rendering (render=true, limited free tier)

Smart URL rewriting:
- github.com/user/repo → README.md (raw, via API for default branch)
- github.com/user/repo/blob/branch/file → raw file content
- github.com/user/repo/tree/branch/dir → README.md in that directory

Environment:
    CLOUDFLARE_ACCOUNT_ID - For Browser Rendering (optional)
    CLOUDFLARE_TOKEN      - For Browser Rendering (optional)

Usage:
    from alpha_sdk.tools.fetch import create_fetch_server

    mcp_servers = {
        "fetch": create_fetch_server()
    }
"""

import base64
import json as json_mod
import os
import re
from typing import Any
import httpx
import logfire

from claude_agent_sdk import tool, create_sdk_mcp_server


# Cloudflare Browser Rendering config (optional)
_CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
_CF_TOKEN = os.getenv("CLOUDFLARE_TOKEN")
_CF_RENDER_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/{_CF_ACCOUNT_ID}/browser-rendering/markdown"
    if _CF_ACCOUNT_ID
    else None
)

# GitHub URL patterns
_GITHUB_REPO_RE = re.compile(
    r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$"
)
_GITHUB_BLOB_RE = re.compile(
    r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$"
)
_GITHUB_TREE_RE = re.compile(
    r"^https?://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+?)/?$"
)


async def _rewrite_github_url(url: str) -> tuple[str, str | None]:
    """Rewrite GitHub URLs to fetch raw content instead of HTML pages.

    Returns:
        Tuple of (rewritten_url, description_of_what_was_done or None)
        If no rewrite applies, returns (original_url, None).
    """
    # github.com/user/repo/blob/branch/path → raw file
    m = _GITHUB_BLOB_RE.match(url)
    if m:
        user, repo, branch, path = m.groups()
        raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
        return raw_url, f"GitHub blob → raw ({user}/{repo}/{path})"

    # github.com/user/repo/tree/branch/dir → README in that dir
    m = _GITHUB_TREE_RE.match(url)
    if m:
        user, repo, branch, path = m.groups()
        raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}/README.md"
        return raw_url, f"GitHub tree → README.md in {path}"

    # github.com/user/repo → README (need API to find default branch)
    m = _GITHUB_REPO_RE.match(url)
    if m:
        user, repo = m.groups()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                api_resp = await client.get(
                    f"https://api.github.com/repos/{user}/{repo}",
                    headers={
                        "Accept": "application/vnd.github.v3+json",
                        "User-Agent": "Alpha/1.0 (https://alphafornow.com)",
                    },
                )
                api_resp.raise_for_status()
                default_branch = api_resp.json().get("default_branch", "main")
        except Exception:
            default_branch = "main"  # Fallback

        raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{default_branch}/README.md"
        return raw_url, f"GitHub repo → README.md ({user}/{repo}, branch: {default_branch})"

    return url, None


async def _try_fetch(url: str) -> tuple[str, bytes, dict[str, str]]:
    """Fetch a URL with markdown preference. Returns (content_type, body, headers)."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        resp = await client.get(
            url,
            headers={
                "Accept": "text/markdown, text/html;q=0.9, image/*;q=0.8, */*;q=0.5",
                "User-Agent": "Alpha/1.0 (https://alphafornow.com)",
            },
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        return content_type, resp.content, dict(resp.headers)


def _ext_from_content_type(content_type: str) -> str:
    """Guess a file extension from a content type."""
    mapping = {
        "application/pdf": ".pdf",
        "application/json": ".json",
        "application/xml": ".xml",
        "application/zip": ".zip",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "video/mp4": ".mp4",
    }
    return mapping.get(content_type, ".bin")


async def _save_to_disk(body: bytes, url: str, ext: str) -> str:
    """Save fetched content to disk. Returns the file path."""
    import hashlib
    from pathlib import Path

    # Save to Alpha-Home/downloads/
    download_dir = Path("/Pondside/Alpha-Home/downloads")
    download_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename from URL hash + extension
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
    # Try to extract a readable name from the URL
    url_path = url.rstrip("/").split("/")[-1].split("?")[0]
    if url_path and len(url_path) < 60:
        # Clean up the filename
        safe_name = "".join(c for c in url_path if c.isalnum() or c in ".-_")
        if safe_name and not safe_name.endswith(ext):
            safe_name = f"{safe_name}{ext}"
        filename = f"{url_hash}_{safe_name}"
    else:
        filename = f"{url_hash}{ext}"

    save_path = download_dir / filename
    save_path.write_bytes(body)
    logfire.info(f"File saved: {save_path} ({len(body):,} bytes)")
    return str(save_path)


async def _html_to_markdown(html_bytes: bytes, encoding: str = "utf-8") -> str:
    """Convert HTML to markdown using html2text."""
    import html2text

    h = html2text.HTML2Text()
    h.body_width = 0  # No line wrapping
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_emphasis = False
    h.skip_internal_links = False

    html_str = html_bytes.decode(encoding, errors="replace")
    return h.handle(html_str)


async def _process_image(image_bytes: bytes, content_type: str) -> tuple[dict[str, Any], str | None]:
    """Process an image: resize, save to disk, return as base64 content block + path.

    Uses the Mind's Eye infrastructure to save a thumbnail so the image can be
    attached to a memory later via `cortex store --image <path>`.

    Returns:
        Tuple of (image_content_block, thumbnail_path or None)
    """
    from ..memories.images import process_inline_image

    # First, base64-encode the raw bytes so process_inline_image can handle it
    raw_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Use the Mind's Eye pipeline: resize to 768px, JPEG/80, save to thumbnails dir
    result = process_inline_image(raw_b64, media_type=content_type)

    if result:
        thumb_b64, thumb_path = result
        logfire.info(f"Fetched image saved: {thumb_path}")
        return {
            "type": "image",
            "data": thumb_b64,
            "mimeType": "image/jpeg",
        }, thumb_path
    else:
        # Fallback: just encode without saving (shouldn't normally happen)
        logfire.warning("Image processing failed, returning raw")
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "type": "image",
            "data": b64_data,
            "mimeType": content_type or "image/jpeg",
        }, None


# Content types that indicate RSS/Atom feeds
_FEED_CONTENT_TYPES = {
    "application/rss+xml",
    "application/atom+xml",
    "application/xml",
    "text/xml",
}


def _parse_feed(body: bytes, url: str) -> str | None:
    """Try to parse bytes as an RSS/Atom feed. Returns formatted text or None."""
    import feedparser

    feed = feedparser.parse(body)

    # feedparser will "parse" anything without erroring — check if it found a real feed
    if not feed.entries and not feed.feed.get("title"):
        return None

    lines = []

    # Feed header
    title = feed.feed.get("title", "Untitled Feed")
    subtitle = feed.feed.get("subtitle", "")
    lines.append(f"# {title}")
    if subtitle:
        lines.append(f"*{subtitle}*")
    lines.append("")

    # Entries — compact format, most recent first (feedparser preserves feed order)
    for entry in feed.entries[:25]:  # Cap at 25 items
        entry_title = entry.get("title", "Untitled")
        entry_link = entry.get("link", "")
        entry_date = entry.get("published", entry.get("updated", ""))

        # Build entry line
        header = f"### {entry_title}"
        if entry_date:
            header += f"  ({entry_date})"
        lines.append(header)

        if entry_link:
            lines.append(entry_link)

        # Summary: prefer summary over full content, truncate to ~300 chars
        summary = entry.get("summary", "")
        if not summary:
            content = entry.get("content", [{}])
            if content:
                summary = content[0].get("value", "")
        if summary:
            # Strip HTML tags from summary (feedparser often leaves them in)
            import re as _re
            clean = _re.sub(r"<[^>]+>", "", summary).strip()
            if len(clean) > 300:
                clean = clean[:297] + "..."
            if clean:
                lines.append(clean)

        lines.append("")

    return "\n".join(lines)


async def _cloudflare_render(url: str) -> str:
    """Fetch markdown via Cloudflare Browser Rendering API."""
    if not _CF_RENDER_URL or not _CF_TOKEN:
        raise RuntimeError(
            "Cloudflare Browser Rendering not configured. "
            "Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_TOKEN."
        )

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            _CF_RENDER_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_CF_TOKEN}",
            },
            json={
                "url": url,
                "gotoOptions": {"waitUntil": "networkidle0"},
            },
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success"):
            errors = data.get("errors", [])
            error_msg = "; ".join(e.get("message", "Unknown") for e in errors)
            raise RuntimeError(f"Cloudflare render failed: {error_msg}")

        return data.get("result", "")


def create_fetch_server():
    """Create the Fetch MCP server.

    Returns:
        MCP server configuration dict
    """

    @tool(
        "fetch",
        "Fetch a URL and return its content. Works with web pages (returns markdown), "
        "images (returns the image so you can see it), and any other URL. "
        "For normal web pages, just provide the URL. "
        "Set render=true for JavaScript-heavy pages that need a real browser "
        "(uses Cloudflare Browser Rendering — limited free tier, use sparingly).",
        {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
                "render": {
                    "type": "boolean",
                    "description": (
                        "Use Cloudflare Browser Rendering for JS-heavy pages (SPAs, "
                        "dynamic content). Slower but renders JavaScript. Limited to "
                        "10 free browser-minutes/month. Default: false."
                    ),
                },
            },
            "required": ["url"],
        },
    )
    async def fetch(args: dict[str, Any]) -> dict[str, Any]:
        """Fetch a URL and return content in the best available format."""
        url = args["url"]
        render = args.get("render", False)

        # Smart URL rewriting (GitHub, etc.) — before any fetching
        original_url = url
        url, rewrite_note = await _rewrite_github_url(url)
        if rewrite_note:
            logfire.info(f"URL rewritten: {rewrite_note}", original=original_url, rewritten=url)

        with logfire.span(
            "mcp.fetch",
            url=url,
            original_url=original_url if rewrite_note else None,
            render=render,
        ) as span:
            if rewrite_note:
                span.set_attribute("rewrite", rewrite_note)

            try:
                # Tier 3: Cloudflare Browser Rendering (explicit opt-in)
                if render:
                    span.set_attribute("tier", "cloudflare_render")
                    markdown = await _cloudflare_render(url)
                    token_estimate = len(markdown) // 4  # Rough estimate
                    span.set_attribute("result_type", "rendered_markdown")
                    span.set_attribute("content_length", len(markdown))
                    return {
                        "content": [
                            {"type": "text", "text": markdown},
                            {"type": "text", "text": f"\n---\n*Rendered via Cloudflare Browser Rendering (~{token_estimate} tokens)*"},
                        ]
                    }

                # Tier 1+2: Try Accept: text/markdown, fall back to html2text
                content_type, body, headers = await _try_fetch(url)
                span.set_attribute("content_type", content_type)
                span.set_attribute("response_size", len(body))

                # Check for markdown token count header (Cloudflare sites)
                md_tokens = headers.get("x-markdown-tokens")
                if md_tokens:
                    span.set_attribute("markdown_tokens", md_tokens)

                # Route by content type
                if content_type == "text/markdown":
                    # Tier 1: Got markdown directly!
                    span.set_attribute("tier", "accept_markdown")
                    span.set_attribute("result_type", "native_markdown")
                    text = body.decode("utf-8", errors="replace")
                    meta = f"\n---\n*Native markdown from {original_url}"
                    if rewrite_note:
                        meta += f" (rewritten: {rewrite_note})"
                    if md_tokens:
                        meta += f" ({md_tokens} tokens)"
                    meta += "*"
                    return {
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "text", "text": meta},
                        ]
                    }

                elif content_type.startswith("image/"):
                    # Image: return as viewable content block + saved path
                    span.set_attribute("tier", "image")
                    span.set_attribute("result_type", "image")
                    image_block, thumb_path = await _process_image(body, content_type)
                    content = [image_block]
                    meta = f"Image from {original_url} ({content_type}, {len(body):,} bytes)"
                    if thumb_path:
                        meta += f"\n[Image saved: {thumb_path}]"
                        span.set_attribute("thumbnail_path", thumb_path)
                    content.append({"type": "text", "text": meta})
                    return {"content": content}

                # Check for RSS/Atom feeds before HTML (some feeds use text/xml)
                if content_type in _FEED_CONTENT_TYPES:
                    feed_text = _parse_feed(body, url)
                    if feed_text:
                        span.set_attribute("tier", "feed")
                        span.set_attribute("result_type", "rss_atom")
                        return {
                            "content": [
                                {"type": "text", "text": feed_text},
                                {"type": "text", "text": f"\n---\n*Feed from {original_url} ({content_type}, {len(body):,} bytes)*"},
                            ]
                        }
                    # Not a real feed — continue to other handlers

                if content_type in ("text/html", "application/xhtml+xml"):
                    # Tier 2: HTML -> markdown via html2text
                    span.set_attribute("tier", "html2text")
                    span.set_attribute("result_type", "converted_markdown")
                    markdown = await _html_to_markdown(body)
                    span.set_attribute("content_length", len(markdown))
                    return {
                        "content": [
                            {"type": "text", "text": markdown},
                            {"type": "text", "text": f"\n---\n*Converted from HTML via html2text ({len(body):,} bytes){' — rewritten: ' + rewrite_note if rewrite_note else ''}*"},
                        ]
                    }

                if content_type in ("application/json", "application/ld+json"):
                    # JSON: return formatted inline
                    span.set_attribute("tier", "json")
                    span.set_attribute("result_type", "json")
                    text = body.decode("utf-8", errors="replace")
                    # Pretty-print if valid JSON
                    try:
                        parsed = json_mod.loads(text)
                        text = json_mod.dumps(parsed, indent=2, ensure_ascii=False)
                    except (json_mod.JSONDecodeError, ValueError):
                        pass  # Return as-is if not valid JSON
                    # Safety valve
                    if len(text) > 500_000:
                        text = text[:500_000] + f"\n\n[Truncated at 500K characters]"
                    meta = f"\n---\n*JSON from {original_url}"
                    if rewrite_note:
                        meta += f" (rewritten: {rewrite_note})"
                    meta += f" ({len(body):,} bytes)*"
                    return {
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "text", "text": meta},
                        ]
                    }

                if content_type == "application/pdf":
                    # PDF: save to disk, return path for Read tool
                    span.set_attribute("tier", "pdf")
                    span.set_attribute("result_type", "saved_file")
                    save_path = await _save_to_disk(body, url, ".pdf")
                    span.set_attribute("save_path", save_path)
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"PDF downloaded and saved to: {save_path}\n"
                                    f"Size: {len(body):,} bytes\n\n"
                                    f"Use the Read tool to view it: Read({save_path})"
                                ),
                            }
                        ]
                    }

                # Fallback: unknown binary type → save to disk
                if not content_type.startswith("text/"):
                    span.set_attribute("tier", "binary_save")
                    span.set_attribute("result_type", "saved_file")
                    ext = _ext_from_content_type(content_type)
                    save_path = await _save_to_disk(body, url, ext)
                    span.set_attribute("save_path", save_path)
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Binary file ({content_type}) saved to: {save_path}\n"
                                    f"Size: {len(body):,} bytes"
                                ),
                            }
                        ]
                    }

                # Fallback: text content, return raw
                span.set_attribute("tier", "raw")
                span.set_attribute("result_type", "raw_text")
                text = body.decode("utf-8", errors="replace")
                if len(text) > 500_000:
                    text = text[:500_000] + f"\n\n[Truncated at 500K characters, full size was {len(body):,} bytes]"
                return {
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "text", "text": f"\n---\n*Raw content from {original_url} ({content_type}){' — rewritten: ' + rewrite_note if rewrite_note else ''}*"},
                    ]
                }

            except httpx.HTTPStatusError as e:
                logfire.warning("Fetch HTTP error", url=url, status=e.response.status_code)
                return {
                    "content": [{"type": "text", "text": f"HTTP {e.response.status_code} fetching {url}"}]
                }
            except httpx.ConnectError:
                logfire.warning("Fetch connection error", url=url)
                return {
                    "content": [{"type": "text", "text": f"Could not connect to {url}"}]
                }
            except httpx.TimeoutException:
                logfire.warning("Fetch timeout", url=url)
                return {
                    "content": [{"type": "text", "text": f"Timeout fetching {url} (30s limit)"}]
                }
            except Exception as e:
                logfire.error("Fetch error", url=url, error=str(e))
                return {
                    "content": [{"type": "text", "text": f"Error fetching {url}: {e}"}]
                }

    # Bundle into MCP server
    return create_sdk_mcp_server(
        name="fetch",
        version="1.0.0",
        tools=[fetch],
    )
