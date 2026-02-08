"""MCP server for VLM image analysis via the Modal-deployed Qwen3-VL endpoint.

Runs as a stdio MCP server. Configure via env vars:
  VLM_ENDPOINT  - Base URL of the VLM endpoint (e.g. https://WORKSPACE--coding-agent-server-serve-vlm.modal.run/v1)
  VLM_MODEL     - Model name (default: Qwen/Qwen3-VL-32B-Thinking-FP8)
  VLM_TIMEOUT   - Request timeout in seconds (default: 300)

Usage:
  VLM_ENDPOINT="https://..." python src/coding_agent_server/vlm_mcp_server.py
"""

import base64
import logging
import mimetypes
import os
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

# Logging to stderr only — stdout is reserved for stdio transport
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Config from env ---

VLM_ENDPOINT = os.environ.get("VLM_ENDPOINT", "")
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-32B-Thinking-FP8")
VLM_TIMEOUT = float(os.environ.get("VLM_TIMEOUT", "300"))

mcp = FastMCP("vlm-analyzer")


def _encode_image(image_path: str) -> tuple[str, str]:
    """Read a local image file and return (mime_type, base64_data)."""
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return mime, data


def _image_content_block(mime: str, data: str) -> dict:
    """Build an OpenAI-compatible image_url content block."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{data}"},
    }


async def _vlm_request(content_blocks: list[dict], max_tokens: int = 2048) -> str:
    """Send a chat completion request to the VLM endpoint."""
    if not VLM_ENDPOINT:
        raise RuntimeError("VLM_ENDPOINT env var is not set")

    payload = {
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": content_blocks}],
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=VLM_TIMEOUT) as client:
        resp = await client.post(
            f"{VLM_ENDPOINT}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


@mcp.tool()
async def analyze_image(image_path: str, prompt: str) -> str:
    """Analyze a local image file with a text prompt.

    Args:
        image_path: Absolute or relative path to an image file.
        prompt: Question or instruction about the image.
    """
    mime, data = _encode_image(image_path)
    content = [
        _image_content_block(mime, data),
        {"type": "text", "text": prompt},
    ]
    return await _vlm_request(content)


@mcp.tool()
async def compare_images(image_paths: list[str], prompt: str) -> str:
    """Compare 2-5 local images with a text prompt.

    Args:
        image_paths: List of 2-5 absolute or relative paths to image files.
        prompt: Question or instruction about the images.
    """
    if len(image_paths) < 2:
        raise ValueError("Need at least 2 images to compare")
    if len(image_paths) > 5:
        raise ValueError("Maximum 5 images per request (server limit)")

    content = []
    for path in image_paths:
        mime, data = _encode_image(path)
        content.append(_image_content_block(mime, data))
    content.append({"type": "text", "text": prompt})

    return await _vlm_request(content)


if __name__ == "__main__":
    mcp.run(transport="stdio")
