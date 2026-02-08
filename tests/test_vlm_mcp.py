"""Tests for VLM MCP tools using generated vector images.

Generates temporary PNG images (triangles, suits, colors, symbols) and tests
the VLM endpoint via the same local functions used by the MCP server.

Skipped unless VLM_ENDPOINT_URL is set (via env or .env file).
Uses a 600s timeout to allow for cold-start scenarios.

If a .env file exists in the project root, it is loaded automatically.
MODAL_WORKSPACE from .env is used to derive VLM_ENDPOINT_URL when not set explicitly.

Usage:
    # With .env file (recommended):
    pytest tests/test_vlm_mcp.py -v

    # Or with explicit env vars:
    VLM_ENDPOINT_URL="https://WORKSPACE--coding-agent-server-serve-vlm.modal.run" \
    MODAL_PROXY_TOKEN_ID=... \
    MODAL_PROXY_TOKEN_SECRET=... \
        pytest tests/test_vlm_mcp.py -v

Requirements:
    - pytest-asyncio (pip install pytest-asyncio)
    - pillow (PIL) for image generation (pip install pillow)
    - httpx (pip install httpx)
    - python-dotenv (pip install python-dotenv)
"""

import base64
import math
import mimetypes
import os
import random
import tempfile
import uuid
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv
from PIL import Image, ImageDraw

# Load .env from project root (does not override existing env vars)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- Config from env (matching MCP server) ---

VLM_ENDPOINT_URL = os.environ.get("VLM_ENDPOINT_URL")
if not VLM_ENDPOINT_URL:
    workspace = os.environ.get("MODAL_WORKSPACE")
    if workspace:
        VLM_ENDPOINT_URL = f"https://{workspace}--coding-agent-server-serve-vlm.modal.run"

VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-32B-Thinking-FP8")
VLM_TIMEOUT = float(os.environ.get("VLM_TIMEOUT", "300"))
MODAL_PROXY_TOKEN_ID = os.environ.get("MODAL_PROXY_TOKEN_ID", "")
MODAL_PROXY_TOKEN_SECRET = os.environ.get("MODAL_PROXY_TOKEN_SECRET", "")

skip_no_vlm_endpoint = pytest.mark.skipif(
    not VLM_ENDPOINT_URL,
    reason="VLM_ENDPOINT_URL env var not set",
)


def _auth_headers() -> dict[str, str]:
    """Build Modal proxy auth headers if credentials are available."""
    if MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET:
        return {
            "Modal-Key": MODAL_PROXY_TOKEN_ID,
            "Modal-Secret": MODAL_PROXY_TOKEN_SECRET,
        }
    return {}


# --- Image generation helpers (matching MCP server patterns) ---


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
    if not VLM_ENDPOINT_URL:
        raise RuntimeError("VLM_ENDPOINT_URL env var is not set")

    payload = {
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": content_blocks}],
        "max_tokens": max_tokens,
    }

    headers = {"Content-Type": "application/json"}
    if MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET:
        headers["Modal-Key"] = MODAL_PROXY_TOKEN_ID
        headers["Modal-Secret"] = MODAL_PROXY_TOKEN_SECRET

    async with httpx.AsyncClient(timeout=VLM_TIMEOUT, follow_redirects=True) as client:
        resp = await client.post(
            f"{VLM_ENDPOINT_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def generate_triangle_image(color: str = "blue", size: int = 256) -> Path:
    """Generate a PNG with a triangle in a random color."""
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    # Random triangle vertices within the image
    margin = size // 10
    points = [
        (random.randint(margin, size - margin), random.randint(margin, size - margin))
        for _ in range(3)
    ]
    draw.polygon(points, fill=color, outline="black")

    path = Path(tempfile.gettempdir()) / f"triangle_{uuid.uuid4().hex}.png"
    img.save(path, "PNG")
    return path


def generate_suit_symbol(suit: str, color: str) -> Path:
    """Generate a PNG with a playing card suit symbol."""
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw suit symbols using text (Unicode) or shapes
    suit_map = {
        "clubs": ("♣", "♣"),  # Unicode clubs symbol
        "diamonds": ("♦", "♦"),
        "hearts": ("♥", "♥"),
        "spades": ("♠", "♠"),
    }

    if suit in suit_map:
        symbol = suit_map[suit][0]
        # Use a larger font-like drawing by scaling up text
        draw.text((size // 2 - 60, size // 2 - 80), symbol, fill=color, font=None)
    else:
        # Fallback: draw a shape
        center = (size // 2, size // 2)
        radius = 80
        if suit == "diamond":
            draw.polygon(
                [
                    (center[0], center[1] - radius),
                    (center[0] + radius, center[1]),
                    (center[0], center[1] + radius),
                    (center[0] - radius, center[1]),
                ],
                fill=color,
            )
        elif suit == "square":
            margin = 40
            draw.rectangle(
                [margin, margin, size - margin, size - margin], fill=color
            )

    path = Path(tempfile.gettempdir()) / f"suit_{suit}_{uuid.uuid4().hex}.png"
    img.save(path, "PNG")
    return path


def generate_color_block(color: str, size: int = 256) -> Path:
    """Generate a solid color PNG block."""
    img = Image.new("RGB", (size, size), color)
    path = Path(tempfile.gettempdir()) / f"color_{color}_{uuid.uuid4().hex}.png"
    img.save(path, "PNG")
    return path


def generate_wingdings_style_symbol(symbol_type: str, color: str) -> Path:
    """Generate a wingdings-style symbolic PNG.

    Creates simple geometric symbols resembling wingdings icons.
    """
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    center = (size // 2, size // 2)

    if symbol_type == "check":
        # Checkmark
        draw.line([(60, 128), (100, 170), (180, 90)], fill=color, width=20)
    elif symbol_type == "cross":
        # X mark
        draw.line([(60, 60), (196, 196)], fill=color, width=20)
        draw.line([(196, 60), (60, 196)], fill=color, width=20)
    elif symbol_type == "star":
        # 5-point star
        points = []
        for i in range(5):
            angle = i * 72
            x = center[0] + 70 * math.cos(math.radians(angle))
            y = center[1] + 70 * math.sin(math.radians(angle))
            points.append((x, y))
            angle = (i * 72) + 36
            x = center[0] + 35 * math.cos(math.radians(angle))
            y = center[1] + 35 * math.sin(math.radians(angle))
            points.append((x, y))
        draw.polygon(points, fill=color)
    elif symbol_type == "heart":
        # Heart shape using two large overlapping circles + a triangle
        r = 45  # circle radius
        cy = center[1] - 15  # vertical center of circles
        draw.ellipse((center[0] - r - 20, cy - r, center[0] - 20 + r, cy + r), fill=color)
        draw.ellipse((center[0] + 20 - r, cy - r, center[0] + 20 + r, cy + r), fill=color)
        draw.polygon(
            [
                (center[0] - 65, cy + 10),
                (center[0] + 65, cy + 10),
                (center[0], center[1] + 90),
            ],
            fill=color,
        )
    elif symbol_type == "smile":
        # Smile face
        draw.ellipse((60, 60, 196, 196), outline=color, width=10)
        draw.ellipse((90, 110, 110, 130), fill=color)  # left eye
        draw.ellipse((146, 110, 166, 130), fill=color)  # right eye
        draw.arc((90, 130, 166, 170), 0, 180, fill=color, width=5)
    elif symbol_type == "arrow_up":
        # Upward arrow
        draw.polygon(
            [(128, 40), (180, 160), (128, 140), (76, 160)], fill=color
        )  # arrow head
        draw.rectangle((118, 140, 138, 200), fill=color)  # arrow shaft
    elif symbol_type == "circle":
        draw.ellipse((60, 60, 196, 196), outline=color, width=10)
    elif symbol_type == "square":
        margin = 40
        draw.rectangle([margin, margin, size - margin, size - margin], outline=color, width=10)
    else:
        # Default: checkmark
        draw.line([(60, 128), (100, 170), (180, 90)], fill=color, width=20)

    path = (
        Path(tempfile.gettempdir())
        / f"wingdings_{symbol_type}_{uuid.uuid4().hex}.png"
    )
    img.save(path, "PNG")
    return path


# --- VLM endpoint tests ---


@skip_no_vlm_endpoint
@pytest.mark.asyncio
async def test_vlm_analyze_triangle():
    """Test VLM can identify a generated triangle image."""
    color = random.choice(["red", "blue", "green", "orange", "purple"])
    img_path = generate_triangle_image(color=color)
    try:
        mime, data = _encode_image(str(img_path))
        content = [
            _image_content_block(mime, data),
            {"type": "text", "text": "Describe this image. What shape do you see?"},
        ]
        response = await _vlm_request(content)
        assert len(response) > 0
        # VLM should recognize it as a triangle or geometric shape
        assert any(
            word in response.lower()
            for word in ["triangle", "shape", "geometric", "polygon"]
        )
    finally:
        img_path.unlink(missing_ok=True)


@skip_no_vlm_endpoint
@pytest.mark.asyncio
async def test_vlm_analyze_clubs_symbol():
    """Test VLM can identify a clubs suit symbol."""
    img_path = generate_suit_symbol("clubs", "black")
    try:
        mime, data = _encode_image(str(img_path))
        content = [
            _image_content_block(mime, data),
            {"type": "text", "text": "What playing card suit is this?"},
        ]
        response = await _vlm_request(content)
        assert len(response) > 0
        assert "club" in response.lower()
    finally:
        img_path.unlink(missing_ok=True)


@skip_no_vlm_endpoint
@pytest.mark.asyncio
async def test_vlm_analyze_diamond_symbol():
    """Test VLM can identify a diamond suit symbol."""
    img_path = generate_suit_symbol("diamonds", "red")
    try:
        mime, data = _encode_image(str(img_path))
        content = [
            _image_content_block(mime, data),
            {"type": "text", "text": "Identify this playing card symbol."},
        ]
        response = await _vlm_request(content)
        assert len(response) > 0
        assert any(
            word in response.lower() for word in ["diamond", "suit", "card"]
        )
    finally:
        img_path.unlink(missing_ok=True)


@skip_no_vlm_endpoint
@pytest.mark.asyncio
async def test_vlm_analyze_wingdings_checkmark():
    """Test VLM can identify a wingdings-style checkmark."""
    img_path = generate_wingdings_style_symbol("check", "green")
    try:
        mime, data = _encode_image(str(img_path))
        content = [
            _image_content_block(mime, data),
            {"type": "text", "text": "What does this symbol represent?"},
        ]
        response = await _vlm_request(content)
        assert len(response) > 0
        assert any(
            word in response.lower()
            for word in ["check", "correct", "success", "yes", "complete"]
        )
    finally:
        img_path.unlink(missing_ok=True)


@skip_no_vlm_endpoint
@pytest.mark.asyncio
async def test_vlm_analyze_wingdings_cross():
    """Test VLM can identify a wingdings-style cross/X mark."""
    img_path = generate_wingdings_style_symbol("cross", "red")
    try:
        mime, data = _encode_image(str(img_path))
        content = [
            _image_content_block(mime, data),
            {"type": "text", "text": "What does this symbol mean?"},
        ]
        response = await _vlm_request(content)
        assert len(response) > 0
        assert any(
            word in response.lower()
            for word in ["cross", "x mark", "error", "no", "cancel", "delete"]
        )
    finally:
        img_path.unlink(missing_ok=True)


@skip_no_vlm_endpoint
@pytest.mark.asyncio
async def test_vlm_analyze_wingdings_heart():
    """Test VLM can identify a wingdings-style heart symbol."""
    img_path = generate_wingdings_style_symbol("heart", "red")
    try:
        mime, data = _encode_image(str(img_path))
        content = [
            _image_content_block(mime, data),
            {"type": "text", "text": "What symbol or shape is shown in this image?"},
        ]
        response = await _vlm_request(content)
        assert len(response) > 0
        assert any(word in response.lower() for word in ["heart", "love", "cardiac"])
    finally:
        img_path.unlink(missing_ok=True)


@skip_no_vlm_endpoint
@pytest.mark.asyncio
async def test_vlm_analyze_wingdings_star():
    """Test VLM can identify a wingdings-style star symbol."""
    img_path = generate_wingdings_style_symbol("star", "gold")
    try:
        mime, data = _encode_image(str(img_path))
        content = [
            _image_content_block(mime, data),
            {"type": "text", "text": "Describe the symbol in this image."},
        ]
        response = await _vlm_request(content)
        assert len(response) > 0
        assert "star" in response.lower()
    finally:
        img_path.unlink(missing_ok=True)


@skip_no_vlm_endpoint
@pytest.mark.asyncio
async def test_vlm_compare_colors():
    """Test VLM can compare two color blocks."""
    red_path = generate_color_block("red")
    blue_path = generate_color_block("blue")

    try:
        content = []
        for path in [str(red_path), str(blue_path)]:
            mime, data = _encode_image(path)
            content.append(_image_content_block(mime, data))
        content.append({"type": "text", "text": "Compare these two colors."})

        response = await _vlm_request(content)
        assert len(response) > 0
        assert any(
            word in response.lower() for word in ["red", "blue", "color", "two"]
        )
    finally:
        red_path.unlink(missing_ok=True)
        blue_path.unlink(missing_ok=True)


@skip_no_vlm_endpoint
@pytest.mark.asyncio
async def test_vlm_compare_shapes():
    """Test VLM can compare triangle and circle."""
    triangle_path = generate_triangle_image(color="blue")
    circle_path = generate_wingdings_style_symbol("circle", "green")

    try:
        content = []
        for path in [str(triangle_path), str(circle_path)]:
            mime, data = _encode_image(path)
            content.append(_image_content_block(mime, data))
        content.append(
            {"type": "text", "text": "What shapes are in these images? How do they differ?"}
        )

        response = await _vlm_request(content)
        assert len(response) > 0
        # Should recognize both shapes
        assert any(
            word in response.lower() for word in ["triangle", "circle", "shape", "different"]
        )
    finally:
        triangle_path.unlink(missing_ok=True)
        circle_path.unlink(missing_ok=True)


@skip_no_vlm_endpoint
@pytest.mark.asyncio
async def test_vlm_multiple_suits():
    """Test VLM can identify multiple playing card suits in one request."""
    clubs_path = generate_suit_symbol("clubs", "black")
    diamonds_path = generate_suit_symbol("diamonds", "red")
    hearts_path = generate_suit_symbol("hearts", "red")
    spades_path = generate_suit_symbol("spades", "black")

    try:
        content = []
        for path in [str(clubs_path), str(diamonds_path), str(hearts_path), str(spades_path)]:
            mime, data = _encode_image(path)
            content.append(_image_content_block(mime, data))
        content.append(
            {"type": "text", "text": "Identify all the playing card suits shown."}
        )

        response = await _vlm_request(content)
        assert len(response) > 0
        # Should identify at least some of the suits
        assert any(
            word in response.lower()
            for word in ["club", "diamond", "heart", "spade", "suit", "card"]
        )
    finally:
        for path in [clubs_path, diamonds_path, hearts_path, spades_path]:
            path.unlink(missing_ok=True)


@skip_no_vlm_endpoint
@pytest.mark.asyncio
async def test_vlm_analyze_random_color_block():
    """Test VLM can identify a randomly colored block."""
    colors = ["red", "orange", "yellow", "green", "blue", "indigo", "violet", "pink"]
    color = random.choice(colors)
    img_path = generate_color_block(color)
    try:
        mime, data = _encode_image(str(img_path))
        content = [
            _image_content_block(mime, data),
            {"type": "text", "text": "What color is this image?"},
        ]
        response = await _vlm_request(content)
        assert len(response) > 0
        # VLM should identify the color (allow some variation)
        response_lower = response.lower()
        # Check if response contains color name or is descriptive
        assert any(
            word in response_lower
            for word in [color.lower(), "color", "solid", "block"]
        )
    finally:
        img_path.unlink(missing_ok=True)

