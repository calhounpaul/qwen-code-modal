"""Smoke tests against live coding-agent-server endpoints.

Skipped unless ENDPOINT_URL / VLM_ENDPOINT_URL env vars are set.
Uses a 600s timeout to allow for cold-start scenarios.

Usage:
    ENDPOINT_URL="https://WORKSPACE--coding-agent-server-serve-coder.modal.run" \
    VLM_ENDPOINT_URL="https://WORKSPACE--coding-agent-server-serve-vlm.modal.run" \
        pytest tests/test_health.py -v
"""

import os

import httpx
import pytest

ENDPOINT_URL = os.environ.get("ENDPOINT_URL")
VLM_ENDPOINT_URL = os.environ.get("VLM_ENDPOINT_URL")
MODEL_NAME = "unsloth/Qwen3-Coder-Next-FP8-Dynamic"
VLM_MODEL_NAME = "Qwen/Qwen3-VL-32B-Thinking-FP8"
TIMEOUT = 600.0  # 10 minutes for cold starts

skip_no_endpoint = pytest.mark.skipif(
    not ENDPOINT_URL,
    reason="ENDPOINT_URL env var not set",
)

skip_no_vlm_endpoint = pytest.mark.skipif(
    not VLM_ENDPOINT_URL,
    reason="VLM_ENDPOINT_URL env var not set",
)


@pytest.fixture
def client():
    with httpx.Client(base_url=ENDPOINT_URL, timeout=TIMEOUT) as c:
        yield c


@pytest.fixture
def vlm_client():
    with httpx.Client(base_url=VLM_ENDPOINT_URL, timeout=TIMEOUT) as c:
        yield c


# --- Coder endpoint tests ---


@skip_no_endpoint
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200


@skip_no_endpoint
def test_models_list(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    model_ids = [m["id"] for m in data["data"]]
    assert MODEL_NAME in model_ids, f"Expected {MODEL_NAME} in {model_ids}"


@skip_no_endpoint
def test_chat_completion(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    assert len(content) > 0, "Expected non-empty response content"


# --- VLM endpoint tests ---


@skip_no_vlm_endpoint
def test_vlm_health(vlm_client):
    resp = vlm_client.get("/health")
    assert resp.status_code == 200


@skip_no_vlm_endpoint
def test_vlm_models_list(vlm_client):
    resp = vlm_client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    model_ids = [m["id"] for m in data["data"]]
    assert VLM_MODEL_NAME in model_ids, f"Expected {VLM_MODEL_NAME} in {model_ids}"
