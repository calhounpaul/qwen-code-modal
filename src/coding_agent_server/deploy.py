"""Modal deployment of vLLM serving coding + VLM models.

Deploy:  modal deploy src/coding_agent_server/deploy.py
Test:    modal run src/coding_agent_server/deploy.py
"""

import json
from typing import Any

import aiohttp
import modal

# --- Coder config (inlined to avoid relative-import issues with Modal) ---

MODEL_NAME = "unsloth/Qwen3-Coder-Next-FP8-Dynamic"
MODEL_DIR = "/model"  # local path where weights are baked into the image
GPU_TYPE = "H200"  # 141 GiB HBM3e — fits 78 GiB FP8 weights + 64K KV cache on 1 GPU
N_GPU = 1
SCALEDOWN_WINDOW = 300  # 5 minutes idle before scale-to-zero
MAX_CONCURRENT_INPUTS = 128  # vLLM handles batching internally
MAX_MODEL_LEN = 131072  # 128K — ~60 GiB free for KV cache on H200
GPU_MEMORY_UTILIZATION = 0.90
KV_CACHE_DTYPE = "fp8"
TOOL_CALL_PARSER = "qwen3_coder"
APP_NAME = "coding-agent-server"
VLLM_PORT = 8000

# --- VLM config ---

VLM_MODEL_NAME = "Qwen/Qwen3-VL-32B-Thinking-FP8"
VLM_MODEL_DIR = "/vlm-model"
VLM_GPU_TYPE = "A100-40GB"
VLM_N_GPU = 1
VLM_MAX_MODEL_LEN = 32768
VLM_GPU_MEMORY_UTILIZATION = 0.90
VLM_MAX_CONCURRENT_INPUTS = 16

# --- Image ---

MINUTES = 60  # seconds


def download_model():
    import os
    import sys

    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_NAME} to {MODEL_DIR}...", flush=True)
    snapshot_download(
        MODEL_NAME,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # prefer safetensors
    )
    # Log what was downloaded
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(MODEL_DIR)
        for f in fns
    )
    print(f"Download complete: {total / 1e9:.1f} GB in {MODEL_DIR}", flush=True)
    sys.stdout.flush()


def download_vlm_model():
    import os
    import sys

    from huggingface_hub import snapshot_download

    print(f"Downloading {VLM_MODEL_NAME} to {VLM_MODEL_DIR}...", flush=True)
    snapshot_download(
        VLM_MODEL_NAME,
        local_dir=VLM_MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],
    )
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(VLM_MODEL_DIR)
        for f in fns
    )
    print(f"Download complete: {total / 1e9:.1f} GB in {VLM_MODEL_DIR}", flush=True)
    sys.stdout.flush()


vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm>=0.15.0",
        "huggingface_hub",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(download_model, timeout=20 * MINUTES)
)

vlm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm>=0.15.0",
        "huggingface_hub",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(download_vlm_model, timeout=20 * MINUTES)
)

# --- App ---

app = modal.App(APP_NAME)


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=10 * MINUTES,
    max_containers=1,
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve_coder():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_DIR,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--served-model-name",
        MODEL_NAME,
        "--tensor-parallel-size",
        str(N_GPU),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--kv-cache-dtype",
        KV_CACHE_DTYPE,
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        TOOL_CALL_PARSER,
        "--trust-remote-code",
        "--enforce-eager",
        "--disable-log-requests",
    ]

    print("Starting vLLM:", *cmd)
    subprocess.Popen(" ".join(cmd), shell=True)


@app.function(
    image=vlm_image,
    gpu=f"{VLM_GPU_TYPE}:{VLM_N_GPU}",
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=10 * MINUTES,
)
@modal.concurrent(max_inputs=VLM_MAX_CONCURRENT_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve_vlm():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        VLM_MODEL_DIR,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--served-model-name",
        VLM_MODEL_NAME,
        "--tensor-parallel-size",
        str(VLM_N_GPU),
        "--max-model-len",
        str(VLM_MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(VLM_GPU_MEMORY_UTILIZATION),
        "--kv-cache-dtype",
        KV_CACHE_DTYPE,
        "--limit-mm-per-prompt",
        "image=5",
        "--trust-remote-code",
        "--enforce-eager",
        "--disable-log-requests",
    ]

    print("Starting vLLM (VLM):", *cmd)
    subprocess.Popen(" ".join(cmd), shell=True)


# --- Local entrypoint (smoke test via `modal run`) ---


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES):
    coder_url = serve_coder.get_web_url()
    vlm_url = serve_vlm.get_web_url()

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function that checks if a number is prime. Be concise."},
    ]

    # --- Test coder endpoint ---
    async with aiohttp.ClientSession(base_url=coder_url) as session:
        print(f"Coder health check: {coder_url}")
        async with session.get(
            "/health", timeout=aiohttp.ClientTimeout(total=test_timeout - MINUTES)
        ) as resp:
            assert resp.status == 200, f"Coder health check failed: {resp.status}"
        print("Coder health check passed")

        print(f"Sending chat completion request to {coder_url}")
        payload: dict[str, Any] = {
            "messages": messages,
            "model": MODEL_NAME,
            "stream": True,
            "max_tokens": 512,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        async with session.post(
            "/v1/chat/completions", json=payload, headers=headers
        ) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[len("data: "):]
                chunk = json.loads(line)
                assert chunk["object"] == "chat.completion.chunk"
                content = chunk["choices"][0]["delta"].get("content", "")
                print(content, end="")
        print("\nCoder smoke test passed!")

    # --- Test VLM endpoint ---
    async with aiohttp.ClientSession(base_url=vlm_url) as session:
        print(f"\nVLM health check: {vlm_url}")
        async with session.get(
            "/health", timeout=aiohttp.ClientTimeout(total=test_timeout - MINUTES)
        ) as resp:
            assert resp.status == 200, f"VLM health check failed: {resp.status}"
        print("VLM health check passed")

        print(f"Sending VLM completion request to {vlm_url}")
        vlm_payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": "Describe what a VLM is in one sentence."}],
            "model": VLM_MODEL_NAME,
            "max_tokens": 128,
        }
        async with session.post(
            "/v1/chat/completions", json=vlm_payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            content = data["choices"][0]["message"]["content"]
            print(f"VLM response: {content}")
        print("VLM smoke test passed!")

    print("\nAll smoke tests passed!")
