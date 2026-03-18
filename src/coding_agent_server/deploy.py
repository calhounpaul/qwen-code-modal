"""Modal deployment of vLLM or llama.cpp serving coding + VLM models.

Deploy:  modal deploy src/coding_agent_server/deploy.py
Test:    modal run src/coding_agent_server/deploy.py

Model Selection:
    Set CODER_MODEL_NAME environment variable to select a different model.

Backend Selection:
    Set INFERENCE_BACKEND to "vllm" or "llamacpp".
    - vllm: All quant formats, fast batching, slow cold start
    - llamacpp: GGUF only, fast cold start, lower concurrency
"""

import json
import sys
from pathlib import Path
from typing import Any

import aiohttp
import modal

# Import configuration from config.py
# Support both `modal deploy` (runs as script) and package imports
try:
    from .config import (
        APP_NAME,
        CODER_MODEL_NAME,
        ENABLE_VLM_MCP,
        GGUF_PATTERN,
        GPU_MEMORY_UTILIZATION,
        GPU_TYPE,
        INFERENCE_BACKEND,
        IS_MULTIMODAL_MODEL,
        KV_CACHE_DTYPE,
        LLAMACPP_CTX_SIZE,
        LLAMACPP_FLASH_ATTN,
        LLAMACPP_N_GPU_LAYERS,
        MAX_CONCURRENT_INPUTS,
        MAX_MODEL_LEN,
        MODEL_DIR,
        MODEL_NAME,
        N_GPU,
        SCALEDOWN_WINDOW,
        SERVER_PORT,
        TOOL_CALL_PARSER,
        VLM_GPU_MEMORY_UTILIZATION,
        VLM_GPU_TYPE,
        VLM_MAX_CONCURRENT_INPUTS,
        VLM_MAX_MODEL_LEN,
        VLM_MODEL_DIR,
        VLM_MODEL_NAME,
        VLM_N_GPU,
        VOLUME_MOUNT_PATH,
        VOLUME_NAME,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from config import (  # type: ignore[no-redef]
        APP_NAME,
        CODER_MODEL_NAME,
        ENABLE_VLM_MCP,
        GGUF_PATTERN,
        GPU_MEMORY_UTILIZATION,
        GPU_TYPE,
        INFERENCE_BACKEND,
        IS_MULTIMODAL_MODEL,
        KV_CACHE_DTYPE,
        LLAMACPP_CTX_SIZE,
        LLAMACPP_FLASH_ATTN,
        LLAMACPP_N_GPU_LAYERS,
        MAX_CONCURRENT_INPUTS,
        MAX_MODEL_LEN,
        MODEL_DIR,
        MODEL_NAME,
        N_GPU,
        SCALEDOWN_WINDOW,
        SERVER_PORT,
        TOOL_CALL_PARSER,
        VLM_GPU_MEMORY_UTILIZATION,
        VLM_GPU_TYPE,
        VLM_MAX_CONCURRENT_INPUTS,
        VLM_MAX_MODEL_LEN,
        VLM_MODEL_DIR,
        VLM_MODEL_NAME,
        VLM_N_GPU,
        VOLUME_MOUNT_PATH,
        VOLUME_NAME,
    )

# --- Images ---

MINUTES = 60  # seconds

config_path = str(Path(__file__).parent / "config.py")

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install("vllm>=0.15.0")
    .add_local_file(config_path, "/root/config.py", copy=True)
)

llamacpp_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git", "cmake", "build-essential")
    .run_commands(
        "ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 "
        "&& cmake --build build --config Release -t llama-server -j$(nproc)",
        "cp /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server",
        "rm /usr/lib/x86_64-linux-gnu/libcuda.so.1",  # remove stub; real driver provides this at runtime
    )
    .add_local_file(config_path, "/root/config.py", copy=True)
)

# Pick image based on backend
coder_image = llamacpp_image if INFERENCE_BACKEND == "llamacpp" else vllm_image

# --- Volume (shared model weight storage) ---

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# --- App ---

app = modal.App(APP_NAME)


def _check_model_status(model_dir: str, model_name: str) -> str | None:
    """Return an error message if model is missing or incomplete, else None."""
    import json
    import os

    status_path = os.path.join(model_dir, "model_metadata.json")
    if not os.path.isdir(model_dir) or not os.path.exists(status_path):
        return (
            f"Model not found at {model_dir}. "
            f"Run 'modal run scripts/download_models.py' first to download {model_name}."
        )
    with open(status_path) as f:
        meta = json.load(f)
    if meta.get("download_status") != "completed":
        return (
            f"Model download incomplete at {model_dir} (status: {meta.get('download_status')}). "
            f"Run 'modal run scripts/download_models.py --force' to retry."
        )
    return None


def _find_gguf_file(model_dir: str, gguf_pattern: str = "") -> str:
    """Find the primary GGUF file in a model directory.

    For sharded models, returns the first shard — llama.cpp auto-detects the rest.
    If gguf_pattern is set, only looks in that subdirectory or for that filename.
    """
    import glob
    import os

    if gguf_pattern:
        # Check subdirectory first (sharded models like UD-IQ2_XXS/)
        subdir = os.path.join(model_dir, gguf_pattern)
        if os.path.isdir(subdir):
            gguf_files = sorted(glob.glob(os.path.join(subdir, "*.gguf")))
            if gguf_files:
                return gguf_files[0]
        # Try as filename pattern
        gguf_files = sorted(glob.glob(os.path.join(model_dir, f"*{gguf_pattern}*.gguf")))
        if gguf_files:
            return gguf_files[0]

    # Fallback: any GGUF file
    gguf_files = sorted(glob.glob(os.path.join(model_dir, "**", "*.gguf"), recursive=True))
    if not gguf_files:
        raise RuntimeError(f"No .gguf files found in {model_dir}")
    return gguf_files[0]


@app.function(
    image=coder_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=10 * MINUTES,
    max_containers=1,
    volumes={VOLUME_MOUNT_PATH: volume},
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
@modal.web_server(port=SERVER_PORT, startup_timeout=10 * MINUTES, requires_proxy_auth=True)
def serve_coder():
    import subprocess

    status = _check_model_status(MODEL_DIR, CODER_MODEL_NAME)
    if status:
        raise RuntimeError(status)

    if INFERENCE_BACKEND == "llamacpp":
        gguf_path = _find_gguf_file(MODEL_DIR, GGUF_PATTERN)
        cmd = [
            "llama-server",
            "-m", gguf_path,
            "--host", "0.0.0.0",
            "--port", str(SERVER_PORT),
            "-ngl", str(LLAMACPP_N_GPU_LAYERS),
            "--ctx-size", str(LLAMACPP_CTX_SIZE),
            "--alias", MODEL_NAME,
        ]
        if LLAMACPP_FLASH_ATTN:
            cmd.extend(["--flash-attn", "on"])
        if N_GPU > 1:
            cmd.extend(["--split-mode", "layer"])
    else:
        cmd = [
            "vllm",
            "serve",
            MODEL_DIR,
            "--host", "0.0.0.0",
            "--port", str(SERVER_PORT),
            "--served-model-name", MODEL_NAME,
            "--tensor-parallel-size", str(N_GPU),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--kv-cache-dtype", KV_CACHE_DTYPE,
            "--enable-auto-tool-choice",
            "--tool-call-parser", TOOL_CALL_PARSER,
            "--trust-remote-code",
            "--enforce-eager",
            "--disable-log-requests",
        ]

    print(f"Starting {INFERENCE_BACKEND}:", *cmd)
    subprocess.Popen(cmd)


# VLM endpoint only if VLM MCP is enabled (not using multimodal model)
# Always uses vLLM regardless of INFERENCE_BACKEND
if ENABLE_VLM_MCP:

    @app.function(
        image=vllm_image,
        gpu=f"{VLM_GPU_TYPE}:{VLM_N_GPU}",
        scaledown_window=SCALEDOWN_WINDOW,
        timeout=10 * MINUTES,
        volumes={VOLUME_MOUNT_PATH: volume},
    )
    @modal.concurrent(max_inputs=VLM_MAX_CONCURRENT_INPUTS)
    @modal.web_server(port=SERVER_PORT, startup_timeout=10 * MINUTES, requires_proxy_auth=True)
    def serve_vlm():
        import subprocess

        status = _check_model_status(VLM_MODEL_DIR, VLM_MODEL_NAME)
        if status:
            raise RuntimeError(status)

        cmd = [
            "vllm",
            "serve",
            VLM_MODEL_DIR,
            "--host", "0.0.0.0",
            "--port", str(SERVER_PORT),
            "--served-model-name", VLM_MODEL_NAME,
            "--tensor-parallel-size", str(VLM_N_GPU),
            "--max-model-len", str(VLM_MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(VLM_GPU_MEMORY_UTILIZATION),
            "--kv-cache-dtype", KV_CACHE_DTYPE,
            "--limit-mm-per-prompt", '{"image":5}',
            "--trust-remote-code",
            "--enforce-eager",
            "--disable-log-requests",
        ]

        print("Starting vLLM (VLM):", *cmd)
        subprocess.Popen(cmd)


# --- Local entrypoint (smoke test via `modal run`) ---

if ENABLE_VLM_MCP:

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

else:
    # Non-multimodal model: only test coder endpoint
    @app.local_entrypoint()
    async def test(test_timeout=10 * MINUTES):
        coder_url = serve_coder.get_web_url()

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
            print(f"\nNote: VLM MCP server disabled - {CODER_MODEL_NAME} has built-in multimodal capabilities.")
