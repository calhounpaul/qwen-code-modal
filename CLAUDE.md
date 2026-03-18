# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**coding-agent-server** is a Modal-deployed inference server hosting AI models for coding, with dual backend support:

1. **Coder LLM** — `unsloth/Qwen3.5-397B-A17B-GGUF` (397B MoE, UD-IQ2_XXS GGUF) on 2x A100-80GB via llama.cpp (default, configurable)
2. **VLM** — `Qwen/Qwen3-VL-32B-Thinking-FP8` (32B dense, FP8) on A100-40GB via vLLM (auto-disabled for multimodal models)

**Inference backends** (select via `INFERENCE_BACKEND` env var):
- `llamacpp` (default) — GGUF models only, fast cold start (~30s), lower concurrency
- `vllm` — All quant formats (FP8/NVFP4/safetensors), fast batching, slow cold start (~5 min)

**Supported models** (select via `CODER_MODEL_NAME` env var):
- `unsloth/Qwen3.5-397B-A17B-GGUF` — 2x A100-80GB, llamacpp, GGUF (default, use `GGUF_PATTERN` to select quant)
- `unsloth/Qwen3-Coder-Next-GGUF` — A100-80GB, llamacpp, GGUF
- `GadflyII/Qwen3-Coder-Next-NVFP4` — A100-80GB, vllm, NVFP4 quantized
- `unsloth/Qwen3-Coder-Next-FP8-Dynamic` — A100-80GB, vllm, FP8
- `Sehyo/Qwen3.5-35B-A3B-NVFP4` — A100-40GB, vllm, multimodal
- `Qwen/Qwen3.5-35B-A3B-FP8` — A100-40GB, vllm, multimodal
- `nvidia/Qwen3.5-397B-A17B-NVFP4` — 4x A100-80GB, vllm, NVFP4

It serves as a fallback coding LLM for Claude Code via an OpenAI-compatible API, with an MCP server for VLM image analysis.

The project also installs **qwen-code** locally as a CLI coding agent backed by this server, with telemetry always disabled.

## Architecture

```
coding-agent-server (Modal App)
├── src/
│   └── coding_agent_server/
│       ├── deploy.py              # Modal app: serve_coder (vllm or llamacpp) + serve_vlm (vllm)
│       ├── config.py              # Model registry, GPU/backend/scaling config, model selection
│       ├── vlm_mcp_server.py      # MCP stdio server for VLM image analysis (auto-disabled for multimodal models)
│       └── __init__.py
├── scripts/
│   ├── download_models.py         # CPU-only model downloader to Modal Volume (GGUF pattern filtering)
│   └── install_qwen_code.sh       # Install qwen-code CLI + disable telemetry + auto-config VLM MCP
├── tests/
│   ├── test_health.py             # Health checks for both endpoints
│   └── test_vlm_mcp.py            # VLM MCP tool tests with generated vector images
└── run.sh                         # All-in-one CLI for deploy/install/test/logs
```

**Modal deployment**: Two `@app.function` entries with `@modal.web_server`:
- **`serve_coder`** — Uses llama.cpp (GGUF) or vLLM depending on `INFERENCE_BACKEND`. `max_containers=1`
- **`serve_vlm`** — Always vLLM. A100-40GB, FP8, image support (only deployed when VLM MCP is enabled)

**Model weight storage**: Shared Modal Volume (`coding-agent-models`) mounted at `/models`. A separate CPU-only script (`scripts/download_models.py`) downloads weights. For GGUF repos with multiple quants, only the selected `GGUF_PATTERN` is downloaded. Each model dir has a `model_metadata.json` tracking download status and model info. Scales to zero after 5 min idle.

**MCP server** (`vlm_mcp_server.py`): FastMCP stdio server with two tools:
- `analyze_image` — Analyze a local image file
- `compare_images` — Compare 2-5 images

Automatically disabled when using multimodal Qwen3.5 models (they have built-in image support).
Registered in `~/.qwen/settings.json` under `mcpServers.vlm-analyzer` via `./run.sh install`.

**Key design decisions**:
- Dual backend: llama.cpp for fast cold starts with GGUF, vLLM for FP8/NVFP4 with high concurrency
- Model weights on shared Modal Volume (not baked into images) — fast deploys, easy model switching
- `model_metadata.json` per model tracks download status, quantization, multimodal, GPU requirements
- FP8 KV cache (`--kv-cache-dtype fp8`) on vLLM endpoints
- `max_containers=1` for coder (single GPU setup serves all requests)
- `scaledown_window=300` (5min idle before scale-to-zero)
- Multimodal models auto-disable the VLM endpoint and MCP server

## Commands

```bash
# Download model weights to Modal Volume (CPU-only, run first)
./run.sh download-models
# or: modal run scripts/download_models.py

# Deploy server to Modal (mounts volume, no download)
./run.sh deploy
# or: modal deploy src/coding_agent_server/deploy.py

# Smoke test endpoints (runs in Modal cloud)
./run.sh smoke

# Install qwen-code CLI + qodal wrapper + VLM MCP server
./run.sh install

# Run tests against live endpoints
./run.sh test

# Check Modal app status
./run.sh logs
```

**Switch models:**
```bash
# Use a different GGUF quant (default: UD-IQ2_XXS)
GGUF_PATTERN=Q4_K_M ./run.sh download-models && GGUF_PATTERN=Q4_K_M ./run.sh deploy

# Switch to vLLM backend with FP8 model
INFERENCE_BACKEND=vllm CODER_MODEL_NAME=GadflyII/Qwen3-Coder-Next-NVFP4 ./run.sh download-models
INFERENCE_BACKEND=vllm CODER_MODEL_NAME=GadflyII/Qwen3-Coder-Next-NVFP4 ./run.sh deploy
```

## Modal Deployment Details

### Coder Endpoint (`serve_coder`)
- **Model**: `unsloth/Qwen3.5-397B-A17B-GGUF` (default, configurable via `CODER_MODEL_NAME`)
- **Backend**: llama.cpp (default) or vLLM (set `INFERENCE_BACKEND`)
- **GPU**: 2x A100-80GB (default) — varies by model
- **Endpoint**: `https://<workspace>--coding-agent-server-serve-coder.modal.run/v1`
- **Concurrency**: `max_containers=1`, `max_inputs=8` (llamacpp) or `128` (vllm)
- **Auth**: `requires_proxy_auth=True` — requires `Modal-Key` / `Modal-Secret` headers

### VLM Endpoint (`serve_vlm`)
- **Model**: `Qwen/Qwen3-VL-32B-Thinking-FP8`
- **Backend**: Always vLLM
- **GPU**: A100-40GB (~17 GiB FP8 weights, ~19 GiB for KV cache)
- **Endpoint**: `https://<workspace>--coding-agent-server-serve-vlm.modal.run/v1`
- **Concurrency**: `max_inputs=16`
- **Features**: Multi-image support (`--limit-mm-per-prompt image=5`), 32K context
- **Auth**: `requires_proxy_auth=True` — requires `Modal-Key` / `Modal-Secret` headers
- **Enabled**: Only when using non-multimodal models (auto-disabled for Qwen3.5 series)

## MCP Server

The VLM MCP server (`vlm_mcp_server.py`) runs as a stdio transport server:

**Tools:**
- `analyze_image(image_path, prompt)` — Read and analyze a local image
- `compare_images(image_paths, prompt)` — Compare 2-5 images

**Note:** Automatically disabled when using multimodal Qwen3.5 models.

**Env vars:**
- `VLM_ENDPOINT` — VLM endpoint base URL (required)
- `VLM_MODEL` — Model name (default: `Qwen/Qwen3-VL-32B-Thinking-FP8`)
- `VLM_TIMEOUT` — Request timeout in seconds (default: 300)
- `MODAL_PROXY_TOKEN_ID` — Modal proxy auth token ID (required for authenticated endpoints)
- `MODAL_PROXY_TOKEN_SECRET` — Modal proxy auth token secret (required for authenticated endpoints)
- `ENABLE_VLM_MCP` — Set to "0" to disable (default: "1")

**Dependencies:** `mcp[cli]`, `httpx`

## Qwen Code Integration

This package installs qwen-code and configures it to:
1. Use the Modal coder endpoint as `OPENAI_BASE_URL`
2. **Always disable telemetry** (`~/.qwen/settings.json` -> `telemetry.enabled: false`)
3. Set model to the configured `CODER_MODEL_NAME`
4. Register `vlm-analyzer` MCP server for image analysis (disabled for multimodal models)

Telemetry disable is enforced in `scripts/install_qwen_code.sh` and should be enforced in any Python install scripts.

## Development Rules

- **No fallbacks or fake data.** Never deploy stub endpoints, mock responses, placeholder models, or dummy data. Every feature must work end-to-end against the real Modal deployment before it ships. If a dependency isn't available yet, don't merge the code that depends on it.
- **No tools that can't work in the target environment.** This runs headless (CLI/SSH). Don't add features that require a display server, GUI session, or browser unless there is a real runtime that provides them.

## Future Plans (not yet implemented)

- MCP servers for embedding tools (semantic code search)
- These will be separate Modal functions added to the same app
