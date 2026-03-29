# qwen-code-modal Project Context

## Project Overview

This is a **Modal-deployed inference server** with dual backend support (llama.cpp + vLLM) hosting AI models for AI-assisted coding:

1. **Qwen3.5-397B-A17B** (397B MoE, UD-IQ2_XXS GGUF) on 2x A100-80GB via llama.cpp — primary coding LLM (default)
2. **Qwen3-VL-32B-Thinking** (32B dense, FP8 quantized) on A100-40GB via vLLM — vision-language model for image analysis

**Inference backends** (select via `INFERENCE_BACKEND` env var):
- `llamacpp` (default) — GGUF models only, fast cold start (~30s), lower concurrency
- `vllm` — All quant formats (FP8/NVFP4/safetensors), fast batching, slow cold start (~5 min)

**Supported models** (select via `CODER_MODEL_NAME` env var):
- `unsloth/Qwen3.5-397B-A17B-GGUF` — 2x A100-80GB, llamacpp (default, `GGUF_PATTERN=UD-IQ2_XXS`)
- `unsloth/Qwen3-Coder-Next-GGUF` — A100-80GB, llamacpp
- `GadflyII/Qwen3-Coder-Next-NVFP4` — A100-80GB, vllm, NVFP4 quantized
- `unsloth/Qwen3-Coder-Next-FP8-Dynamic` — A100-80GB, vllm, FP8
- `Sehyo/Qwen3.5-35B-A3B-NVFP4` — A100-40GB, vllm, multimodal
- `Qwen/Qwen3.5-35B-A3B-FP8` — A100-40GB, vllm, multimodal
- `nvidia/Qwen3.5-397B-A17B-NVFP4` — 4x A100-80GB, vllm, NVFP4

The server exposes an **OpenAI-compatible API** and serves as:
- A fallback coding LLM for **Claude Code**
- The primary backend for **qwen-code** CLI (via the `qodal` wrapper)

**Key architectural decisions:**
- Dual backend: **llama.cpp** for GGUF (fast cold start) + **vLLM** for FP8/NVFP4 (fast batching)
- Model weights are stored on a **shared Modal Volume** (`coding-agent-models`), downloaded via a separate CPU-only script
- **Scales to zero** after 5 minutes of inactivity (per-second billing)
- **Proxy Auth required** for all web endpoints (Modal-Key/Modal-Secret headers)
- **VLM MCP server disabled automatically** when using multimodal Qwen3.5 models (they have built-in multimodal capabilities)

## Project Structure

```
<project-root>/
├── src/coding_agent_server/
│   ├── deploy.py              # Modal app with serve_coder (conditional A100) + serve_vlm (conditional)
│   ├── config.py              # Configuration constants (models, GPUs, scaling, model selection)
│   ├── vlm_mcp_server.py      # MCP stdio server for VLM image analysis tools (auto-disabled for multimodal models)
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_health.py         # Health check pytest tests for both endpoints
│   └── test_vlm_mcp.py        # VLM MCP tool tests with generated vector images
├── scripts/
│   ├── download_models.py     # CPU-only model downloader to Modal Volume
│   └── install_qwen_code.sh   # Install qwen-code CLI + disable telemetry + auto-config VLM MCP
├── .env.example               # Environment variable template
├── run.sh                     # All-in-one CLI for deploy/install/test
├── .gitignore
├── LICENSE
├── README.md
├── PLAN.md                      # Implementation roadmap
├── CLAUDE.md                    # Claude Code-specific context
└── QWEN.md                      # This file
```

## Building and Running

### Prerequisites
- Python 3.10+ (project uses Python 3.12 in Docker image)
- npm (required for qwen-code installation)
- Modal account (`pip install modal && modal setup`)
- Modal workspace with A100-80GB (or A100-40GB for multimodal models) GPU access

### Configuration

1. Copy the environment template and configure:
   ```bash
   cp .env.example .env
   # Edit .env with your Modal workspace and proxy auth tokens
   ```

2. Set required values in `.env`:
   - `MODAL_WORKSPACE` — your Modal workspace name
   - `MODAL_PROXY_TOKEN_ID` — from Modal dashboard → Settings → Proxy Auth Tokens
   - `MODAL_PROXY_TOKEN_SECRET` — from Modal dashboard → Settings → Proxy Auth Tokens

### Deployment

```bash
# First time: download model weights to Modal Volume (CPU-only)
./run.sh download-models

# Deploy both endpoints to Modal (coder + VLM)
./run.sh deploy

# Or manually:
modal run scripts/download_models.py
modal deploy src/coding_agent_server/deploy.py
```

Model weights are stored on a shared Modal Volume. The first `download-models` run downloads weights (~44 GiB for coder + ~17 GiB for VLM). Subsequent runs are idempotent (skipped unless `--force`). Deploys are fast since they only build the CUDA+vLLM image and mount the volume.

**Note:** When using multimodal models (Qwen3.5-35B series), the VLM endpoint is automatically skipped since the model has built-in multimodal capabilities.

### Installation (for qwen-code)

```bash
# Install qwen-code CLI, create qodal wrapper, register VLM MCP server (if needed)
./run.sh install
```

This:
- Installs qwen-code if not present
- Creates `~/.local/bin/qodal` wrapper pointing to your Modal endpoint
- Installs MCP server dependencies (`mcp[cli]`, `httpx`) if using VLM MCP
- Registers `vlm-analyzer` MCP server in `~/.qwen/settings.json` (if using non-multimodal model)
- Disables telemetry in qwen-code settings

**Note:** The VLM MCP server is automatically disabled when using multimodal Qwen3.5 models.

### Testing

```bash
# Smoke test via modal run (runs in cloud, not local pytest)
./run.sh smoke

# Run pytest health checks against live endpoints
./run.sh test

# Or manually:
ENDPOINT_URL="https://WORKSPACE--coding-agent-server-serve-coder.modal.run" \
VLM_ENDPOINT_URL="https://WORKSPACE--coding-agent-server-serve-vlm.modal.run" \
MODAL_PROXY_TOKEN_ID=... \
MODAL_PROXY_TOKEN_SECRET=... \
    pytest tests/test_health.py -v
```

**Note:** For multimodal models, skip the VLM endpoint tests since the VLM MCP server is disabled.

### VLM MCP Tool Tests

Tests for the VLM MCP tools (`test_vlm_mcp.py`) generate temporary vector images and test the VLM endpoint:

- **Triangle images** - Random triangles with configurable colors
- **Suit symbols** - Playing card suits (♣, ♦, ♥, ♠)
- **Color blocks** - Solid color blocks for color recognition
- **Wingdings-style symbols** - Checkmarks, crosses, stars, hearts, arrows, etc.

**Note:** These tests are **skipped automatically** when using multimodal Qwen3.5 models (the VLM MCP server is disabled in that case).

**Requirements:**
```bash
pip install pytest-asyncio pillow
```

**Usage:**
```bash
VLM_ENDPOINT_URL="https://WORKSPACE--coding-agent-server-serve-vlm.modal.run" \
VLM_MODEL="Qwen/Qwen3-VL-32B-Thinking-FP8" \
MODAL_PROXY_TOKEN_ID=YOUR_TOKEN_ID \
MODAL_PROXY_TOKEN_SECRET=YOUR_TOKEN_SECRET \
    pytest tests/test_vlm_mcp.py -v
```

The tests use the same `_vlm_request()` function as the MCP server to send requests to the Modal VLM endpoint.

### Usage

```bash
# Launch qwen-code with Modal backend
qodal

# Or use the raw OpenAI-compatible API (requires proxy auth headers):
curl https://YOUR-WORKSPACE--coding-agent-server-serve-coder.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Modal-Key: YOUR_PROXY_TOKEN_ID" \
  -H "Modal-Secret: YOUR_PROXY_TOKEN_SECRET" \
  -d '{
    "model": "unsloth/Qwen3-Coder-Next-FP8-Dynamic",
    "messages": [{"role": "user", "content": "Write a Python prime checker"}],
    "max_tokens": 4096
  }'

# View logs
./run.sh logs
```

## API Endpoints

### Coder Endpoint (`serve_coder`)

| Setting | Value |
|---------|-------|
| **Model** | `GadflyII/Qwen3-Coder-Next-NVFP4` (default) |
| **GPU** | A100-80GB (single) or A100-40GB (multimodal) or 4x A100-80GB (397B) |
| **Context** | 128K tokens |
| **Concurrency** | 128 inputs (vLLM handles internal batching) |
| **Containers** | max 1 (single GPU serves all requests) |
| **Scale-down** | 5 min idle |
| **Auth** | Proxy auth required (`Modal-Key`/`Modal-Secret`) |
| **Port** | 8000 |
| **Features** | Tool calling enabled (`--tool-call-parser qwen3_coder`), FP8 KV cache |

**Note:** The default model is `GadflyII/Qwen3-Coder-Next-NVFP4`. Select other models via `CODER_MODEL_NAME` env var.

### VLM Endpoint (`serve_vlm`)

| Setting | Value |
|---------|-------|
| **Model** | `Qwen/Qwen3-VL-32B-Thinking-FP8` |
| **GPU** | A100-40GB (17 GiB FP8 weights) |
| **Context** | 32K tokens |
| **Concurrency** | 16 inputs |
| **Scale-down** | 5 min idle |
| **Auth** | Proxy auth required |
| **Port** | 8000 |
| **Features** | Multi-image support (max 5), FP8 KV cache, no tool calling |
| **Enabled** | Only when using non-multimodal models (disabled for Qwen3.5 series) |

**Note:** The VLM endpoint is automatically skipped during deployment when using multimodal Qwen3.5 models.

## Development Conventions

### Code Style
- **Python** — Follows PEP 8 with modern features (async/await, type hints)
- **Shell scripts** — Use `bash` with `set -e` for error handling
- **Logging** — Use standard `logging` module; `FastMCP` handles MCP logging
- **No fallbacks** — Never deploy stub endpoints, mock responses, or placeholder models

### Environment Management
- Virtual environment at `.venv/` (gitignored)
- Environment variables loaded from `.env` (gitignored)
- Template at `.env.example` (committed)
- Proxy auth tokens required for all authenticated endpoints

### Testing Practices
- Health checks via `pytest` in `tests/test_health.py`
- Tests are **skipped** unless `ENDPOINT_URL` / `VLM_ENDPOINT_URL` env vars are set
- Smoke tests run via `modal run` (executes in Modal cloud)
- Timeout set to 600s to accommodate cold starts

### MCP Server
The VLM MCP server (`vlm_mcp_server.py`) provides:
- **`analyze_image(image_path, prompt)`** — Analyze a single local image
- **`compare_images(image_paths, prompt)`** — Compare 2-5 images

**Note:** This server is **disabled automatically** when using multimodal Qwen3.5 models (`Sehyo/Qwen3.5-35B-A3B-NVFP4` or `Qwen/Qwen3.5-35B-A3B-FP8`). These models have built-in multimodal capabilities and can process images directly via the OpenAI-compatible API.

**Env vars:**
- `VLM_ENDPOINT` — VLM endpoint base URL (required)
- `VLM_MODEL` — Model name (default: `Qwen/Qwen3-VL-32B-Thinking-FP8`)
- `VLM_TIMEOUT` — Request timeout in seconds (default: 300)
- `MODAL_PROXY_TOKEN_ID` — Modal proxy auth token ID
- `MODAL_PROXY_TOKEN_SECRET` — Modal proxy auth token secret
- `ENABLE_VLM_MCP` — Set to "0" to disable (default: "1")

## Key Configuration Files

### Model Configuration (`config.py`)
- Coder: A100-80GB (default), 128K context, 128 max inputs, FP8 KV cache, tool calling enabled
- VLM: A100-40GB (~17 GiB FP8 weights), 32K context, 16 max inputs, FP8 KV cache, max 5 images
- **Model Selection:** Set `CODER_MODEL_NAME` to one of:
  - `GadflyII/Qwen3-Coder-Next-NVFP4` (A100-80GB, default)
  - `unsloth/Qwen3-Coder-Next-FP8-Dynamic` (A100-80GB)
  - `Sehyo/Qwen3.5-35B-A3B-NVFP4` (A100-40GB, multimodal)
  - `Qwen/Qwen3.5-35B-A3B-FP8` (A100-40GB, multimodal)
  - `unsloth/Qwen3.5-397B-A17B-GGUF-Q3_K_M` (4x A100-80GB, ~177 GB GGUF)
  - `nvidia/Qwen3.5-397B-A17B-NVFP4` (4x A100-80GB, native NVFP4)
- **Multimodal models** automatically disable the VLM MCP server

### Deployment (`deploy.py`)
- Two `@app.function` entries with `@modal.web_server`
- vLLM serve command with FP8-specific flags
- Startup timeout: 10 minutes (allows time for cold starts)

## Requirements & Dependencies

**Runtime:**
- Python 3.12 (in Docker image)
- CUDA 12.8.0
- vLLM >= 0.15.0
- huggingface_hub

**Python deps (project):**
- modal >= 0.73
- aiohttp (for smoke tests)
- httpx (for MCP server)
- mcp (for MCP stdio transport)

**Test deps:**
- pytest (for test framework)
- pytest-asyncio (for async test support)
- pillow (for generating test images)

**System deps:**
- npm (for qwen-code installation)
- curl (for qwen-code installer)

## Future Plans

Per `PLAN.md`:
- Embedding MCP server (semantic code search)
- Code execution sandbox via Modal Sandboxes
- Web search / RAG pipeline

## Related Documentation

- `README.md` — User-facing quick start guide
- `PLAN.md` — Implementation roadmap and decision log
- `CLAUDE.md` — Claude Code-specific context
- `.env.example` — Environment variable template
