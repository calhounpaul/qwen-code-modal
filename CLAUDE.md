# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**coding-agent-server** is a Modal-deployed vLLM inference server hosting two models:
1. `unsloth/Qwen3-Coder-Next-FP8-Dynamic` (80B MoE, 3B active params, 256K context) — coding LLM on H200
2. `Qwen/Qwen3-VL-32B-Thinking-FP8` (32B dense, FP8) — vision-language model on A100-40GB

It serves as a fallback coding LLM for Claude Code via an OpenAI-compatible API, with an MCP server for VLM image analysis.

The project also installs **qwen-code** locally as a CLI coding agent backed by this server, with telemetry always disabled.

## Architecture

```
coding-agent-server (Modal App)
├── src/
│   └── coding_agent_server/
│       ├── deploy.py              # Modal app: serve_coder (H200) + serve_vlm (A100-40GB)
│       ├── config.py              # Model/GPU/scaling configuration
│       └── vlm_mcp_server.py      # MCP stdio server for VLM image analysis
├── scripts/
│   └── install_qwen_code.sh       # Install qwen-code CLI + disable telemetry
└── tests/
    └── test_health.py             # Health checks for both endpoints
```

**Modal deployment**: Two `@app.function` entries with `@modal.web_server`, each running vLLM:
- **`serve_coder`** — H200, FP8 coder model, tool calling enabled, `max_containers=1` (vLLM handles batching)
- **`serve_vlm`** — A100-40GB, FP8 VLM, image support via `--limit-mm-per-prompt image=5`

Model weights are downloaded during image build via `.run_function()` and baked directly into the container images for fast cold starts. Scales to zero after 5 min idle.

**MCP server** (`vlm_mcp_server.py`): FastMCP stdio server with three tools:
- `analyze_image` — Analyze a local image file
- `analyze_screenshot` — Capture and analyze current screen
- `compare_images` — Compare 2-5 images

Registered in `~/.qwen/settings.json` under `mcpServers.vlm-analyzer` via `./run.sh install`.

**Key design decisions**:
- vLLM (not llama.cpp) for FP8 dynamic quantization + continuous batching
- FP8 KV cache (`--kv-cache-dtype fp8`) on both endpoints
- `@modal.concurrent(max_inputs=128)` for coder, `max_inputs=16` for VLM
- `max_containers=1` for coder (single H200 handles all concurrent requests via vLLM batching)
- `scaledown_window=300` (5min idle before scale-to-zero)

## Reference Materials

All research docs and prior implementations are in `docs/tmp_data/` (gitignored):
- `docs/tmp_data/research/modal_docs_1.md` - Modal platform overview and getting started
- `docs/tmp_data/research/modal_docs_2.md` - Complete vLLM deployment guide with FP8 config, cost analysis, and troubleshooting
- `docs/tmp_data/non-modal-repo/` - Prior local deployment using llama.cpp + GGUF (Docker/llama-server approach)
- `docs/tmp_data/qwen-code-repo/` - Cloned qwen-code CLI source (github.com/QwenLM/qwen-code)

## Commands

```bash
# Deploy both endpoints to Modal
modal deploy src/coding_agent_server/deploy.py

# Smoke test both endpoints (runs on Modal cloud)
modal run src/coding_agent_server/deploy.py

# Install qwen-code CLI + qodal wrapper + VLM MCP server
./run.sh install

# Run tests against live endpoints
./run.sh test

# Check Modal app status
modal app list
modal app logs coding-agent-server
```

## Modal Deployment Details

### Coder Endpoint (`serve_coder`)
- **Model**: `unsloth/Qwen3-Coder-Next-FP8-Dynamic`
- **GPU**: H200 (141 GiB HBM3e)
- **Inference engine**: vLLM >= 0.15.0
- **Endpoint**: `https://<workspace>--coding-agent-server-serve-coder.modal.run/v1`
- **Concurrency**: `max_containers=1`, `max_inputs=128` (vLLM handles batching internally)
- **Features**: Tool calling (`--tool-call-parser qwen3_coder`), FP8 KV cache

### VLM Endpoint (`serve_vlm`)
- **Model**: `Qwen/Qwen3-VL-32B-Thinking-FP8`
- **GPU**: A100-40GB (~17 GiB FP8 weights, ~19 GiB for KV cache)
- **Endpoint**: `https://<workspace>--coding-agent-server-serve-vlm.modal.run/v1`
- **Concurrency**: `max_inputs=16`
- **Features**: Multi-image support (`--limit-mm-per-prompt image=5`), 32K context

### Context/Memory Trade-offs on H200 (141 GiB)

| Context Length | ~GPU Memory | Concurrent Users |
|----------------|-------------|------------------|
| 64K tokens     | ~95GB       | 20+              |
| 128K tokens    | ~115GB      | 10-15            |
| 256K tokens    | ~135GB      | 2-4              |

## MCP Server

The VLM MCP server (`vlm_mcp_server.py`) runs as a stdio transport server:

**Tools:**
- `analyze_image(image_path, prompt)` — Read and analyze a local image
- `analyze_screenshot(prompt)` — Capture screen and analyze
- `compare_images(image_paths, prompt)` — Compare 2-5 images

**Env vars:**
- `VLM_ENDPOINT` — VLM endpoint base URL (required)
- `VLM_MODEL` — Model name (default: `Qwen/Qwen3-VL-32B-Thinking-FP8`)
- `VLM_TIMEOUT` — Request timeout in seconds (default: 300)

**Dependencies:** `mcp[cli]`, `httpx`, `pillow`

## Qwen Code Integration

This package installs qwen-code and configures it to:
1. Use the Modal coder endpoint as `OPENAI_BASE_URL`
2. **Always disable telemetry** (`~/.qwen/settings.json` -> `telemetry.enabled: false`)
3. Set model to `unsloth/Qwen3-Coder-Next-FP8-Dynamic`
4. Register `vlm-analyzer` MCP server for image analysis

Telemetry disable is enforced in `scripts/install_qwen_code.sh` and should be enforced in any Python install scripts.

## Future Plans (not yet implemented)

- MCP servers for embedding tools (semantic code search)
- These will be separate Modal functions added to the same app
