# qwen-code-modal

Modal-deployed vLLM inference server hosting two models for AI-assisted coding:

1. **Qwen3-Coder-Next** (80B MoE, FP8) on H200 — coding LLM with tool calling
2. **Qwen3-VL-32B-Thinking** (32B dense, FP8) on A100-40GB — vision-language model for image analysis

Serves as an OpenAI-compatible backend for [qwen-code](https://github.com/QwenLM/qwen-code) and as a fallback coding LLM for Claude Code. Scales to zero when idle.

## Quick Start

### 1. Configure

```bash
cp .env.example .env
# Edit .env and set MODAL_WORKSPACE to your Modal workspace name
```

### 2. Deploy

```bash
./run.sh deploy
```

This deploys both endpoints to Modal. First deploy builds container images with baked-in model weights (~78 GiB coder + ~17 GiB VLM).

### 3. Install qwen-code CLI

```bash
./run.sh install
```

Installs qwen-code, creates a `qodal` wrapper command pointing at your Modal endpoint, installs VLM MCP server dependencies, and registers the VLM image analysis tools.

### 4. Use

```bash
# Use qwen-code with the Modal backend
qodal

# Or via the OpenAI-compatible API
curl https://YOUR-WORKSPACE--coding-agent-server-serve-coder.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3-Coder-Next-FP8-Dynamic",
    "messages": [{"role": "user", "content": "Write a Python prime checker"}],
    "max_tokens": 4096
  }'
```

## Architecture

```
coding-agent-server (Modal App)
├── src/coding_agent_server/
│   ├── deploy.py              # Modal app: serve_coder (H200) + serve_vlm (A100-40GB)
│   ├── config.py              # Model/GPU/scaling configuration
│   └── vlm_mcp_server.py      # MCP stdio server for VLM image analysis
├── scripts/
│   └── install_qwen_code.sh   # Install qwen-code CLI + disable telemetry
├── tests/
│   └── test_health.py         # Health checks for both endpoints
└── run.sh                     # All-in-one CLI (deploy, install, test, etc.)
```

### Endpoints

| Endpoint | Model | GPU | Context | Concurrency |
|----------|-------|-----|---------|-------------|
| `serve_coder` | `unsloth/Qwen3-Coder-Next-FP8-Dynamic` | H200 (141 GiB) | 128K | 128 inputs, 1 container |
| `serve_vlm` | `Qwen/Qwen3-VL-32B-Thinking-FP8` | A100-40GB | 32K | 16 inputs |

Both endpoints use FP8 KV cache, scale to zero after 5 minutes idle, and serve an OpenAI-compatible API.

### MCP Server

The VLM MCP server (`vlm_mcp_server.py`) provides image analysis tools for qwen-code:

- **`analyze_image`** — Analyze a local image file
- **`analyze_screenshot`** — Capture and analyze the current screen
- **`compare_images`** — Compare 2-5 images side by side

Registered automatically via `./run.sh install`.

## Commands

```bash
./run.sh deploy    # Deploy both endpoints to Modal
./run.sh install   # Install qwen-code + qodal wrapper + VLM MCP server
./run.sh smoke     # Smoke test via modal run
./run.sh test      # Run pytest health checks against live endpoints
./run.sh logs      # Tail Modal app logs
./run.sh env       # Print configured env vars
./run.sh qwen      # Launch qwen-code with endpoint configured
```

## Requirements

- Python 3.10+
- [Modal](https://modal.com) account (`pip install modal && modal setup`)
- Modal workspace with H200 and A100-40GB GPU access

## License

MIT
