# PLAN.md - Implementation Roadmap

## Phase 1: Core Server Deployment (Done)

Deploy `unsloth/Qwen3-Coder-Next-FP8-Dynamic` on Modal with an OpenAI-compatible API.

- [x] `src/coding_agent_server/config.py` — Model, GPU, scaling configuration
- [x] `src/coding_agent_server/deploy.py` — Modal app with vLLM on H200
- [x] `scripts/install_qwen_code.sh` — Install qwen-code CLI + disable telemetry
- [x] `.env.example` — Environment template
- [x] `tests/test_health.py` — Health + chat completion smoke tests
- [x] Deploy and validate on H200

### Key Configuration

```python
MODEL_NAME = "unsloth/Qwen3-Coder-Next-FP8-Dynamic"
GPU_TYPE = "H200"  # 141 GiB HBM3e
MAX_MODEL_LEN = 131072  # 128K context
MAX_CONCURRENT_INPUTS = 128  # vLLM handles batching internally
```

### Notes

- H200 (141 GiB) fits 78 GiB FP8 weights + 42 GiB KV cache (~27x concurrency at 128K)
- FP8 KV cache (`--kv-cache-dtype fp8`) maximizes context window
- `max_containers=1` — single H200 handles all concurrent requests via vLLM batching
- Model weights baked into container image for fast cold starts (no volumes needed)

---

## Phase 2: VLM Endpoint + MCP Server (Done)

Deploy `Qwen/Qwen3-VL-32B-Thinking-FP8` on A100-40GB for image analysis, with an MCP server for qwen-code integration.

- [x] `serve_vlm` function in `deploy.py` — A100-40GB, 32K context, multi-image support
- [x] `src/coding_agent_server/vlm_mcp_server.py` — FastMCP stdio server with image analysis tools
- [x] VLM health tests in `test_health.py`
- [x] MCP server registration in `run.sh install`
- [x] `qodal` wrapper command

### VLM Details

- A100-40GB: 17 GiB FP8 weights, ~19 GiB for KV cache
- `--limit-mm-per-prompt image=5` (max 5 images per request)
- No tool calling (VLM is text+image only)
- MCP tools: `analyze_image`, `compare_images`

---

## Phase 3: Future (Planned, Not Yet)

### Embedding MCP Server
- Deploy an embedding model on Modal (e.g., `nomic-embed-text` or `bge-large`)
- Expose as MCP tool for semantic code search
- Could share the same Modal app with a separate `@app.function`

### Other Potential MCP Tools
- Code execution sandbox (Modal Sandboxes)
- Web search / RAG pipeline

---

## Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Inference engine | vLLM | FP8 dynamic support, continuous batching, tool calling |
| Quantization | FP8 dynamic (Unsloth) | Best quality for H200/A100 |
| Cloud platform | Modal | Serverless GPU, scale-to-zero, per-second billing |
| Coder GPU | H200 | 141 GiB fits 80B MoE FP8 + large KV cache |
| VLM GPU | A100-40GB | 32B dense FP8 fits comfortably |
| Default context | 128K (coder), 32K (VLM) | Balance of context vs concurrency |
| Local CLI | qwen-code | Open-source, works with OpenAI-compatible APIs |
| Telemetry | Always off | Privacy requirement |
