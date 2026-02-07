"""Configuration constants for the coding-agent-server Modal deployment."""

# --- Coder model ---
MODEL_NAME = "unsloth/Qwen3-Coder-Next-FP8-Dynamic"
MODEL_DIR = "/model"  # local path where weights are baked into the image

# GPU — H200 (141 GiB HBM3e) fits 78 GiB FP8 weights + KV cache on 1 GPU
GPU_TYPE = "H200"
N_GPU = 1

# Scaling
SCALEDOWN_WINDOW = 300  # 5 minutes idle before scale-to-zero
MAX_CONCURRENT_INPUTS = 128  # vLLM handles batching internally

# vLLM engine
MAX_MODEL_LEN = 131072  # 128K — ~60 GiB free for KV cache on H200
GPU_MEMORY_UTILIZATION = 0.90
KV_CACHE_DTYPE = "fp8"
TOOL_CALL_PARSER = "qwen3_coder"

# --- VLM model ---
VLM_MODEL_NAME = "Qwen/Qwen3-VL-32B-Thinking-FP8"
VLM_MODEL_DIR = "/vlm-model"

# GPU — A100-40GB: 17 GiB FP8 weights leaves ~19 GiB for KV cache
VLM_GPU_TYPE = "A100-40GB"
VLM_N_GPU = 1

# VLM scaling
VLM_MAX_CONCURRENT_INPUTS = 16

# VLM vLLM engine
VLM_MAX_MODEL_LEN = 32768  # 32K — conservative for image analysis one-shots
VLM_GPU_MEMORY_UTILIZATION = 0.90

# --- Modal app ---
APP_NAME = "coding-agent-server"

# Server
VLLM_PORT = 8000
