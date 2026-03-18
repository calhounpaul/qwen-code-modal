"""Configuration constants for the coding-agent-server Modal deployment.

Model Selection:
    Set CODER_MODEL_NAME to one of the supported models below.
    GPU requirements will be automatically selected based on the model.

Inference Backend:
    Set INFERENCE_BACKEND to "vllm" or "llamacpp".
    - vllm: All quant formats (FP8/NVFP4/safetensors), fast concurrency, slow cold start
    - llamacpp: GGUF only, fast cold start, lower VRAM, limited concurrency

Multimodal Models:
    The Qwen3.5 series (Sehyo/Qwen3.5-35B-A3B-NVFP4, Qwen/Qwen3.5-35B-A3B-FP8)
    have built-in multimodal capabilities. When selected, the VLM MCP server
    is automatically disabled and qwen-code can use image input directly.
"""

import os

# --- Coder model configuration ---
CODER_MODEL_NAME = os.environ.get("CODER_MODEL_NAME", "unsloth/Qwen3.5-397B-A17B-GGUF")

# --- Inference backend ---
# "vllm" — all formats, fast batching, slow cold start (~5 min)
# "llamacpp" — GGUF only, fast cold start (~30s), lower concurrency
INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "llamacpp")

# --- Volume configuration ---
VOLUME_NAME = "coding-agent-models"
VOLUME_MOUNT_PATH = "/models"

# --- Model registry ---
# Each entry: {gpu_type, n_gpu, multimodal, quantization, hf_url, backend}
# backend: "any" (works with both), "vllm" (vllm only), "llamacpp" (llamacpp only)
# GGUF file selection for multi-quant repos (set via env var)
# Only used when model repo contains multiple GGUF files
# GGUF file/pattern for multi-quant repos
# For sharded models, use the subdirectory name (e.g. "UD-IQ2_XXS")
GGUF_PATTERN = os.environ.get("GGUF_PATTERN", "UD-IQ2_XXS")

MODEL_REGISTRY = {
    # --- llama.cpp (GGUF) models ---
    "unsloth/Qwen3.5-397B-A17B-GGUF": {
        "gpu_type": "A100-80GB",
        "n_gpu": 3,  # 115GB weights + 96K ctx KV cache needs ~125GB headroom
        "multimodal": True,  # Qwen3.5 has built-in vision — no VLM MCP needed
        "quantization": "GGUF",
        "backend": "llamacpp",
        "gguf_pattern": GGUF_PATTERN,  # UD-IQ2_XXS=115GB(3xA100-80GB)
        "hf_url": "https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF",
    },
    "unsloth/Qwen3-Coder-Next-GGUF": {
        "gpu_type": "A100-80GB",
        "n_gpu": 1,
        "multimodal": False,
        "quantization": "GGUF",
        "backend": "llamacpp",
        "gguf_pattern": GGUF_PATTERN,  # Q4_K_M=48.5GB, Q3_K_M=38.3GB
        "hf_url": "https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF",
    },
    # --- vLLM models ---
    "GadflyII/Qwen3-Coder-Next-NVFP4": {
        "gpu_type": "A100-80GB",
        "n_gpu": 1,
        "multimodal": False,
        "quantization": "NVFP4",
        "backend": "vllm",
        "hf_url": "https://huggingface.co/GadflyII/Qwen3-Coder-Next-NVFP4",
    },
    "unsloth/Qwen3-Coder-Next-FP8-Dynamic": {
        "gpu_type": "A100-80GB",
        "n_gpu": 1,
        "multimodal": False,
        "quantization": "FP8",
        "backend": "vllm",
        "hf_url": "https://huggingface.co/unsloth/Qwen3-Coder-Next-FP8-Dynamic",
    },
    "Sehyo/Qwen3.5-35B-A3B-NVFP4": {
        "gpu_type": "A100-40GB",
        "n_gpu": 1,
        "multimodal": True,
        "quantization": "NVFP4",
        "backend": "vllm",
        "hf_url": "https://huggingface.co/Sehyo/Qwen3.5-35B-A3B-NVFP4",
    },
    "Qwen/Qwen3.5-35B-A3B-FP8": {
        "gpu_type": "A100-40GB",
        "n_gpu": 1,
        "multimodal": True,
        "quantization": "FP8",
        "backend": "vllm",
        "hf_url": "https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8",
    },
    "nvidia/Qwen3.5-397B-A17B-NVFP4": {
        "gpu_type": "A100-80GB",
        "n_gpu": 4,
        "multimodal": False,
        "quantization": "NVFP4",
        "backend": "vllm",
        "hf_url": "https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4",
    },
}

# VLM model metadata (always vLLM — llama.cpp doesn't support VL models well)
VLM_MODEL_REGISTRY = {
    "Qwen/Qwen3-VL-32B-Thinking-FP8": {
        "gpu_type": "A100-40GB",
        "n_gpu": 1,
        "multimodal": True,
        "quantization": "FP8",
        "backend": "vllm",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking-FP8",
    },
}

# Derive GPU config from registry
_model_info = MODEL_REGISTRY.get(CODER_MODEL_NAME, {"gpu_type": "A100-80GB", "n_gpu": 1, "multimodal": False, "backend": "vllm"})
GPU_TYPE = _model_info["gpu_type"]
N_GPU = _model_info["n_gpu"]

# Validate backend compatibility
_model_backend = _model_info.get("backend", "any")
if _model_backend not in ("any", INFERENCE_BACKEND):
    raise ValueError(
        f"Model {CODER_MODEL_NAME} requires backend '{_model_backend}' "
        f"but INFERENCE_BACKEND='{INFERENCE_BACKEND}'"
    )

# Backward compatibility alias
MODEL_NAME = CODER_MODEL_NAME

# Check if the selected model has built-in multimodal capabilities
# When True, the VLM MCP server is disabled
IS_MULTIMODAL_MODEL = _model_info.get("multimodal", False)

# Scaling
SCALEDOWN_WINDOW = 300  # 5 minutes idle before scale-to-zero
MAX_CONCURRENT_INPUTS = 128 if INFERENCE_BACKEND == "vllm" else 8

# vLLM engine settings (only used when INFERENCE_BACKEND == "vllm")
MAX_MODEL_LEN = 131072  # 128K — ~60 GiB free for KV cache on A100-80GB
GPU_MEMORY_UTILIZATION = 0.90
KV_CACHE_DTYPE = "fp8"
TOOL_CALL_PARSER = "qwen3_coder"

# llama.cpp settings (only used when INFERENCE_BACKEND == "llamacpp")
LLAMACPP_CTX_SIZE = 98304  # 96K — fits in 3x A100-80GB with IQ2_XXS weights
LLAMACPP_N_GPU_LAYERS = 999  # offload all layers to GPU
LLAMACPP_FLASH_ATTN = True  # use flash attention if supported

# --- VLM model ---
# Only enabled when using non-multimodal models
# When IS_MULTIMODAL_MODEL=True, the VLM MCP server is disabled
ENABLE_VLM_MCP = not IS_MULTIMODAL_MODEL

VLM_MODEL_NAME = "Qwen/Qwen3-VL-32B-Thinking-FP8"

# GPU — A100-40GB: 17 GiB FP8 weights leaves ~19 GiB for KV cache
VLM_GPU_TYPE = "A100-40GB"
VLM_N_GPU = 1

# VLM scaling
VLM_MAX_CONCURRENT_INPUTS = 16

# VLM vLLM engine
VLM_MAX_MODEL_LEN = 32768  # 32K — conservative for image analysis one-shots
VLM_GPU_MEMORY_UTILIZATION = 0.90

# --- Model directories (on volume) ---
MODEL_DIR = f"{VOLUME_MOUNT_PATH}/{CODER_MODEL_NAME}"
VLM_MODEL_DIR = f"{VOLUME_MOUNT_PATH}/{VLM_MODEL_NAME}"

# Models to download to volume
MODELS_TO_DOWNLOAD = [CODER_MODEL_NAME]
if ENABLE_VLM_MCP:
    MODELS_TO_DOWNLOAD.append(VLM_MODEL_NAME)

# --- Modal app ---
APP_NAME = "coding-agent-server"

# Server
SERVER_PORT = 8000
