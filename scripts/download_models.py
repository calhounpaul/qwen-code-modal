"""Download model weights to a shared Modal Volume (CPU-only, no GPU needed).

Usage:
    modal run scripts/download_models.py                    # download all configured models
    modal run scripts/download_models.py --model NAME       # download a specific model
    modal run scripts/download_models.py --force            # re-download even if present

Each model directory gets a model_metadata.json file tracking:
  - download_status: "incomplete" while downloading, "completed" after success
  - model_name, hf_url, quantization, multimodal, gpu_type, n_gpu, backend
  - size_gb, downloaded_at timestamp
  - gguf_pattern (for GGUF models): the subdirectory/file pattern downloaded
Models with missing or incomplete metadata are re-downloaded automatically.

For GGUF repos with multiple quant files, only the configured gguf_pattern is downloaded.
"""

import sys
from pathlib import Path

import modal

# Import config — handle running from repo root via `modal run scripts/download_models.py`
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "coding_agent_server"))
    from config import (
        MODEL_REGISTRY,
        MODELS_TO_DOWNLOAD,
        VLM_MODEL_REGISTRY,
        VOLUME_MOUNT_PATH,
        VOLUME_NAME,
    )
except ImportError:
    VOLUME_NAME = "coding-agent-models"
    VOLUME_MOUNT_PATH = "/models"
    MODELS_TO_DOWNLOAD = ["unsloth/Qwen3.5-397B-A17B-GGUF"]
    MODEL_REGISTRY = {}
    VLM_MODEL_REGISTRY = {}

METADATA_FILE = "model_metadata.json"

app = modal.App("download-models")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


def _get_model_info(model_name: str) -> dict:
    """Look up model metadata from the registry (local entrypoint only)."""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    if model_name in VLM_MODEL_REGISTRY:
        return VLM_MODEL_REGISTRY[model_name]
    return {}


def _read_metadata(model_dir: str) -> dict | None:
    import json
    import os

    path = os.path.join(model_dir, METADATA_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _write_metadata(model_dir: str, download_status: str, info: dict, size_gb: float = 0.0):
    """Write model_metadata.json. `info` is the model registry entry."""
    import json
    import os
    from datetime import datetime, timezone

    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, METADATA_FILE)
    meta = {
        "model_name": info.get("model_name", ""),
        "hf_url": info.get("hf_url", ""),
        "quantization": info.get("quantization", "unknown"),
        "multimodal": info.get("multimodal", False),
        "gpu_type": info.get("gpu_type", "unknown"),
        "n_gpu": info.get("n_gpu", 1),
        "backend": info.get("backend", "vllm"),
        "size_gb": round(size_gb, 1),
        "download_status": download_status,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    if info.get("gguf_pattern"):
        meta["gguf_pattern"] = info["gguf_pattern"]
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def _dir_size_gb(path: str) -> float:
    import os

    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(path)
        for f in fns
        if f != METADATA_FILE
    )
    return total / 1e9


@app.function(
    image=download_image,
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=30 * 60,  # 30 minutes
)
def download_model(model_name: str, info: dict, force: bool = False):
    """Download a model to the volume. `info` is the registry entry dict."""
    from huggingface_hub import snapshot_download

    model_dir = f"{VOLUME_MOUNT_PATH}/{model_name}"
    gguf_pattern = info.get("gguf_pattern", "")

    # Check if already completed
    if not force:
        meta = _read_metadata(model_dir)
        if meta and meta.get("download_status") == "completed":
            if gguf_pattern and meta.get("gguf_pattern") != gguf_pattern:
                print(f"GGUF pattern changed: {meta.get('gguf_pattern')} -> {gguf_pattern}, re-downloading...")
            else:
                size = _dir_size_gb(model_dir)
                print(f"Already downloaded: {model_name} ({size:.1f} GB in {model_dir})")
                return

    # Write metadata as incomplete before starting
    _write_metadata(model_dir, "incomplete", info)
    volume.commit()

    if gguf_pattern:
        # GGUF: download only files matching the pattern (subdir like "UD-IQ2_XXS")
        allow = [f"{gguf_pattern}/**", f"*{gguf_pattern}*.gguf"]
        print(f"Downloading {model_name} [{gguf_pattern}] to {model_dir}...", flush=True)
        snapshot_download(
            model_name,
            local_dir=model_dir,
            allow_patterns=allow,
        )
    else:
        # Non-GGUF: download entire repo (safetensors)
        print(f"Downloading {model_name} to {model_dir}...", flush=True)
        snapshot_download(
            model_name,
            local_dir=model_dir,
            ignore_patterns=["*.pt", "*.bin"],
        )

    size = _dir_size_gb(model_dir)
    print(f"Download complete: {model_name} ({size:.1f} GB in {model_dir})", flush=True)

    _write_metadata(model_dir, "completed", info, size)
    volume.commit()


@app.local_entrypoint()
def main(model: str = "", force: bool = False):
    models = [model] if model else MODELS_TO_DOWNLOAD

    print(f"Models to download: {models}")
    print(f"Volume: {VOLUME_NAME} -> {VOLUME_MOUNT_PATH}")
    print(f"Force: {force}")
    print()

    for m in models:
        info = _get_model_info(m)
        info["model_name"] = m
        if not info.get("hf_url"):
            info["hf_url"] = f"https://huggingface.co/{m}"
        print(f"  {m}: gguf_pattern={info.get('gguf_pattern', '(none)')}")
        download_model.remote(m, info=info, force=force)
