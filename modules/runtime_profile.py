import os


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_runtime_profile() -> dict:
    """Return a unified runtime profile consumed by all model/session builders."""
    return {
        "low_vram": _env_flag("DLC_LOW_VRAM") or _env_flag("LOW_VRAM"),
        "prefer_fp16": _env_flag("DLC_FP16") or _env_flag("ORT_FP16"),
        "ort_intra_threads": int(os.environ.get("DLC_ORT_INTRA_THREADS", "0") or 0),
        "ort_inter_threads": int(os.environ.get("DLC_ORT_INTER_THREADS", "1") or 1),
        "gpu_memory_limit_mb": int(os.environ.get("DLC_GPU_MEM_MB", "0") or 0),
        "enable_tensorrt": _env_flag("DLC_TENSORRT"),
    }


def should_force_low_vram() -> bool:
    return get_runtime_profile()["low_vram"]


def should_prefer_fp16() -> bool:
    return get_runtime_profile()["prefer_fp16"]
