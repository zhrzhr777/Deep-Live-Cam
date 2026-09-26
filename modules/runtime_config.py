import os

try:
    import onnxruntime
except Exception:  # pragma: no cover
    onnxruntime = None


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_low_vram_mode() -> bool:
    """Return True when the user has requested a lower-memory execution mode."""
    return (
        _truthy(os.environ.get("DLC_LOW_VRAM"))
        or _truthy(os.environ.get("LOW_VRAM"))
    )


def prefer_fp16() -> bool:
    """Return True when FP16 inference should be preferred where supported."""
    return (
        _truthy(os.environ.get("DLC_FP16"))
        or _truthy(os.environ.get("ORT_FP16"))
    )


def _clamp_threads(value: int, fallback: int) -> int:
    if value is None:
        return fallback
    return max(1, min(value, fallback))


def get_ort_thread_count() -> int:
    """Pick a conservative ORT thread count for low-vram / low-latency runs."""
    cpu_count = os.cpu_count() or 1
    if is_low_vram_mode():
        limit = max(1, min(2, cpu_count))
        env_value = os.environ.get("DLC_ORT_INTRA_THREADS")
        if env_value:
            try:
                return _clamp_threads(int(env_value), limit)
            except ValueError:
                pass
        return limit

    env_value = os.environ.get("DLC_ORT_INTRA_THREADS")
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass
    return max(1, cpu_count)


def apply_ort_session_options(session_options) -> None:
    """Apply a more stable ORT configuration, especially for low-vram setups."""
    if onnxruntime is None:
        return

    session_options.intra_op_num_threads = get_ort_thread_count()
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session_options.enable_cpu_mem_arena = True
    session_options.enable_mem_pattern = not is_low_vram_mode()

    if is_low_vram_mode():
        session_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
    else:
        session_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )


def provider_order(providers):
    """Prefer GPU first but keep CPU fallback for low-vram stable runs."""
    if not providers:
        return providers
    if is_low_vram_mode():
        ordered = []
        for provider in providers:
            if isinstance(provider, tuple):
                ordered.append(provider)
            else:
                ordered.append(provider)
        # Keep the first provider available but ensure CPU remains a safe fallback.
        if "CPUExecutionProvider" not in [
            p[0] if isinstance(p, tuple) else p for p in ordered
        ]:
            ordered.append("CPUExecutionProvider")
        return ordered
    return providers
