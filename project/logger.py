"""Logging utilities for experiments."""

import logging
from typing import Any

try:
    from pynvml import (  # type: ignore
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlSystemGetDriverVersion,
        nvmlInit,
        nvmlShutdown,
    )
except Exception:  # pragma: no cover - optional GPU dependency at dev time
    nvmlDeviceGetHandleByIndex = None  # type: ignore[assignment]
    nvmlDeviceGetMemoryInfo = None  # type: ignore[assignment]
    nvmlDeviceGetName = None  # type: ignore[assignment]
    nvmlSystemGetDriverVersion = None  # type: ignore[assignment]
    nvmlInit = None  # type: ignore[assignment]
    nvmlShutdown = None  # type: ignore[assignment]


def get_logger(name: str = "cloud_research") -> logging.Logger:
    """Create and return a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _decode_if_bytes(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def classify_runtime_environment(gpu_name: str, total_memory_mb: float) -> str:
    """Heuristic environment classification from observable GPU signals."""
    name_lower = gpu_name.lower()
    if total_memory_mb <= 8192:
        return "virtual (likely vGPU slice)"
    if any(token in name_lower for token in ["virtual", "vgpu", "grid", "shared"]):
        return "virtual (name indicates sharing/virtualization)"
    if total_memory_mb >= 24576:
        return "physical (full-size memory profile)"
    return "unknown"


def get_gpu_static_info(gpu_index: int = 0) -> dict[str, Any]:
    """Return static GPU identity details for experiment credibility records."""
    if not all(
        [
            nvmlInit,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetName,
            nvmlDeviceGetMemoryInfo,
            nvmlSystemGetDriverVersion,
            nvmlShutdown,
        ]
    ):
        return {
            "available": False,
            "error": "pynvml not available",
        }

    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        gpu_name = _decode_if_bytes(nvmlDeviceGetName(handle))
        memory = nvmlDeviceGetMemoryInfo(handle)
        driver_version = _decode_if_bytes(nvmlSystemGetDriverVersion())
        total_memory_mb = float(memory.total) / (1024 * 1024)
        inferred_environment = classify_runtime_environment(gpu_name, total_memory_mb)
        return {
            "available": True,
            "gpu_name": gpu_name,
            "total_memory_mb": round(total_memory_mb, 2),
            "driver_version": driver_version,
            "inferred_environment": inferred_environment,
        }
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
        }
    finally:
        try:
            nvmlShutdown()
        except Exception:
            pass


def log_runtime_environment(log: logging.Logger, declared_environment: str, gpu_index: int = 0) -> dict[str, Any]:
    """Log declared environment and measured runtime GPU identity details."""
    gpu_info = get_gpu_static_info(gpu_index=gpu_index)
    log.info("Declared config environment: %s", declared_environment)

    if gpu_info.get("available"):
        log.info("GPU name: %s", gpu_info["gpu_name"])
        log.info("GPU total memory (MB): %.2f", gpu_info["total_memory_mb"])
        log.info("NVIDIA driver version: %s", gpu_info["driver_version"])
        log.info("Runtime environment inference: %s", gpu_info["inferred_environment"])
    else:
        log.warning("GPU runtime details unavailable: %s", gpu_info.get("error", "unknown error"))

    return gpu_info