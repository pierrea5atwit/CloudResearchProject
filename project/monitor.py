"""Telemetry monitoring module using NVML."""

from __future__ import annotations

import threading
import time
from typing import Any

try:
    from pynvml import (  # type: ignore
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetUtilizationRates,
        nvmlInit,
        nvmlShutdown,
    )
except Exception:  # pragma: no cover - optional GPU dependency at dev time
    nvmlDeviceGetHandleByIndex = None  # type: ignore[assignment]
    nvmlDeviceGetMemoryInfo = None  # type: ignore[assignment]
    nvmlDeviceGetUtilizationRates = None  # type: ignore[assignment]
    nvmlInit = None  # type: ignore[assignment]
    nvmlShutdown = None  # type: ignore[assignment]


class Monitor:
    """Background telemetry monitor skeleton with NVML sampling."""

    def __init__(self, sample_interval_sec: float = 1.0, gpu_index: int = 0) -> None:
        self.sample_interval_sec = sample_interval_sec
        self.gpu_index = gpu_index
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []

    def _collect_loop(self) -> None:
        if not all([nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, nvmlShutdown]):
            self._samples.append(
                {
                    "timestamp": time.time(),
                    "error": "pynvml not available",
                }
            )
            return

        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
            while not self._stop_event.is_set():
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                self._samples.append(
                    {
                        "timestamp": time.time(),
                        "gpu_utilization_pct": float(util.gpu),
                        "memory_utilization_pct": float(util.memory),
                        "memory_used_mb": float(mem.used) / (1024 * 1024),
                    }
                )
                time.sleep(self.sample_interval_sec)
        except Exception as exc:
            self._samples.append(
                {
                    "timestamp": time.time(),
                    "error": str(exc),
                }
            )
        finally:
            try:
                nvmlShutdown()
            except Exception:
                pass

    def start(self) -> None:
        """Start background telemetry collection."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        """Stop telemetry collection and return all collected samples."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(1.0, self.sample_interval_sec * 2))
        return list(self._samples)

    def collect(self) -> dict:
        """Collect-once skeleton API for compatibility with prior interface."""
        samples = self.stop()
        return {
            "sample_count": len(samples),
            "samples": samples,
        }
