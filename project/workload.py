"""Workload module for cuML inference logic."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from logger import get_logger

_log = get_logger("cloud_research.workload")

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - optional at development time
    cp = None  # type: ignore[assignment]

try:
    from cuml.neighbors import KNeighborsClassifier  # type: ignore
except Exception:  # pragma: no cover - optional at development time
    KNeighborsClassifier = None  # type: ignore[assignment]


@dataclass(frozen=True)
class WorkloadConfig:
    """Configuration for synthetic data generation and inference loops."""

    n_samples: int = 200_000
    n_features: int = 64
    n_classes: int = 4
    k_neighbors: int = 5
    batch_size: int = 1_000
    inference_loops: int = 500
    random_seed: int = 42
    backend: str = "numpy"  # "numpy" or "cupy"
    workload_type: str = "knn"  # "knn" for inference, "gpu_kernel" for GPU matrix ops
    warmup_seconds: float = 0.0  # seconds of warmup to discard before measurement


class Workload:
    """cuML kNN inference workload runner."""

    def __init__(self, config: dict[str, Any] | WorkloadConfig) -> None:
        if isinstance(config, WorkloadConfig):
            self.config = config
        else:
            self.config = WorkloadConfig(**config)

    def _resolve_backend(self) -> str:
        requested = self.config.backend.lower().strip()
        if requested not in {"numpy", "cupy"}:
            raise ValueError("backend must be either 'numpy' or 'cupy'")
        if requested == "cupy" and cp is None:
            return "numpy"
        return requested

    def _generate_synthetic_dataset(self, backend: str) -> tuple[Any, Any, Any]:
        """Generate train and inference inputs on selected backend."""
        n_samples = int(self.config.n_samples)
        n_features = int(self.config.n_features)
        batch_size = int(self.config.batch_size)
        n_classes = int(self.config.n_classes)
        seed = int(self.config.random_seed)

        if backend == "cupy":
            assert cp is not None
            cp.random.seed(seed)
            x_train = cp.random.random((n_samples, n_features), dtype=cp.float32)
            y_train = cp.random.randint(0, n_classes, size=n_samples, dtype=cp.int32)
            x_infer = cp.random.random((batch_size, n_features), dtype=cp.float32)
            return x_train, y_train, x_infer

        rng = np.random.default_rng(seed)
        x_train = rng.random((n_samples, n_features), dtype=np.float32)
        y_train = rng.integers(0, n_classes, size=n_samples, dtype=np.int32)
        x_infer = rng.random((batch_size, n_features), dtype=np.float32)
        return x_train, y_train, x_infer

    def _latency_stats(self, latencies_ms: list[float]) -> dict[str, float]:
        lat_arr = np.array(latencies_ms, dtype=np.float64)
        return {
            "latency_mean_ms": float(np.mean(lat_arr)),
            "latency_std_ms": float(np.std(lat_arr)),
            "latency_min_ms": float(np.min(lat_arr)),
            "latency_p50_ms": float(np.percentile(lat_arr, 50)),
            "latency_p95_ms": float(np.percentile(lat_arr, 95)),
            "latency_p99_ms": float(np.percentile(lat_arr, 99)),
            "latency_max_ms": float(np.max(lat_arr)),
        }

    def _predict_numpy_fallback(self, x_train: np.ndarray, y_train: np.ndarray, x_infer: np.ndarray) -> np.ndarray:
        """Lightweight kNN fallback used only when cuML is unavailable."""
        max_train = min(4096, x_train.shape[0])
        max_batch = min(256, x_infer.shape[0])
        x_train_small = x_train[:max_train]
        y_train_small = y_train[:max_train]
        x_infer_small = x_infer[:max_batch]

        # Squared Euclidean distance via vectorized broadcast.
        distances = np.sum((x_infer_small[:, None, :] - x_train_small[None, :, :]) ** 2, axis=2)
        k = min(int(self.config.k_neighbors), max_train)
        nearest_idx = np.argpartition(distances, kth=max(0, k - 1), axis=1)[:, :k]
        nearest_labels = y_train_small[nearest_idx]
        preds = np.empty(nearest_labels.shape[0], dtype=np.int32)
        for i, row in enumerate(nearest_labels):
            counts = np.bincount(row.astype(np.int32), minlength=int(self.config.n_classes))
            preds[i] = int(np.argmax(counts))
        return preds

    def _gpu_kernel_matmul(self) -> dict[str, Any]:
        """Run GPU-accelerated matrix multiplication workload using CuPy.

        Produces sustained GPU load without requiring heavy ML frameworks.
        """
        if cp is None:
            raise RuntimeError("CuPy not available. Install via: pip install cupy-cuda12x (or appropriate CUDA version)")

        loops = int(self.config.inference_loops)
        matrix_dim = int(int(self.config.n_samples) ** 0.5)
        latencies_ms: list[float] = []

        # Pre-allocate GPU matrices for sustained operations
        cp.random.seed(int(self.config.random_seed))
        a = cp.random.random((matrix_dim, matrix_dim), dtype=cp.float32)
        b = cp.random.random((matrix_dim, matrix_dim), dtype=cp.float32)

        _log.info(
            "Workload | type=gpu_kernel | backend=cupy | matrix_dim=%d | loops=%d | warmup=%.1fs",
            matrix_dim, loops, float(self.config.warmup_seconds),
        )

        warmup_sec = float(self.config.warmup_seconds)
        if warmup_sec > 0:
            warmup_deadline = time.perf_counter() + warmup_sec
            while time.perf_counter() < warmup_deadline:
                _ = cp.dot(a, b)
                cp.cuda.Stream.null.synchronize()

        start_total = time.perf_counter()
        last_progress_ts = start_total
        for loop_idx in range(loops):
            start = time.perf_counter()
            _ = cp.dot(a, b)  # GPU matrix multiplication
            cp.cuda.Stream.null.synchronize()  # Ensure GPU work completes
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)

            if loop_idx % max(1, loops // 10) == 0 or loop_idx == loops - 1:
                elapsed_so_far = time.perf_counter() - start_total
                avg_loop_sec = elapsed_so_far / float(loop_idx + 1)
                loops_remaining = loops - (loop_idx + 1)

        elapsed_total = time.perf_counter() - start_total
        total_requests = loops
        flops_per_matmul = 2 * (matrix_dim ** 3)  # Approximate FLOPs
        total_flops = total_requests * flops_per_matmul
        gflops_per_sec = (total_flops / 1e9) / elapsed_total if elapsed_total > 0 else 0.0

        stats = self._latency_stats(latencies_ms)
        stats.update(
            {
                "backend_requested": self.config.backend,
                "backend_used": "cupy",
                "model_backend": "gpu-kernel-matmul",
                "matrix_dimension": matrix_dim,
                "total_requests": total_requests,
                "total_flops": int(total_flops),
                "gflops_per_sec": gflops_per_sec,
                "elapsed_total_sec": elapsed_total,
                "throughput_requests_per_sec": total_requests / elapsed_total if elapsed_total > 0 else 0.0,
            }
        )
        return stats

    def run(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        progress_interval_sec: float = 5.0,
    ) -> dict[str, Any]:
        """Execute workload and return throughput and latency statistics.

        Workload type is determined by config.workload_type:
        - "knn": kNN inference (existing behavior)
        - "gpu_kernel": GPU-accelerated matrix multiplication (requires CuPy)
        """
        workload_type = str(self.config.workload_type).lower().strip()

        if workload_type == "gpu_kernel":
            return self._gpu_kernel_matmul()

        # Default to kNN workload
        return self._run_knn_inference(progress_callback=progress_callback, progress_interval_sec=progress_interval_sec)

    def _run_knn_inference(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        progress_interval_sec: float = 5.0,
    ) -> dict[str, Any]:
        """Execute kNN inference workload and return throughput and latency statistics."""
        backend = self._resolve_backend()
        x_train, y_train, x_infer = self._generate_synthetic_dataset(backend)

        model_backend = "cuml"
        model: Any | None = None
        if KNeighborsClassifier is not None:
            model = KNeighborsClassifier(n_neighbors=int(self.config.k_neighbors))
            model.fit(x_train, y_train)
        else:
            model_backend = "numpy-fallback"
            if backend == "cupy" and cp is not None:
                x_train = cp.asnumpy(x_train)
                y_train = cp.asnumpy(y_train)
                x_infer = cp.asnumpy(x_infer)

        _log.info(
            "Workload | type=knn | backend=%s | model=%s | k=%d | loops=%d | warmup=%.1fs",
            backend, model_backend, int(self.config.k_neighbors),
            int(self.config.inference_loops), float(self.config.warmup_seconds),
        )

        loops = int(self.config.inference_loops)
        if loops <= 0:
            raise ValueError("inference_loops must be > 0")

        warmup_sec = float(self.config.warmup_seconds)
        if warmup_sec > 0:
            warmup_deadline = time.perf_counter() + warmup_sec
            while time.perf_counter() < warmup_deadline:
                if model_backend == "cuml":
                    _ = model.predict(x_infer)
                else:
                    _ = self._predict_numpy_fallback(x_train, y_train, x_infer)

        latencies_ms: list[float] = []
        start_total = time.perf_counter()
        last_progress_ts = start_total

        for loop_idx in range(loops):
            start = time.perf_counter()
            if model_backend == "cuml":
                _ = model.predict(x_infer)
            else:
                _ = self._predict_numpy_fallback(x_train, y_train, x_infer)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)

            if progress_callback is not None and progress_interval_sec > 0:
                now = time.perf_counter()
                should_emit = (now - last_progress_ts) >= progress_interval_sec or (loop_idx + 1) == loops
                if should_emit:
                    elapsed_so_far = now - start_total
                    avg_loop_sec = elapsed_so_far / float(loop_idx + 1)
                    loops_remaining = loops - (loop_idx + 1)
                    progress_callback(
                        {
                            "loops_completed": loop_idx + 1,
                            "loops_total": loops,
                            "progress_pct": ((loop_idx + 1) / loops) * 100.0,
                            "elapsed_sec": elapsed_so_far,
                            "estimated_remaining_sec": loops_remaining * avg_loop_sec,
                        }
                    )
                    last_progress_ts = now

        elapsed_total = time.perf_counter() - start_total
        total_requests = loops
        total_predictions = loops * int(self.config.batch_size)
        requests_per_sec = total_requests / elapsed_total if elapsed_total > 0 else 0.0
        samples_per_sec = total_predictions / elapsed_total if elapsed_total > 0 else 0.0

        stats = self._latency_stats(latencies_ms)
        stats.update(
            {
                "backend_requested": self.config.backend,
                "backend_used": backend,
                "model_backend": model_backend,
                "total_requests": total_requests,
                "batch_size": int(self.config.batch_size),
                "total_predictions": total_predictions,
                "elapsed_total_sec": elapsed_total,
                "throughput_requests_per_sec": requests_per_sec,
                "throughput_samples_per_sec": samples_per_sec,
            }
        )
        return stats
