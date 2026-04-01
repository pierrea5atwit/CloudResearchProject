"""Entry point for running cloud research experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from config import load_config
from logger import get_logger, log_runtime_environment
from monitor import Monitor
from runner import run_concurrent_workers
from workload import Workload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment scenarios")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index for monitoring and sanity checks")
    parser.add_argument(
        "--sanity-duration",
        type=int,
        default=5,
        help="Seconds to sample GPU utilization during sanity check",
    )
    parser.add_argument(
        "--dev-skip-vgpu-gate",
        dest="require_virtual",
        action="store_false",
        help="[DEV ONLY] Skip strict virtual/vGPU environment checks. Used for local testing when NVML is unavailable.",
    )
    parser.add_argument(
        "--progress-interval-sec",
        type=float,
        default=5.0,
        help="Worker progress reporting interval in seconds (throttled to reduce overhead)",
    )
    parser.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable periodic worker progress logs",
    )
    parser.set_defaults(require_virtual=True)
    parser.set_defaults(show_progress=True)
    return parser.parse_args()


def _aggregate_results(worker_results: list[dict[str, Any]], telemetry_samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate worker outputs and telemetry into quick run-level summary."""
    ok_results = [item for item in worker_results if item.get("status") == "ok"]
    error_results = [item for item in worker_results if item.get("status") != "ok"]

    total_requests = sum(int(item.get("result", {}).get("total_requests", 0)) for item in ok_results)
    total_predictions = sum(int(item.get("result", {}).get("total_predictions", 0)) for item in ok_results)
    aggregate_rps = sum(float(item.get("result", {}).get("throughput_requests_per_sec", 0.0)) for item in ok_results)

    return {
        "worker_count": len(worker_results),
        "workers_ok": len(ok_results),
        "workers_error": len(error_results),
        "telemetry_samples": len(telemetry_samples),
        "total_requests": total_requests,
        "total_predictions": total_predictions,
        "aggregate_throughput_requests_per_sec": round(aggregate_rps, 4),
        "errors": [{"worker_id": e.get("worker_id"), "error": e.get("error")} for e in error_results],
    }


def _resolve_config_path(raw_config_path: str) -> Path:
    """Resolve config path from common run locations.

    Supports running from either workspace root or project directory.
    """
    raw = Path(raw_config_path).expanduser()
    if raw.is_absolute() and raw.exists():
        return raw.resolve()

    module_dir = Path(__file__).resolve().parent
    candidates = [
        (Path.cwd() / raw),
        (module_dir / raw),
        (module_dir.parent / raw),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Config path does not exist: '{raw_config_path}'. Searched: {searched}")


def _validate_runtime_gpu_activity(
    telemetry_samples: list[dict[str, Any]],
    min_expected_gpu_utilization: float = 30.0,
) -> dict[str, Any]:
    """Validate GPU activity from telemetry collected during worker execution.

    Expected to run AFTER workers complete, using samples captured DURING load.
    """
    gpu_utils = [
        item.get("gpu_utilization_pct", 0.0)
        for item in telemetry_samples
        if "gpu_utilization_pct" in item and "error" not in item
    ]
    if not gpu_utils:
        return {
            "valid": False,
            "peak_gpu_utilization_during_run": None,
            "samples_with_util_data": 0,
            "reason": "No GPU utilization data collected during run",
        }
    peak = max(gpu_utils)
    valid = peak >= min_expected_gpu_utilization
    return {
        "valid": valid,
        "peak_gpu_utilization_during_run": peak,
        "mean_gpu_utilization_during_run": sum(gpu_utils) / len(gpu_utils),
        "samples_with_util_data": len(gpu_utils),
        "threshold": min_expected_gpu_utilization,
        "reason": None if valid else f"GPU utilization peaked at {peak}%, expected >={min_expected_gpu_utilization}%",
    }


def _estimate_parallel_runtime(dry_run_result: dict[str, Any], num_workers: int) -> dict[str, float]:
    """Estimate expected runtime and throughput from single-worker baseline.

    This is a heuristic only; real vGPU contention may reduce throughput more heavily.
    """
    single_elapsed = float(dry_run_result.get("elapsed_total_sec", 0.0))
    single_rps = float(dry_run_result.get("throughput_requests_per_sec", 0.0))
    if single_elapsed <= 0 or single_rps <= 0 or num_workers <= 0:
        return {
            "single_worker_elapsed_sec": 0.0,
            "optimistic_wall_sec": 0.0,
            "estimated_wall_sec": 0.0,
            "estimated_aggregate_rps": 0.0,
        }

    # Lower bound assumes ideal overlap. Estimated wall uses mild contention factor.
    contention_factor = 1.0 + 0.35 * max(0, num_workers - 1)
    estimated_wall = single_elapsed * contention_factor
    estimated_aggregate_rps = (single_rps * num_workers) / contention_factor

    return {
        "single_worker_elapsed_sec": single_elapsed,
        "optimistic_wall_sec": single_elapsed,
        "estimated_wall_sec": estimated_wall,
        "estimated_aggregate_rps": estimated_aggregate_rps,
    }


def main() -> None:
    """Run the configured experiment workflow."""
    args = _parse_args()
    config_path = _resolve_config_path(args.config)

    config = load_config(config_path)
    log = get_logger("cloud_research.main")

    declared_environment = str(config.get("environment", "unknown")).lower()
    gpu_info = log_runtime_environment(log, declared_environment=declared_environment, gpu_index=args.gpu_index)
    inferred_environment = str(gpu_info.get("inferred_environment", "unknown")).lower()

    if declared_environment != "virtual":
        raise RuntimeError("This workflow is configured to run virtual experiments only. Set environment: virtual in config.")

    if args.require_virtual:
        if not gpu_info.get("available"):
            raise RuntimeError("Virtual runtime check failed: NVML GPU details unavailable.")
        if "virtual" not in inferred_environment:
            raise RuntimeError(
                f"Virtual runtime check failed: inferred environment is '{inferred_environment}', expected a virtual/vGPU signal."
            )
        log.info("Virtual environment gate passed (NVIDIA vGPU signal detected).")

    log.info("Baseline dry run for runtime estimation.")
    dry_run_result = Workload(config["workload"]).run()
    log.info("Single-worker dry run complete:")
    for key, value in dry_run_result.items():
        log.info("  %s: %s", key, value)

    num_workers = int(config["concurrency"].get("num_workers", 1))
    estimate = _estimate_parallel_runtime(dry_run_result, num_workers=num_workers)
    log.info(
        "Runtime estimate | workers=%d | single=%.2fs | optimistic=%.2fs | estimated=%.2fs | est aggregate req/s=%.2f",
        num_workers,
        estimate["single_worker_elapsed_sec"],
        estimate["optimistic_wall_sec"],
        estimate["estimated_wall_sec"],
        estimate["estimated_aggregate_rps"],
    )

    log.info("GPU activity validation will occur post-run using telemetry collected during worker execution.")

    monitor_interval = float(config["monitoring"].get("sample_interval_sec", 1))
    monitor = Monitor(sample_interval_sec=monitor_interval, gpu_index=args.gpu_index)
    monitor.start()
    log.info("Telemetry monitor started in background.")

    worker_results: list[dict[str, Any]] = []
    telemetry_samples: list[dict[str, Any]] = []
    try:
        def _on_worker_result(item: dict[str, Any]) -> None:
            if item.get("status") == "ok":
                result = item.get("result", {})
                log.info(
                    "Worker %s complete | req/s=%.2f | p95=%.2f ms",
                    item.get("worker_id"),
                    float(result.get("throughput_requests_per_sec", 0.0)),
                    float(result.get("latency_p95_ms", 0.0)),
                )
            else:
                log.error("Worker %s failed: %s", item.get("worker_id"), item.get("error"))

        def _on_worker_event(item: dict[str, Any]) -> None:
            if not args.show_progress:
                return
            if item.get("event") != "progress":
                return
            log.info(
                "Worker %s progress %d/%d (%.1f%%) | elapsed=%.1fs | eta=%.1fs",
                item.get("worker_id"),
                int(item.get("loops_completed", 0)),
                int(item.get("loops_total", 0)),
                float(item.get("progress_pct", 0.0)),
                float(item.get("elapsed_sec", 0.0)),
                float(item.get("estimated_remaining_sec", 0.0)),
            )

        worker_results = run_concurrent_workers(
            num_workers=num_workers,
            workload_config=dict(config["workload"]),
            on_result=_on_worker_result,
            on_event=_on_worker_event,
            progress_interval_sec=max(2.0, float(args.progress_interval_sec)),
        )
        log.info("Concurrent run finished with %d worker result entries.", len(worker_results))
    finally:
        telemetry_samples = monitor.stop()
        log.info("Telemetry monitor stopped with %d samples.", len(telemetry_samples))

    gpu_activity_valid = _validate_runtime_gpu_activity(
        telemetry_samples,
        min_expected_gpu_utilization=30.0,
    )
    log.info("GPU activity validation result: %s", gpu_activity_valid)
    if not gpu_activity_valid.get("valid", False):
        # Check if workload was GPU-bound (gpu_kernel type)
        workload_type = str(config.get("workload", {}).get("workload_type", "knn")).lower().strip()
        if workload_type == "gpu_kernel":
            # GPU workload was requested but didn't show activity: this is a real error
            if args.require_virtual:
                raise RuntimeError(
                    f"Experiment GPU activity check failed: {gpu_activity_valid.get('reason')}. "
                    "This may indicate the workload is not GPU-bound or GPU resources are unavailable."
                )
            log.warning("GPU activity check failed; dev-skip-vgpu-gate override active. Continuing for local testability.")
        else:
            # CPU workload (knn): GPU activity not required, just log info
            log.info("GPU activity low (expected for CPU-based workload type '%s'). GPU check skipped.", workload_type)

    summary = _aggregate_results(worker_results, telemetry_samples)
    log.info("Final aggregation: %s", summary)


if __name__ == "__main__":
    main()
