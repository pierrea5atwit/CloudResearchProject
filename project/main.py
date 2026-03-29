"""Entry point for running cloud research experiments."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

from config import load_config, run_gpu_sanity_check
from logger import get_logger, log_runtime_environment
from monitor import Monitor


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
    return parser.parse_args()


def _worker_skeleton(worker_id: int, config: dict[str, Any]) -> dict[str, Any]:
    """Multiprocessing worker skeleton for future inference implementation."""
    _ = config
    start = time.time()
    time.sleep(0.05)
    elapsed = time.time() - start
    return {
        "worker_id": worker_id,
        "status": "skeleton",
        "elapsed_sec": round(elapsed, 4),
        "requests": 0,
        "throughput": 0.0,
    }


def _launch_workers_skeleton(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Launch worker pool skeleton and return worker-level placeholders."""
    num_workers = int(config["concurrency"].get("num_workers", 1))
    worker_ids = list(range(num_workers))
    with mp.get_context("spawn").Pool(processes=num_workers) as pool:
        return pool.starmap(_worker_skeleton, [(worker_id, config) for worker_id in worker_ids])


def _aggregate_results_skeleton(worker_results: list[dict[str, Any]], telemetry_samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregation skeleton for final KPI implementation."""
    return {
        "worker_count": len(worker_results),
        "telemetry_samples": len(telemetry_samples),
        "total_requests": sum(int(result.get("requests", 0)) for result in worker_results),
        "notes": "Aggregation is currently a skeleton. Replace with KPI calculations.",
    }


def main() -> None:
    """Run the configured experiment workflow."""
    args = _parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config path does not exist: {config_path}")

    config = load_config(config_path)
    log = get_logger("cloud_research.main")

    declared_environment = str(config.get("environment", "unknown"))
    gpu_info = log_runtime_environment(log, declared_environment=declared_environment, gpu_index=args.gpu_index)
    if gpu_info.get("available") and "virtual" in str(gpu_info.get("inferred_environment", "")).lower() and declared_environment == "physical":
        log.warning("Declared environment is physical, but runtime signals look virtualized.")

    log.info("GPU sanity check protocol: step 1 single-worker dry run, step 2 utilization validation (>50%%).")
    single_worker_result = _worker_skeleton(0, config)
    log.info("Single-worker skeleton run complete: %s", single_worker_result)

    sanity = run_gpu_sanity_check(
        duration_seconds=args.sanity_duration,
        min_expected_gpu_utilization=50.0,
        gpu_index=args.gpu_index,
    )
    log.info("GPU sanity check result: %s", sanity)
    if not sanity.get("valid", False):
        log.error("Experiment credibility check failed: observed GPU utilization is below threshold (>50%% expected).")

    monitor_interval = float(config["monitoring"].get("sample_interval_sec", 1))
    monitor = Monitor(sample_interval_sec=monitor_interval, gpu_index=args.gpu_index)
    monitor.start()
    log.info("Telemetry monitor started in background (skeleton).")

    worker_results: list[dict[str, Any]] = []
    try:
        worker_results = _launch_workers_skeleton(config)
        log.info("Concurrent worker skeleton finished with %d worker result entries.", len(worker_results))
    finally:
        telemetry_samples = monitor.stop()
        log.info("Telemetry monitor stopped with %d samples.", len(telemetry_samples))

    summary = _aggregate_results_skeleton(worker_results, telemetry_samples)
    log.info("Final aggregation skeleton: %s", summary)


if __name__ == "__main__":
    main()
