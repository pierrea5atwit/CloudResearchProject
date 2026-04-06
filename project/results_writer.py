"""Persist experiment results to CSV using pandas.

Writes two append-only files into the configured output directory:
  - worker_results.csv  : one row per worker per run (latency + throughput)
  - telemetry.csv       : one row per NVML sample per run (GPU utilization + memory)

Both files share a `run_id` column (ISO timestamp of run start) so they can be
joined for visualizations described in TESTING.md.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_WORKER_COLUMNS = [
    "run_id",
    "timestamp",
    "environment",
    "num_workers",
    "worker_id",
    "status",
    "workload_type",
    "latency_mean_ms",
    "latency_std_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "throughput_requests_per_sec",
    "total_requests",
    "elapsed_total_sec",
]

_TELEMETRY_COLUMNS = [
    "run_id",
    "timestamp",
    "environment",
    "num_workers",
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "memory_used_mb",
]


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_csv(df: pd.DataFrame, path: Path, log: logging.Logger) -> None:
    """Append DataFrame rows to CSV, writing the header only on first creation."""
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)
    action = "created" if write_header else "appended"
    log.info("Results %s: %s (%d rows)", action, path, len(df))


def build_worker_dataframe(
    worker_results: list[dict[str, Any]],
    run_id: str,
    environment: str,
    num_workers: int,
    workload_type: str,
) -> pd.DataFrame:
    """Build a per-worker summary DataFrame from runner output."""
    rows: list[dict[str, Any]] = []
    for item in worker_results:
        result = item.get("result") or {}
        row: dict[str, Any] = {
            "run_id": run_id,
            "timestamp": run_id,
            "environment": environment,
            "num_workers": num_workers,
            "worker_id": item.get("worker_id"),
            "status": item.get("status", "unknown"),
            "workload_type": workload_type,
            "latency_mean_ms": result.get("latency_mean_ms"),
            "latency_std_ms": result.get("latency_std_ms"),
            "latency_p50_ms": result.get("latency_p50_ms"),
            "latency_p95_ms": result.get("latency_p95_ms"),
            "latency_p99_ms": result.get("latency_p99_ms"),
            "throughput_requests_per_sec": result.get("throughput_requests_per_sec"),
            "total_requests": result.get("total_requests"),
            "elapsed_total_sec": result.get("elapsed_total_sec"),
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=_WORKER_COLUMNS)
    return df


def build_telemetry_dataframe(
    telemetry_samples: list[dict[str, Any]],
    run_id: str,
    environment: str,
    num_workers: int,
) -> pd.DataFrame:
    """Build a time-series telemetry DataFrame from monitor output."""
    rows: list[dict[str, Any]] = []
    for sample in telemetry_samples:
        if "error" in sample:
            continue
        row: dict[str, Any] = {
            "run_id": run_id,
            "timestamp": datetime.fromtimestamp(
                float(sample["timestamp"]), tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "environment": environment,
            "num_workers": num_workers,
            "gpu_utilization_pct": sample.get("gpu_utilization_pct"),
            "memory_utilization_pct": sample.get("memory_utilization_pct"),
            "memory_used_mb": sample.get("memory_used_mb"),
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=_TELEMETRY_COLUMNS)
    return df


def write_run_results(
    worker_results: list[dict[str, Any]],
    telemetry_samples: list[dict[str, Any]],
    config: dict[str, Any],
    log: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write worker results and telemetry to append-only CSVs in the output directory.

    Returns (worker_df, telemetry_df) for any in-process use after writing.
    """
    output_dir = Path(config.get("logging", {}).get("output_dir", "results/"))
    if not output_dir.is_absolute():
        # Resolve relative to the project root (parent of this file's directory)
        output_dir = Path(__file__).resolve().parent.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = _iso_now()
    environment = str(config.get("environment", "unknown"))
    num_workers = int(config.get("concurrency", {}).get("num_workers", 1))
    workload_type = str(config.get("workload", {}).get("workload_type", "knn"))

    worker_df = build_worker_dataframe(
        worker_results,
        run_id=run_id,
        environment=environment,
        num_workers=num_workers,
        workload_type=workload_type,
    )
    telemetry_df = build_telemetry_dataframe(
        telemetry_samples,
        run_id=run_id,
        environment=environment,
        num_workers=num_workers,
    )

    _append_csv(worker_df, output_dir / "worker_results.csv", log)
    _append_csv(telemetry_df, output_dir / "telemetry.csv", log)

    # Log a brief summary so it appears in the existing structured log stream
    ok_workers = int((worker_df["status"] == "ok").sum())
    if not telemetry_df.empty:
        mean_util = telemetry_df["gpu_utilization_pct"].mean()
        peak_util = telemetry_df["gpu_utilization_pct"].max()
        log.info(
            "Run %s | workers_ok=%d/%d | gpu_util mean=%.1f%% peak=%.1f%% | telemetry_rows=%d",
            run_id, ok_workers, num_workers, mean_util, peak_util, len(telemetry_df),
        )
    else:
        log.info(
            "Run %s | workers_ok=%d/%d | no GPU telemetry rows written",
            run_id, ok_workers, num_workers,
        )

    return worker_df, telemetry_df
