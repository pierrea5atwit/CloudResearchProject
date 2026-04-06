#!/usr/bin/env python
"""Concurrency sweep: runs gpu_kernel workload under 2, 4, 6, and 8 workers.

Each scenario is an independent experiment run. Worker counts increase to reveal
the saturation knee, contention effects, and throughput variance described in
TESTING.md. Results from all scenarios accumulate in results/worker_results.csv
and results/telemetry.csv, keyed by run_id, for downstream analysis.

Usage (from workspace root):

    # On a vGPU instance (strict mode):
    python run_scenarios.py

    # Local dev without a GPU:
    python run_scenarios.py --dev-skip-vgpu-gate --no-progress

    # Run a subset of worker counts:
    python run_scenarios.py --workers 2 4

    # Stop on first failure instead of continuing:
    python run_scenarios.py --fail-fast
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent
_PROJECT_DIR = _WORKSPACE / "project"
_MAIN = _PROJECT_DIR / "main.py"

_DEFAULT_WORKERS = [2, 4, 6, 8]


def _config_path(num_workers: int) -> Path:
    return _PROJECT_DIR / "configs" / f"scenario_workers_{num_workers}.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run concurrency sweep across worker counts")
    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=_DEFAULT_WORKERS,
        metavar="N",
        help=f"Worker counts to sweep (default: {_DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--dev-skip-vgpu-gate",
        action="store_true",
        help="[DEV ONLY] Skip vGPU environment checks (pass-through to main.py)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-worker progress logs (pass-through to main.py)",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU device index (pass-through to main.py)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the sweep on the first failed scenario",
    )
    return parser.parse_args()


def _build_cmd(config: Path, args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(_MAIN), "--config", str(config), "--gpu-index", str(args.gpu_index)]
    if args.dev_skip_vgpu_gate:
        cmd.append("--dev-skip-vgpu-gate")
    if args.no_progress:
        cmd.append("--no-progress")
    return cmd


def main() -> None:
    args = _parse_args()
    worker_counts = sorted(set(args.workers))

    print(f"Concurrency sweep | workers={worker_counts}", flush=True)
    print(f"Results directory : {_WORKSPACE / 'results'}", flush=True)

    passed: list[int] = []
    failed: list[int] = []

    for num_workers in worker_counts:
        config = _config_path(num_workers)
        divider = "=" * 64
        print(f"\n{divider}", flush=True)
        print(f"Scenario: {num_workers} workers | config: {config.name}", flush=True)
        print(divider, flush=True)

        if not config.exists():
            print(f"  [SKIP] Config not found: {config}", flush=True)
            failed.append(num_workers)
            if args.fail_fast:
                break
            continue

        cmd = _build_cmd(config, args)
        result = subprocess.run(cmd, cwd=str(_PROJECT_DIR))

        if result.returncode == 0:
            print(f"\n  [PASS] workers={num_workers}", flush=True)
            passed.append(num_workers)
        else:
            print(f"\n  [FAIL] workers={num_workers} — exit code {result.returncode}", flush=True)
            failed.append(num_workers)
            if args.fail_fast:
                print("  --fail-fast set, stopping sweep.", flush=True)
                break

    print(f"\n{'=' * 64}", flush=True)
    print(f"Sweep complete | passed={passed} | failed={failed}", flush=True)
    if passed:
        print(f"Results written to: {_WORKSPACE / 'results'}", flush=True)
        print("  worker_results.csv — per-worker latency and throughput", flush=True)
        print("  telemetry.csv      — GPU utilization and memory time series", flush=True)
    print(flush=True)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
