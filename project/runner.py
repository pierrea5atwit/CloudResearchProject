"""Multiprocessing runner for workload workers."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from typing import Any, Callable

from workload import Workload


def _worker_entry(
    worker_id: int,
    workload_config: dict[str, Any],
    result_queue: mp.Queue,
    progress_interval_sec: float,
) -> None:
    """Execute one worker workload and post result to queue."""
    started_at = time.time()
    try:
        def _progress(payload: dict[str, Any]) -> None:
            result_queue.put(
                {
                    "event": "progress",
                    "worker_id": worker_id,
                    **payload,
                }
            )

        stats = Workload(workload_config).run(progress_callback=_progress, progress_interval_sec=progress_interval_sec)
        result_queue.put(
            {
                "event": "result",
                "worker_id": worker_id,
                "status": "ok",
                "started_at": started_at,
                "finished_at": time.time(),
                "result": stats,
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "event": "result",
                "worker_id": worker_id,
                "status": "error",
                "started_at": started_at,
                "finished_at": time.time(),
                "error": str(exc),
            }
        )


def run_concurrent_workers(
    num_workers: int,
    workload_config: dict[str, Any],
    queue_timeout_sec: float = 1.0,
    on_result: Callable[[dict[str, Any]], None] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    progress_interval_sec: float = 5.0,
) -> list[dict[str, Any]]:
    """Spawn N workers, run workloads, and collect queue results."""
    if num_workers <= 0:
        raise ValueError("num_workers must be > 0")

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    workers: list[mp.Process] = []

    for worker_id in range(num_workers):
        worker_payload = dict(workload_config)
        process = ctx.Process(
            target=_worker_entry,
            args=(worker_id, worker_payload, result_queue, progress_interval_sec),
            name=f"workload-worker-{worker_id}",
        )
        process.start()
        workers.append(process)

    results: list[dict[str, Any]] = []
    while len(results) < num_workers:
        try:
            item = result_queue.get(timeout=queue_timeout_sec)
            if on_event is not None:
                on_event(item)

            if item.get("event") != "result":
                continue

            results.append(item)
            if on_result is not None:
                on_result(item)
        except queue.Empty:
            alive = sum(1 for process in workers if process.is_alive())
            if alive == 0:
                break

    for process in workers:
        process.join(timeout=5)

    for process in workers:
        if process.exitcode not in (0, None):
            already_reported = any(
                result.get("worker_id") == int(process.name.rsplit("-", maxsplit=1)[-1])
                and result.get("status") == "error"
                for result in results
            )
            if not already_reported:
                results.append(
                    {
                        "worker_id": int(process.name.rsplit("-", maxsplit=1)[-1]),
                        "status": "error",
                        "error": f"worker exited with code {process.exitcode}",
                    }
                )

    return sorted(results, key=lambda item: int(item.get("worker_id", -1)))
