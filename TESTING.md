# Testing Plan

This document is the home base for the project testing workflow and the source of truth for implementation planning.

## 1) Success Criteria

The accelerator-aware control framework is considered successful when all targets below are met in multi-tenant vGPU environments.

| Area | Target |
| --- | --- |
| Throughput variance reduction | >=30% reduction |
| Telemetry overhead | <3% CPU utilization |
| GPU contention detection | >=85% accuracy |
| Bottleneck attribution | >=80% correct attribution across experiments |
| Scaling efficiency | >70% when scaling from 1 to 4 vGPU instances |

## 2) Metrics and Why They Matter

| Category | Metric(s) | Reason |
| --- | --- | --- |
| Training Throughput | Tokens/sec or Samples/sec | Measures useful GPU work output. If utilization rises while throughput stalls, this suggests contention or scheduler inefficiency. |
| Throughput Variance | Standard deviation of tokens/sec across repeated runs | Captures run-to-run stability. High variance indicates unstable scheduling or hidden multi-tenant interference. |
| GPU Utilization | Tensor Core utilization %, Streaming Multiprocessor (SM) utilization % | Shows whether compute resources are effectively used. High utilization with low throughput may indicate memory or contention bottlenecks. |
| Resource Contention | Memory bandwidth utilization, VRAM usage, kernel wait time | Detects process competition on shared GPU resources that can cause slowdowns and unpredictability. |
| Monitoring Overhead | CPU utilization of monitoring agent, GPU overhead from telemetry | Ensures observability does not distort system behavior or invalidate conclusions. |
| Scaling Efficiency (future work) | Speedup ratio when adding GPUs | Measures multi-GPU scaling effectiveness when additional accelerators are available. |

## 3) Methodology Summary

1. Environment setup
Install CUDA drivers and configure Python with RAPIDS (cuML), NumPy, Pandas, and Matplotlib. Verify GPU access via NVML or nvidia-smi.

2. Dataset preparation
Generate synthetic NumPy datasets with configurable sizes (for example, 100k to 1M samples) for kNN training/inference.

3. Model initialization
Train a cuML k-Nearest Neighbors classifier once and reuse it across scenarios to isolate inference behavior.

4. Workload generation
Run multiple concurrent Python processes that repeatedly call model.predict() on batch inputs. Adjust process count by scenario.

5. Metric logging
Per process, log latency and throughput. In parallel, poll NVML on a fixed interval for GPU utilization and memory signals.

6. Data aggregation
Consolidate logs with Pandas and compute cross-scenario summaries, including latency, requests/sec, and utilization.

7. Performance analysis
Visualize latency distributions, throughput trends, and utilization behavior to identify saturation points and contention effects.

## 4) Data Visualization Plan

### Tooling

| Component | Description |
| --- | --- |
| Visualization | Matplotlib charts generated from Python scripts/notebooks |
| Data processing | Pandas for aggregation and preprocessing |
| GPU telemetry source | NVIDIA Management Library via pynvml |

### Metric-to-visual mapping

Each primary metric should map to a chart that communicates scaling behavior, saturation limits, and stability.

| Visual | Chart Type | Purpose | Axes |
| --- | --- | --- | --- |
| Throughput scaling (knee curve) | Line graph | Show throughput (requests/sec) vs concurrent jobs and identify saturation knee where added load no longer increases throughput. | X: Concurrent GPU jobs, Y: Throughput (requests/sec) |
| Throughput variance | Box plot or violin plot | Show spread and jitter of throughput per scenario to reveal instability under contention. | X: Scenario/concurrent jobs, Y: Throughput (requests/sec) |
| Utilization vs throughput | Dual line graph | Compare hardware saturation and application output in one view. | X: Concurrent jobs or time, Y-left: Throughput, Y-right: GPU utilization (%) |
| Monitoring comparison | Comparative line graph or overlaid latency curves | Compare behavior before vs after control/monitoring logic to assess stability improvements and collapse prevention. | X: Time or concurrent jobs, Y: Latency or throughput |

## 5) Context to Carry Into Implementation Planning

### Assumptions currently implied

- Workload is inference-focused with a pretrained/reused cuML kNN model.
- Primary evaluation is on shared/virtualized GPU scenarios under rising concurrency.
- Monitoring must be lightweight enough to stay below the overhead budget.

### Open planning decisions (next)

- Define exact scenario matrix: process counts, batch sizes, and run durations.
- Define ground-truth labeling approach for contention and bottleneck attribution accuracy metrics.
- Define logging schema and result file layout in results/ for repeatable analysis.
- Define pass/fail rule implementation for all success criteria.

### Immediate implementation work packages

1. Config package — **complete**
YAML configs exist for virtual, physical, smoke, and concurrency scenarios (2, 4, 6, 8 workers).
All scenario configs enforce the same workload parameters; only `num_workers` varies per AGENT.MD.

2. Workload runner — **complete**
Concurrent multiprocessing workers run the `gpu_kernel` (CuPy matmul) or `knn` workload.
Warmup phase (time-based, driven by `experiment.warmup_seconds`) discards cold-start loops before measurement begins.
Per-worker metrics returned: latency percentiles, throughput req/s, GFLOPs/s, total requests, elapsed time.

3. Telemetry monitor — **complete**
NVML sampling loop runs in a background thread during worker execution.
Samples: `gpu_utilization_pct`, `memory_utilization_pct`, `memory_used_mb` at configurable interval (default 0.5s for scenario configs).
GPU activity validated post-run; behavior is workload-aware (0% is expected and non-fatal for CPU kNN).

4. Logging/aggregation — **complete**
Structured log lines emitted at workload resolution, per-worker completion, telemetry stop, and run summary.
Results persisted to `results/` as two append-only CSVs via `project/results_writer.py`:

**`results/worker_results.csv`** — one row per worker per run

| Column | Description |
| --- | --- |
| `run_id` | ISO UTC write-time timestamp; join key with `telemetry.csv` |
| `timestamp` | ISO UTC finish time of this worker process |
| `environment` | `virtual` or `physical` |
| `num_workers` | Concurrent workers in this run |
| `worker_id` | Zero-based worker index |
| `status` | `ok` or `error` |
| `workload_type` | `gpu_kernel` or `knn` |
| `backend_used` | `cupy` or `numpy` (post-fallback) |
| `model_backend` | `gpu-kernel-matmul`, `cuml`, or `numpy-fallback` |
| `latency_mean_ms` | Mean per-loop latency (measurement phase) |
| `latency_std_ms` | Latency standard deviation |
| `latency_p50_ms` | Median latency |
| `latency_p95_ms` | 95th-percentile latency |
| `latency_p99_ms` | 99th-percentile latency |
| `throughput_requests_per_sec` | Loops/sec (measurement phase only, warmup excluded) |
| `gflops_per_sec` | GPU compute throughput (`gpu_kernel` only; NaN for kNN) |
| `total_requests` | Total loops completed |
| `elapsed_total_sec` | Measurement phase wall time |

**`results/telemetry.csv`** — one row per NVML sample per run

| Column | Description |
| --- | --- |
| `run_id` | Join key with `worker_results.csv` |
| `timestamp` | ISO UTC time of this NVML sample |
| `environment` | `virtual` or `physical` |
| `num_workers` | Concurrent workers in this run |
| `gpu_utilization_pct` | SM utilization % (`nvmlDeviceGetUtilizationRates`) |
| `memory_utilization_pct` | Memory bus utilization % |
| `memory_used_mb` | VRAM in use (MB) |

5. Evaluation module — **pending**
Compute KPI outcomes against success criteria and emit a final experiment verdict.
Inputs will be `results/worker_results.csv` and `results/telemetry.csv` accumulated across the concurrency sweep.

### Concurrency sweep

The scenario runner (`run_scenarios.py`) drives four sequential experiments at 2, 4, 6, and 8 workers
using identical `gpu_kernel` workload parameters. Results accumulate in the same CSVs for joint analysis.

```
python run_scenarios.py                              # vGPU instance
python run_scenarios.py --dev-skip-vgpu-gate --no-progress  # local dev
```



