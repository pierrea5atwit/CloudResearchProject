# VIRTUALIZATION & TESTING, NVIDIA GPU UNDER ARTIFICIAL LOAD

Author: Andrew Pierre
Cloud Computing, Wentworth Institute

# Purpose / Problem Statement

## Motivating Example: The Cost of Hidden GPU Inefficiency

### Scenario: An Early-Stage AI Researcher

Maya is training a transformer-based medical imaging model on a limited research budget.
Her lab relies on four rented cloud virtual GPU (vGPU) instances that advertise identical
compute capacity. Training runs are distributed across these instances and often take days.

As experimentation scales, Maya observes four recurring problems:

- Completion time varies by as much as 30% across similar runs.
- Reported GPU utilization remains high while end-to-end training time worsens.
- Scaling from 2 to 4 vGPUs delivers little throughput gain.
- Training stalls are more frequent during peak cloud usage windows.

Because the GPUs are virtualized, the root cause is hidden. Maya cannot directly determine
whether slowdown comes from tenant co-location, memory bandwidth pressure, or uneven
hypervisor-level resource sharing. She can verify allocation, but not behavior.

This is the core systems gap: cloud schedulers allocate accelerator resources, but they often
do not provide workload-aware visibility or contention-sensitive feedback for tuning AI jobs.
In practice, vGPU resources are managed as static units, even though their performance is
dynamic and sensitive to co-tenant interference.

## Proposed Direction

This project explores an accelerator-aware observability and control layer that operates above
the hypervisor and focuses on workload behavior rather than allocation status alone.

Desired capabilities include:

- Continuous GPU telemetry collection (kernel latency, utilization, memory pressure)
- Workload classification by compute and memory behavior
- Detection of multi-tenant contention across virtual GPU slices
- Scheduling or workload adjustment recommendations to reduce interference
- Scaling-efficiency diagnostics across distributed training nodes

Expected outputs include:

- Bottleneck attribution reports
- Variance tracking over time
- Contention alerts
- Cross-instance performance comparisons

With this feedback loop, researchers like Maya can stabilize throughput, improve scaling
efficiency, and reduce cost per experiment.

## Research Focus

This work does not replace cloud infrastructure. It augments existing virtualization and
scheduling systems with low-overhead, accelerator-sensitive observability.

Research question:

Can a distributed accelerator-aware control framework reduce contention, stabilize runtime
variance, and improve training efficiency in multi-tenant AI environments?


## Project Goals / Success Crit
Success Statement:

The project will be considered successful if the accelerator-aware control framework reduces training throughput variance by at least 30% in multi-tenant virtual GPU environments while maintaining telemetry overhead below 3% CPU utilization. Additionally, the system must detect GPU contention events with ≥85% accuracy, attribute performance bottlenecks correctly in ≥80% of experiments, and maintain scaling efficiency above 70%


## Setup

Install dependencies:

```
pip install -r requirements.txt
```

**Optional: GPU Kernel Workload Support**

For GPU-accelerated matrix multiply workload on vGPU instances:

```
pip install cupy-cuda12x  # CUDA 12.x
# or
pip install cupy-cuda11x  # CUDA 11.x
```

Verify your CUDA version: `nvidia-smi`

## Usage

Run from the workspace root:

python project/main.py --config configs/virtual.yaml

### Workload Types

The project supports two workload modes configured in the YAML via `workload.workload_type`:

**GPU-Accelerated Matrix Multiplication** (`gpu_kernel`) — default for `virtual.yaml`
- Requires CuPy: `pip install cupy-cuda12x` (adjust CUDA version as needed)
- Produces sustained GPU load via CuPy `cp.dot` matrix multiplication
- Records GFLOPs/sec in addition to latency and throughput
- Required for telemetry-observable GPU utilization on vGPU instances

**kNN Inference** (`knn`)
- Uses cuML `KNeighborsClassifier` if available; falls back to a lightweight numpy implementation
- CPU-bound in fallback mode — GPU utilization will read 0%, which is expected
- Useful for CPU baseline measurements and local development without a GPU

To run the kNN config explicitly:

python project/main.py --config configs/virtual.yaml  # after setting workload_type: knn

## Command-Line Arguments

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| --config PATH | Yes | None | Path to YAML experiment config file. Supports absolute paths and relative paths from workspace root or project directory. |
| --gpu-index INT | No | 0 | GPU index used for runtime environment checks and telemetry sampling. |
| --dev-skip-vgpu-gate | No | Off | [DEV ONLY] Bypasses strict virtual/vGPU environment verification. Use only when testing locally and NVML or vGPU indicators are unavailable. |
| --progress-interval-sec FLOAT | No | 5.0 | Interval (seconds) for worker progress updates in CLI logs. Internally throttled to avoid high overhead. |
| --no-progress | No | Off | Disables periodic worker progress logging. Final worker summaries are still printed. |

## Common Commands

Strict virtual run (default, enforces vGPU validation):

python project/main.py --config configs/virtual.yaml

Virtual run with custom progress interval:

python project/main.py --config configs/virtual.yaml --progress-interval-sec 3

Run with explicit GPU index:

python project/main.py --config configs/virtual.yaml --gpu-index 0

Local development mode (non-vGPU override, for testing when NVML unavailable):

python project/main.py --config configs/virtual.yaml --dev-skip-vgpu-gate

Local dev mode without progress spam:

python project/main.py --config configs/virtual.yaml --dev-skip-vgpu-gate --no-progress

## Concurrency Sweep

To run the full scenario matrix (2, 4, 6, and 8 concurrent workers) in sequence:

python run_scenarios.py

Each scenario uses the `gpu_kernel` workload with a 1000×1000 CuPy matrix. All results
accumulate in `results/` across scenarios, keyed by `run_id`, for knee-curve analysis.

```
# Local dev (no GPU):
python run_scenarios.py --dev-skip-vgpu-gate --no-progress

# Run a subset:
python run_scenarios.py --workers 4 8

# Stop on first failure:
python run_scenarios.py --fail-fast
```

## Results Output

After each run, two CSV files are written (or appended) to the directory set by
`logging.output_dir` in the config (default: `results/`).

### `results/worker_results.csv`

One row per worker per run.

| Column | Type | Description |
| --- | --- | --- |
| `run_id` | string | ISO UTC timestamp at write time — join key with `telemetry.csv` |
| `timestamp` | string | ISO UTC finish time of this specific worker process |
| `environment` | string | Value of `config.environment` (`virtual` or `physical`) |
| `num_workers` | int | Total concurrent workers in this run |
| `worker_id` | int | Zero-based worker index |
| `status` | string | `ok` or `error` |
| `workload_type` | string | `gpu_kernel` or `knn` |
| `backend_used` | string | `cupy` or `numpy` — actual backend after fallback resolution |
| `model_backend` | string | `gpu-kernel-matmul`, `cuml`, or `numpy-fallback` |
| `latency_mean_ms` | float | Mean per-loop latency over the measurement phase |
| `latency_std_ms` | float | Standard deviation of per-loop latency |
| `latency_p50_ms` | float | Median latency |
| `latency_p95_ms` | float | 95th-percentile latency |
| `latency_p99_ms` | float | 99th-percentile latency |
| `throughput_requests_per_sec` | float | Loops completed per second (measurement phase only) |
| `gflops_per_sec` | float | GPU throughput in GFLOPs/s (`gpu_kernel` only; NaN for kNN) |
| `total_requests` | int | Total inference loops completed |
| `elapsed_total_sec` | float | Wall time of the measurement phase (excludes warmup) |

### `results/telemetry.csv`

One row per NVML sample per run.

| Column | Type | Description |
| --- | --- | --- |
| `run_id` | string | ISO UTC timestamp at write time — join key with `worker_results.csv` |
| `timestamp` | string | ISO UTC time of this specific NVML sample |
| `environment` | string | Value of `config.environment` |
| `num_workers` | int | Total concurrent workers in this run |
| `gpu_utilization_pct` | float | SM utilization % from `nvmlDeviceGetUtilizationRates` |
| `memory_utilization_pct` | float | Memory bus utilization % |
| `memory_used_mb` | float | VRAM in use (MB) from `nvmlDeviceGetMemoryInfo` |

### Loading results for analysis

```python
import pandas as pd

workers = pd.read_csv("results/worker_results.csv")
telemetry = pd.read_csv("results/telemetry.csv")

# Aggregate throughput per concurrency level (knee curve)
knee = workers[workers["status"] == "ok"].groupby("num_workers")["throughput_requests_per_sec"].sum()

# Mean GPU utilization per run, joined to worker throughput
util_per_run = telemetry.groupby("run_id")["gpu_utilization_pct"].mean().reset_index()
merged = workers.merge(util_per_run, on="run_id")
```

## Notes

- The workflow is **strict virtual-only** by default. Config must have `environment: virtual`.
- In strict mode, runtime must show virtual/vGPU indicators AND GPU activity during worker execution.
- GPU activity validation occurs **post-run** using telemetry collected during active worker execution (not idle time).
- `virtual.yaml` defaults to `workload_type: gpu_kernel`. For CPU-only local testing, set `workload_type: knn` and add `--dev-skip-vgpu-gate`.
- kNN in fallback mode (no cuML) caps training samples at 4096 and batch size at 256 regardless of config values; it is not suitable for GPU load testing.
- GPU-accelerated cuML is not installed; kNN GPU inference requires a RAPIDS environment.
