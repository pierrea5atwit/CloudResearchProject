# Fix Summary: Logging and GPU Activity Check

## Changes Made to `project/main.py`

### Fix #1: Dry Run Logging - Print Metrics on Separate Lines

**Location:** Lines 178-182  
**Problem:** Single dry-run result dict was printed on one long line, making metrics hard to read  
**Solution:** Loop through result dict and log each metric-value pair on its own line with indentation

```python
log.info("Single-worker dry run complete:")
for key, value in dry_run_result.items():
    log.info("  %s: %s", key, value)
```

**Before:**
```
Single-worker dry run complete: {'latency_mean_ms': 186.48, 'latency_std_ms': 38.81, ...}
```

**After:**
```
Single-worker dry run complete:
  latency_mean_ms: 186.48839399999864
  latency_std_ms: 38.816026757372676
  latency_min_ms: 145.1260999999704
  ... (each metric on its own line)
```

---

### Fix #2: GPU Activity Check - Workload-Aware Validation

**Location:** Lines 250-263  
**Problem:** RuntimeError when default CPU-based kNN workload produces 0% GPU utilization  
**Solution:** Check workload type and only enforce GPU activity check for GPU-bound workloads

```python
if not gpu_activity_valid.get("valid", False):
    # Check if workload was GPU-bound (gpu_kernel type)
    workload_type = str(config.get("workload", {}).get("workload_type", "knn")).lower().strip()
    if workload_type == "gpu_kernel":
        # GPU workload was requested but didn't show activity: this is a real error
        if args.require_virtual:
            raise RuntimeError(f"Experiment GPU activity check failed: {gpu_activity_valid.get('reason')}. ...")
        log.warning("GPU activity check failed; dev-skip-vgpu-gate override active. Continuing...")
    else:
        # CPU workload (knn): GPU activity not required, just log info
        log.info("GPU activity low (expected for CPU-based workload type '%s'). GPU check skipped.", workload_type)
```

**Behavior:**
- **For `workload_type: "knn"` (CPU-based):**  
  Logs info message and continues normally (GPU activity not expected)
  ```
  GPU activity low (expected for CPU-based workload type 'knn'). GPU check skipped.
  ```

- **For `workload_type: "gpu_kernel"` (GPU-based):**  
  Enforces GPU activity check (raises RuntimeError if GPU util < 30%)
  ```
  Experiment GPU activity check failed: GPU utilization peaked at X%, expected >=30.0%. 
  This may indicate the workload is not GPU-bound or GPU resources are unavailable.
  ```

---

## Testing

Two test scripts have been created to verify the fixes independently:

1. **`test_fix2.py`** - Demonstrates GPU activity check logic for different workload types
   - Shows how CPU workloads skip the check
   - Shows how GPU workloads enforce the check

2. **Actual experiment output** - Shows Fix #1 working (formatted metrics logging)
   - Run with: `python project/main.py --config configs/virtual.yaml --dev-skip-vgpu-gate --no-progress`

---

## Code Changes Summary

| Issue | Files Changed | Lines | Type |
|-------|---------------|-------|------|
| Dry run logging readability | `project/main.py` | 178-182 | Enhancement |
| GPU activity check RuntimeError | `project/main.py` | 250-263 | Bug Fix |

Both changes are minimal, focused, and backward compatible.

---

# Fix Summary: Warmup Phase and GPU Kernel Workload

## Changes Made

### Fix #3: Warmup Phase — Discard Startup Latency

**Problem:** All inference loops were measured from the start of execution, including the first loops where the CUDA context is cold, JIT kernels are compiling, and memory allocation is occurring. These cold-start measurements inflate latency and deflate throughput, producing unrepresentative results.

**Files changed:**

- `project/workload.py` — Added `warmup_seconds: float = 0.0` field to `WorkloadConfig`
- `project/workload.py` — `_gpu_kernel_matmul`: runs warmup loop for `warmup_seconds`, then resets `start_total` and `last_progress_ts` so only post-warmup loops are measured
- `project/workload.py` — `_run_knn_inference`: same warmup loop inserted before the measurement loop
- `project/main.py` — Reads `experiment.warmup_seconds` from config and injects it into the workload config dict (`workload_cfg`) used for both the dry run and all concurrent workers

**Behavior:**

Warmup is time-based. For `warmup_seconds: 5` (the value in all configs), the workload executes the same operation continuously for 5 seconds before any latency is recorded. `start_total` is reset immediately after warmup completes, so elapsed time and throughput calculations are based solely on the measurement phase.

```
[warmup: 5s, discarded] → [measurement: inference_loops iterations, recorded]
```

No config changes are required to enable warmup — it is automatically read from `experiment.warmup_seconds` which already exists in all YAML configs.

---

### Fix #4: Switch `virtual.yaml` Default Workload to GPU Kernel

**Problem:** `virtual.yaml` had no `workload_type` key, defaulting to `"knn"`. Since cuML is not installed, kNN ran as a numpy CPU fallback — producing 0% GPU utilization. The telemetry monitor was correctly reporting no GPU activity because none was occurring.

**File changed:** `project/configs/virtual.yaml`

**Added to `workload` section:**
```yaml
backend: "cupy"
workload_type: "gpu_kernel"
```

This switches the default `virtual` experiment from numpy kNN inference to CuPy matrix multiplication (`cp.dot`), which runs directly on the GPU and produces telemetry-observable utilization.

---

## Usage Changes

### `virtual.yaml` now requires CuPy

Running with the default config now requires CuPy:

```
pip install cupy-cuda12x   # CUDA 12.x
# or
pip install cupy-cuda11x   # CUDA 11.x
```

Verify your CUDA version first: `nvidia-smi`

If CuPy is unavailable (local dev without a GPU), use `--dev-skip-vgpu-gate` and switch back to kNN by removing the `workload_type` and `backend` lines from `virtual.yaml`, or run the dedicated CPU config:

```
python project/main.py --config configs/virtual.yaml --dev-skip-vgpu-gate
```

### Warmup is automatic — no new arguments needed

Warmup is sourced from `experiment.warmup_seconds` in the YAML config (default `5` in all existing configs). No command-line changes are required. The dry-run log will show the active warmup duration:

```
Baseline dry run for runtime estimation (warmup=5.0s).
```

To disable warmup, set `warmup_seconds: 0` in the config's `experiment` section.

---

## Full Change Index

| Issue | Files Changed | Type |
|-------|---------------|------|
| Dry run logging readability | `project/main.py` | Enhancement |
| GPU activity check RuntimeError for CPU workloads | `project/main.py` | Bug Fix |
| Warmup phase (discard cold-start latency) | `project/workload.py`, `project/main.py` | Feature |
| Switch `virtual.yaml` to GPU kernel workload | `project/configs/virtual.yaml` | Config Change |
| Pandas CSV result persistence | `project/results_writer.py` (new), `project/main.py` | Feature |

---

# Fix Summary: Pandas Result Persistence

## Changes Made

### Fix #5: Structured CSV Output via Pandas

**Problem:** All experiment output was written only to stdout via the logger. No persistent record existed across runs, and the data was not in a form usable for the visualizations required by TESTING.md (knee curves, latency distributions, utilization vs throughput).

**Files changed:**

- `project/results_writer.py` *(new)* — builds and appends DataFrames to CSV
- `project/main.py` — imports `write_run_results` and calls it at the end of `main()`

---

### How It Works

Two append-only CSV files are written to the directory specified by `logging.output_dir` in the config (default: `results/` relative to the workspace root). Both files share a `run_id` column (ISO UTC timestamp of run completion) so they can be joined for cross-metric analysis.

**`results/worker_results.csv`** — one row per worker per run

| Column | Source |
|--------|--------|
| `run_id` | ISO UTC timestamp assigned at write time |
| `timestamp` | Same as run_id |
| `environment` | `config.environment` |
| `num_workers` | `config.concurrency.num_workers` |
| `worker_id` | Worker process index |
| `status` | `"ok"` or `"error"` |
| `workload_type` | `config.workload.workload_type` |
| `latency_mean_ms` | Per-worker latency stats |
| `latency_std_ms` | |
| `latency_p50_ms` | |
| `latency_p95_ms` | |
| `latency_p99_ms` | |
| `throughput_requests_per_sec` | Per-worker throughput |
| `total_requests` | Inference loops completed |
| `elapsed_total_sec` | Wall time for measurement phase |

**`results/telemetry.csv`** — one row per NVML sample per run

| Column | Source |
|--------|--------|
| `run_id` | Shared with worker_results for join |
| `timestamp` | Sample wall-clock time (ISO UTC) |
| `environment` | `config.environment` |
| `num_workers` | `config.concurrency.num_workers` |
| `gpu_utilization_pct` | `nvmlDeviceGetUtilizationRates().gpu` |
| `memory_utilization_pct` | `nvmlDeviceGetUtilizationRates().memory` |
| `memory_used_mb` | `nvmlDeviceGetMemoryInfo().used` |

Samples with `"error"` keys (NVML unavailable or exception) are silently skipped. The file is still written; it will be empty-columned if no valid samples exist.

---

### Append Behavior

Both CSVs are opened in append mode (`mode='a'`). The header row is written only when the file does not yet exist. This means:

- First run: file is created with headers
- Subsequent runs: rows are appended, headers are not repeated
- Multiple `num_workers` configurations can accumulate in the same file and be grouped/filtered in analysis by `num_workers` or `run_id`

---

### Log Output Added

After writing, a summary line is emitted through the existing logger:

```
Run 2026-04-05T14:23:01Z | workers_ok=4/4 | gpu_util mean=72.3% peak=89.0% | telemetry_rows=63
```

---

## Usage Changes

No command-line changes required. The output directory is read from `logging.output_dir` in the YAML config (already present in all configs as `"results/"`). The directory is created automatically if it does not exist.

To load results for analysis:

```python
import pandas as pd

workers = pd.read_csv("results/worker_results.csv")
telemetry = pd.read_csv("results/telemetry.csv")

# Join on run_id for utilization vs throughput analysis
merged = workers.merge(telemetry.groupby("run_id")["gpu_utilization_pct"].mean().reset_index(), on="run_id")
```

---

# Fix Summary: Workload Identity Logging and CSV Schema Extension

## Changes Made

### Fix #6: Workload Identity Logging at Runtime

**Problem:** When a run produced results, no log line indicated which workload type, backend, or model was actually active. On a vGPU instance this distinction matters: `gpu_kernel` with `cupy`, `knn` with `cuml`, and `knn` with `numpy-fallback` produce fundamentally different resource profiles. Without a log entry at the point of resolution, a 0% GPU utilization result was ambiguous.

**Files changed:** `project/workload.py`

**Implementation:**

A module-level logger is instantiated at import time under the existing `cloud_research` hierarchy:

```python
from logger import get_logger
_log = get_logger("cloud_research.workload")
```

No new logger configuration is needed — `get_logger` reuses the handler and formatter already set up by `logger.py`. The `cloud_research.workload` name places it in the same namespace as `cloud_research.main`, so log output is consistent across modules.

**Log line emitted by `_gpu_kernel_matmul`** (after GPU matrices are allocated, all facts confirmed):

```
Workload | type=gpu_kernel | backend=cupy | matrix_dim=1000 | loops=200 | warmup=5.0s
```

`matrix_dim` is logged here rather than `n_samples` because it is the operationally meaningful value — it is the dimension of the matrices that will actually be multiplied on the GPU.

**Log line emitted by `_run_knn_inference`** (after model/backend resolution, before the warmup phase):

```
Workload | type=knn | backend=numpy | model=numpy-fallback | k=5 | loops=500 | warmup=5.0s
```

The `model=` field distinguishes `cuml` from `numpy-fallback` at runtime, which is the key signal for whether GPU inference was actually used.

Both lines are emitted once per worker process (each spawned worker initializes its own `Workload` instance), so in a 4-worker run the log will contain 4 identical lines confirming all workers resolved the same configuration.

---

### Fix #7: Extend `worker_results.csv` with Backend and GFLOPs Columns

**Problem:** `results/worker_results.csv` captured throughput and latency but omitted the workload identity fields already present in the `result` dict returned by `Workload.run()`. Without `backend_used` and `model_backend` in the CSV, post-run analysis could not distinguish runs by compute backend, and GFLOPs data from GPU kernel runs was silently discarded.

**File changed:** `project/results_writer.py`

**Columns added to `_WORKER_COLUMNS` and `build_worker_dataframe`:**

| Column | Source | Notes |
|--------|--------|-------|
| `backend_used` | `result["backend_used"]` | `"cupy"` or `"numpy"` — the backend that was actually used after fallback resolution |
| `model_backend` | `result["model_backend"]` | `"gpu-kernel-matmul"`, `"cuml"`, or `"numpy-fallback"` — the specific compute path |
| `gflops_per_sec` | `result["gflops_per_sec"]` | Populated for `gpu_kernel` runs; `NaN` for kNN runs (key absent, pandas fills gracefully) |

These columns are positioned in the schema between `workload_type` and the latency columns so workload identity fields are grouped together:

```
run_id | timestamp | environment | num_workers | worker_id | status |
workload_type | backend_used | model_backend |
latency_mean_ms | ... | throughput_requests_per_sec | gflops_per_sec |
total_requests | elapsed_total_sec
```

`gflops_per_sec` is placed adjacent to `throughput_requests_per_sec` since both measure compute output rate, just in different units (requests/s for kNN, GFLOPs/s for matrix kernels).

**Unused import removed:** `import time` was present in `results_writer.py` but never called. Removed.

---

### Fix #8: Concurrency Sweep Configs and Scenario Runner

**Problem:** The experiment required running the same heavy-load workload under 2, 4, 6, and 8 concurrent workers to produce the knee curve and contention data described in TESTING.md. There was no mechanism to do this in a single command — each scenario required a manual invocation and a separate config file.

**Files added:**

- `project/configs/scenario_workers_2.yaml`
- `project/configs/scenario_workers_4.yaml`
- `project/configs/scenario_workers_6.yaml`
- `project/configs/scenario_workers_8.yaml`
- `run_scenarios.py` *(workspace root)*

**Scenario config design:**

All four configs use identical workload parameters. Per AGENT.MD: *"The ONLY intended variable is: Number of concurrent GPU workers."* Varying the workload across scenarios would confound the contention signal.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `workload_type` | `gpu_kernel` | CuPy `cp.dot` — only workload that produces measurable GPU utilization without cuML |
| `backend` | `cupy` | Required for `gpu_kernel` |
| `n_samples` | `1000000` | `matrix_dim = sqrt(1000000) = 1000` → 1000×1000 float32 matrices (~2 GFLOPs/op) |
| `inference_loops` | `200` | Sufficient measurement phase without excessive wall time per scenario |
| `warmup_seconds` | `5` | Consistent with all other configs; primes CUDA context before measurement |
| `sample_interval_sec` | `0.5` | Doubled telemetry resolution (vs default 1s) for finer contention detection |

The 1000×1000 matrix size was chosen to produce real GPU pressure: at even a moderate 1 TFLOPS effective throughput, each matmul takes ~2ms, meaning 200 loops ≈ 400ms of pure compute per worker. With 8 workers all issuing `cp.dot` concurrently, this creates genuine contention on the GPU's compute schedulers.

**`run_scenarios.py` — implementation details:**

The runner drives `main.py` as a subprocess for each scenario, inheriting the full initialization path (NVML checks, virtual gate, dry run, monitor lifecycle). This avoids duplicating or reimporting the setup logic from `main.py`.

Key behaviors:
- Scenarios run sequentially (not in parallel) — parallel execution would conflate the per-scenario contention signal
- `--fail-fast` flag stops the sweep on first non-zero exit code; default is to continue and report all pass/fail at the end
- `--workers` accepts a custom list to run a subset (e.g. `--workers 4 8`)
- All pass-through flags (`--dev-skip-vgpu-gate`, `--no-progress`, `--gpu-index`) are forwarded to each `main.py` invocation unchanged

**Usage:**

```bash
# Full sweep on vGPU instance:
python run_scenarios.py

# Local dev (no GPU):
python run_scenarios.py --dev-skip-vgpu-gate --no-progress

# Partial sweep with early exit on failure:
python run_scenarios.py --workers 4 8 --fail-fast
```

After a full sweep, `results/worker_results.csv` contains rows for all four concurrency levels, filterable by `num_workers`, ready for knee-curve plotting:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/worker_results.csv")
agg = df[df["status"] == "ok"].groupby("num_workers")["throughput_requests_per_sec"].sum().reset_index()
plt.plot(agg["num_workers"], agg["throughput_requests_per_sec"], marker="o")
plt.xlabel("Concurrent workers"); plt.ylabel("Aggregate req/s"); plt.title("Throughput Knee Curve")
plt.show()
```

---

## Full Change Index

| # | Description | Files | Type |
|---|-------------|-------|------|
| 1 | Dry run logging readability | `project/main.py` | Enhancement |
| 2 | GPU activity check RuntimeError for CPU workloads | `project/main.py` | Bug Fix |
| 3 | Warmup phase (discard cold-start latency) | `project/workload.py`, `project/main.py` | Feature |
| 4 | Switch `virtual.yaml` default to GPU kernel workload | `project/configs/virtual.yaml` | Config Change |
| 5 | Pandas CSV result persistence (`worker_results.csv`, `telemetry.csv`) | `project/results_writer.py` *(new)*, `project/main.py` | Feature |
| 6 | Workload identity log lines at backend/model resolution | `project/workload.py` | Feature |
| 7 | Extend CSV schema: `backend_used`, `model_backend`, `gflops_per_sec` | `project/results_writer.py` | Enhancement |
| 8 | Concurrency sweep configs and scenario runner | `project/configs/scenario_workers_*.yaml` *(new × 4)*, `run_scenarios.py` *(new)* | Feature |
