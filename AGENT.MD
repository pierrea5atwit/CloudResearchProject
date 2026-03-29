# Experiment Execution Plan

This project implements a controlled GPU contention benchmark. The goal is to measure how inference performance degrades as concurrent GPU workloads increase.

The system MUST follow a strict experimental structure to ensure valid results.

---

## Core Principle

All experiments must:

- Use the SAME code paths
- Use the SAME workload logic
- Only vary configuration parameters (via YAML)

The ONLY intended variable is:
- Number of concurrent GPU workers
- Environment (virtual vs physical GPU)

---

## Execution Flow (High-Level)

1. Load configuration from YAML
2. Initialize logging system
3. Initialize GPU telemetry monitor (background process/thread)
4. Warm up GPU (optional but recommended)
5. Launch concurrent worker processes
6. Each worker runs repeated inference loops
7. Collect latency and throughput per worker
8. Continuously sample GPU metrics (utilization, memory)
9. Aggregate results at end of run
10. Write structured output logs (CSV)

---

## Worker Behavior

Each worker process must:

1. Initialize its own cuML kNN model
2. Generate or receive dataset (GPU-resident if possible)
3. Perform repeated inference calls:
   - model.predict(batch)
4. Measure:
   - per-request latency
   - total requests completed
5. Return summary metrics at completion

Workers must:
- Avoid unnecessary CPU-GPU data transfer
- Keep data on GPU where possible
- Minimize Python overhead inside loops

---

## Concurrency Model

- Use Python multiprocessing (NOT threading)
- Each worker is an independent process
- Workers simulate independent tenants using the same GPU

Concurrency levels to test:
- 1 worker (baseline)
- 2 workers
- 4 workers
- 6+ workers (stress)

---

## Timing Model

Each experiment run must include:

- Warmup phase (ignored in metrics)
- Measurement phase (timed)

Metrics should only be recorded AFTER warmup.

Example:
- warmup_seconds = 5
- duration_seconds = 60

---

## Metrics to Collect

### Per Worker:
- total_requests_completed
- total_time_seconds
- average_latency_ms
- throughput_requests_per_sec

### System-wide:
- GPU utilization (%)
- GPU memory usage (MB)
- timestamped samples (1 Hz)

---

## Logging Requirements

All logs must be written in structured CSV format.

Each row should include:

timestamp,
environment,
num_workers,
worker_id,
latency_ms,
throughput_rps,
gpu_utilization,
gpu_memory_MB

Logs must be:
- append-only
- consistent across runs
- written to /results directory

---

## Telemetry Monitor

The monitor must:

- Run in parallel with workload execution
- Sample GPU metrics every N seconds (default = 1)
- Use NVML via pynvml
- Record:
  - utilization.gpu
  - memory.used

The monitor must NOT:
- block workers
- introduce significant CPU overhead

---

## Experiment Loop (Critical)

The system must support running multiple experiment configurations:

Example loop:

for workers in [1, 2, 4, 8]:
    update config.num_workers
    run experiment
    save results

Each configuration must be run multiple times:

repeat_runs = 3–5

---

## Expected Behavior

As num_workers increases:

- throughput initially increases
- then plateaus (saturation point)
- latency increases sharply after saturation
- GPU utilization approaches ~100%

This produces the "knee curve" in analysis.

---

## Failure Conditions to Detect

The system should log warnings if:

- GPU utilization is low (<30%) during test
  → likely CPU bottleneck

- Throughput does not change with concurrency
  → workload not GPU-bound

- Workers crash or hang
  → resource contention too high

---

## Important Constraints

- Do NOT introduce heavy ML models for monitoring
- Monitoring must be lightweight (rule-based)
- Avoid excessive logging inside tight loops
- Ensure reproducibility across environments

---

## Output Expectations

At the end of all runs, the system should produce:

- Multiple CSV files (one per run or scenario)
- Data suitable for:
  - throughput vs concurrency plots
  - latency distributions
  - utilization vs throughput graphs

---

## Goal of Implementation

The system is not just running workloads.

It must enable analysis of:

- GPU contention behavior
- performance scaling limits
- differences between virtual and physical GPUs