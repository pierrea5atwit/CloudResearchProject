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


## Usage

Run from the workspace root:

python project/main.py --config configs/virtual.yaml

## Command-Line Arguments

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| --config PATH | Yes | None | Path to YAML experiment config file. Supports absolute paths and relative paths from workspace root or project directory. |
| --gpu-index INT | No | 0 | GPU index used for runtime environment checks and telemetry sampling. |
| --sanity-duration INT | No | 5 | Number of seconds to sample GPU utilization for the sanity check. |
| --allow-non-virtual | No | Off | Bypasses strict virtual/vGPU gate checks for local testing when NVML or vGPU signals are unavailable. |
| --progress-interval-sec FLOAT | No | 5.0 | Interval (seconds) for worker progress updates in CLI logs. Internally throttled to avoid high overhead. |
| --no-progress | No | Off | Disables periodic worker progress logging. Final worker summaries are still printed. |

## Common Commands

Strict virtual run (default behavior):

python project/main.py --config configs/virtual.yaml

Virtual run with custom progress interval:

python project/main.py --config configs/virtual.yaml --progress-interval-sec 3

Run with explicit GPU index:

python project/main.py --config configs/virtual.yaml --gpu-index 0

Local test mode (non-virtual override):

python project/main.py --config configs/virtual.yaml --allow-non-virtual

Local test mode without progress spam:

python project/main.py --config configs/virtual.yaml --allow-non-virtual --no-progress

## Notes

- The workflow expects environment: virtual in the selected config.
- In strict mode, runtime must show virtual/vGPU indicators or execution stops.
- Use --allow-non-virtual only for local/debug environments where validation signals are incomplete.
