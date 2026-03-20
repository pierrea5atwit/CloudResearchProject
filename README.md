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
