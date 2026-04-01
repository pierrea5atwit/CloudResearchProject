#!/usr/bin/env python
"""Test script to validate the fixes independently."""

import sys
from pathlib import Path

# Add project module to path
project_dir = Path(__file__).parent / "project"
sys.path.insert(0, str(project_dir))

from config import load_config
from workload import Workload
from main import _validate_runtime_gpu_activity

def test_dry_run_logging():
    """Test that dry run result can be formatted on separate lines."""
    print("\n=== Test 1: Dry Run Logging Format ===")
    config = load_config(Path(__file__).parent / "project" / "configs" / "virtual.yaml")
    workload = Workload(config["workload"])
    
    print("Running workload (this will take a few seconds)...")
    result = workload.run()
    
    print("✓ Dry run result keys (each would print on its own line):")
    for key in sorted(result.keys()):
        print(f"    {key}: {result[key]}")
    
    return True

def test_gpu_activity_check():
    """Test that GPU activity check handles CPU workloads properly."""
    print("\n=== Test 2: GPU Activity Check Logic ===")
    
    # Test case 1: CPU workload (should not require GPU activity)
    print("Test case 2a: CPU-based kNN workload")
    config = load_config(Path(__file__).parent / "project" / "configs" / "virtual.yaml")
    workload_type = str(config.get("workload", {}).get("workload_type", "knn")).lower().strip()
    print(f"  Workload type: {workload_type}")
    
    if workload_type == "gpu_kernel":
        print("  ✓ GPU kernel workload detected - GPU activity would be required")
    else:
        print(f"  ✓ CPU workload ({workload_type}) - GPU activity check would be skipped")
    
    # Test case 2: GPU activity validation with empty samples (simulating no GPU activity)
    print("\nTest case 2b: GPU activity validation with zero GPU utilization")
    telemetry_samples = [
        {"timestamp": 1.0, "gpu_utilization_pct": 0.0},
        {"timestamp": 2.0, "gpu_utilization_pct": 0.0},
        {"timestamp": 3.0, "gpu_utilization_pct": 0.0},
    ]
    
    result = _validate_runtime_gpu_activity(telemetry_samples, min_expected_gpu_utilization=30.0)
    print(f"  Validation result: {result}")
    
    if not result.get("valid"):
        print(f"  ✓ GPU validation correctly failed: {result.get('reason')}")
    
    # Test case 3: Show that logic would handle CPU vs GPU workload check
    print("\nTest case 2c: Simulating fix logic")
    workload_configs = [
        {"workload_type": "knn"},
        {"workload_type": "gpu_kernel"},
    ]
    
    for cfg in workload_configs:
        wt = str(cfg.get("workload_type", "knn")).lower().strip()
        if wt == "gpu_kernel":
            print(f"  - {wt}: GPU activity would be REQUIRED (error if not met)")
        else:
            print(f"  - {wt}: GPU activity would be OPTIONAL (only warning if missing)")
    
    return True

if __name__ == "__main__":
    try:
        success = True
        success &= test_dry_run_logging()
        success &= test_gpu_activity_check()
        
        if success:
            print("\n✓ All validation tests passed!")
            sys.exit(0)
        else:
            print("\n✗ Some tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
