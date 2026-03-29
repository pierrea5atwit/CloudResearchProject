"""Experiment configuration helpers."""

from pathlib import Path
from typing import Any

import yaml

try:
	from pynvml import (  # type: ignore
		nvmlDeviceGetHandleByIndex,
		nvmlDeviceGetUtilizationRates,
		nvmlInit,
		nvmlShutdown,
	)
except Exception:  # pragma: no cover - optional GPU dependency at dev time
	nvmlDeviceGetHandleByIndex = None  # type: ignore[assignment]
	nvmlDeviceGetUtilizationRates = None  # type: ignore[assignment]
	nvmlInit = None  # type: ignore[assignment]
	nvmlShutdown = None  # type: ignore[assignment]


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
VIRTUAL_CONFIG = CONFIG_DIR / "virtual.yaml"
PHYSICAL_CONFIG = CONFIG_DIR / "physical.yaml"
SUPPORTED_ENVIRONMENTS = {"virtual", "physical"}


def load_config(config_path: str | Path) -> dict[str, Any]:
	"""Load a YAML experiment config with structural checks."""
	path = Path(config_path).expanduser().resolve()
	if not path.exists():
		raise FileNotFoundError(f"Config file not found: {path}")
	if path.suffix.lower() not in {".yaml", ".yml"}:
		raise ValueError(f"Config must be a YAML file (.yaml/.yml): {path}")

	with path.open("r", encoding="utf-8") as handle:
		payload = yaml.safe_load(handle)

	if not isinstance(payload, dict):
		raise ValueError("Config must parse to a dictionary-like object")

	required_sections = {"environment", "experiment", "workload", "concurrency", "monitoring", "logging"}
	missing = sorted(required_sections.difference(payload.keys()))
	if missing:
		raise ValueError(f"Config missing required sections: {', '.join(missing)}")

	environment = str(payload.get("environment", "")).strip().lower()
	if environment not in SUPPORTED_ENVIRONMENTS:
		supported = ", ".join(sorted(SUPPORTED_ENVIRONMENTS))
		raise ValueError(f"Invalid environment '{environment}'. Expected one of: {supported}")

	return payload


def run_gpu_sanity_check(
	duration_seconds: int = 5,
	min_expected_gpu_utilization: float = 50.0,
	gpu_index: int = 0,
) -> dict[str, Any]:
	"""Sample GPU utilization and validate experiment credibility threshold."""
	if duration_seconds <= 0:
		raise ValueError("duration_seconds must be > 0")

	if not all([nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlShutdown]):
		return {
			"valid": False,
			"peak_gpu_utilization": None,
			"samples_collected": 0,
			"reason": "pynvml not available",
		}

	import time

	samples: list[float] = []
	try:
		nvmlInit()
		handle = nvmlDeviceGetHandleByIndex(gpu_index)
		for _ in range(duration_seconds):
			util = nvmlDeviceGetUtilizationRates(handle)
			samples.append(float(util.gpu))
			time.sleep(1)
	except Exception as exc:
		return {
			"valid": False,
			"peak_gpu_utilization": None,
			"samples_collected": len(samples),
			"reason": f"sanity check failed: {exc}",
		}
	finally:
		try:
			nvmlShutdown()
		except Exception:
			pass

	peak = max(samples) if samples else None
	valid = peak is not None and peak >= min_expected_gpu_utilization
	return {
		"valid": valid,
		"peak_gpu_utilization": peak,
		"samples_collected": len(samples),
		"threshold": min_expected_gpu_utilization,
		"reason": None if valid else "GPU utilization stayed below credibility threshold",
	}
