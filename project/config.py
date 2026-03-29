"""Experiment configuration helpers."""

from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
VIRTUAL_CONFIG = CONFIG_DIR / "virtual.yaml"
PHYSICAL_CONFIG = CONFIG_DIR / "physical.yaml"
