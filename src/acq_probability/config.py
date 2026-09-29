from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        if not self.train_end < self.validation_end < self.test_end:
            raise ValueError("Split dates must satisfy train_end < validation_end < test_end")


@dataclass(frozen=True)
class ModelConfig:
    split: SplitConfig
    horizon_months: int = 12
    top_fraction: float = 0.10
    random_seed: int = 42
    bootstrap_iterations: int = 300

    def __post_init__(self) -> None:
        if self.horizon_months != 12:
            raise ValueError("This model's label is fixed at 12 months; horizon_months must be 12")
        if not 0 < self.top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1]")
        if self.bootstrap_iterations < 1:
            raise ValueError("bootstrap_iterations must be positive")


def load_config(path: str | Path) -> ModelConfig:
    """Load the versioned TOML configuration used by the CLI."""
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    split = raw["split"]
    model = raw.get("model", {})
    return ModelConfig(
        split=SplitConfig(
            train_end=pd.Timestamp(split["train_end"]),
            validation_end=pd.Timestamp(split["validation_end"]),
            test_end=pd.Timestamp(split["test_end"]),
        ),
        horizon_months=int(model.get("horizon_months", 12)),
        top_fraction=float(model.get("top_fraction", 0.10)),
        random_seed=int(model.get("random_seed", 42)),
        bootstrap_iterations=int(model.get("bootstrap_iterations", 300)),
    )
