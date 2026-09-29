"""Small reporting helpers: one JSON summary and one decile chart."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def write_json(payload: dict, path: Path) -> None:
    def normalize(value):
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return value

    path.write_text(
        json.dumps(normalize(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def plot_deciles(predictions: pd.DataFrame, path: Path) -> None:
    # Import plotting only when exporting a chart, never during labeling or scoring.
    os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = predictions.groupby("predicted_decile")["acquired_within_12m"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#E9C46A" if decile == 1 else "#15324B" for decile in rates.index]
    ax.bar(rates.index.astype(str), rates, color=colors)
    ax.axhline(
        predictions["acquired_within_12m"].mean(),
        color="#6B7280",
        linestyle="--",
        label="Universe base rate",
    )
    ax.set(
        title="Held-out acquisition frequency by predicted decile",
        xlabel="Predicted decile (1 = highest propensity)",
        ylabel="Acquisition frequency",
    )
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
