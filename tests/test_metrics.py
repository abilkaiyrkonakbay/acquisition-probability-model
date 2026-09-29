import numpy as np
import pandas as pd

from acq_probability.metrics import aggregate_backtest, assign_deciles, ranking_backtest


def test_periodic_top_bucket_lift_and_capture() -> None:
    dates = pd.Series(["2023-12-31"] * 10 + ["2024-12-31"] * 10)
    actual = pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = pd.Series(np.linspace(1.0, 0.1, 20))

    backtest = ranking_backtest(actual, scores, dates, top_fraction=0.20)
    aggregate = aggregate_backtest(backtest)

    assert aggregate["base_rate"] == 0.20
    assert aggregate["top_decile_acquisition_rate"] == 0.75
    assert aggregate["top_decile_lift"] == 3.75
    assert aggregate["top_decile_capture_rate"] == 0.75


def test_deciles_are_assigned_within_date() -> None:
    custom_index = range(100, 120)
    scores = pd.Series(list(range(10, 0, -1)) * 2, index=custom_index, dtype=float)
    dates = pd.Series(
        ["2023-12-31"] * 10 + ["2024-12-31"] * 10,
        index=custom_index,
    )
    deciles = assign_deciles(scores, dates)

    assert deciles.tolist() == list(range(1, 11)) * 2
    assert deciles.index.tolist() == list(custom_index)
