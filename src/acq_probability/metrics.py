from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def _validate_fraction(top_fraction: float) -> None:
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")


def ranking_backtest(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    dates: pd.Series | np.ndarray,
    *,
    top_fraction: float = 0.10,
) -> pd.DataFrame:
    """Evaluate a periodically rebalanced top-ranked target screen."""
    _validate_fraction(top_fraction)
    data = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(np.asarray(dates)),
            "actual": np.asarray(y_true, dtype=int),
            "score": np.asarray(scores, dtype=float),
        }
    )
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for observation_date, period in data.groupby("observation_date", sort=True):
        ranked = period.sort_values("score", ascending=False, kind="mergesort")
        top_n = max(1, math.ceil(len(ranked) * top_fraction))
        selected = ranked.iloc[:top_n]
        base_rate = float(ranked["actual"].mean())
        top_rate = float(selected["actual"].mean())
        events = int(ranked["actual"].sum())
        selected_events = int(selected["actual"].sum())
        rows.append(
            {
                "observation_date": observation_date,
                "universe_size": int(len(ranked)),
                "acquisitions": events,
                "base_rate": base_rate,
                "top_n": top_n,
                "top_decile_acquisitions": selected_events,
                "top_decile_rate": top_rate,
                "lift": top_rate / base_rate if base_rate > 0 else np.nan,
                "capture_rate": selected_events / events if events > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def aggregate_backtest(backtest: pd.DataFrame) -> dict[str, float]:
    observations = float(backtest["universe_size"].sum())
    events = float(backtest["acquisitions"].sum())
    selected = float(backtest["top_n"].sum())
    selected_events = float(backtest["top_decile_acquisitions"].sum())
    base_rate = events / observations if observations else np.nan
    top_rate = selected_events / selected if selected else np.nan
    return {
        "base_rate": base_rate,
        "top_decile_acquisition_rate": top_rate,
        "top_decile_lift": top_rate / base_rate if base_rate > 0 else np.nan,
        "top_decile_capture_rate": selected_events / events if events > 0 else np.nan,
    }


def assign_deciles(scores: pd.Series, dates: pd.Series) -> pd.Series:
    """Assign 1 (highest score) through 10 (lowest) within each date."""
    score_series = pd.Series(scores, copy=True)
    if isinstance(dates, pd.Series):
        date_series = pd.to_datetime(dates).reindex(score_series.index)
    else:
        date_series = pd.Series(pd.to_datetime(dates), index=score_series.index)
    frame = pd.DataFrame({"score": score_series, "date": date_series})
    result = pd.Series(index=frame.index, dtype="int64", name="predicted_decile")
    for _, period_index in frame.groupby("date", sort=False).groups.items():
        ordinal_rank = frame.loc[period_index, "score"].rank(method="first", ascending=False)
        result.loc[period_index] = (np.floor((ordinal_rank - 1) * 10 / len(period_index)) + 1).clip(
            1, 10
        )
    return result.astype(int)


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    dates: pd.Series | np.ndarray,
    *,
    top_fraction: float = 0.10,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    actual = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1 - 1e-8)
    if len(actual) == 0:
        raise ValueError("Evaluation set is empty")

    backtest = ranking_backtest(actual, probability, dates, top_fraction=top_fraction)
    metrics: dict[str, float | int] = {
        "n_observations": int(len(actual)),
        "acquisitions": int(actual.sum()),
        "roc_auc": float(roc_auc_score(actual, probability))
        if len(np.unique(actual)) == 2
        else np.nan,
        "average_precision": float(average_precision_score(actual, probability))
        if actual.sum()
        else np.nan,
        "brier_score": float(brier_score_loss(actual, probability)),
        "log_loss": float(log_loss(actual, probability, labels=[0, 1])),
    }
    metrics.update(aggregate_backtest(backtest))
    return metrics, backtest


def bootstrap_lift_interval(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    dates: pd.Series | np.ndarray,
    *,
    top_fraction: float = 0.10,
    iterations: int = 300,
    random_seed: int = 42,
    company_ids: pd.Series | np.ndarray,
) -> tuple[float, float]:
    """Resample whole firms, preserving repeated observations but not regime uncertainty."""
    data = pd.DataFrame(
        {
            "actual": np.asarray(y_true, dtype=int),
            "score": np.asarray(probabilities, dtype=float),
            "date": pd.to_datetime(np.asarray(dates)),
        }
    )
    rng = np.random.default_rng(random_seed)
    values: list[float] = []
    data["company_id"] = np.asarray(company_ids)
    if data["company_id"].isna().any() or data.empty:
        raise ValueError("Cluster bootstrap requires nonmissing company IDs and a nonempty sample")
    histories = [g for _, g in data.groupby("company_id", sort=True)]
    for _ in range(iterations):
        sampled = [histories[i] for i in rng.integers(0, len(histories), len(histories))]
        draw = pd.concat(sampled, ignore_index=True)
        bt = ranking_backtest(
            draw["actual"], draw["score"], draw["date"], top_fraction=top_fraction
        )
        lift = aggregate_backtest(bt)["top_decile_lift"]
        if np.isfinite(lift):
            values.append(float(lift))
    if not values:
        return np.nan, np.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def enrich_predictions(predictions: pd.DataFrame, *, top_fraction: float = 0.1) -> pd.DataFrame:
    """One ranked output for both scoring and backtesting; no duplicate shortlist CSV."""
    _validate_fraction(top_fraction)
    result = predictions.copy()
    result["predicted_decile"] = assign_deciles(
        result["predicted_probability"], result["observation_date"]
    )
    by_date = result.groupby("observation_date")["predicted_probability"]
    rank = by_date.rank(method="first", ascending=False)
    result["selected_flag"] = (rank <= np.ceil(by_date.transform("size") * top_fraction)).astype(
        int
    )
    return result.sort_values(
        ["observation_date", "predicted_probability"], ascending=[True, False], kind="stable"
    ).reset_index(drop=True)
