"""The complete workflow: prepare data, fit models, evaluate, save results."""

from pathlib import Path

import joblib
import pandas as pd

from acq_probability.config import ModelConfig
from acq_probability.data import AVAILABILITY_COLUMNS, generate_synthetic_panel
from acq_probability.features import NUMERIC_FEATURES, engineer_features
from acq_probability.labels import require_mature_labels
from acq_probability.metrics import bootstrap_lift_interval, enrich_predictions
from acq_probability.modeling import ModelResult, temporal_split, train_and_evaluate
from acq_probability.reporting import plot_deciles, write_json


def run_pipeline(
    panel: pd.DataFrame,
    config: ModelConfig,
    output_dir: str | Path,
    *,
    as_of: str | pd.Timestamp | None = None,
    data_source: str = "user_supplied",
) -> ModelResult:
    features = engineer_features(panel)
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        if pd.isna(cutoff):
            raise ValueError("as_of must be a valid outcome-coverage date")
        coverage = features.get(
            "outcomes_observed_through", pd.Series(cutoff, index=features.index)
        )
        features["outcomes_observed_through"] = coverage.clip(upper=cutoff)
    require_mature_labels(features)
    split = temporal_split(features, config.split)
    result = train_and_evaluate(
        split, top_fraction=config.top_fraction, random_seed=config.random_seed
    )
    predictions = enrich_predictions(result.test_predictions, top_fraction=config.top_fraction)
    result.test_predictions = predictions
    lower, upper = bootstrap_lift_interval(
        predictions["acquired_within_12m"],
        predictions["predicted_probability"],
        predictions["observation_date"],
        company_ids=predictions["company_id"],
        top_fraction=config.top_fraction,
        iterations=config.bootstrap_iterations,
        random_seed=config.random_seed,
    )
    result.test_metrics.update(
        {
            "selected_model": result.model_name,
            "data_source": data_source,
            "top_fraction": config.top_fraction,
            "lift_ci_lower": lower,
            "lift_ci_upper": upper,
            "mean_predicted_probability": float(predictions["predicted_probability"].mean()),
            "protocol": result.protocol,
            "minimum_outcome_coverage": features["outcomes_observed_through"].min(),
            "availability_timestamp_coverage": {
                name: float(pd.to_datetime(panel[name]).notna().mean()) if name in panel else 0.0
                for name in AVAILABILITY_COLUMNS
            },
        }
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    coverage = (
        pd.DataFrame(
            {
                "initial_training_nonmissing_fraction": split.train[NUMERIC_FEATURES]
                .notna()
                .mean(),
                "test_nonmissing_fraction": split.test[NUMERIC_FEATURES].notna().mean(),
            }
        )
        .rename_axis("feature")
        .reset_index()
    )
    for filename, table in {
        "test_predictions.csv": predictions,
        "backtest_by_year.csv": result.backtest,
        "model_selection.csv": result.candidate_metrics,
        "selection_by_fold.csv": result.fold_metrics,
        "feature_coverage.csv": coverage,
    }.items():
        table.to_csv(destination / filename, index=False, date_format="%Y-%m-%d")
    write_json(result.test_metrics, destination / "metrics.json")
    joblib.dump(
        {
            "schema_version": 3,
            "model": result.model,
            "top_fraction": config.top_fraction,
            "features": list(result.model.feature_names_in_),
            "earliest_score_date": result.protocol["calibration_label_end"],
            "data_source": data_source,
        },
        destination / "model.joblib",
    )
    plot_deciles(predictions, destination / "decile_chart.png")
    return result


def run_demo(
    config: ModelConfig,
    output_dir: str | Path,
    *,
    n_companies: int = 650,
) -> ModelResult:
    panel = generate_synthetic_panel(
        n_companies=n_companies,
        start_year=2012,
        end_year=config.split.test_end.year,
        random_seed=config.random_seed,
    )
    result = run_pipeline(panel, config, output_dir, data_source="synthetic")
    # A reproducible example input, not another copy saved for every real-data run.
    panel.to_csv(Path(output_dir) / "company_panel.csv", index=False, date_format="%Y-%m-%d")
    return result
