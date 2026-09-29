import json

import joblib
import numpy as np
import pandas as pd
import pytest

from acq_probability.cli import main
from acq_probability.config import ModelConfig, SplitConfig
from acq_probability.data import RESEARCH_INPUTS, generate_synthetic_panel
from acq_probability.features import engineer_features
from acq_probability.pipeline import run_pipeline


@pytest.mark.parametrize("include_research", [True, False])
def test_pipeline_writes_reproducible_portfolio_artifacts(tmp_path, include_research) -> None:
    panel = generate_synthetic_panel(n_companies=220, start_year=2012, end_year=2024)
    if not include_research:
        panel = panel.drop(columns=RESEARCH_INPUTS)
    config = ModelConfig(
        split=SplitConfig(
            train_end=pd.Timestamp("2017-12-31"),
            validation_end=pd.Timestamp("2020-12-31"),
            test_end=pd.Timestamp("2024-12-31"),
        ),
        bootstrap_iterations=20,
    )

    run_pipeline(panel, config, tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())

    assert metrics["top_decile_lift"] > 0
    assert metrics["n_observations"] > 0
    assert {p.name for p in tmp_path.iterdir()} == {
        "model.joblib",
        "decile_chart.png",
        "feature_coverage.csv",
        "metrics.json",
        "test_predictions.csv",
        "backtest_by_year.csv",
        "model_selection.csv",
        "selection_by_fold.csv",
    }
    protocol = metrics["protocol"]
    assert protocol["selection_dates"] == ["2018-12-31", "2019-12-31"]
    assert protocol["calibration_date"] not in protocol["selection_dates"]
    assert protocol["calibration_label_end"] <= protocol["first_test_date"]
    folds = pd.read_csv(tmp_path / "selection_by_fold.csv")
    assert (folds["latest_train_label_end"] <= folds["fold_date"]).all()
    assert "baseline_logistic" in set(folds["model"])
    assert "research_without_underperformance" in set(folds["model"])
    bundle = joblib.load(tmp_path / "model.joblib")
    latest = panel.loc[panel["observation_date"] == panel["observation_date"].max()].copy()
    latest_path = tmp_path / "latest.csv"
    latest.drop(columns=["acquired_within_12m", "acquisition_date"]).to_csv(
        latest_path, index=False
    )
    score_path = tmp_path / "scored.csv"
    main(
        [
            "score",
            "--input",
            str(latest_path),
            "--model",
            str(tmp_path / "model.joblib"),
            "--output",
            str(score_path),
        ]
    )
    scores = pd.read_csv(score_path).set_index("company_id")
    features = engineer_features(latest, require_target=False)
    expected = pd.Series(
        bundle["model"].predict_proba(features[bundle["features"]])[:, 1],
        index=features["company_id"],
    )
    np.testing.assert_allclose(scores["predicted_probability"], expected.reindex(scores.index))
