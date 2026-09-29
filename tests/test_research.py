import json

import numpy as np
import pandas as pd
import pytest

from acq_probability.cli import main
from acq_probability.config import ModelConfig, SplitConfig
from acq_probability.data import RESEARCH_INPUTS, generate_synthetic_panel, validate_panel
from acq_probability.features import MODEL_FEATURES, engineer_features
from acq_probability.labels import build_labels, require_mature_labels
from acq_probability.metrics import bootstrap_lift_interval, calculate_metrics, enrich_predictions
from acq_probability.modeling import mature_history, temporal_split, train_and_evaluate
from acq_probability.reporting import write_json


@pytest.fixture(scope="module")
def panel():
    return generate_synthetic_panel(n_companies=120, start_year=2012, end_year=2024)


def test_fractional_labels_are_rejected(panel):
    frame = panel.copy()
    frame["acquired_within_12m"] = frame["acquired_within_12m"].astype(float)
    frame.loc[0, "acquired_within_12m"] = 0.5
    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_panel(frame)


def test_future_source_timestamps_are_rejected(panel):
    frame = panel.copy()
    frame.loc[0, "industry_available_date"] = frame.loc[0, "observation_date"] + pd.Timedelta(
        days=1
    )
    with pytest.raises(ValueError, match="unavailable"):
        engineer_features(frame)


def test_event_label_mismatch_rejected(panel):
    frame = panel.copy()
    frame.loc[0, "acquisition_date"] = frame.loc[0, "observation_date"] + pd.Timedelta(days=1)
    frame.loc[0, "acquired_within_12m"] = 0
    with pytest.raises(ValueError, match="conflicts"):
        validate_panel(frame)


def test_missing_research_inputs_remain_missing(panel):
    features = engineer_features(panel.drop(columns=RESEARCH_INPUTS))
    for feature in ["asset_tangibility", "rd_to_sales", "shock_x_credit_capacity"]:
        assert features[feature].isna().all()
    assert "acquisition_date" not in MODEL_FEATURES
    assert "deal_type" not in MODEL_FEATURES
    assert "label_window_end" not in MODEL_FEATURES


def test_research_formulas(panel):
    frame = panel.iloc[[0]].copy()
    frame["total_assets"], frame["ppe"] = 100, 40
    frame["sector_deal_count_1y"], frame["sector_deal_count_3y"] = 6, 12
    frame["sector_sales_growth"], frame["credit_spread_bps"] = -0.2, 100
    frame["institutional_ownership"], frame["institutional_ownership_1y_ago"] = 0.6, 0.5
    features = engineer_features(frame).iloc[0]
    assert features["asset_tangibility"] == pytest.approx(0.4)
    assert features["deal_activity_acceleration"] == pytest.approx(0.5)
    assert features["shock_x_credit_capacity"] == pytest.approx(0.1)
    assert features["institutional_ownership_change"] == pytest.approx(0.1)


def test_announcement_labels_remove_known_targets_and_censored_windows(panel):
    frame = pd.concat([panel.iloc[[0]].copy()] * 4, ignore_index=True)
    frame["company_id"] = ["already", "target", "late", "censored"]
    frame["observation_date"] = pd.to_datetime(["2020-12-31"] * 3 + ["2021-06-30"])
    events = pd.DataFrame(
        {
            "company_id": ["already", "target", "late", "censored"],
            "announcement_date": ["2020-12-31", "2021-12-31", "2022-01-01", "2021-07-01"],
        }
    )
    # Remove original demo provenance, which belongs to different dates.
    frame = frame.drop(columns=[c for c in frame if c.endswith("available_date")])
    labeled, audit = build_labels(frame, events, as_of="2021-12-31")
    assert labeled.set_index("company_id")["acquired_within_12m"].to_dict() == {
        "late": 0,
        "target": 1,
    }
    assert audit["already_announced_rows"] == 1
    assert audit["censored_rows_excluding_already_announced"] == 1
    require_mature_labels(labeled)


def test_training_requires_complete_outcome_coverage(panel):
    with pytest.raises(ValueError, match="outcomes_observed_through"):
        require_mature_labels(panel.drop(columns="outcomes_observed_through"))
    frame = panel.copy()
    frame["outcomes_observed_through"] = frame["observation_date"] + pd.Timedelta(days=5)
    with pytest.raises(ValueError, match="Incomplete"):
        require_mature_labels(frame)


def test_label_end_cannot_be_shortened_to_evade_purge(panel):
    frame = panel.copy()
    frame["label_window_end"] = frame["observation_date"]
    clean = validate_panel(frame)
    pd.testing.assert_series_equal(
        clean["label_window_end"],
        clean["observation_date"] + pd.DateOffset(months=12),
        check_names=False,
    )


def test_monthly_overlap_is_purged():
    frame = pd.DataFrame(
        {"observation_date": pd.to_datetime(["2018-12-31", "2019-01-31", "2019-12-31"])}
    )
    mature = mature_history(frame, pd.Timestamp("2019-12-31"))
    assert mature.index.tolist() == [0]


def test_calibration_outcomes_must_mature_before_test(panel):
    features = engineer_features(panel)
    features.loc[features["observation_date"] == pd.Timestamp("2021-12-31"), "observation_date"] = (
        pd.Timestamp("2021-06-30")
    )
    split = temporal_split(
        features,
        SplitConfig(
            pd.Timestamp("2017-12-31"), pd.Timestamp("2020-12-31"), pd.Timestamp("2024-12-31")
        ),
    )
    with pytest.raises(ValueError, match="Calibration outcomes overlap"):
        train_and_evaluate(split)


def test_cluster_bootstrap_is_deterministic():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.2, 0.1] * 2)
    dates = np.array(["2020-12-31"] * 4 + ["2021-12-31"] * 4)
    ids = np.array(["a", "b", "c", "d"] * 2)
    first = bootstrap_lift_interval(y, score, dates, company_ids=ids, iterations=50)
    second = bootstrap_lift_interval(y, score, dates, company_ids=ids, iterations=50)
    assert first == second
    assert 0 <= first[0] <= first[1]


def test_zero_event_slice_and_nested_json(tmp_path):
    metrics, _ = calculate_metrics([0, 0], [0.1, 0.2], ["2020-12-31"] * 2)
    assert np.isnan(metrics["top_decile_lift"])
    assert np.isnan(metrics["roc_auc"])
    output = tmp_path / "metrics.json"
    write_json({"nested": metrics}, output)
    assert json.loads(output.read_text())["nested"]["top_decile_lift"] is None


def test_custom_bucket_matches_backtest():
    data = pd.DataFrame(
        {"predicted_probability": np.linspace(1, 0, 11), "observation_date": "2020-12-31"}
    )
    enriched = enrich_predictions(data, top_fraction=0.2)
    assert enriched["selected_flag"].sum() == 3
    assert (enriched["predicted_decile"] == 1).sum() == 2


def test_invalid_fraction_is_rejected():
    predictions = pd.DataFrame({"predicted_probability": [0.1], "observation_date": ["2020-12-31"]})
    with pytest.raises(ValueError, match="top_fraction"):
        enrich_predictions(predictions, top_fraction=0)


def test_unsupported_horizon_and_bad_dates_rejected():
    config = SplitConfig(pd.Timestamp("2017"), pd.Timestamp("2020"), pd.Timestamp("2024"))
    with pytest.raises(ValueError, match="12 months"):
        ModelConfig(config, horizon_months=6)
    with pytest.raises(ValueError, match="Split dates"):
        SplitConfig(pd.Timestamp("2020"), pd.Timestamp("2017"), pd.Timestamp("2024"))


def test_cli_missing_file_is_actionable(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["train", "--input", "this-file-does-not-exist.csv"])
    assert exc.value.code == 2
    assert "acq-model demo" in capsys.readouterr().err
