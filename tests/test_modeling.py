import numpy as np
import pandas as pd
import pytest

from acq_probability.config import SplitConfig
from acq_probability.data import generate_synthetic_panel
from acq_probability.features import MODEL_FEATURES, engineer_features
from acq_probability.modeling import temporal_split, train_and_evaluate


def test_temporal_split_has_strict_nonoverlapping_windows() -> None:
    panel = generate_synthetic_panel(n_companies=180, start_year=2012, end_year=2024)
    features = engineer_features(panel)
    split = temporal_split(
        features,
        SplitConfig(
            train_end=pd.Timestamp("2018-12-31"),
            validation_end=pd.Timestamp("2020-12-31"),
            test_end=pd.Timestamp("2024-12-31"),
        ),
    )

    assert split.train["observation_date"].max() == pd.Timestamp("2018-12-31")
    assert split.validation["observation_date"].min() == pd.Timestamp("2019-12-31")
    assert split.validation["observation_date"].max() == pd.Timestamp("2020-12-31")
    assert split.test["observation_date"].min() == pd.Timestamp("2021-12-31")
    assert set(split.train.index).isdisjoint(split.validation.index)


def test_empty_partition_is_rejected() -> None:
    panel = generate_synthetic_panel(n_companies=120, start_year=2017, end_year=2019)
    features = engineer_features(panel)

    with pytest.raises(ValueError, match="partition is empty"):
        temporal_split(
            features,
            SplitConfig(
                train_end=pd.Timestamp("2016-12-31"),
                validation_end=pd.Timestamp("2018-12-31"),
                test_end=pd.Timestamp("2019-12-31"),
            ),
        )


def test_calibration_and_test_outcomes_cannot_select_model():
    panel = generate_synthetic_panel(n_companies=220, start_year=2012, end_year=2024)
    features = engineer_features(panel)
    config = SplitConfig(
        pd.Timestamp("2017-12-31"), pd.Timestamp("2020-12-31"), pd.Timestamp("2024-12-31")
    )
    first = train_and_evaluate(temporal_split(features, config))
    modified = features.copy()
    for date, rows in modified.groupby("observation_date").groups.items():
        if date >= pd.Timestamp("2020-12-31"):
            modified.loc[rows, "acquired_within_12m"] = np.roll(
                modified.loc[rows, "acquired_within_12m"].to_numpy(), 7
            )
    second = train_and_evaluate(temporal_split(modified, config))
    assert first.model_name == second.model_name
    pd.testing.assert_frame_equal(first.candidate_metrics, second.candidate_metrics)
    pd.testing.assert_frame_equal(first.fold_metrics, second.fold_metrics)
    test = features.loc[features["observation_date"] > config.validation_end, MODEL_FEATURES]
    np.testing.assert_allclose(
        first.model.base_model.predict_proba(test), second.model.base_model.predict_proba(test)
    )
