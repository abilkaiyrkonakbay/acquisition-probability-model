import pandas as pd
import pytest

from acq_probability.data import generate_synthetic_panel, validate_panel
from acq_probability.features import engineer_features


def test_future_cross_section_does_not_change_past_features() -> None:
    panel = generate_synthetic_panel(n_companies=120, start_year=2017, end_year=2019)
    past = panel.loc[panel["observation_date"] <= pd.Timestamp("2018-12-31")].copy()

    past_features = engineer_features(past).sort_values(["company_id", "observation_date"])
    full_features = engineer_features(panel)
    comparable = full_features.loc[
        full_features["observation_date"] <= pd.Timestamp("2018-12-31")
    ].sort_values(["company_id", "observation_date"])

    pd.testing.assert_frame_equal(
        past_features.reset_index(drop=True), comparable.reset_index(drop=True), check_exact=True
    )


def test_duplicate_company_date_is_rejected() -> None:
    panel = generate_synthetic_panel(n_companies=100, start_year=2017, end_year=2018)
    panel = panel.drop(columns=["acquisition_date"])
    duplicated = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate company-date"):
        validate_panel(duplicated)
