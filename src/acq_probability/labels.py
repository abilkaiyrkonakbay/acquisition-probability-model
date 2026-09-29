"""Announcement-based labels with explicit, complete observation horizons."""

from __future__ import annotations

import pandas as pd

from acq_probability.data import TARGET_COLUMN, validate_panel


def build_labels(
    panel: pd.DataFrame, events: pd.DataFrame, *, as_of: str | pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Events must be a complete register of qualifying target announcements.

    Include withdrawn offers; the outcome is an announcement, not completion.
    Minority purchases, rumors and other exclusions must be filtered upstream.
    `as_of` asserts event-register coverage, not just today's date.
    """
    clean = validate_panel(panel, require_target=False)
    required = {"company_id", "announcement_date"}
    if not required.issubset(events.columns):
        raise ValueError("Events require company_id and announcement_date")
    registry = events.copy()
    registry["announcement_date"] = pd.to_datetime(registry["announcement_date"], errors="raise")
    cutoff = pd.Timestamp(as_of)
    if pd.isna(cutoff) or registry[list(required)].isna().any().any():
        raise ValueError("Event identifiers, dates and as_of cannot be missing")
    registry = registry.loc[registry["announcement_date"] <= cutoff]
    first = registry.groupby("company_id")["announcement_date"].min()
    clean["acquisition_date"] = clean["company_id"].map(first)
    already_announced = clean["acquisition_date"] <= clean["observation_date"]
    censored = clean["label_window_end"] > cutoff
    audit = {
        "input_rows": len(clean),
        "already_announced_rows": int(already_announced.sum()),
        "censored_rows_excluding_already_announced": int((censored & ~already_announced).sum()),
        "retained_rows": int((~already_announced & ~censored).sum()),
    }
    # Remove incomplete windows even when a positive event is already visible.
    clean = clean.loc[~already_announced & ~censored].copy()
    clean[TARGET_COLUMN] = (clean["acquisition_date"] <= clean["label_window_end"]).astype(int)
    clean = clean.drop(columns=["deal_type"], errors="ignore")
    clean["outcomes_observed_through"] = cutoff
    return validate_panel(clean), audit


def require_mature_labels(panel: pd.DataFrame) -> None:
    if "outcomes_observed_through" not in panel:
        raise ValueError(
            "Training needs outcomes_observed_through or --as-of YYYY-MM-DD. "
            "This date must reflect complete event coverage, not just today's date."
        )
    coverage = pd.to_datetime(panel["outcomes_observed_through"])
    if coverage.isna().any() or (panel["label_window_end"] > coverage).any():
        raise ValueError(
            "Incomplete 12-month outcome windows: use acq-model label to remove censored rows"
        )
