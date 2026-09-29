from pathlib import Path

import duckdb
import pandas as pd
import pytest

from acq_probability.data import generate_synthetic_panel, validate_panel
from acq_probability.labels import build_labels
from acq_probability.metrics import enrich_predictions, ranking_backtest


@pytest.fixture
def mart():
    raw = generate_synthetic_panel(n_companies=100, start_year=2017, end_year=2019)
    conn = duckdb.connect()
    prices = raw[
        ["company_id", "observation_date", "market_cap", "return_12m", "sector_return_12m"]
    ].rename(columns={"observation_date": "month_end", "return_12m": "total_return_12m"})
    prices["is_month_end"] = True
    master = raw[["company_id", "observation_date", "company_name", "sector"]].copy()
    master["is_public"] = True
    master["available_date"] = master["observation_date"]
    fundamentals = raw[
        [
            "company_id",
            "filing_available_date",
            "enterprise_value",
            "revenue",
            "ebitda",
            "net_income",
            "cash",
            "total_debt",
            "sales_growth_1y",
            "sales_growth_3y",
            "total_assets",
            "ppe",
            "free_cash_flow",
            "rd_expense",
        ]
    ].copy()
    fundamentals["fiscal_period_end"] = fundamentals["filing_available_date"] - pd.DateOffset(
        months=1
    )
    ownership = raw[
        [
            "company_id",
            "observation_date",
            "institutional_ownership",
            "insider_ownership",
            "analyst_coverage",
            "institutional_ownership_1y_ago",
            "outside_block_ownership",
            "poison_pill",
            "staggered_board",
            "ownership_available_date",
        ]
    ].rename(columns={"ownership_available_date": "available_date"})
    assets = raw[
        [
            "company_id",
            "observation_date",
            "patent_intensity",
            "intangible_asset_ratio",
            "patent_stock",
            "patent_citations_to_date",
            "innovation_available_date",
        ]
    ].rename(columns={"innovation_available_date": "available_date"})
    industry = (
        raw[
            [
                "sector",
                "observation_date",
                "sector_hhi",
                "sector_deal_count_3y",
                "sector_deal_count_1y",
                "sector_sales_growth",
                "credit_spread_bps",
                "industry_available_date",
            ]
        ]
        .drop_duplicates()
        .rename(columns={"industry_available_date": "available_date"})
    )
    events = raw.loc[raw["acquisition_date"].notna(), ["company_id", "acquisition_date"]].rename(
        columns={"company_id": "target_company_id", "acquisition_date": "announcement_date"}
    )
    events["is_qualifying_control_announcement"] = True
    coverage = pd.DataFrame({"outcomes_observed_through": [pd.Timestamp("2019-12-31")]})
    for name, frame in {
        "prices_monthly": prices,
        "company_master_snapshot": master,
        "fundamentals_quarterly": fundamentals,
        "ownership_snapshot": ownership,
        "strategic_assets_snapshot": assets,
        "industry_snapshot": industry,
        "mna_events": events,
        "data_coverage": coverage,
    }.items():
        conn.register("input_frame", frame)
        conn.execute(f"CREATE TABLE {name} AS SELECT * FROM input_frame")
    query = (Path(__file__).parents[1] / "sql" / "feature_mart.sql").read_text()
    yield conn, query, raw, events
    conn.close()


def test_sql_labels_match_python_and_exclude_incomplete_windows(mart):
    conn, query, raw, events = mart
    conn.execute(query)
    actual = validate_panel(conn.table("acquisition_feature_mart").df())
    expected, _ = build_labels(
        raw, events.rename(columns={"target_company_id": "company_id"}), as_of="2019-12-31"
    )
    columns = ["company_id", "observation_date", "acquired_within_12m"]
    pd.testing.assert_frame_equal(actual[columns], expected[columns], check_dtype=False)


def test_sql_excludes_future_ownership_and_already_announced_targets(mart):
    conn, query, raw, _ = mart
    company = raw.iloc[0]["company_id"]
    conn.execute("UPDATE ownership_snapshot SET available_date = DATE '2030-01-01'")
    conn.execute("INSERT INTO mna_events VALUES (?, DATE '2017-01-01', TRUE)", [company])
    conn.execute(query)
    actual = conn.table("acquisition_feature_mart").df()
    assert company not in set(actual["company_id"])
    assert actual["institutional_ownership"].isna().all()
    assert actual["ownership_available_date"].isna().all()


def test_sql_csv_backtest_matches_python(tmp_path, monkeypatch):
    predictions = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(["2023-12-31"] * 5 + ["2024-12-31"] * 5),
            "predicted_probability": [0.8, 0.6, 0.4, 0.2, 0.1] * 2,
            "acquired_within_12m": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        }
    )
    predictions = enrich_predictions(predictions, top_fraction=0.4)
    output = tmp_path / "outputs" / "demo"
    output.mkdir(parents=True)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    query = (Path(__file__).parents[1] / "sql" / "top_decile_backtest.sql").read_text()
    monkeypatch.chdir(tmp_path)
    with duckdb.connect() as conn:
        actual = conn.execute(query).df()
    expected = ranking_backtest(
        predictions["acquired_within_12m"],
        predictions["predicted_probability"],
        predictions["observation_date"],
        top_fraction=0.4,
    )
    assert actual["selected_count"].tolist() == expected["top_n"].tolist()
    assert actual["selected_acquisitions"].tolist() == expected["top_decile_acquisitions"].tolist()
    assert actual["lift"].tolist() == expected["lift"].tolist()
