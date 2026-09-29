from __future__ import annotations

import numpy as np
import pandas as pd

from acq_probability.data import TARGET_COLUMN, validate_panel

BASELINE_NUMERIC_FEATURES = [
    "valuation_discount_ev_ebitda",
    "valuation_discount_ev_sales",
    "log_market_cap",
    "sales_growth_1y",
    "sales_growth_3y",
    "institutional_ownership",
    "insider_ownership",
    "analyst_coverage",
    "cash_to_ev",
    "net_debt_to_ebitda",
    "sector_hhi",
    "log_sector_deal_count_3y",
    "relative_return_12m",
    "ebitda_margin",
    "net_margin",
    "patent_intensity",
    "intangible_asset_ratio",
]
RESEARCH_FEATURE_BLOCKS = {
    "industry_waves": [
        "deal_activity_acceleration",
        "industry_shock",
        "credit_spread_bps",
        "shock_x_credit_capacity",
    ],
    "ownership_and_assets": [
        "institutional_ownership_change",
        "outside_block_ownership",
        "poison_pill",
        "staggered_board",
        "asset_tangibility",
    ],
    "financing_and_value": [
        "free_cash_flow_to_ev",
        "cash_x_low_growth",
        "discount_x_underperformance",
    ],
    "innovation": ["rd_to_sales", "log_patent_stock", "citations_per_patent"],
}
NUMERIC_FEATURES = BASELINE_NUMERIC_FEATURES + [
    feature for block in RESEARCH_FEATURE_BLOCKS.values() for feature in block
]
CATEGORICAL_FEATURES = ["sector"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
METADATA_COLUMNS = ["company_id", "company_name", "observation_date", "sector"]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float) / denominator.astype(float).replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def engineer_features(frame: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    """Build point-in-time, economically motivated features.

    Sector valuation medians are computed only among companies sharing the same
    observation date, so future rows never influence an earlier cross-section.
    """
    clean = validate_panel(frame, require_target=require_target)
    result = clean.copy()

    result["ev_to_sales"] = _safe_divide(result["enterprise_value"], result["revenue"])
    positive_ebitda = result["ebitda"].where(result["ebitda"] > 0)
    result["ev_to_ebitda"] = _safe_divide(result["enterprise_value"], positive_ebitda)

    peer_group = ["observation_date", "sector"]
    median_ev_sales = result.groupby(peer_group, observed=True)["ev_to_sales"].transform("median")
    median_ev_ebitda = result.groupby(peer_group, observed=True)["ev_to_ebitda"].transform("median")
    result["valuation_discount_ev_sales"] = (
        _safe_divide(median_ev_sales, result["ev_to_sales"]) - 1
    ).clip(-3, 3)
    result["valuation_discount_ev_ebitda"] = (
        _safe_divide(median_ev_ebitda, result["ev_to_ebitda"]) - 1
    ).clip(-3, 3)

    result["log_market_cap"] = np.log1p(result["market_cap"].clip(lower=0))
    result["cash_to_ev"] = _safe_divide(result["cash"], result["enterprise_value"]).clip(0, 2)
    result["net_debt_to_ebitda"] = _safe_divide(
        result["total_debt"] - result["cash"], positive_ebitda
    ).clip(-5, 10)
    result["log_sector_deal_count_3y"] = np.log1p(result["sector_deal_count_3y"].clip(lower=0))
    result["relative_return_12m"] = result["return_12m"] - result["sector_return_12m"]
    result["ebitda_margin"] = _safe_divide(result["ebitda"], result["revenue"]).clip(-2, 2)
    result["net_margin"] = _safe_divide(result["net_income"], result["revenue"]).clip(-2, 2)

    result["deal_activity_acceleration"] = (
        _safe_divide(result["sector_deal_count_1y"], result["sector_deal_count_3y"] / 3) - 1
    ).clip(-1, 2)
    result["industry_shock"] = result["sector_sales_growth"].abs()
    credit_capacity = 1 / (1 + result["credit_spread_bps"].clip(lower=0) / 100)
    result["shock_x_credit_capacity"] = result["industry_shock"] * credit_capacity
    result["institutional_ownership_change"] = (
        result["institutional_ownership"] - result["institutional_ownership_1y_ago"]
    )
    result["asset_tangibility"] = _safe_divide(result["ppe"], result["total_assets"]).clip(0, 1)
    result["free_cash_flow_to_ev"] = _safe_divide(
        result["free_cash_flow"], result["enterprise_value"]
    ).clip(-2, 2)
    result["cash_x_low_growth"] = result["cash_to_ev"] * (-result["sales_growth_1y"]).clip(0, 1)
    result["discount_x_underperformance"] = result["valuation_discount_ev_sales"].clip(lower=0) * (
        -result["relative_return_12m"]
    ).clip(lower=0)
    result["rd_to_sales"] = _safe_divide(result["rd_expense"], result["revenue"]).clip(0, 2)
    result["log_patent_stock"] = np.log1p(result["patent_stock"].clip(lower=0))
    result["citations_per_patent"] = _safe_divide(
        result["patent_citations_to_date"], result["patent_stock"]
    ).clip(0, 100)

    keep = list(dict.fromkeys(METADATA_COLUMNS + MODEL_FEATURES))
    if require_target:
        keep += [TARGET_COLUMN]
    keep += [
        column
        for column in [
            "acquisition_date",
            "label_window_end",
            "outcomes_observed_through",
        ]
        if column in result
    ]
    return result.loc[:, keep]


def model_matrix(feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model inputs and the binary forward-event target."""
    return feature_frame[MODEL_FEATURES].copy(), feature_frame[TARGET_COLUMN].astype(int).copy()
