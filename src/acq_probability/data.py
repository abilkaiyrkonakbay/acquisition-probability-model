from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

IDENTIFIER_COLUMNS = ["company_id", "company_name", "observation_date", "sector"]
RAW_FEATURE_COLUMNS = [
    "market_cap",
    "enterprise_value",
    "revenue",
    "ebitda",
    "net_income",
    "cash",
    "total_debt",
    "sales_growth_1y",
    "sales_growth_3y",
    "return_12m",
    "sector_return_12m",
    "institutional_ownership",
    "insider_ownership",
    "analyst_coverage",
    "patent_intensity",
    "intangible_asset_ratio",
    "sector_hhi",
    "sector_deal_count_3y",
]
TARGET_COLUMN = "acquired_within_12m"
# Optional research inputs. Unknown values stay missing, never an assumed zero.
RESEARCH_INPUTS = [
    "total_assets",
    "ppe",
    "free_cash_flow",
    "institutional_ownership_1y_ago",
    "outside_block_ownership",
    "poison_pill",
    "staggered_board",
    "sector_deal_count_1y",
    "sector_sales_growth",
    "credit_spread_bps",
    "rd_expense",
    "patent_stock",
    "patent_citations_to_date",
]
AVAILABILITY_COLUMNS = [
    "filing_available_date",
    "ownership_available_date",
    "industry_available_date",
    "innovation_available_date",
]


def required_columns(require_target: bool = True) -> list[str]:
    columns = IDENTIFIER_COLUMNS + RAW_FEATURE_COLUMNS
    if require_target:
        columns.append(TARGET_COLUMN)
    return columns


def validate_panel(frame: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    """Validate and normalize a point-in-time company panel."""
    missing = sorted(set(required_columns(require_target)) - set(frame.columns))
    if missing:
        raise ValueError(f"Input panel is missing required columns: {', '.join(missing)}")

    clean = frame.copy()
    clean["observation_date"] = pd.to_datetime(clean["observation_date"], errors="raise")
    if clean[IDENTIFIER_COLUMNS].isna().any().any():
        raise ValueError("Identifiers, sector and observation_date cannot be missing")
    if "acquisition_date" in clean:
        clean["acquisition_date"] = pd.to_datetime(clean["acquisition_date"], errors="raise")
    for column in AVAILABILITY_COLUMNS:
        if column in clean:
            clean[column] = pd.to_datetime(clean[column], errors="raise")
            if (clean[column] > clean["observation_date"]).any():
                raise ValueError(f"{column} contains information unavailable at observation_date")
    if "outcomes_observed_through" in clean:
        clean["outcomes_observed_through"] = pd.to_datetime(
            clean["outcomes_observed_through"], errors="raise"
        )
    # Always reconstruct this field; never trust a supplied shortened label window.
    clean["label_window_end"] = clean["observation_date"] + pd.DateOffset(months=12)

    duplicates = clean.duplicated(["company_id", "observation_date"])
    if duplicates.any():
        example = clean.loc[duplicates, ["company_id", "observation_date"]].iloc[0].to_dict()
        raise ValueError(f"Duplicate company-date observations are not allowed: {example}")

    for column in RAW_FEATURE_COLUMNS + RESEARCH_INPUTS:
        if column not in clean:
            clean[column] = np.nan
        clean[column] = pd.to_numeric(clean[column], errors="raise").replace(
            [np.inf, -np.inf], np.nan
        )

    if require_target:
        clean[TARGET_COLUMN] = pd.to_numeric(clean[TARGET_COLUMN], errors="raise")
        invalid_targets = ~clean[TARGET_COLUMN].isin([0, 1])
        if invalid_targets.any():
            raise ValueError("acquired_within_12m must contain only 0 and 1")
        clean[TARGET_COLUMN] = clean[TARGET_COLUMN].astype(int)
        if "acquisition_date" in clean:
            known = clean["acquisition_date"].notna()
            if (clean.loc[known, "acquisition_date"] <= clean.loc[known, "observation_date"]).any():
                raise ValueError("Already-announced targets must be removed from the universe")
            in_window = clean["acquisition_date"] <= clean["label_window_end"]
            if (clean.loc[known, TARGET_COLUMN] != in_window.loc[known].astype(int)).any():
                raise ValueError("Acquisition date conflicts with the 12-month target label")

    invalid_ownership = (
        (clean["institutional_ownership"].dropna() < 0).any()
        or (clean["institutional_ownership"].dropna() > 1).any()
        or (clean["insider_ownership"].dropna() < 0).any()
        or (clean["insider_ownership"].dropna() > 1).any()
    )
    if invalid_ownership:
        raise ValueError("Ownership fields must be decimal fractions between 0 and 1")
    for column in ["institutional_ownership_1y_ago", "outside_block_ownership", "sector_hhi"]:
        if not clean[column].dropna().between(0, 1).all():
            raise ValueError(f"{column} must be a decimal fraction between 0 and 1")
    for column in ["poison_pill", "staggered_board"]:
        if not clean[column].dropna().isin([0, 1]).all():
            raise ValueError(f"{column} must be 0, 1, or missing")

    return clean.sort_values(["observation_date", "company_id"]).reset_index(drop=True)


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -30, 30))))


def generate_synthetic_panel(
    *,
    n_companies: int = 650,
    start_year: int = 2012,
    end_year: int = 2024,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Create a deterministic, economically structured demo panel.

    The synthetic data is explicitly a software demonstration. It embeds plausible
    relationships, not evidence about real-world expected returns.
    """
    if n_companies < 100:
        raise ValueError("n_companies must be at least 100 for stable cross-sectional ranks")
    if end_year <= start_year:
        raise ValueError("end_year must be later than start_year")

    rng = np.random.default_rng(random_seed)
    sectors = np.array(
        [
            "Business Services",
            "Consumer",
            "Energy",
            "Financials",
            "Healthcare",
            "Industrials",
            "Technology",
            "Telecom & Media",
        ]
    )
    sector_weights = np.array([0.13, 0.14, 0.08, 0.14, 0.15, 0.14, 0.15, 0.07])
    sector_for_company = rng.choice(sectors, size=n_companies, p=sector_weights)
    quality = rng.normal(0, 1, n_companies)
    strategic = rng.normal(0, 1, n_companies)
    ownership_style = rng.normal(0, 1, n_companies)
    base_log_market_cap = rng.normal(7.35, 1.05, n_companies)
    base_revenue_scale = rng.normal(7.15, 0.95, n_companies)
    alive = np.ones(n_companies, dtype=bool)

    sector_growth = {sector: rng.normal(0.0, 0.018) for sector in sectors}
    sector_margin = {sector: rng.uniform(0.10, 0.25) for sector in sectors}
    sector_multiple = {sector: rng.uniform(1.5, 4.2) for sector in sectors}
    sector_hhi_base = {sector: rng.uniform(0.08, 0.24) for sector in sectors}

    company_names = [
        f"{['Apex', 'Beacon', 'Cobalt', 'Delta', 'Evergreen', 'Frontier'][i % 6]} "
        f"{sector_for_company[i].split()[0]} {i + 1:04d}"
        for i in range(n_companies)
    ]
    records: list[dict[str, object]] = []

    for year in range(start_year, end_year + 1):
        cycle = 0.025 * np.sin((year - start_year) / 1.7)
        sector_year_return = {
            sector: rng.normal(0.08 + cycle + sector_growth[sector], 0.09) for sector in sectors
        }
        sector_year_hhi = {
            sector: float(
                np.clip(
                    sector_hhi_base[sector] + 0.003 * (year - start_year) + rng.normal(0, 0.008),
                    0.06,
                    0.36,
                )
            )
            for sector in sectors
        }
        sector_deals = {
            sector: int(rng.poisson(7 + 52 * sector_year_hhi[sector] + 0.3 * (year - start_year)))
            for sector in sectors
        }

        for company_idx in np.flatnonzero(alive):
            sector = str(sector_for_company[company_idx])
            age = year - start_year
            growth_1y = float(
                np.clip(
                    0.055
                    + sector_growth[sector]
                    + 0.025 * strategic[company_idx]
                    + cycle
                    + rng.normal(0, 0.11),
                    -0.45,
                    0.65,
                )
            )
            growth_3y = float(
                np.clip(0.7 * growth_1y + 0.3 * (0.05 + rng.normal(0, 0.08)), -0.35, 0.45)
            )
            revenue = float(
                np.exp(base_revenue_scale[company_idx] + age * (0.035 + growth_3y * 0.35))
            )
            ebitda_margin = float(
                np.clip(
                    sector_margin[sector] + 0.035 * quality[company_idx] + rng.normal(0, 0.06),
                    -0.25,
                    0.55,
                )
            )
            net_margin = float(
                np.clip(ebitda_margin - rng.uniform(0.04, 0.13) + rng.normal(0, 0.025), -0.40, 0.40)
            )
            ebitda = revenue * ebitda_margin
            net_income = revenue * net_margin

            relative_return = float(rng.normal(0.05 * quality[company_idx] - 0.02 * age / 10, 0.23))
            return_12m = float(sector_year_return[sector] + relative_return)
            base_log_market_cap[company_idx] += float(
                0.3 * growth_1y + 0.18 * return_12m + rng.normal(0, 0.06)
            )
            market_cap = float(np.exp(base_log_market_cap[company_idx]))
            cash_ratio = float(
                np.clip(0.10 + 0.025 * strategic[company_idx] + rng.normal(0, 0.055), 0.005, 0.42)
            )
            leverage_ratio = float(
                np.clip(0.28 - 0.035 * quality[company_idx] + rng.normal(0, 0.14), 0.0, 0.95)
            )
            cash = market_cap * cash_ratio
            total_debt = market_cap * leverage_ratio

            valuation_noise = float(rng.normal(0, 0.28))
            ev_to_sales = sector_multiple[sector] * np.exp(
                0.75 * (growth_1y - 0.05) + 0.8 * (ebitda_margin - 0.15) + valuation_noise
            )
            enterprise_value = float(
                max(revenue * ev_to_sales, market_cap + total_debt - cash, 25.0)
            )
            market_cap = float(max(enterprise_value - total_debt + cash, 10.0))

            institutional_ownership = float(
                np.clip(
                    0.52
                    + 0.075 * ownership_style[company_idx]
                    + 0.035 * (base_log_market_cap[company_idx] - 7)
                    + rng.normal(0, 0.09),
                    0.03,
                    0.96,
                )
            )
            insider_ownership = float(
                np.clip(
                    0.16 - 0.035 * ownership_style[company_idx] + rng.normal(0, 0.06), 0.0, 0.65
                )
            )
            analyst_coverage = int(
                np.clip(
                    np.rint(3 + 2.0 * (base_log_market_cap[company_idx] - 6.5) + rng.normal(0, 2)),
                    0,
                    35,
                )
            )
            patent_intensity = float(
                np.clip(
                    0.35
                    + 0.22 * strategic[company_idx]
                    + (0.24 if sector in {"Technology", "Healthcare"} else 0)
                    + rng.normal(0, 0.12),
                    0,
                    1.5,
                )
            )
            intangible_asset_ratio = float(
                np.clip(0.20 + 0.075 * strategic[company_idx] + rng.normal(0, 0.08), 0, 0.75)
            )
            hhi = sector_year_hhi[sector]
            deal_count = sector_deals[sector]

            # The event process creates a rare target with a transparent economic signal.
            # Noise keeps the demonstration realistic enough to reward ranking
            # rather than memorization.
            signal = (
                0.60 * (-valuation_noise)
                + 0.30 * (-(base_log_market_cap[company_idx] - 7.35) / 1.05)
                + 0.38 * ((growth_1y - 0.055) / 0.11)
                + 0.25 * ((institutional_ownership - 0.52) / 0.12)
                - 0.22 * ((insider_ownership - 0.16) / 0.08)
                + 0.28 * ((cash_ratio - 0.10) / 0.06)
                - 0.28 * ((leverage_ratio - 0.28) / 0.16)
                + 0.48 * ((hhi - 0.16) / 0.07)
                + 0.34 * ((deal_count - 16) / 7)
                - 0.58 * (relative_return / 0.23)
                + 0.24 * ((ebitda_margin - sector_margin[sector]) / 0.07)
                + 0.50 * strategic[company_idx]
                + rng.normal(0, 0.72)
            )
            event_probability = _sigmoid(-4.15 + signal)
            acquired = int(rng.random() < event_probability)
            acquisition_date = (
                pd.Timestamp(year=year + 1, month=1, day=1)
                + timedelta(days=int(rng.integers(20, 340)))
                if acquired
                else pd.NaT
            )

            records.append(
                {
                    "company_id": f"C{company_idx + 1:05d}",
                    "company_name": company_names[company_idx],
                    "observation_date": pd.Timestamp(year=year, month=12, day=31),
                    "sector": sector,
                    "market_cap": market_cap,
                    "enterprise_value": enterprise_value,
                    "revenue": revenue,
                    "ebitda": ebitda,
                    "net_income": net_income,
                    "cash": cash,
                    "total_debt": total_debt,
                    "sales_growth_1y": growth_1y,
                    "sales_growth_3y": growth_3y,
                    "return_12m": return_12m,
                    "sector_return_12m": sector_year_return[sector],
                    "institutional_ownership": institutional_ownership,
                    "insider_ownership": insider_ownership,
                    "analyst_coverage": analyst_coverage,
                    "patent_intensity": patent_intensity,
                    "intangible_asset_ratio": intangible_asset_ratio,
                    "sector_hhi": hhi,
                    "sector_deal_count_3y": deal_count,
                    "acquired_within_12m": acquired,
                    "acquisition_date": acquisition_date,
                }
            )
            if acquired:
                alive[company_idx] = False

    # Separate RNG: extending demo features cannot change existing financials or labels.
    panel = pd.DataFrame.from_records(records)
    research_rng = np.random.default_rng(random_seed + 10_000)
    n = len(panel)
    panel["total_assets"] = panel["revenue"] * research_rng.uniform(0.8, 2.5, n)
    panel["ppe"] = panel["total_assets"] * research_rng.uniform(0.05, 0.65, n)
    panel["free_cash_flow"] = panel["net_income"] * research_rng.uniform(0.4, 1.2, n)
    panel["institutional_ownership_1y_ago"] = panel.groupby("company_id")[
        "institutional_ownership"
    ].shift()
    panel["outside_block_ownership"] = research_rng.uniform(0, 0.3, n)
    panel["poison_pill"] = research_rng.binomial(1, 0.15, n)
    panel["staggered_board"] = research_rng.binomial(1, 0.3, n)
    sector_dates = panel[["observation_date", "sector", "sector_deal_count_3y"]].drop_duplicates()
    sector_dates["sector_deal_count_1y"] = research_rng.binomial(
        sector_dates["sector_deal_count_3y"].astype(int), 1 / 3
    )
    sector_dates["sector_sales_growth"] = research_rng.normal(0.03, 0.12, len(sector_dates))
    panel = panel.merge(sector_dates, on=["observation_date", "sector", "sector_deal_count_3y"])
    years = sorted(panel["observation_date"].unique())
    spreads = dict(zip(years, research_rng.uniform(100, 600, len(years)), strict=True))
    panel["credit_spread_bps"] = panel["observation_date"].map(spreads)
    panel["rd_expense"] = panel["revenue"] * panel["patent_intensity"] * 0.06
    panel["patent_stock"] = (panel["patent_intensity"] * 100).round()
    panel["patent_citations_to_date"] = panel["patent_stock"] * research_rng.integers(1, 9, n)
    for column in AVAILABILITY_COLUMNS:
        panel[column] = panel["observation_date"] - pd.Timedelta(days=60)
    panel["outcomes_observed_through"] = pd.Timestamp(end_year + 1, 12, 31)
    return validate_panel(panel)
