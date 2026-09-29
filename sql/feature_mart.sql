-- DuckDB reference mart; upstream contracts: docs/data_dictionary.md.
-- data_coverage has EXACTLY ONE row: outcomes_observed_through (DATE).
-- Snapshot tables have at most one row per entity/date and reflect historical,
-- not current, membership. Resolve revisions and snapshot duplication upstream.
-- mna_events.is_qualifying_control_announcement must NOT depend on eventual completion.

CREATE OR REPLACE TABLE acquisition_feature_mart AS
WITH qualifying_events AS (
    SELECT target_company_id, MIN(announcement_date) AS acquisition_date
    FROM mna_events, data_coverage
    WHERE is_qualifying_control_announcement = TRUE
      AND announcement_date <= outcomes_observed_through
    GROUP BY target_company_id
),
available_fundamentals AS (
    SELECT
        p.company_id,
        p.month_end AS observation_date,
        f.* EXCLUDE (company_id, fiscal_period_end),
        ROW_NUMBER() OVER (
            PARTITION BY p.company_id, p.month_end
            ORDER BY f.filing_available_date DESC, f.fiscal_period_end DESC
        ) AS recency_rank
    FROM prices_monthly p
    JOIN fundamentals_quarterly f
      ON p.company_id = f.company_id
     AND f.filing_available_date <= p.month_end
     AND f.filing_available_date > p.month_end - INTERVAL 18 MONTH
    WHERE p.is_month_end = TRUE
)
SELECT
    p.company_id,
    c.company_name,
    p.month_end AS observation_date,
    c.sector,
    p.market_cap,
    p.market_cap + f.total_debt - f.cash AS enterprise_value,
    f.revenue,
    f.ebitda,
    f.net_income,
    f.cash,
    f.total_debt,
    f.sales_growth_1y,
    f.sales_growth_3y,
    f.total_assets,
    f.ppe,
    f.free_cash_flow,
    f.rd_expense,
    f.filing_available_date,
    p.total_return_12m AS return_12m,
    p.sector_return_12m,
    o.institutional_ownership,
    o.insider_ownership,
    o.analyst_coverage,
    o.institutional_ownership_1y_ago,
    o.outside_block_ownership,
    o.poison_pill,
    o.staggered_board,
    o.available_date AS ownership_available_date,
    a.patent_intensity,
    a.intangible_asset_ratio,
    a.patent_stock,
    a.patent_citations_to_date,
    a.available_date AS innovation_available_date,
    i.sector_hhi,
    i.sector_deal_count_3y,
    i.sector_deal_count_1y,
    i.sector_sales_growth,
    i.credit_spread_bps,
    i.available_date AS industry_available_date,
    CASE WHEN m.acquisition_date > p.month_end
              AND m.acquisition_date <= p.month_end + INTERVAL 12 MONTH
         THEN 1 ELSE 0 END AS acquired_within_12m,
    m.acquisition_date,
    p.month_end + INTERVAL 12 MONTH AS label_window_end,
    d.outcomes_observed_through
FROM prices_monthly p
JOIN company_master_snapshot c
  ON p.company_id = c.company_id AND p.month_end = c.observation_date
 AND c.available_date <= p.month_end
JOIN available_fundamentals f
  ON p.company_id = f.company_id AND p.month_end = f.observation_date
 AND f.recency_rank = 1
LEFT JOIN ownership_snapshot o
  ON p.company_id = o.company_id AND p.month_end = o.observation_date
 AND o.available_date <= p.month_end
LEFT JOIN strategic_assets_snapshot a
  ON p.company_id = a.company_id AND p.month_end = a.observation_date
 AND a.available_date <= p.month_end
LEFT JOIN industry_snapshot i
  ON c.sector = i.sector AND p.month_end = i.observation_date
 AND i.available_date <= p.month_end
LEFT JOIN qualifying_events m ON p.company_id = m.target_company_id
CROSS JOIN data_coverage d
WHERE p.is_month_end = TRUE
  AND c.is_public = TRUE
  AND p.market_cap >= 50
  AND p.month_end + INTERVAL 12 MONTH <= d.outcomes_observed_through
  AND (m.acquisition_date IS NULL OR m.acquisition_date > p.month_end);
