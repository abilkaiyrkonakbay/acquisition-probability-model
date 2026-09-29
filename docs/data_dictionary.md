# Data dictionary

One row per `company_id,observation_date`. Use identical currencies, units and sector
definitions in training and scoring; the demo uses USD millions. Ownership, returns
and growth are decimal fractions, not percentages. Unknown numeric values are blank,
not zero. Extra text/deal-type columns are not used by this version.

## Required identifiers and baseline inputs

- `company_id`, `company_name`, `observation_date` (ISO date), `sector`.
- `market_cap`, `enterprise_value`, `revenue`, `ebitda`, `net_income`, `cash`, `total_debt`.
- `sales_growth_1y`, `sales_growth_3y`, `return_12m`, `sector_return_12m`.
- `institutional_ownership`, `insider_ownership`: fractions in [0, 1].
- `analyst_coverage`: number of covering analysts.
- `patent_intensity`, `intangible_asset_ratio`: consistently defined strategic-asset proxies.
- `sector_hhi`: concentration index in [0, 1].
- `sector_deal_count_3y`: trailing three-year qualifying announced deal count.

Accounting and ownership inputs must have been publicly available at the observation
date. Corporate actions, inactive securities, currencies and entity mapping must be
handled upstream. The project does not provide licensed market or transaction data.

## Outcomes and availability

- `acquired_within_12m`: required for training, strictly 0 or 1. Not required for scoring.
- `acquisition_date`: optional **announcement** date; known dates must agree with labels.
- `outcomes_observed_through`: complete event-register coverage date, covering every
  row's 12-month horizon. Required for training unless supplied via `--as-of`.
- `label_window_end`: reconstructed as observation date + 12 months; input values
  cannot shorten the horizon. Never a predictor.
- Optional `filing_available_date`, `ownership_available_date`, `industry_available_date`,
  `innovation_available_date`: the latest availability date of inputs in that block.
  Future timestamps are rejected. Missing coverage is reported in `metrics.json`.

The `label` command requires a complete register with `company_id,announcement_date`.
First qualifying announcements determine the label. Already-announced targets and
incomplete windows are excluded. Filter deal taxonomy upstream, without conditioning
on eventual completion. Supplying a coverage date does not prove register completeness.

## Optional research inputs and formulas

- `total_assets`, `ppe`: asset tangibility = PPE/assets, clipped [0, 1].
- `free_cash_flow`: consistently defined CFO less capex; FCF/EV clipped [-2, 2].
- `institutional_ownership_1y_ago`: fraction known at the one-year-ago observation;
  change = current minus prior disclosed fraction, not percentage growth.
- `outside_block_ownership`: non-overlapping unaffiliated blockholder fraction;
  document the block threshold, e.g. 5%.
- `poison_pill`, `staggered_board`: 1 for verified presence, 0 for verified absence,
  missing for unknown.
- `sector_deal_count_1y`: same taxonomy as the 3y count. Acceleration =
  count_1y / (count_3y / 3) - 1, clipped [-1, 2]. Zero denominator gives missing.
- `sector_sales_growth`: trailing sector growth; absolute value is the shock proxy.
- `credit_spread_bps`: contemporary credit spread in basis points. Capacity proxy =
  1 / (1 + max(spread, 0)/100), interacted with absolute sector growth.
- `rd_expense`: R&D/revenue clipped [0, 2].
- `patent_stock`: publicly known patent count; feature = log(1 + count).
- `patent_citations_to_date`: citations known by observation date, not lifetime totals;
  citations/patent clipped [0, 100], with missing for zero patent stock.

Other features use sector-date peer valuation discounts, log size, cash/EV, net
debt/positive EBITDA, relative returns and margins. Extra interactions are positive
valuation discount times negative relative return, and cash/EV times negative growth.
Training-only median imputation handles missing fields. `feature_coverage.csv` shows
coverage separately for the initial training window and testing.

## SQL contracts

`sql/feature_mart.sql` uses DuckDB and assumes these upstream tables:

- `prices_monthly`: company, `month_end`, `is_month_end`, market cap and trailing returns.
- `company_master_snapshot`: company/date name, sector, historical `is_public`, `available_date`.
- `fundamentals_quarterly`: accounting inputs, `fiscal_period_end`, `filing_available_date`.
- `ownership_snapshot`: company/date stakes, coverage, changes, defenses, `available_date`.
- `strategic_assets_snapshot`: company/date patent/intangible inputs, `available_date`.
- `industry_snapshot`: sector/date concentration, deal counts, growth, spread, `available_date`.
- `mna_events`: `target_company_id`, `announcement_date`, `is_qualifying_control_announcement`.
- `data_coverage`: exactly one row with `outcomes_observed_through`.

Resolve duplicate vintages upstream; snapshots are unique on entity/date. Missing
optional columns must exist as typed NULLs in SQL. Fundamentals use the most recent
available filing within 18 months. The example EV definition is observation-date
market cap + known debt - cash; extend for other claims consistently if necessary.
The mart excludes incomplete horizons and known targets. Tests compare its labels
with Python. `sql/top_decile_backtest.sql` queries the predictions CSV directly.
