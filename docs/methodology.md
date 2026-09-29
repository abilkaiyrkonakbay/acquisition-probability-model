# Methodology

## Target and universe

One row represents a company on an end-of-day observation date. The target is 1
for a qualifying control announcement strictly after that date and on or before
12 calendar months later. It is not a completion label. The label builder excludes
observations on/after the first qualifying announcement and all incomplete windows,
including early visible positives. Define event exclusions upstream; do not drop
announced offers because they were subsequently withdrawn.

Preserve the naturally rare-event denominator, including inactive firms. Do not
balance validation/test samples. `outcomes_observed_through` asserts completeness
of the historical register, not the date on which it was downloaded.

## Features and models

The baseline uses 17 numeric features plus sector: relative valuation, size, growth,
ownership, cash, leverage, industry concentration/activity, relative returns,
profitability and strategic-asset proxies. Research models add 15 financial inputs
and interactions covering industry shocks/credit, governance, tangibility, ownership
changes, cash flow and innovation. See the data dictionary for formulas.

All peer medians use the same sector and observation date. Nonpositive EBITDA
produces a missing multiple. Missing optional inputs are reported, not treated as
known zeros. Imputation, scaling and sector encoding are fitted on training history.
Supplied future availability dates are rejected; absent timestamps are reported as
unverified. Code checks cannot certify vendor vintages or missing-event coverage.

Four fixed candidates compete: baseline logistic, research logistic, research
histogram boosting, and research logistic without relative-return terms. Class
weights apply only during fitting. Boosting uses no random early-stopping split.

## Timeline and leakage controls

- Initial training window: 2012–2017.
- Selection: expanding 2018 and 2019 folds. Each fit includes only earlier snapshots
  whose complete 12-month labels have matured by that date.
- Rank candidates by pooled validation lift, then average precision. Selection
  outcomes must be mature by the calibration observation date.
- Fit the winning base model on history mature at the 2020 calibration date.
  Use only the separate 2020 cross-section for unweighted sigmoid calibration.
- Test on 2021–2024 snapshots with the model frozen. Calibration outcomes must be
  known by the first test observation (2021-12-31 in the demo).

The same maturity purge applies to monthly panels. A calibration/test boundary
without a full 12-month gap is rejected. One usable selection fold is allowed but
flagged; none is an error. Calibration requires both classes. A negative calibration
slope triggers a recorded constant-prior forecast, not a reversed ranking.

## Evaluation

Select `ceil(N * top_fraction)` names per date, default 10%. Ties follow input
company-ID order after validation. Lift is pooled selected-event frequency divided
by pooled universe-event frequency, not an average of yearly lift. Capture is the
share of positive company-date observations selected. Monthly panels can include
multiple pre-event positives for one eventual target; counts are not unique deals.

Average precision, ROC AUC, Brier score and log loss supplement lift. Model-selection
scores are uncalibrated; final test probabilities are calibrated but can drift.
The 95% interval resamples entire company histories and reranks each date. It is
conditional on observed regimes and the fitted model, and omits model-selection
uncertainty, common market shocks and data errors. Undefined metrics become JSON null.

`selected_flag` follows the configured fraction. Deciles always mean ten buckets.
For compatibility, metric keys named `top_decile_*` refer to the configured fraction;
check `top_fraction` in `metrics.json`. `backtest_by_year.csv` actually has one row per
observation date, including when observations are more frequent than annual.

## What this version does not do

No text-similarity engine, subtype classifiers, automatic permutation importance,
return backtest or vendor ingestion. Synthetic outcomes are unchanged from v0.2;
the simpler original-feature benchmark still wins the demo. Never tune candidates
against these test years and then describe the same period as untouched evidence.
Sources and retained research adaptations are in `RESEARCH_IMPLEMENTATION.txt`.
