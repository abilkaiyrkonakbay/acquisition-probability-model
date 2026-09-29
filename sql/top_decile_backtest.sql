-- DuckDB: reads the single predictions CSV; no separate SQLite copy is needed.
-- selected_flag follows the configured fraction, 10% by default, within each date.

SELECT
    observation_date,
    COUNT(*) AS universe_size,
    SUM(acquired_within_12m) AS acquisitions,
    AVG(acquired_within_12m) AS base_rate,
    SUM(selected_flag) AS selected_count,
    SUM(CASE WHEN selected_flag = 1 THEN acquired_within_12m ELSE 0 END) AS selected_acquisitions,
    AVG(CASE WHEN selected_flag = 1 THEN acquired_within_12m END) AS selected_rate,
    AVG(CASE WHEN selected_flag = 1 THEN acquired_within_12m END)
        / NULLIF(AVG(acquired_within_12m), 0) AS lift
FROM read_csv_auto('outputs/demo/test_predictions.csv')
GROUP BY observation_date
ORDER BY observation_date;
