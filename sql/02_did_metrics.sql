-- Case 2 metrics mirror: paid-search naive ROI and simple DiD readout.
WITH panel AS (
    SELECT *
    FROM read_csv_auto('data/case2_did_panel.csv')
),
naive AS (
    SELECT SUM(paid_revenue) / SUM(spend) AS naive_paid_click_roi
    FROM panel
    WHERE treated_market = 1 AND post = 1
),
eligible AS (
    SELECT *
    FROM panel
    WHERE eligible_control = 1
),
means AS (
    SELECT
        treated_market,
        post,
        AVG(total_clicks) AS avg_total_clicks
    FROM eligible
    GROUP BY 1, 2
),
did AS (
    SELECT
        (MAX(CASE WHEN treated_market = 1 AND post = 1 THEN avg_total_clicks END)
        - MAX(CASE WHEN treated_market = 1 AND post = 0 THEN avg_total_clicks END))
        - (MAX(CASE WHEN treated_market = 0 AND post = 1 THEN avg_total_clicks END)
        - MAX(CASE WHEN treated_market = 0 AND post = 0 THEN avg_total_clicks END))
            AS simple_did_incremental_clicks
    FROM means
)
SELECT *
FROM naive
CROSS JOIN did;
