-- Case 3 metrics mirror: naive directed-search gap and matched ATT.
WITH sessions AS (
    SELECT *
    FROM read_csv_auto('data/case3_sessions.csv')
),
matches AS (
    SELECT *
    FROM read_csv_auto('data/case3_psm_matches.csv')
),
naive AS (
    SELECT
        AVG(CASE WHEN directed_search_usage = 1 THEN sales END)
        - AVG(CASE WHEN directed_search_usage = 0 THEN sales END)
            AS naive_sales_lift
    FROM sessions
),
psm AS (
    SELECT
        AVG(difference) AS matched_att,
        COUNT(*) AS matched_pairs
    FROM matches
)
SELECT *
FROM naive
CROSS JOIN psm;
