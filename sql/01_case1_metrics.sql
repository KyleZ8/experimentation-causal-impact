-- Case 1 (A/B test readout) metric definitions.
-- Assumes a table/view `case1_users` is registered in the DuckDB connection,
-- e.g. `duckdb.sql("CREATE VIEW case1_users AS SELECT * FROM read_csv_auto('data/case1/case1_users.csv')")`.
-- The notebook runs these as an independent cross-check against the Python
-- (scipy/statsmodels) results in src/causal_toolkit/ab.py.

-- Sample ratio: row counts per arm, for the SRM check.
CREATE OR REPLACE VIEW case1_arm_counts AS
SELECT
    treated,
    COUNT(*) AS n
FROM case1_users
GROUP BY treated;

-- Primary metric: posting rate by arm.
CREATE OR REPLACE VIEW case1_primary_metric AS
SELECT
    treated,
    COUNT(*) AS n_users,
    SUM(posted) AS n_posted,
    SUM(posted) * 1.0 / COUNT(*) AS posting_rate
FROM case1_users
GROUP BY treated;

-- Guardrail 1: average post length by arm, posters only.
CREATE OR REPLACE VIEW case1_guardrail_length AS
SELECT
    treated,
    COUNT(*) AS n_posters,
    AVG(post_length) AS avg_post_length
FROM case1_users
WHERE posted = 1
GROUP BY treated;

-- Guardrail 2: content-flag rate by arm, posters only.
CREATE OR REPLACE VIEW case1_guardrail_flag_rate AS
SELECT
    treated,
    COUNT(*) AS n_posters,
    SUM(flagged) AS n_flagged,
    SUM(flagged) * 1.0 / COUNT(*) AS flag_rate
FROM case1_users
WHERE posted = 1
GROUP BY treated;
