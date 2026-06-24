-- Singular test: every row in mart_aml_alert_queue must have risk_score >= 25.
-- dbt fails the test if this query returns any rows.

SELECT
    transaction_id,
    risk_score
FROM {{ ref('mart_aml_alert_queue') }}
WHERE risk_score < 25
