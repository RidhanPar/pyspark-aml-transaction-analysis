-- Customer risk profile mart: per-customer aggregation of AML risk metrics.
-- Aggregates ALL transactions (not just alerts) so totals reflect the complete
-- customer history. max_risk_score drives the customer-level risk tier.

WITH risk_scored AS (

    SELECT
        *,

        -- Weighted composite score (same formula as Python pipeline)
        (CAST(flag_structuring        AS INT64) * 30
         + CAST(flag_high_velocity    AS INT64) * 25
         + CAST(flag_high_risk_country AS INT64) * 20
         + CAST(flag_amount_spike     AS INT64) * 15
         + CAST(flag_rapid_succession AS INT64) * 10) AS risk_score,

        -- Number of typology flags triggered on this transaction
        (CAST(flag_structuring        AS INT64)
         + CAST(flag_high_velocity    AS INT64)
         + CAST(flag_high_risk_country AS INT64)
         + CAST(flag_amount_spike     AS INT64)
         + CAST(flag_rapid_succession AS INT64)) AS flags_triggered

    FROM {{ ref('int_risk_flagged') }}

)

SELECT
    customer_id,
    COUNT(*)                           AS total_transactions,
    ROUND(SUM(amount), 2)             AS total_amount,
    ROUND(AVG(amount), 2)             AS avg_amount,
    SUM(flags_triggered)              AS flags_triggered,
    MAX(risk_score)                   AS max_risk_score,
    MAX(transaction_timestamp)        AS latest_transaction_timestamp,

    CASE
        WHEN MAX(risk_score) >= 75 THEN 'Critical'
        WHEN MAX(risk_score) >= 50 THEN 'High'
        WHEN MAX(risk_score) >= 25 THEN 'Medium'
        ELSE                           'Low'
    END AS risk_tier

FROM risk_scored
GROUP BY customer_id
ORDER BY
    max_risk_score DESC,
    total_amount   DESC
