-- Intermediate model: ephemeral (no physical table — inlined as CTE by dbt).
-- Applies the five AML typology rules using BigQuery window functions,
-- mirroring scripts/run_typology_detection.py and bigquery/risk_score_final.sql
-- exactly: same country list, same thresholds, same window specs.

WITH window_features AS (

    SELECT
        transaction_id,
        customer_id,
        originator_account,
        beneficiary_account,
        amount,
        currency,
        channel,
        originator_country,
        beneficiary_country,
        transaction_timestamp,
        is_flagged_source,

        -- 7-day rolling volume (UNIX_SECONDS RANGE for sub-second precision)
        SUM(amount) OVER (
            PARTITION BY customer_id
            ORDER BY UNIX_SECONDS(transaction_timestamp)
            RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
        ) AS rolling_7d_amount,

        COUNT(*) OVER (
            PARTITION BY customer_id
            ORDER BY UNIX_SECONDS(transaction_timestamp)
            RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
        ) AS rolling_7d_count,

        AVG(amount) OVER (
            PARTITION BY customer_id
            ORDER BY UNIX_SECONDS(transaction_timestamp)
            RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
        ) AS rolling_7d_avg,

        -- Previous transaction timestamp for inter-arrival time calculation
        LAG(transaction_timestamp) OVER (
            PARTITION BY customer_id
            ORDER BY transaction_timestamp
        ) AS prev_timestamp

    FROM {{ ref('stg_transactions') }}

),

derived AS (

    SELECT
        *,

        -- Amount-to-average ratio (NULL when no prior history in window)
        CASE
            WHEN rolling_7d_avg > 0 THEN amount / rolling_7d_avg
            ELSE NULL
        END AS amount_vs_7d_avg_ratio,

        -- Seconds elapsed since the customer's previous transaction
        TIMESTAMP_DIFF(transaction_timestamp, prev_timestamp, SECOND)
            AS seconds_since_last_txn

    FROM window_features

),

flagged AS (

    SELECT
        transaction_id,
        customer_id,
        originator_account,
        beneficiary_account,
        amount,
        currency,
        channel,
        originator_country,
        beneficiary_country,
        transaction_timestamp,
        is_flagged_source,

        ROUND(rolling_7d_amount,      2) AS rolling_7d_amount,
        rolling_7d_count,
        ROUND(rolling_7d_avg,         2) AS rolling_7d_avg,
        ROUND(amount_vs_7d_avg_ratio, 4) AS amount_vs_7d_avg_ratio,
        seconds_since_last_txn,

        -- T1: Structuring — sub-threshold amount with ≥3 similar txns in 7 days
        (amount BETWEEN 9000 AND 9999.99
         AND rolling_7d_count >= 3)
            AS flag_structuring,

        -- T2: High-velocity layering — ≥5 txns and ≥20k volume in 7 days
        (rolling_7d_count >= 5
         AND rolling_7d_amount >= 20000)
            AS flag_high_velocity,

        -- T3: High-risk country routing (matches HIGH_RISK_COUNTRIES in Python)
        (originator_country  IN ('CY', 'MT', 'PAN', 'BVI', 'SCH')
         OR beneficiary_country IN ('CY', 'MT', 'PAN', 'BVI', 'SCH'))
            AS flag_high_risk_country,

        -- T4: Behavioural amount spike — ≥3× the customer's 7-day average
        (amount_vs_7d_avg_ratio IS NOT NULL
         AND amount_vs_7d_avg_ratio >= 3.0)
            AS flag_amount_spike,

        -- T5: Rapid succession — under 5 minutes since previous transaction
        (seconds_since_last_txn IS NOT NULL
         AND seconds_since_last_txn <= 300)
            AS flag_rapid_succession

    FROM derived

)

SELECT * FROM flagged
