{{
  config(
    materialized = 'table',
    partition_by = {
      'field': 'transaction_timestamp',
      'data_type': 'timestamp',
      'granularity': 'day'
    }
  )
}}

-- Alert queue mart: scored transactions with risk_score >= 25.
-- Partitioned by DATE(transaction_timestamp) for cost-efficient time-range queries.
-- Weights and tiers match scripts/run_typology_detection.py exactly.

WITH risk_scored AS (

    SELECT
        *,

        -- Weighted composite score (same formula as Python pipeline)
        (CAST(flag_structuring        AS INT64) * 30
         + CAST(flag_high_velocity    AS INT64) * 25
         + CAST(flag_high_risk_country AS INT64) * 20
         + CAST(flag_amount_spike     AS INT64) * 15
         + CAST(flag_rapid_succession AS INT64) * 10) AS risk_score,

        (CAST(flag_structuring        AS INT64)
         + CAST(flag_high_velocity    AS INT64)
         + CAST(flag_high_risk_country AS INT64)
         + CAST(flag_amount_spike     AS INT64)
         + CAST(flag_rapid_succession AS INT64)) AS flag_count

    FROM {{ ref('int_risk_flagged') }}

)

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

    rolling_7d_amount,
    rolling_7d_count,
    rolling_7d_avg,
    amount_vs_7d_avg_ratio,
    seconds_since_last_txn,

    flag_structuring,
    flag_high_velocity,
    flag_high_risk_country,
    flag_amount_spike,
    flag_rapid_succession,
    flag_count,

    risk_score,

    CASE
        WHEN risk_score >= 50 THEN 'HIGH'
        WHEN risk_score >= 25 THEN 'MEDIUM'
        WHEN risk_score >   0 THEN 'LOW'
        ELSE                       'NONE'
    END AS risk_tier

FROM risk_scored
WHERE risk_score >= 25
ORDER BY
    risk_score DESC,
    amount     DESC
