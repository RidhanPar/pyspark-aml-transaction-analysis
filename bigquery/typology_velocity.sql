/*
  Typology: High-Velocity Layering
  ─────────────────────────────────────────────────────────────────────────────
  Detects accounts with unusually high transaction frequency and volume over a
  7-day rolling window, a common indicator of layering in money-laundering
  schemes.

  Flag condition  : ≥ 5 transactions AND ≥ 20,000 total volume within a
                    rolling 7-day window ending at the current transaction.

  Mirrors PySpark rule in scripts/run_typology_detection.py:
      flag_high_velocity = (rolling_7d_count >= 5) & (rolling_7d_amount >= 20_000)

  Replace YOUR_PROJECT and YOUR_DATASET before running.
*/

WITH rolling_stats AS (

  SELECT
    transaction_id,
    customer_id,
    amount,
    currency,
    channel,
    originator_account,
    beneficiary_account,
    originator_country,
    beneficiary_country,
    timestamp,

    -- Transaction count in rolling 7-day window (604,800 s).
    COUNT(*) OVER (
      PARTITION BY customer_id
      ORDER BY UNIX_SECONDS(timestamp)
      RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_count,

    -- Volume in the same window.
    SUM(amount) OVER (
      PARTITION BY customer_id
      ORDER BY UNIX_SECONDS(timestamp)
      RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_amount,

    -- Average transaction amount in the same window (for context).
    AVG(amount) OVER (
      PARTITION BY customer_id
      ORDER BY UNIX_SECONDS(timestamp)
      RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_avg

  FROM `YOUR_PROJECT.YOUR_DATASET.transactions`

)

SELECT
  transaction_id,
  customer_id,
  amount,
  currency,
  channel,
  originator_country,
  beneficiary_country,
  timestamp,
  rolling_7d_count,
  ROUND(rolling_7d_amount, 2) AS rolling_7d_amount,
  ROUND(rolling_7d_avg,    2) AS rolling_7d_avg,
  'high_velocity_layering'    AS typology
FROM rolling_stats
WHERE
  rolling_7d_count  >= 5
  AND rolling_7d_amount >= 20000
ORDER BY
  rolling_7d_amount DESC,
  customer_id,
  timestamp;
