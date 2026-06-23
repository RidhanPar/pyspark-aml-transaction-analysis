/*
  Typology: Structuring / Smurfing
  ─────────────────────────────────────────────────────────────────────────────
  Detects customers who break up larger amounts into sub-threshold transactions
  (9,000–9,999) to avoid reporting requirements.

  Flag condition  : transaction amount between 9,000 and 9,999.99
                    AND the customer has ≥ 3 such transactions within any
                    rolling 7-day window ending at the current transaction.

  Mirrors PySpark rule in scripts/run_typology_detection.py:
      flag_structuring = amount.between(9_000, 9_999.99) & (rolling_7d_count >= 3)

  Replace YOUR_PROJECT and YOUR_DATASET before running.
*/

WITH rolling_sub_threshold AS (

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

    -- Count of sub-threshold transactions for this customer in the
    -- preceding 7 days (604,800 seconds), including the current row.
    COUNT(*) OVER (
      PARTITION BY customer_id
      ORDER BY UNIX_SECONDS(timestamp)
      RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
    ) AS sub_threshold_count_7d,

    -- Total volume of sub-threshold transactions in the same window.
    SUM(amount) OVER (
      PARTITION BY customer_id
      ORDER BY UNIX_SECONDS(timestamp)
      RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
    ) AS sub_threshold_volume_7d

  FROM `YOUR_PROJECT.YOUR_DATASET.transactions`
  WHERE amount BETWEEN 9000 AND 9999.99

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
  sub_threshold_count_7d,
  ROUND(sub_threshold_volume_7d, 2) AS sub_threshold_volume_7d,
  'structuring'                     AS typology
FROM rolling_sub_threshold
WHERE sub_threshold_count_7d >= 3
ORDER BY
  customer_id,
  timestamp;
