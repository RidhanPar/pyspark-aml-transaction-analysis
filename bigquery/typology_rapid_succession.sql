/*
  Typology: Rapid Succession
  ─────────────────────────────────────────────────────────────────────────────
  Detects transactions that occur within a very short time of the same
  customer's previous transaction, which can indicate automated or scripted
  fund movement.

  Flag condition  : current transaction occurs within 300 seconds (5 minutes)
                    of the immediately preceding transaction by the same
                    customer (NULL for the customer's very first transaction).

  Mirrors PySpark rule in scripts/run_typology_detection.py:
      flag_rapid_succession = (seconds_since_last_txn <= 300) & isNotNull()
      where seconds_since_last_txn = unix(current) - unix(LAG(timestamp))

  Replace YOUR_PROJECT and YOUR_DATASET before running.
*/

WITH with_lag AS (

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

    -- Timestamp of the previous transaction by the same customer.
    LAG(timestamp) OVER (
      PARTITION BY customer_id
      ORDER BY timestamp
    ) AS prev_timestamp,

    -- 7-day rolling volume for context.
    SUM(amount) OVER (
      PARTITION BY customer_id
      ORDER BY UNIX_SECONDS(timestamp)
      RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_amount,

    COUNT(*) OVER (
      PARTITION BY customer_id
      ORDER BY UNIX_SECONDS(timestamp)
      RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_count

  FROM `YOUR_PROJECT.YOUR_DATASET.transactions`

),

with_gap AS (

  SELECT
    *,
    -- Gap in seconds since the previous transaction; NULL for the first one.
    TIMESTAMP_DIFF(timestamp, prev_timestamp, SECOND) AS seconds_since_last_txn
  FROM with_lag

)

SELECT
  transaction_id,
  customer_id,
  amount,
  currency,
  channel,
  originator_country,
  beneficiary_country,
  prev_timestamp,
  timestamp                AS current_timestamp,
  seconds_since_last_txn,
  ROUND(rolling_7d_amount, 2) AS rolling_7d_amount,
  rolling_7d_count,
  'rapid_succession'          AS typology
FROM with_gap
WHERE
  seconds_since_last_txn IS NOT NULL
  AND seconds_since_last_txn <= 300
ORDER BY
  seconds_since_last_txn ASC,
  customer_id,
  timestamp;
