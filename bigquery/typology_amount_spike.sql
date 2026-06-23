/*
  Typology: Behavioural Amount Spike
  ─────────────────────────────────────────────────────────────────────────────
  Detects transactions where the amount is significantly higher than the
  customer's own recent behaviour, indicating a sudden unexplained increase
  in activity volume.

  Flag condition  : current amount ≥ 3× the customer's 7-day rolling average,
                    provided the rolling average is positive (at least one prior
                    transaction exists in the window).

  Mirrors PySpark rule in scripts/run_typology_detection.py:
      flag_amount_spike = (amount_vs_7d_avg_ratio >= 3.0) & ratio.isNotNull()
      where amount_vs_7d_avg_ratio = amount / rolling_7d_avg  (when avg > 0)

  Replace YOUR_PROJECT and YOUR_DATASET before running.
*/

WITH rolling_avg AS (

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

    -- 7-day rolling average for this customer (including current row).
    AVG(amount) OVER (
      PARTITION BY customer_id
      ORDER BY UNIX_SECONDS(timestamp)
      RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_avg,

    -- Count of transactions in the window (to confirm a baseline exists).
    COUNT(*) OVER (
      PARTITION BY customer_id
      ORDER BY UNIX_SECONDS(timestamp)
      RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_count,

    -- Total volume for context.
    SUM(amount) OVER (
      PARTITION BY customer_id
      ORDER BY UNIX_SECONDS(timestamp)
      RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_total

  FROM `YOUR_PROJECT.YOUR_DATASET.transactions`

),

with_ratio AS (

  SELECT
    *,
    -- Ratio is NULL when rolling average is zero (first-ever transaction).
    CASE
      WHEN rolling_7d_avg > 0 THEN ROUND(amount / rolling_7d_avg, 4)
      ELSE NULL
    END AS amount_vs_7d_avg_ratio
  FROM rolling_avg

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
  ROUND(rolling_7d_avg,    2) AS rolling_7d_avg,
  rolling_7d_count,
  amount_vs_7d_avg_ratio,
  'amount_spike'              AS typology
FROM with_ratio
WHERE
  amount_vs_7d_avg_ratio IS NOT NULL
  AND amount_vs_7d_avg_ratio >= 3.0
ORDER BY
  amount_vs_7d_avg_ratio DESC,
  customer_id,
  timestamp;
