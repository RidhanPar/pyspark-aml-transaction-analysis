/*
  Risk Score – All Typologies Combined
  ─────────────────────────────────────────────────────────────────────────────
  Applies all five AML typology rules to every transaction and produces a
  composite weighted risk score and tier for downstream alert triage.

  Typology weights (match scripts/run_typology_detection.py):
    flag_structuring       → 30 pts
    flag_high_velocity     → 25 pts
    flag_high_risk_country → 20 pts
    flag_amount_spike      → 15 pts
    flag_rapid_succession  → 10 pts

  Risk tiers:
    HIGH   : risk_score >= 50
    MEDIUM : risk_score >= 25
    LOW    : risk_score >  0
    NONE   : risk_score =  0

  Replace YOUR_PROJECT and YOUR_DATASET before running.
*/

-- ── Step 1: rolling window features ─────────────────────────────────────────
WITH window_features AS (

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
    is_flagged_source,

    -- 7-day rolling volume (seconds-based RANGE for precision).
    SUM(amount) OVER (
      PARTITION BY customer_id
      ORDER BY UNIX_SECONDS(timestamp)
      RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_amount,

    COUNT(*) OVER (
      PARTITION BY customer_id
      ORDER BY UNIX_SECONDS(timestamp)
      RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_count,

    AVG(amount) OVER (
      PARTITION BY customer_id
      ORDER BY UNIX_SECONDS(timestamp)
      RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_avg,

    -- Previous transaction timestamp for gap calculation.
    LAG(timestamp) OVER (
      PARTITION BY customer_id
      ORDER BY timestamp
    ) AS prev_timestamp

  FROM `YOUR_PROJECT.YOUR_DATASET.transactions`

),

-- ── Step 2: derived features ─────────────────────────────────────────────────
derived AS (

  SELECT
    *,

    -- Amount-to-average ratio (NULL when no prior average exists).
    CASE
      WHEN rolling_7d_avg > 0 THEN amount / rolling_7d_avg
      ELSE NULL
    END AS amount_vs_7d_avg_ratio,

    -- Seconds elapsed since the customer's previous transaction.
    TIMESTAMP_DIFF(timestamp, prev_timestamp, SECOND) AS seconds_since_last_txn

  FROM window_features

),

-- ── Step 3: typology flags ───────────────────────────────────────────────────
flags AS (

  SELECT
    *,

    -- T1 – Structuring: sub-threshold amount with 3+ similar txns in 7 days.
    (amount BETWEEN 9000 AND 9999.99
     AND rolling_7d_count >= 3)
      AS flag_structuring,

    -- T2 – High-velocity layering: ≥5 txns and ≥20k volume in 7 days.
    (rolling_7d_count >= 5
     AND rolling_7d_amount >= 20000)
      AS flag_high_velocity,

    -- T3 – High-risk country routing.
    (originator_country  IN ('CY', 'MT', 'PAN', 'BVI', 'SCH')
     OR beneficiary_country IN ('CY', 'MT', 'PAN', 'BVI', 'SCH'))
      AS flag_high_risk_country,

    -- T4 – Behavioural amount spike: ≥3× the 7-day average.
    (amount_vs_7d_avg_ratio IS NOT NULL
     AND amount_vs_7d_avg_ratio >= 3.0)
      AS flag_amount_spike,

    -- T5 – Rapid succession: under 5 minutes since previous transaction.
    (seconds_since_last_txn IS NOT NULL
     AND seconds_since_last_txn <= 300)
      AS flag_rapid_succession

  FROM derived

),

-- ── Step 4: weighted risk score ──────────────────────────────────────────────
scored AS (

  SELECT
    *,

    -- Composite score: sum of weights for each active flag.
    (CAST(flag_structuring       AS INT64) * 30
     + CAST(flag_high_velocity     AS INT64) * 25
     + CAST(flag_high_risk_country AS INT64) * 20
     + CAST(flag_amount_spike      AS INT64) * 15
     + CAST(flag_rapid_succession  AS INT64) * 10)
      AS risk_score,

    -- Total flag count for triage prioritisation.
    (CAST(flag_structuring       AS INT64)
     + CAST(flag_high_velocity     AS INT64)
     + CAST(flag_high_risk_country AS INT64)
     + CAST(flag_amount_spike      AS INT64)
     + CAST(flag_rapid_succession  AS INT64))
      AS flag_count

  FROM flags

)

-- ── Final output ─────────────────────────────────────────────────────────────
SELECT
  transaction_id,
  customer_id,
  amount,
  currency,
  channel,
  originator_country,
  beneficiary_country,
  timestamp,
  is_flagged_source,

  -- Window features.
  ROUND(rolling_7d_amount, 2)    AS rolling_7d_amount,
  rolling_7d_count,
  ROUND(rolling_7d_avg,    2)    AS rolling_7d_avg,
  ROUND(amount_vs_7d_avg_ratio, 4) AS amount_vs_7d_avg_ratio,
  seconds_since_last_txn,

  -- Flags.
  flag_structuring,
  flag_high_velocity,
  flag_high_risk_country,
  flag_amount_spike,
  flag_rapid_succession,
  flag_count,

  -- Composite score.
  risk_score,

  -- Risk tier.
  CASE
    WHEN risk_score >= 50 THEN 'HIGH'
    WHEN risk_score >= 25 THEN 'MEDIUM'
    WHEN risk_score >   0 THEN 'LOW'
    ELSE                       'NONE'
  END AS risk_tier

FROM scored
ORDER BY
  risk_score DESC,
  amount     DESC;
