/*
  Typology: High-Risk Country Routing
  ─────────────────────────────────────────────────────────────────────────────
  Detects transactions where either the originator or beneficiary is located in
  a jurisdiction with known AML/CFT weaknesses.

  High-risk jurisdictions : CY (Cyprus), MT (Malta), PAN (Panama),
                            BVI (British Virgin Islands), SCH (Seychelles)

  Flag condition  : originator_country OR beneficiary_country is in the
                    high-risk set.

  Mirrors PySpark rule in scripts/run_typology_detection.py:
      flag_high_risk_country = involves_high_risk_country
      where involves_high_risk_country checks both country fields.

  Replace YOUR_PROJECT and YOUR_DATASET before running.
*/

WITH high_risk_tagged AS (

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

    -- Identify which leg of the transaction involves a high-risk jurisdiction.
    originator_country IN ('CY', 'MT', 'PAN', 'BVI', 'SCH') AS originator_is_high_risk,
    beneficiary_country IN ('CY', 'MT', 'PAN', 'BVI', 'SCH') AS beneficiary_is_high_risk,

    -- 7-day rolling volume and count for context (all transactions, same customer).
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

)

SELECT
  transaction_id,
  customer_id,
  amount,
  currency,
  channel,
  originator_country,
  originator_is_high_risk,
  beneficiary_country,
  beneficiary_is_high_risk,
  timestamp,
  ROUND(rolling_7d_amount, 2) AS rolling_7d_amount,
  rolling_7d_count,
  'high_risk_country_routing'  AS typology
FROM high_risk_tagged
WHERE
  originator_is_high_risk
  OR beneficiary_is_high_risk
ORDER BY
  amount DESC,
  customer_id,
  timestamp;
