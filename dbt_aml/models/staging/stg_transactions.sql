-- Staging model: light column renames and type casts on the raw BigQuery table.
-- Source: aml_transactions.transactions (loaded by bigquery/load_to_bq.py)
-- Materialised as view (default for staging layer).

WITH source AS (

    SELECT * FROM {{ source('aml_raw', 'transactions') }}

),

renamed AS (

    SELECT
        transaction_id,
        customer_id,
        originator_account,
        beneficiary_account,
        CAST(amount AS FLOAT64)       AS amount,
        currency,
        channel,
        originator_country,
        beneficiary_country,
        CAST(timestamp AS TIMESTAMP)  AS transaction_timestamp,
        is_flagged_source

    FROM source

)

SELECT * FROM renamed
