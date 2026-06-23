"""
Step 2 – Typology Detection: feature engineering, five AML rule flags, weighted risk scoring.
Input:  data/raw/transactions/
Output: data/processed/txn_scored/  and  data/processed/customer_risk/
"""
import os

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

RAW_PATH       = os.environ.get("AML_RAW_PATH",       "/opt/airflow/data/raw/transactions")
SCORED_PATH    = os.environ.get("AML_SCORED_PATH",    "/opt/airflow/data/processed/txn_scored")
CUST_RISK_PATH = os.environ.get("AML_CUST_RISK_PATH", "/opt/airflow/data/processed/customer_risk")

HIGH_RISK_COUNTRIES = {"CY", "MT", "PAN", "BVI", "SCH"}

WEIGHTS = {
    "flag_structuring":       30,
    "flag_high_velocity":     25,
    "flag_high_risk_country": 20,
    "flag_amount_spike":      15,
    "flag_rapid_succession":  10,
}


def main():
    spark = (
        SparkSession.builder
        .appName("AML_Typology_Detection")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    txn_df = spark.read.parquet(RAW_PATH)
    print(f"Loaded {txn_df.count():,} transactions from {RAW_PATH}")

    # ── Feature engineering ────────────────────────────────────────────────
    txn_enriched = (
        txn_df
        .withColumn("date",        F.to_date("timestamp"))
        .withColumn("hour",        F.hour("timestamp"))
        .withColumn("day_of_week", F.dayofweek("timestamp"))
        .withColumn("is_weekend",  F.col("day_of_week").isin(1, 7))
        .withColumn("is_offhours", (F.col("hour") < 6) | (F.col("hour") >= 22))
        .withColumn(
            "involves_high_risk_country",
            F.col("originator_country").isin(list(HIGH_RISK_COUNTRIES))
            | F.col("beneficiary_country").isin(list(HIGH_RISK_COUNTRIES)),
        )
    )

    W_7D = (
        Window.partitionBy("customer_id")
        .orderBy(F.unix_timestamp("timestamp"))
        .rangeBetween(-7 * 86400, 0)
    )
    W_CUMUL = (
        Window.partitionBy("customer_id")
        .orderBy(F.unix_timestamp("timestamp"))
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    W_LAG = Window.partitionBy("customer_id").orderBy("timestamp")

    txn_features = (
        txn_enriched
        .withColumn("rolling_7d_amount",      F.sum("amount").over(W_7D))
        .withColumn("rolling_7d_count",       F.count("transaction_id").over(W_7D))
        .withColumn("rolling_7d_avg",         F.avg("amount").over(W_7D))
        .withColumn("cumulative_amount",      F.sum("amount").over(W_CUMUL))
        .withColumn("cumulative_count",       F.count("transaction_id").over(W_CUMUL))
        .withColumn("prev_ts",                F.lag("timestamp", 1).over(W_LAG))
        .withColumn(
            "seconds_since_last_txn",
            F.unix_timestamp("timestamp") - F.unix_timestamp("prev_ts"),
        )
        .withColumn(
            "amount_vs_7d_avg_ratio",
            F.when(F.col("rolling_7d_avg") > 0, F.col("amount") / F.col("rolling_7d_avg"))
            .otherwise(None),
        )
        .drop("prev_ts")
    )

    # ── Typology flags ─────────────────────────────────────────────────────
    txn_flagged = (
        txn_features
        .withColumn(
            "flag_structuring",
            F.col("amount").between(9_000, 9_999.99) & (F.col("rolling_7d_count") >= 3),
        )
        .withColumn(
            "flag_high_velocity",
            (F.col("rolling_7d_count") >= 5) & (F.col("rolling_7d_amount") >= 20_000),
        )
        .withColumn("flag_high_risk_country", F.col("involves_high_risk_country"))
        .withColumn(
            "flag_amount_spike",
            (F.col("amount_vs_7d_avg_ratio") >= 3.0)
            & F.col("amount_vs_7d_avg_ratio").isNotNull(),
        )
        .withColumn(
            "flag_rapid_succession",
            (F.col("seconds_since_last_txn") <= 300)
            & F.col("seconds_since_last_txn").isNotNull(),
        )
        .withColumn(
            "flag_count",
            F.col("flag_structuring").cast("int")
            + F.col("flag_high_velocity").cast("int")
            + F.col("flag_high_risk_country").cast("int")
            + F.col("flag_amount_spike").cast("int")
            + F.col("flag_rapid_succession").cast("int"),
        )
    )

    # ── Risk scoring ───────────────────────────────────────────────────────
    risk_score_expr = sum(F.col(f).cast("int") * w for f, w in WEIGHTS.items())

    txn_scored = (
        txn_flagged
        .withColumn("risk_score", risk_score_expr)
        .withColumn(
            "risk_tier",
            F.when(F.col("risk_score") >= 50, "HIGH")
            .when(F.col("risk_score") >= 25, "MEDIUM")
            .when(F.col("risk_score") >   0, "LOW")
            .otherwise("NONE"),
        )
    )

    txn_scored.groupBy("risk_tier").count().orderBy(F.desc("count")).show()

    # ── Customer-level aggregation ─────────────────────────────────────────
    customer_risk = (
        txn_scored
        .groupBy("customer_id")
        .agg(
            F.count("transaction_id")                          .alias("total_transactions"),
            F.sum("amount")                                    .alias("total_volume_eur"),
            F.avg("amount")                                    .alias("avg_transaction_amount"),
            F.max("amount")                                    .alias("max_transaction_amount"),
            F.sum(F.col("risk_score"))                         .alias("cumulative_risk_score"),
            F.max("risk_score")                                .alias("peak_risk_score"),
            F.sum(F.col("flag_structuring").cast("int"))       .alias("structuring_flags"),
            F.sum(F.col("flag_high_velocity").cast("int"))     .alias("velocity_flags"),
            F.sum(F.col("flag_high_risk_country").cast("int")) .alias("high_risk_country_flags"),
            F.countDistinct("beneficiary_country")             .alias("distinct_beneficiary_countries"),
            F.countDistinct("beneficiary_account")             .alias("distinct_beneficiary_accounts"),
        )
        .withColumn(
            "customer_risk_tier",
            F.when(F.col("peak_risk_score") >= 50, "HIGH")
            .when(F.col("peak_risk_score") >= 25, "MEDIUM")
            .when(F.col("peak_risk_score") >   0, "LOW")
            .otherwise("NONE"),
        )
        .orderBy(F.desc("cumulative_risk_score"))
    )

    for path in (SCORED_PATH, CUST_RISK_PATH):
        os.makedirs(path, exist_ok=True)

    txn_scored.coalesce(4).write.mode("overwrite").parquet(SCORED_PATH)
    print(f"Scored transactions written to {SCORED_PATH}")

    customer_risk.coalesce(1).write.mode("overwrite").parquet(CUST_RISK_PATH)
    print(f"Customer risk profiles written to {CUST_RISK_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
