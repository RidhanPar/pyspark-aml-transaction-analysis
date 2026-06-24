"""
streaming/structured_streaming_consumer.py

PySpark Structured Streaming consumer for the AML pipeline.

Reads from Kafka topic "aml_transactions", applies the same feature
engineering and five typology rules as scripts/run_typology_detection.py
via foreachBatch, and writes scored micro-batch output to Parquet every
10 seconds.

Usage:
    spark-submit \\
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \\
      streaming/structured_streaming_consumer.py

Environment variables:
    KAFKA_BOOTSTRAP_SERVERS   default: localhost:9092
    STREAMING_ALERT_OUTPUT    default: output/streaming_alerts
    STREAMING_CHECKPOINT      default: output/streaming_checkpoints
    TRIGGER_INTERVAL          default: 10 seconds
    AML_ALERT_THRESHOLD       default: 25

Note: window functions inside foreachBatch are computed on each micro-batch
snapshot — the 7-day rolling window reflects data within the batch, not the
full historical dataset. This is the standard streaming trade-off vs batch.
"""

import os

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType, DoubleType, StringType,
    StructField, StructType,
)

# ── Configuration ──────────────────────────────────────────────────────────────
TOPIC             = "aml_transactions"
BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
ALERT_OUTPUT      = os.environ.get("STREAMING_ALERT_OUTPUT",  "output/streaming_alerts")
CHECKPOINT_DIR    = os.environ.get("STREAMING_CHECKPOINT",    "output/streaming_checkpoints")
TRIGGER_INTERVAL  = os.environ.get("TRIGGER_INTERVAL",        "10 seconds")
ALERT_THRESHOLD   = int(os.environ.get("AML_ALERT_THRESHOLD", "25"))

# ── JSON schema (matches streaming/producer.py + ingest_transactions.py) ───────
# timestamp is a String here; cast to TimestampType inside foreachBatch
# so that rangeBetween window functions work correctly on unix_timestamp().
JSON_SCHEMA = StructType([
    StructField("transaction_id",      StringType(),  nullable=False),
    StructField("customer_id",         StringType(),  nullable=False),
    StructField("originator_account",  StringType(),  nullable=True),
    StructField("beneficiary_account", StringType(),  nullable=True),
    StructField("amount",              DoubleType(),  nullable=False),
    StructField("currency",            StringType(),  nullable=False),
    StructField("channel",             StringType(),  nullable=False),
    StructField("originator_country",  StringType(),  nullable=True),
    StructField("beneficiary_country", StringType(),  nullable=True),
    StructField("timestamp",           StringType(),  nullable=False),
    StructField("is_flagged_source",   BooleanType(), nullable=False),
])

HIGH_RISK_COUNTRIES = {"CY", "MT", "PAN", "BVI", "SCH"}

# Weights match scripts/run_typology_detection.py exactly
WEIGHTS = {
    "flag_structuring":       30,
    "flag_high_velocity":     25,
    "flag_high_risk_country": 20,
    "flag_amount_spike":      15,
    "flag_rapid_succession":  10,
}


# ── foreachBatch handler ───────────────────────────────────────────────────────

def apply_typology_and_score(batch_df, batch_id: int) -> None:
    """
    Applies the full batch-pipeline typology logic to one Structured Streaming
    micro-batch and writes scored transactions to Parquet.

    Steps mirror scripts/run_typology_detection.py:
      1. Feature engineering  (date/time parts, high-risk-country flag)
      2. Rolling window features  (7-day amount / count / avg, cumulative, lag)
      3. Five AML typology flags
      4. Weighted risk score + tier assignment
      5. Parquet write (append) + console summary
    """
    if batch_df.rdd.isEmpty():
        print(f"Batch {batch_id}: 0 records received")
        return

    # Cast ISO-8601 string from producer to TimestampType
    txn_df = batch_df.withColumn("timestamp", F.to_timestamp(F.col("timestamp")))

    # ── Feature engineering ────────────────────────────────────────────────────
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

    # Window specs match scripts/run_typology_detection.py exactly
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
        .withColumn("rolling_7d_amount",  F.sum("amount").over(W_7D))
        .withColumn("rolling_7d_count",   F.count("transaction_id").over(W_7D))
        .withColumn("rolling_7d_avg",     F.avg("amount").over(W_7D))
        .withColumn("cumulative_amount",  F.sum("amount").over(W_CUMUL))
        .withColumn("cumulative_count",   F.count("transaction_id").over(W_CUMUL))
        .withColumn("prev_ts",            F.lag("timestamp", 1).over(W_LAG))
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

    # ── Typology flags ─────────────────────────────────────────────────────────
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

    # ── Risk scoring ───────────────────────────────────────────────────────────
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

    # ── Persist and summarise ──────────────────────────────────────────────────
    total  = txn_scored.count()
    alerts = txn_scored.filter(F.col("risk_score") >= ALERT_THRESHOLD).count()
    print(f"Batch {batch_id}: processed {total} records, {alerts} alerts (score >= {ALERT_THRESHOLD})")

    os.makedirs(ALERT_OUTPUT, exist_ok=True)
    txn_scored.write.mode("append").parquet(ALERT_OUTPUT)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    spark = (
        SparkSession.builder
        .appName("AML_Structured_Streaming")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(ALERT_OUTPUT,   exist_ok=True)

    # ── Read raw bytes from Kafka ──────────────────────────────────────────────
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # Decode value bytes → JSON → flat columns
    parsed_stream = (
        raw_stream
        .select(
            F.from_json(F.col("value").cast("string"), JSON_SCHEMA).alias("data")
        )
        .select("data.*")
        .filter(F.col("transaction_id").isNotNull())
    )

    # ── Write stream with foreachBatch ────────────────────────────────────────
    query = (
        parsed_stream.writeStream
        .foreachBatch(apply_typology_and_score)
        .option("checkpointLocation", CHECKPOINT_DIR)
        .trigger(processingTime=TRIGGER_INTERVAL)
        .start()
    )

    print(f"Streaming consumer started.")
    print(f"  Topic:      {TOPIC} @ {BOOTSTRAP_SERVERS}")
    print(f"  Output:     {ALERT_OUTPUT}")
    print(f"  Checkpoint: {CHECKPOINT_DIR}")
    print(f"  Trigger:    every {TRIGGER_INTERVAL}  |  Ctrl+C to stop")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\nStopping streaming query...")
        query.stop()
        spark.stop()
        print("Consumer stopped.")


if __name__ == "__main__":
    main()
