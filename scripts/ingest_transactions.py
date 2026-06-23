"""
Step 1 – Ingest: generate synthetic AML transactions, validate, and persist as Parquet.
Output: data/raw/transactions/
"""
import os
import random
from datetime import datetime, timedelta

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType, DoubleType, StringType,
    StructField, StructType, TimestampType,
)

OUTPUT_PATH = os.environ.get("AML_RAW_PATH", "/opt/airflow/data/raw/transactions")

random.seed(42)

N_TRANSACTIONS = 10_000
N_CUSTOMERS    = 500
N_ACCOUNTS     = 600
START_DATE     = datetime(2024, 1, 1)

CURRENCIES          = ["EUR", "USD", "GBP", "SEK", "CHF"]
CHANNELS            = ["SEPA", "SWIFT", "FASTER_PAYMENTS", "INTERNAL", "CARD"]
HIGH_RISK_COUNTRIES = {"CY", "MT", "PAN", "BVI", "SCH"}
COUNTRY_POOL        = ["LV", "EE", "LT", "FI", "SE", "DE", "GB", "CY", "MT", "PAN", "BVI"]

customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(N_CUSTOMERS)]
account_ids  = [f"ACC{str(i).zfill(6)}"  for i in range(N_ACCOUNTS)]

SCHEMA = StructType([
    StructField("transaction_id",      StringType(),    nullable=False),
    StructField("customer_id",         StringType(),    nullable=False),
    StructField("originator_account",  StringType(),    nullable=True),
    StructField("beneficiary_account", StringType(),    nullable=True),
    StructField("amount",              DoubleType(),    nullable=False),
    StructField("currency",            StringType(),    nullable=False),
    StructField("channel",             StringType(),    nullable=False),
    StructField("originator_country",  StringType(),    nullable=True),
    StructField("beneficiary_country", StringType(),    nullable=True),
    StructField("timestamp",           TimestampType(), nullable=False),
    StructField("is_flagged_source",   BooleanType(),   nullable=False),
])


def random_amount(suspicious: bool = False) -> float:
    if suspicious:
        return round(random.uniform(9_000, 9_999), 2)
    return round(random.expovariate(1 / 2500), 2)


def make_transaction(txn_id: int) -> dict:
    s  = random.random() < 0.08
    ts = START_DATE + timedelta(
        days=random.randint(0, 364),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )
    return {
        "transaction_id":      f"TXN{str(txn_id).zfill(8)}",
        "customer_id":         random.choice(customer_ids),
        "originator_account":  random.choice(account_ids),
        "beneficiary_account": random.choice(account_ids),
        "amount":              random_amount(s),
        "currency":            random.choice(CURRENCIES),
        "channel":             random.choice(CHANNELS),
        "originator_country":  random.choice(COUNTRY_POOL),
        "beneficiary_country": (
            random.choice(COUNTRY_POOL) if s
            else random.choice(["LV", "EE", "LT", "FI", "SE", "DE", "GB"])
        ),
        "timestamp":         ts,
        "is_flagged_source": s,
    }


def main():
    spark = (
        SparkSession.builder
        .appName("AML_Ingest_Transactions")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    records = [make_transaction(i) for i in range(N_TRANSACTIONS)]
    print(f"Generated {len(records):,} synthetic transactions.")

    txn_df = spark.createDataFrame(records, schema=SCHEMA)

    # Data quality checks – fail fast on critical issues
    critical_cols = ["transaction_id", "customer_id", "amount", "currency", "timestamp"]
    null_counts = txn_df.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in critical_cols]
    ).collect()[0].asDict()
    for col, cnt in null_counts.items():
        if cnt > 0:
            raise ValueError(f"Null values in critical column '{col}': {cnt}")

    invalid_amounts = txn_df.filter(F.col("amount") <= 0).count()
    if invalid_amounts > 0:
        raise ValueError(f"Found {invalid_amounts} transactions with non-positive amounts.")

    duplicates = txn_df.count() - txn_df.select("transaction_id").distinct().count()
    if duplicates > 0:
        raise ValueError(f"Found {duplicates} duplicate transaction IDs.")

    print("Data quality checks passed.")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    txn_df.coalesce(1).write.mode("overwrite").parquet(OUTPUT_PATH)
    print(f"Raw transactions written to {OUTPUT_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
