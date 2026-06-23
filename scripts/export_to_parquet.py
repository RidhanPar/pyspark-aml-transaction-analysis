"""
Step 3 – Export: filter alert queue from scored transactions and write final Parquet outputs.
Input:  data/processed/txn_scored/  and  data/processed/customer_risk/
Output: output/alert_queue/  and  output/customer_risk_profiles/
"""
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

SCORED_PATH        = os.environ.get("AML_SCORED_PATH",       "/opt/airflow/data/processed/txn_scored")
CUST_RISK_PATH     = os.environ.get("AML_CUST_RISK_PATH",    "/opt/airflow/data/processed/customer_risk")
ALERT_OUT_PATH     = os.environ.get("AML_ALERT_OUT_PATH",    "/opt/airflow/output/alert_queue")
CUST_RISK_OUT_PATH = os.environ.get("AML_CUST_RISK_OUT_PATH","/opt/airflow/output/customer_risk_profiles")

ALERT_THRESHOLD = int(os.environ.get("AML_ALERT_THRESHOLD", "25"))


def main():
    spark = (
        SparkSession.builder
        .appName("AML_Export_Parquet")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    txn_scored    = spark.read.parquet(SCORED_PATH)
    customer_risk = spark.read.parquet(CUST_RISK_PATH)

    total = txn_scored.count()

    alert_queue = (
        txn_scored
        .filter(F.col("risk_score") >= ALERT_THRESHOLD)
        .select(
            "transaction_id", "customer_id", "timestamp", "amount", "currency",
            "channel", "originator_country", "beneficiary_country",
            "risk_score", "risk_tier", "flag_count",
            "rolling_7d_amount", "rolling_7d_count",
        )
        .orderBy(F.desc("risk_score"), F.desc("amount"))
    )

    n_alerts = alert_queue.count()
    alert_rate = n_alerts / total if total else 0
    print(f"Alerts: {n_alerts:,} / {total:,} ({alert_rate:.1%} alert rate)")

    for path in (ALERT_OUT_PATH, CUST_RISK_OUT_PATH):
        os.makedirs(path, exist_ok=True)

    alert_queue.coalesce(1).write.mode("overwrite").parquet(ALERT_OUT_PATH)
    print(f"Alert queue written to {ALERT_OUT_PATH}")

    customer_risk.coalesce(1).write.mode("overwrite").parquet(CUST_RISK_OUT_PATH)
    print(f"Customer risk profiles written to {CUST_RISK_OUT_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
