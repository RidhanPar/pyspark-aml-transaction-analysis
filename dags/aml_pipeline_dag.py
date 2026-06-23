"""
AML Pipeline DAG – three sequential tasks that run the PySpark AML pipeline.

Task order:
  ingest_transactions  →  run_typology_detection  →  export_to_parquet

Each task submits the corresponding script via spark-submit inside the container.
Environment variables control I/O paths so the scripts remain reusable outside Airflow.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

SPARK_SUBMIT = "spark-submit --master local[*]"
SCRIPTS_DIR  = "/opt/airflow/scripts"

# Shared path env-vars passed to every task so each script knows where to read/write.
_PATH_ENV = (
    "AML_RAW_PATH=/opt/airflow/data/raw/transactions "
    "AML_SCORED_PATH=/opt/airflow/data/processed/txn_scored "
    "AML_CUST_RISK_PATH=/opt/airflow/data/processed/customer_risk "
    "AML_ALERT_OUT_PATH=/opt/airflow/output/alert_queue "
    "AML_CUST_RISK_OUT_PATH=/opt/airflow/output/customer_risk_profiles"
)

default_args = {
    "owner":            "ridhanpar",
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="aml_pipeline",
    description="End-to-end AML transaction monitoring pipeline (PySpark)",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["aml", "pyspark", "compliance"],
) as dag:

    ingest_transactions = BashOperator(
        task_id="ingest_transactions",
        bash_command=(
            f"env {_PATH_ENV} "
            f"{SPARK_SUBMIT} {SCRIPTS_DIR}/ingest_transactions.py"
        ),
        doc_md=(
            "**Ingest Transactions** – generate 10 k synthetic AML transactions, "
            "run data-quality checks (nulls, duplicates, negative amounts), "
            "and write raw Parquet to `data/raw/transactions/`."
        ),
    )

    run_typology_detection = BashOperator(
        task_id="run_typology_detection",
        bash_command=(
            f"env {_PATH_ENV} "
            f"{SPARK_SUBMIT} {SCRIPTS_DIR}/run_typology_detection.py"
        ),
        doc_md=(
            "**Typology Detection** – read raw transactions, compute rolling-window "
            "features, apply five AML rule flags (structuring, velocity, high-risk "
            "country, amount spike, rapid succession), assign weighted risk scores "
            "(HIGH/MEDIUM/LOW/NONE), aggregate to customer level, and persist "
            "processed Parquet to `data/processed/`."
        ),
    )

    export_to_parquet = BashOperator(
        task_id="export_to_parquet",
        bash_command=(
            f"env {_PATH_ENV} "
            f"{SPARK_SUBMIT} {SCRIPTS_DIR}/export_to_parquet.py"
        ),
        doc_md=(
            "**Export to Parquet** – filter the alert queue (risk_score ≥ 25), "
            "write `output/alert_queue/` and `output/customer_risk_profiles/` "
            "Parquet files ready for downstream consumption."
        ),
    )

    ingest_transactions >> run_typology_detection >> export_to_parquet
