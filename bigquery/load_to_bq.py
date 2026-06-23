"""
Load AML transaction Parquet output to a Google BigQuery dataset.

Usage
-----
    python bigquery/load_to_bq.py \\
        --project  my-gcp-project \\
        --dataset  aml_transactions \\
        --parquet  data/raw/transactions

    # Create the dataset automatically on first run:
    python bigquery/load_to_bq.py \\
        --project  my-gcp-project \\
        --dataset  aml_transactions \\
        --parquet  data/raw/transactions \\
        --create-dataset

    # Load a different table (e.g. the alert queue):
    python bigquery/load_to_bq.py \\
        --project  my-gcp-project \\
        --dataset  aml_transactions \\
        --parquet  output/alert_queue \\
        --table    alert_queue

Authentication
--------------
    The script uses Application Default Credentials (ADC).
    Authenticate before running:
        gcloud auth application-default login

Dependencies
------------
    pip install google-cloud-bigquery pyarrow pandas db-dtypes
"""
import argparse
import glob
import os
import sys

from google.cloud import bigquery
from google.cloud.bigquery import LoadJobConfig, SchemaField, SourceFormat

# Explicit schema matches schema.json and ingest_transactions.py SCHEMA.
# FLOAT64 maps to PySpark DoubleType; TIMESTAMP to TimestampType.
TRANSACTIONS_SCHEMA: list[SchemaField] = [
    SchemaField("transaction_id",      "STRING",    mode="REQUIRED",
                description="Unique transaction identifier"),
    SchemaField("customer_id",         "STRING",    mode="REQUIRED",
                description="Customer who initiated the transaction"),
    SchemaField("originator_account",  "STRING",    mode="NULLABLE",
                description="Sending account reference"),
    SchemaField("beneficiary_account", "STRING",    mode="NULLABLE",
                description="Receiving account reference"),
    SchemaField("amount",              "FLOAT64",   mode="REQUIRED",
                description="Transaction amount"),
    SchemaField("currency",            "STRING",    mode="REQUIRED",
                description="ISO 4217 currency code"),
    SchemaField("channel",             "STRING",    mode="REQUIRED",
                description="Payment channel"),
    SchemaField("originator_country",  "STRING",    mode="NULLABLE",
                description="ISO 3166-1 alpha-2 country of the sender"),
    SchemaField("beneficiary_country", "STRING",    mode="NULLABLE",
                description="ISO 3166-1 alpha-2 country of the receiver"),
    SchemaField("timestamp",           "TIMESTAMP", mode="REQUIRED",
                description="UTC timestamp of the transaction"),
    SchemaField("is_flagged_source",   "BOOL",      mode="REQUIRED",
                description="Synthetic ground-truth label"),
]


def _collect_parquet_files(path: str) -> list[str]:
    """Return all .parquet part-files under a directory, or the path itself."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
        if not files:
            raise FileNotFoundError(f"No .parquet files found under {path!r}")
        return files
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Parquet path not found: {path!r}")
    return [path]


def load_parquet_to_bq(
    client: bigquery.Client,
    project: str,
    dataset: str,
    parquet_path: str,
    table: str,
    schema: list[SchemaField],
    write_disposition: str = "WRITE_TRUNCATE",
) -> None:
    table_ref = f"{project}.{dataset}.{table}"
    files = _collect_parquet_files(parquet_path)
    print(f"Loading {len(files)} Parquet file(s) → {table_ref} …")

    job_config = LoadJobConfig(
        schema=schema,
        source_format=SourceFormat.PARQUET,
        write_disposition=write_disposition,
        # PySpark writes timestamps as INT96 or int64 micros; auto-detect handles both.
        parquet_options=bigquery.format_options.ParquetOptions(
            enable_list_inference=True,
        ),
    )

    for i, filepath in enumerate(files, start=1):
        print(f"  [{i}/{len(files)}] {os.path.basename(filepath)}")
        with open(filepath, "rb") as fh:
            job = client.load_table_from_file(fh, table_ref, job_config=job_config)
            job.result()  # wait for completion; raises on error
        # Only the first file truncates; subsequent parts append.
        job_config.write_disposition = "WRITE_APPEND"

    loaded_table = client.get_table(table_ref)
    print(f"Done. {loaded_table.num_rows:,} rows now in {table_ref}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load AML Parquet outputs to Google BigQuery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project",        required=True,
                        help="GCP project ID")
    parser.add_argument("--dataset",        required=True,
                        help="BigQuery dataset name (created if --create-dataset is set)")
    parser.add_argument("--parquet",        required=True,
                        help="Path to a Parquet file or directory of part-files")
    parser.add_argument("--table",          default="transactions",
                        help="Target BigQuery table name")
    parser.add_argument("--location",       default="US",
                        help="Dataset location (only used when creating a new dataset)")
    parser.add_argument("--create-dataset", action="store_true",
                        help="Create the BigQuery dataset if it does not exist")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    client = bigquery.Client(project=args.project)

    if args.create_dataset:
        dataset_ref = bigquery.Dataset(f"{args.project}.{args.dataset}")
        dataset_ref.location = args.location
        client.create_dataset(dataset_ref, exists_ok=True)
        print(f"Dataset {args.project}.{args.dataset} ({args.location}) ready.")

    # Use the explicit TRANSACTIONS_SCHEMA for the canonical transactions table;
    # fall back to auto-detection for any other table (e.g. alert_queue).
    schema = TRANSACTIONS_SCHEMA if args.table == "transactions" else []

    load_parquet_to_bq(
        client=client,
        project=args.project,
        dataset=args.dataset,
        parquet_path=args.parquet,
        table=args.table,
        schema=schema,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
