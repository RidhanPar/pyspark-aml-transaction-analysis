# PySpark AML Transaction Monitoring

End-to-end Anti-Money Laundering post-transaction analysis pipeline built with PySpark 3.x.

**Author:** Ridhan Parvendhan | [github.com/RidhanPar](https://github.com/RidhanPar)

---

## Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `scripts/ingest_transactions.py` | Generate 10 k synthetic transactions, run DQ checks, write raw Parquet |
| 2 | `scripts/run_typology_detection.py` | Window-function features, 5 AML rule flags, weighted risk scoring, customer aggregation |
| 3 | `scripts/export_to_parquet.py` | Filter alert queue (score ≥ 25), export final Parquet outputs |

### AML Typologies Detected

| Typology | Rule |
|---|---|
| Structuring | Amount 9,000–9,999 AND ≥ 3 txns in 7 days |
| High-velocity layering | ≥ 5 txns and ≥ 20 k volume in 7 days |
| High-risk country routing | Originator/beneficiary in weak-AML jurisdiction |
| Amount spike | Current amount ≥ 3× customer 7-day average |
| Rapid succession | < 5 minutes since previous transaction |

---

## Pipeline Orchestration

The pipeline is orchestrated with **Apache Airflow 2.9** using a `LocalExecutor` backed by PostgreSQL. Three sequential Airflow tasks map 1-to-1 to the scripts above.

```
ingest_transactions → run_typology_detection → export_to_parquet
```

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/) installed
- Ports 8080 (Airflow UI) and 5432 (Postgres) free on your machine

### Start the Airflow Stack

```bash
# 1. Build and start all services (first run initialises the DB and creates admin user)
docker compose -f docker-compose.airflow.yml up -d

# 2. Tail logs to watch the init container finish (takes ~60 s on first run)
docker compose -f docker-compose.airflow.yml logs -f airflow-init
```

The Airflow UI is available at **http://localhost:8080**
Default credentials: `admin` / `admin`

### Trigger the DAG

**Via the UI**

1. Open http://localhost:8080 and log in.
2. Find the `aml_pipeline` DAG and toggle it **on**.
3. Click **Trigger DAG** (▶) to run immediately.

**Via the CLI**

```bash
docker compose -f docker-compose.airflow.yml exec airflow-scheduler \
  airflow dags trigger aml_pipeline
```

### Monitor a Run

```bash
# Stream scheduler logs
docker compose -f docker-compose.airflow.yml logs -f airflow-scheduler

# List recent DAG runs
docker compose -f docker-compose.airflow.yml exec airflow-scheduler \
  airflow dags list-runs -d aml_pipeline
```

### Output Locations

After a successful run the following directories are populated on the host (bind-mounted from the container):

```
output/
  alert_queue/                   # Parquet – transactions with risk_score ≥ 25
  customer_risk_profiles/        # Parquet – per-customer aggregated risk metrics
data/
  raw/transactions/              # Parquet – raw synthetic transactions
  processed/txn_scored/          # Parquet – scored transactions (all typology flags)
  processed/customer_risk/       # Parquet – intermediate customer aggregation
```

### Stop the Stack

```bash
docker compose -f docker-compose.airflow.yml down
# Add -v to also remove the Postgres volume (resets all Airflow metadata)
docker compose -f docker-compose.airflow.yml down -v
```

---

## Running Scripts Directly (without Airflow)

```bash
pip install pyspark==3.5.1

spark-submit scripts/ingest_transactions.py
spark-submit scripts/run_typology_detection.py
spark-submit scripts/export_to_parquet.py
```

Path defaults (`/opt/airflow/data/…`) can be overridden with environment variables:

| Variable | Default |
|---|---|
| `AML_RAW_PATH` | `/opt/airflow/data/raw/transactions` |
| `AML_SCORED_PATH` | `/opt/airflow/data/processed/txn_scored` |
| `AML_CUST_RISK_PATH` | `/opt/airflow/data/processed/customer_risk` |
| `AML_ALERT_OUT_PATH` | `/opt/airflow/output/alert_queue` |
| `AML_CUST_RISK_OUT_PATH` | `/opt/airflow/output/customer_risk_profiles` |
| `AML_ALERT_THRESHOLD` | `25` |

---

## Tech Stack

| Component | Version |
|---|---|
| Python | 3.10+ |
| PySpark | 3.5.1 |
| Apache Airflow | 2.9.3 |
| PostgreSQL | 15 (Airflow metadata DB) |

> **Disclaimer:** all transaction data is entirely synthetic and generated for demonstration purposes only.
