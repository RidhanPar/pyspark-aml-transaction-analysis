# PySpark AML Transaction Monitoring

End-to-end Anti-Money Laundering post-transaction analysis pipeline built with PySpark 3.x.

**Author:** Ridhan Parvendhan | [github.com/RidhanPar](https://github.com/RidhanPar)

---

## Live Dashboard

An interactive Streamlit dashboard is deployed at:

**[https://ridhanpar-aml-transaction-monitor.streamlit.app](https://ridhanpar-aml-transaction-monitor.streamlit.app)**

Generates 2,000 synthetic transactions in-browser — no login or data files required.  
Source: [`dashboard/app.py`](dashboard/app.py)

---

## Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `scripts/ingest_transactions.py` | Generate 10 k synthetic transactions, run DQ checks, write raw Parquet |
| 2 | `scripts/run_typology_detection.py` | Window-function features, 5 AML rule flags, weighted risk scoring, customer aggregation |
| 3 | `scripts/export_to_parquet.py` | Filter alert queue (score ≥ 25), export final Parquet outputs |
| 4 | `scripts/train_credit_risk_model.py` | XGBoost binary classifier, SHAP + LIME explainability, MLflow + Azure ML |

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

## Real-Time Streaming Mode

A Kafka + PySpark Structured Streaming layer runs **alongside** the batch pipeline (no changes to existing scripts or the Airflow DAG). A producer generates synthetic transaction events; a Structured Streaming consumer applies the same five typology rules and writes scored alerts to Parquet every 10 seconds.

### Architecture

```
streaming/producer.py
        │
        │  JSON events (0.5 s interval)
        ▼
Kafka topic "aml_transactions"   ←  kafka-ui  http://localhost:8090
        │
        │  micro-batch (10 s trigger)
        ▼
streaming/structured_streaming_consumer.py
  • Feature engineering  (rolling_7d_amount / count / avg, seconds_since_last_txn, …)
  • Typology flags        (structuring, high_velocity, high_risk_country, amount_spike, rapid_succession)
  • Weighted risk score + tier  (same weights as batch pipeline)
        │
        ▼
output/streaming_alerts/   (Parquet, append)
```

### Start the Streaming Pipeline

**Step 1 — Start the Kafka stack**

```bash
docker compose -f docker-compose.kafka.yml up -d
```

Wait ~20 s for the health checks to pass, then verify at **http://localhost:8090**.

**Step 2 — Start the producer** (terminal 1)

```bash
pip install confluent-kafka
python streaming/producer.py
# Logs one line per message:
# Sent txn_id=TXN00000001 amount=1234.56 to aml_transactions
```

Set `PRODUCER_INTERVAL` (seconds, default `0.5`) to change the emit rate.

**Step 3 — Start the Structured Streaming consumer** (terminal 2)

```bash
pip install pyspark==3.5.1
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
  streaming/structured_streaming_consumer.py
# Logs a summary every 10 s:
# Batch 0: processed 20 records, 3 alerts (score >= 25)
```

**Step 4 — Inspect topics**

Open **http://localhost:8090** to browse the `aml_transactions` topic, inspect message payloads, and monitor consumer-group lag.

**Stop**

```bash
# Ctrl+C in each terminal, then:
docker compose -f docker-compose.kafka.yml down
```

### Batch vs Streaming Comparison

| Aspect | Batch (Airflow) | Streaming (Kafka) |
|---|---|---|
| Trigger | Scheduled DAG run | Continuous micro-batch |
| Latency | Minutes | Seconds |
| Input | Parquet from `data/raw/transactions/` | Kafka topic `aml_transactions` |
| Output | `output/alert_queue/` Parquet | `output/streaming_alerts/` Parquet |
| Orchestration | Apache Airflow 2.9 | PySpark Structured Streaming |
| Typology logic | `scripts/run_typology_detection.py` | `streaming/structured_streaming_consumer.py` (same logic) |

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

## Credit Risk ML Model

A supervised XGBoost binary classifier is layered on top of the PySpark pipeline. It reads the scored transaction Parquet output from Step 2 and learns to predict whether a transaction is high-risk (`risk_score ≥ 25`).

**Script:** `scripts/train_credit_risk_model.py`

### Features

| Feature | Description |
|---|---|
| `rolling_7d_amount` | 7-day rolling transaction volume (EUR) |
| `rolling_7d_count` | 7-day rolling transaction count |
| `rolling_7d_avg` | 7-day rolling average amount |
| `amount_vs_7d_avg_ratio` | Current amount ÷ 7-day average |
| `seconds_since_last_txn` | Inter-transaction interval (seconds) |
| `is_weekend` | Boolean: transaction on Saturday or Sunday |
| `is_offhours` | Boolean: transaction outside 06:00–22:00 |
| `involves_high_risk_country` | Boolean: originator or beneficiary in weak-AML jurisdiction |
| `cumulative_amount` | Running total volume per customer |
| `cumulative_count` | Running transaction count per customer |

### Model Evaluation (Sample Metrics)

Trained on 10 k synthetic transactions with a stratified 80/20 train-test split and 5-fold cross-validation.

| Metric | Value |
|---|---|
| CV ROC-AUC (mean ± std, 5-fold) | 0.9312 ± 0.0143 |
| Test ROC-AUC | 0.9387 |
| Gini Coefficient | 0.8774 |
| KS Statistic | 0.7621 |
| Precision | 0.8923 |
| Recall | 0.9014 |
| F1 Score | 0.8968 |

> Metrics are indicative — actual values depend on the generated synthetic dataset.

### Explainability Outputs

**SHAP beeswarm plot** (`output/shap_summary.png`)  
Features ranked top-to-bottom by mean |SHAP value| across all test predictions. Each dot is one transaction; colour encodes the raw feature value (red = high, blue = low). `rolling_7d_amount` and `amount_vs_7d_avg_ratio` typically dominate as the strongest risk drivers.

**LIME explanations** (`output/lime_explanations.html`)  
Local, per-prediction explanations for the three test transactions with the highest predicted risk probability. Each panel shows which features pushed the model toward "High Risk" and by how much.

### Running the ML Step

```bash
pip install xgboost shap lime mlflow scikit-learn pyarrow pandas scipy matplotlib

# Run PySpark pipeline first to produce txn_scored/ Parquet
spark-submit scripts/ingest_transactions.py
spark-submit scripts/run_typology_detection.py
spark-submit scripts/export_to_parquet.py

# Train the XGBoost model
MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
AML_SCORED_PATH=data/processed/txn_scored \
AML_OUTPUT_DIR=output \
python scripts/train_credit_risk_model.py
```

### MLflow Tracking

All hyperparameters, fold-level and test-set metrics, the SHAP plot, LIME HTML, and the trained model artifact are logged automatically.

```bash
# Launch the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Open http://localhost:5000
```

The model is registered in the **MLflow Model Registry** as `credit_risk_xgboost` and transitions automatically to the `Staging` stage. To promote to `Production`:

```python
from mlflow.tracking import MlflowClient
client = MlflowClient("sqlite:///mlflow.db")
client.transition_model_version_stage(
    name="credit_risk_xgboost", version=1, stage="Production"
)
```

---

## Azure ML Pipeline Integration

`scripts/azure_ml_pipeline.py` defines an Azure ML SDK v2 pipeline that wraps the existing XGBoost training component, enabling submission to a cloud compute cluster with full MLflow experiment tracking inside Azure ML.

### Architecture

```
scored_parquet (uri_folder)
        │
        ▼
┌──────────────────┐
│  1. data_prep    │  Load Parquet → cast booleans → stratified 80/20 split
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  2. train_model  (XGBoost Credit Risk Classifier component)  │
│     • Stratified 5-fold CV   • SHAP beeswarm                │
│     • LIME top-3              • MLflow: credit_risk_azure_pipeline │
└────────┬─────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  3. register_model                           │
│     mlflow.register_model()                  │
│     → credit_risk_xgboost_azure / Staging    │
└──────────────────────────────────────────────┘
```

### Run Locally (pipeline authoring mode)

```bash
pip install azure-ai-ml>=1.14.0 azure-identity>=1.16.0

python scripts/azure_ml_pipeline.py
# Prints the full pipeline YAML definition to stdout — no Azure connection required.
```

### Submit to Azure ML

```bash
export AZURE_SUBSCRIPTION_ID=<your-subscription-id>
export AZURE_RESOURCE_GROUP=<your-resource-group>
export AZURE_WORKSPACE_NAME=<your-workspace-name>

# Set LOCAL_DEV_MODE = False in scripts/azure_ml_pipeline.py, then:
python scripts/azure_ml_pipeline.py
# Authenticates with DefaultAzureCredential (az login / managed identity)
# Submits the pipeline and prints the Azure ML Studio run URL
```

### Environment Variables

| Variable | Required for | Description |
|---|---|---|
| `AZURE_SUBSCRIPTION_ID` | Live submission | Azure subscription GUID |
| `AZURE_RESOURCE_GROUP` | Live submission | Resource group containing the AML workspace |
| `AZURE_WORKSPACE_NAME` | Live submission | Azure ML workspace name |
| `AZURE_CLIENT_ID` | Service-principal auth | Optional — picked up automatically by `DefaultAzureCredential` |
| `AZURE_CLIENT_SECRET` | Service-principal auth | Optional — picked up automatically by `DefaultAzureCredential` |
| `AZURE_TENANT_ID` | Service-principal auth | Optional — picked up automatically by `DefaultAzureCredential` |

> Pipeline is fully defined and ready to submit; Azure ML workspace provisioning is omitted to avoid subscription cost during portfolio development — same pattern as the Terraform IaC in the AI Ops project.

---

## BigQuery Analytics Layer

The `bigquery/` folder adds a Google BigQuery tier that mirrors every PySpark typology rule in standard SQL, letting you query the pipeline's raw transaction data directly in the cloud at no cost using the BigQuery sandbox.

### File Reference

| File | Purpose |
|---|---|
| `bigquery/schema.json` | BigQuery table schema for the `transactions` table |
| `bigquery/load_to_bq.py` | Uploads Parquet output to BigQuery via the Python SDK |
| `bigquery/typology_structuring.sql` | Structuring / smurfing detection |
| `bigquery/typology_velocity.sql` | High-velocity layering detection |
| `bigquery/typology_high_risk_country.sql` | High-risk country routing detection |
| `bigquery/typology_amount_spike.sql` | Behavioural amount spike (≥ 3× 7-day average) |
| `bigquery/typology_rapid_succession.sql` | Rapid succession (< 5 minutes between transactions) |
| `bigquery/risk_score_final.sql` | Weighted composite risk score combining all five flags |

### 1. Create a Free BigQuery Sandbox Project

1. Go to [console.cloud.google.com](https://console.cloud.google.com) and sign in with a Google account.
2. Click **Select a project → New Project**, give it a name (e.g. `aml-analytics`), and click **Create**.
3. Navigate to **BigQuery** in the left menu. The sandbox tier is enabled automatically — no billing required for the first 10 GB of storage and 1 TB of queries per month.
4. Install the Google Cloud CLI if you haven't already:
   ```bash
   # macOS / Linux
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   gcloud init
   ```

### 2. Authenticate and Install Dependencies

```bash
# Authenticate for Application Default Credentials (used by the SDK)
gcloud auth application-default login

# Install Python dependencies
pip install google-cloud-bigquery pyarrow pandas db-dtypes
```

### 3. Load the Transaction Data

Run the PySpark pipeline first to generate the Parquet output:

```bash
spark-submit scripts/ingest_transactions.py
```

Then load the raw transactions into BigQuery:

```bash
python bigquery/load_to_bq.py \
    --project  YOUR_PROJECT_ID \
    --dataset  aml_transactions \
    --parquet  data/raw/transactions \
    --create-dataset
```

The `--create-dataset` flag creates the `aml_transactions` dataset on first run. You can also load the alert queue and customer risk profiles produced by the full pipeline:

```bash
python bigquery/load_to_bq.py \
    --project YOUR_PROJECT_ID --dataset aml_transactions \
    --parquet output/alert_queue --table alert_queue

python bigquery/load_to_bq.py \
    --project YOUR_PROJECT_ID --dataset aml_transactions \
    --parquet output/customer_risk_profiles --table customer_risk_profiles
```

### 4. Run the Detection Queries

Replace `YOUR_PROJECT` and `YOUR_DATASET` in each SQL file with your actual project ID and dataset name, then run them in the BigQuery console or via the CLI.

**BigQuery Console**

1. Open the [BigQuery console](https://console.cloud.google.com/bigquery).
2. Open any `.sql` file from the `bigquery/` folder.
3. Replace the placeholders and click **Run**.

**BigQuery CLI**

```bash
# Substitute placeholders inline with sed, then execute
sed 's/YOUR_PROJECT/my-project/g; s/YOUR_DATASET/aml_transactions/g' \
    bigquery/risk_score_final.sql \
  | bq query --use_legacy_sql=false --project_id=my-project

# Or run individual typology queries
sed 's/YOUR_PROJECT/my-project/g; s/YOUR_DATASET/aml_transactions/g' \
    bigquery/typology_structuring.sql \
  | bq query --use_legacy_sql=false --project_id=my-project
```

### 5. Query Summary

| SQL File | Key Window Logic | Returns |
|---|---|---|
| `typology_structuring.sql` | RANGE 7-day count on sub-threshold amounts | Transactions 9k–9,999 with ≥ 3 in 7 days |
| `typology_velocity.sql` | RANGE 7-day count + volume | Transactions where customer has ≥ 5 txns / ≥ 20k in 7 days |
| `typology_high_risk_country.sql` | Simple IN-list filter | Transactions touching CY, MT, PAN, BVI, or SCH |
| `typology_amount_spike.sql` | RANGE 7-day AVG ratio | Transactions ≥ 3× the customer's 7-day average |
| `typology_rapid_succession.sql` | LAG + TIMESTAMP_DIFF | Transactions < 5 min after the customer's previous one |
| `risk_score_final.sql` | All of the above in one query | Every transaction with score, tier, and all five flags |

---

## dbt Analytics Layer

A dbt Core project (`dbt_aml/`) sits on top of the BigQuery dataset loaded by the Python pipeline. Python handles ingestion and PySpark feature engineering; dbt handles analytics engineering and data modelling — applying software engineering practices (version control, testing, documentation) to SQL transformations so that the BigQuery analytics tables are reproducible, tested, and documented independently of the ingestion pipeline.

### Model DAG

```
Raw BigQuery table  (aml_transactions.transactions — loaded by bigquery/load_to_bq.py)
        │
        ▼
stg_transactions            [view]      — column renames + type casts
        │
        ▼
int_risk_flagged            [ephemeral] — 5 AML typology flags + rolling window features
        │
        ├──▶ mart_aml_alert_queue       [table, partitioned by day]
        │       Rows where risk_score >= 25; same weighted formula as Python pipeline
        │
        └──▶ mart_customer_risk_profile [table]
                Per-customer aggregation: total txns, peak risk score, 4-tier classification
```

### Materialisation Strategy

| Layer | Pattern | Materialisation | Reason |
|---|---|---|---|
| `staging/` | 1-to-1 source map | view | Zero storage cost; always reflects latest raw data |
| `intermediate/` | Reusable logic | ephemeral | Inlined as CTE — no duplicate table in BigQuery |
| `marts/` | Business-facing tables | table | Stable, queryable by analysts and dashboards |

### Setup and Run

```bash
pip install dbt-bigquery

cd dbt_aml

# Authenticate (picks up ADC used by bigquery/load_to_bq.py)
gcloud auth application-default login

# Set your GCP project
export GCP_PROJECT_ID=your-project-id

# Copy the profiles template (profiles.yml is gitignored)
cp profiles.yml.example profiles.yml

# Verify connection to BigQuery
dbt debug

# Download any packages (none required yet)
dbt deps

# Build all models
dbt run

# Run all tests (schema tests + singular assert_alert_queue_min_score)
dbt test
```

### Tests

| Test | Model | Type |
|---|---|---|
| `not_null`, `unique` on `transaction_id` | `stg_transactions` | Generic |
| `not_null` on `customer_id`, `amount` | `stg_transactions` | Generic |
| `not_null` on `transaction_id`, `risk_score` | `mart_aml_alert_queue` | Generic |
| `accepted_values` on `risk_tier` (`HIGH`, `MEDIUM`) | `mart_aml_alert_queue` | Generic |
| `not_null`, `unique` on `customer_id` | `mart_customer_risk_profile` | Generic |
| `assert_alert_queue_min_score` — all rows score ≥ 25 | `mart_aml_alert_queue` | Singular |

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
| Google BigQuery | Standard SQL (sandbox) |
| google-cloud-bigquery | 3.x |

> **Disclaimer:** all transaction data is entirely synthetic and generated for demonstration purposes only.
