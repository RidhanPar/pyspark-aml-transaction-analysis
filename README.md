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
