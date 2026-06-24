# Azure ML Pipeline – Credit Risk XGBoost

This directory integrates the PySpark AML pipeline's credit risk model with **Azure Machine Learning** using the Azure AI ML SDK v2. The pipeline is **defined but not provisioned by default** — no cloud resources are created or billed until you explicitly run `credit_risk_pipeline.py`.

---

## Directory Structure

```
azure-ml/
  credit_risk_pipeline.py          # Pipeline definition and submission script
  environment.yml                  # Conda environment for Azure ML compute jobs
  config.json.example              # Workspace config template (copy → config.json)
  README.md                        # This file
  components/
    train/
      train_component.yml          # Component YAML – training step
      train_script.py              # Training script (wraps train_credit_risk_model.py)
    evaluate/
      evaluate_component.yml       # Component YAML – evaluation step
      evaluate_script.py           # Evaluation + Azure ML metrics logging
```

---

## Step 1 – Create a Free Azure Account

1. Go to [azure.microsoft.com/free](https://azure.microsoft.com/free/) and sign up.  
   The free tier includes **$200 credit for 30 days** and several always-free services.
2. Complete identity verification and accept the Azure subscription agreement.
3. You will receive a **Subscription ID** (a UUID) — note it down.

---

## Step 2 – Create an Azure ML Workspace

1. In the [Azure Portal](https://portal.azure.com), search for **Machine Learning** and click **Create**.
2. Fill in:
   - **Subscription**: your subscription
   - **Resource group**: create new, e.g. `aml-aml-rg`
   - **Workspace name**: e.g. `aml-credit-risk-ws`
   - **Region**: pick the one closest to you
3. Click **Review + create → Create** (takes ~2 minutes).
4. Once created, go to the workspace → **Download config.json** (top-right menu).  
   Copy the downloaded file to `azure-ml/config.json`.  
   **`config.json` is gitignored — never commit it.**

---

## Step 3 – Install Prerequisites

### Azure CLI

```bash
# Linux / WSL
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# macOS
brew install azure-cli

# Windows – download the MSI from https://aka.ms/installazurecliwindows
```

### Python dependencies

```bash
pip install -r azure-requirements.txt
```

---

## Step 4 – Authenticate

```bash
# Interactive browser login (works on your local machine)
az login

# If running in a CI/non-interactive environment, use a service principal instead:
# az login --service-principal -u $AZURE_CLIENT_ID -p $AZURE_CLIENT_SECRET --tenant $AZURE_TENANT_ID
```

`DefaultAzureCredential` (used by `credit_risk_pipeline.py`) automatically picks up the `az login` session, managed identity, or service-principal environment variables — whichever is present.

---

## Step 5 – Configure the Workspace

Copy the example config and fill in your values:

```bash
cp azure-ml/config.json.example azure-ml/config.json
# Edit azure-ml/config.json and replace the placeholders with your actual values.
```

**Or** set environment variables instead of a file:

```bash
export AZURE_SUBSCRIPTION_ID="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
export AZURE_RESOURCE_GROUP="aml-aml-rg"
export AZURE_WORKSPACE_NAME="aml-credit-risk-ws"
```

---

## Step 6 – Run the PySpark Pipeline First

The Azure ML pipeline reads from `data/processed/txn_scored/`. Generate it locally:

```bash
pip install pyspark==3.5.1

spark-submit scripts/ingest_transactions.py
spark-submit scripts/run_typology_detection.py
spark-submit scripts/export_to_parquet.py
```

---

## Step 7 – Submit the Azure ML Pipeline

```bash
python azure-ml/credit_risk_pipeline.py
```

The script will:

| Step | Action |
|------|--------|
| 1 | Connect to your Azure ML workspace |
| 2 | Register `credit-risk-env:1` conda environment (first run builds Docker image ~5 min) |
| 3 | Provision `cpu-cluster` (Standard_DS3_v2, 0→4 nodes, auto-scales to 0 when idle) |
| 4 | Register `data/processed/txn_scored/` as the `aml_txn_scored` data asset |
| 5 | Load the two components from their YAML definitions |
| 6 | Submit the pipeline to Azure ML and stream logs |
| 7 | Register the trained model as `credit_risk_xgboost` in the Azure ML Model Registry |

### Monitor in the Studio

After submission, a **Studio URL** is printed. Open it to:
- Watch the pipeline graph and per-step logs in real time
- View logged metrics on the **Metrics** tab
- Download the SHAP plot and LIME HTML from the **Outputs + logs** tab

---

## Pipeline Architecture

```
scored_parquet (Data Asset)
        │
        ▼
┌─────────────────────────────────┐
│  train_credit_risk_model (Step 1)│
│  ─ 5-fold stratified CV         │
│  ─ Final XGBoost model          │
│  ─ SHAP beeswarm plot           │
│  ─ LIME explanations (top 3)    │
│  ─ Metrics logged via MLflow    │
└────────┬────────────────────────┘
         │  model_dir (uri_folder)
         ▼
┌─────────────────────────────────┐
│  evaluate_credit_risk_model (2) │
│  ─ Reload model + test split    │
│  ─ AUC, Gini, KS, F1           │
│  ─ Metrics logged via MLflow    │
│  ─ JSON eval report output      │
└─────────────────────────────────┘
         │
         ▼
  Model Registry (credit_risk_xgboost → Staging)
```

---

## Cost Estimate

| Resource | Cost |
|---|---|
| Azure ML workspace | Free (pay only for compute and storage) |
| `cpu-cluster` Standard_DS3_v2 | ~$0.19 / hr, scales to 0 when idle |
| Training job (single run, ~10 k rows) | < $0.05 |
| Blob Storage for artifacts | < $0.01 / run |

The compute cluster **auto-scales to 0 nodes** after 2 minutes of idle time, so you are only charged while jobs are running. The free $200 credit is more than enough for many pipeline runs.

---

## Promoting the Model to Production

After reviewing the evaluation metrics in the Studio, promote the model from Staging to Production:

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(),
    path="azure-ml/config.json",
)

# List versions in the registry
for v in ml_client.models.list("credit_risk_xgboost"):
    print(v.version, v.tags)

# Update the tag on the desired version
from azure.ai.ml.entities import Model
model = ml_client.models.get("credit_risk_xgboost", version="1")
model.tags["stage"] = "Production"
ml_client.models.create_or_update(model)
```

---

## Notes

- `azure-ml/config.json` is gitignored. Never commit subscription credentials.
- The pipeline is idempotent — re-running it registers a new model version.
- To change the compute size, edit `COMPUTE_NAME` / `size` in `credit_risk_pipeline.py`.
- To use serverless compute instead of a cluster, set `pipeline_job.settings.default_compute = "serverless"`.
