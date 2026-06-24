"""
Azure ML SDK v2 pipeline – Credit Risk XGBoost Classifier.

This script:
  1. Connects to an Azure ML workspace (config.json or env-vars).
  2. Creates the conda environment in the workspace (credit-risk-env:1).
  3. Provisions a cpu-cluster if one does not already exist.
  4. Registers the scored-transaction Parquet files as an Azure ML data asset.
  5. Loads the train and evaluate components from their YAML definitions.
  6. Assembles and submits the two-step pipeline to Azure ML.
  7. Waits for completion, then registers the trained model in the Azure ML
     Model Registry with AUC, Gini, and KS version tags.

This file is DEFINED but not provisioned by default — no cloud resources
are created or billed until you explicitly run it (see azure-ml/README.md).

Usage:
    python azure-ml/credit_risk_pipeline.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from azure.ai.ml import Input, MLClient, load_component
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import AmlCompute, Data, Environment, Model
from azure.identity import DefaultAzureCredential

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
REPO_ROOT = HERE.parent
DATA_PATH = REPO_ROOT / "data" / "processed" / "txn_scored"

TRAIN_COMPONENT_YAML    = HERE / "components" / "train"    / "train_component.yml"
EVALUATE_COMPONENT_YAML = HERE / "components" / "evaluate" / "evaluate_component.yml"
CONDA_ENV_FILE          = HERE / "environment.yml"

# config.json is searched in azure-ml/ first, then the repo root.
_CONFIG_CANDIDATES = [HERE / "config.json", REPO_ROOT / "config.json"]

# ── Azure ML settings ──────────────────────────────────────────────────────────
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "")
RESOURCE_GROUP  = os.getenv("AZURE_RESOURCE_GROUP",  "")
WORKSPACE_NAME  = os.getenv("AZURE_WORKSPACE_NAME",  "")

ENV_NAME        = "credit-risk-env"
ENV_VERSION     = "1"
COMPUTE_NAME    = "cpu-cluster"
EXPERIMENT_NAME = "credit_risk_classification"
AML_MODEL_NAME  = "credit_risk_xgboost"
DATA_ASSET_NAME = "aml_txn_scored"


# ── Workspace connection ───────────────────────────────────────────────────────

def get_ml_client() -> MLClient:
    """Return an authenticated MLClient from config.json or environment variables."""
    credential = DefaultAzureCredential()

    for config_path in _CONFIG_CANDIDATES:
        if config_path.exists():
            print(f"  Using workspace config: {config_path}")
            return MLClient.from_config(credential=credential, path=str(config_path))

    if SUBSCRIPTION_ID and RESOURCE_GROUP and WORKSPACE_NAME:
        return MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME,
        )

    raise EnvironmentError(
        "No Azure ML workspace configuration found.\n"
        "Provide one of:\n"
        "  (A) azure-ml/config.json  (see azure-ml/config.json.example)\n"
        "  (B) Environment variables: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP,"
        " AZURE_WORKSPACE_NAME\n"
        "Then authenticate with:  az login"
    )


# ── Setup helpers ──────────────────────────────────────────────────────────────

def ensure_environment(ml_client: MLClient) -> None:
    """Register the conda environment in the workspace (idempotent)."""
    env = Environment(
        name=ENV_NAME,
        version=ENV_VERSION,
        description="Credit risk XGBoost training and evaluation environment",
        conda_file=str(CONDA_ENV_FILE),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )
    ml_client.environments.create_or_update(env)
    print(f"  Environment  '{ENV_NAME}:{ENV_VERSION}'  registered.")


def ensure_compute(ml_client: MLClient) -> None:
    """Create a cpu-cluster (Standard_DS3_v2, 0–4 nodes) if it does not exist."""
    try:
        ml_client.compute.get(COMPUTE_NAME)
        print(f"  Compute cluster '{COMPUTE_NAME}'  already exists.")
    except Exception:
        print(f"  Creating compute cluster '{COMPUTE_NAME}' …")
        cluster = AmlCompute(
            name=COMPUTE_NAME,
            type="amlcompute",
            size="Standard_DS3_v2",
            min_instances=0,
            max_instances=4,
            idle_time_before_scale_down=120,
        )
        ml_client.compute.begin_create_or_update(cluster).result()
        print(f"  Compute cluster '{COMPUTE_NAME}'  created.")


def register_data_asset(ml_client: MLClient) -> str:
    """Register the txn_scored Parquet folder as an Azure ML data asset.

    Returns the azureml: URI string for use as a pipeline input.
    """
    asset = Data(
        name=DATA_ASSET_NAME,
        version="1",
        description=(
            "AML scored transactions Parquet produced by "
            "scripts/run_typology_detection.py"
        ),
        type=AssetTypes.URI_FOLDER,
        path=str(DATA_PATH),
    )
    registered = ml_client.data.create_or_update(asset)
    uri = f"azureml:{registered.name}:{registered.version}"
    print(f"  Data asset   '{registered.name}:{registered.version}'  registered.")
    return uri


# ── Pipeline factory ───────────────────────────────────────────────────────────

def create_pipeline(train_comp: Any, eval_comp: Any):
    """Return an Azure ML @pipeline function that wires train → evaluate."""

    @pipeline(
        name="credit_risk_pipeline",
        description=(
            "AML credit risk pipeline: XGBoost training with SHAP/LIME "
            "explainability followed by held-out evaluation."
        ),
        tags={"project": "aml-transaction-analysis", "model": AML_MODEL_NAME},
    )
    def _pipeline(scored_parquet: Input(type=AssetTypes.URI_FOLDER)):  # type: ignore[valid-type]
        train_step = train_comp(scored_parquet=scored_parquet)
        eval_step  = eval_comp(
            model_dir=train_step.outputs.model_dir,
            scored_parquet=scored_parquet,
        )
        return {
            "model_dir":  train_step.outputs.model_dir,
            "eval_report": eval_step.outputs.eval_report,
        }

    return _pipeline


# ── Model registration ─────────────────────────────────────────────────────────

def register_model(
    ml_client: MLClient,
    job_name: str,
    metrics: dict[str, Any],
) -> None:
    """Register the pipeline's model_dir output in the Azure ML Model Registry."""
    model = Model(
        path=f"azureml://jobs/{job_name}/outputs/model_dir",
        name=AML_MODEL_NAME,
        description=(
            "XGBoost binary classifier for AML credit risk, trained on "
            "rolling-window and behavioural features from scored transactions."
        ),
        type="custom_model",
        tags={
            "auc":   f"{metrics.get('test_auc',  'n/a')}",
            "gini":  f"{metrics.get('test_gini', 'n/a')}",
            "ks":    f"{metrics.get('test_ks',   'n/a')}",
            "stage": "Staging",
        },
    )
    registered = ml_client.models.create_or_update(model)
    print(
        f"  Model '{AML_MODEL_NAME}' v{registered.version} registered "
        f"→ Staging  (AUC={metrics.get('test_auc', 'n/a')})"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Azure ML Credit Risk Pipeline")
    print("=" * 60)

    # 1. Connect
    print("\n[1/6] Connecting to workspace …")
    ml_client = get_ml_client()
    print(f"  Workspace: {ml_client.workspace_name}")
    print(f"  Resource group: {ml_client.resource_group_name}")

    # 2. Environment
    print("\n[2/6] Registering conda environment …")
    ensure_environment(ml_client)

    # 3. Compute
    print("\n[3/6] Ensuring compute cluster …")
    ensure_compute(ml_client)

    # 4. Data asset
    print("\n[4/6] Registering data asset …")
    data_uri = register_data_asset(ml_client)

    # 5. Build pipeline
    print("\n[5/6] Building pipeline …")
    train_component    = load_component(source=str(TRAIN_COMPONENT_YAML))
    evaluate_component = load_component(source=str(EVALUATE_COMPONENT_YAML))

    pipeline_fn  = create_pipeline(train_component, evaluate_component)
    pipeline_job = pipeline_fn(
        scored_parquet=Input(path=data_uri, type=AssetTypes.URI_FOLDER)
    )
    pipeline_job.settings.default_compute   = COMPUTE_NAME
    pipeline_job.settings.default_datastore = "workspaceblobstore"

    # 6. Submit
    print("\n[6/6] Submitting pipeline …")
    returned_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=EXPERIMENT_NAME
    )
    print(f"  Job name  : {returned_job.name}")
    print(f"  Studio URL: {returned_job.studio_url}")
    print("\nWaiting for pipeline to complete (may take several minutes) …")
    ml_client.jobs.stream(returned_job.name)

    # Register model after pipeline finishes
    print("\nRegistering model in Azure ML Model Registry …")
    metrics: dict[str, Any] = {}
    try:
        ml_client.jobs.download(
            name=returned_job.name,
            output_name="eval_report",
            download_path="./azure-ml-outputs",
        )
        import glob as _glob
        report_files = _glob.glob("./azure-ml-outputs/**/eval_report*", recursive=True)
        if report_files:
            with open(report_files[0]) as fh:
                metrics = json.load(fh)
    except Exception as exc:
        print(f"  Could not download eval report ({exc}); registering without metrics.")

    register_model(ml_client, returned_job.name, metrics)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully.")
    print(f"View results at: {returned_job.studio_url}")
    print("=" * 60)


if __name__ == "__main__":
    main()
