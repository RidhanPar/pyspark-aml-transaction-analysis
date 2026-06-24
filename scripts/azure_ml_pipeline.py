"""
scripts/azure_ml_pipeline.py

Azure ML SDK v2 pipeline for the credit risk XGBoost classifier.

Goal: pipeline authoring and Azure ML experiment-tracking pattern —
not a live Azure submission during portfolio development.

LOCAL_DEV_MODE = True  (default)
    Skips MLClient authentication entirely.
    Calls build_pipeline() and prints the serialised pipeline component
    definition (yaml.dump) to stdout, then exits with an instructions message.

LOCAL_DEV_MODE = False
    Authenticates with DefaultAzureCredential.
    Instantiates MLClient from environment variables.
    Assembles and submits the pipeline job; prints the run URL.

Environment variables required for live submission:
    AZURE_SUBSCRIPTION_ID
    AZURE_RESOURCE_GROUP
    AZURE_WORKSPACE_NAME
"""

from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
import mlflow
import yaml
import os

# ── Mode switch ────────────────────────────────────────────────────────────────
LOCAL_DEV_MODE = True

# ── Constants ──────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "credit_risk_azure_pipeline"
AML_MODEL_NAME    = "credit_risk_xgboost_azure"
AML_ENVIRONMENT   = "azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:latest"
SCRIPTS_DIR       = os.path.dirname(os.path.abspath(__file__))

# ── CommandComponent definition ────────────────────────────────────────────────
# Defined at module level so it is reusable in both the LOCAL_DEV_MODE path
# (serialisation) and the live submission path (pipeline step call).
credit_risk_training_component = command(
    name="credit_risk_training",
    display_name="XGBoost Credit Risk Classifier",
    description=(
        "XGBoost with stratified 5-fold CV, SHAP beeswarm, LIME top-3, "
        "MLflow tracking, Model Registry → Staging"
    ),
    inputs={
        "script":     Input(type="uri_file"),
        "train_data": Input(type="uri_file"),
    },
    outputs={
        "model_dir": Output(type="uri_folder"),
    },
    command="python ${{inputs.script}} --output-dir ${{outputs.model_dir}}",
    environment=AML_ENVIRONMENT,
)


# ── Pipeline assembly ──────────────────────────────────────────────────────────

def build_pipeline() -> dict:
    """
    Assembles the 3-step credit risk pipeline and returns a plain dict
    that can be serialised directly with yaml.dump.

    Step 1  data_prep      – feature engineering + train/test split
    Step 2  train_model    – XGBoost component; logs ROC-AUC / Gini / KS
                             as MLflow run metrics under MLFLOW_EXPERIMENT
    Step 3  register_model – mlflow.register_model() → AML_MODEL_NAME / Staging
    """

    # ── Step 1: data_prep ──────────────────────────────────────────────────────
    step_data_prep = {
        "display_name": "Data Preparation",
        "description": (
            "Load scored-transaction Parquet from AML_SCORED_PATH. "
            "Cast boolean feature columns (is_weekend, is_offhours, "
            "involves_high_risk_country) to float. Apply stratified 80/20 "
            "train-test split with random_state=42. Write the processed "
            "feature matrix as Parquet to the step output directory."
        ),
        "inputs": {
            "scored_parquet":  {"type": "uri_folder", "path": "azureml:aml_txn_scored:1"},
            "label_threshold": {"type": "integer",    "default": 25},
            "test_size":       {"type": "number",     "default": 0.20},
            "random_state":    {"type": "integer",    "default": 42},
        },
        "outputs": {
            "processed_parquet": {"type": "uri_folder"},
        },
        "environment": AML_ENVIRONMENT,
        "command": (
            "python scripts/run_typology_detection.py"
            " --scored-path ${{inputs.scored_parquet}}"
            " --output-path ${{outputs.processed_parquet}}"
        ),
    }

    # ── Step 2: train_model ────────────────────────────────────────────────────
    # Azure ML auto-sets MLFLOW_TRACKING_URI on compute; mlflow.log_metric()
    # calls inside train_credit_risk_model.py are directed to the workspace
    # experiment server under MLFLOW_EXPERIMENT.
    step_train_model = {
        "display_name": credit_risk_training_component.display_name,
        "component":    credit_risk_training_component.name,
        "description":  credit_risk_training_component.description,
        "inputs": {
            "script": {
                "type": "uri_file",
                "path": os.path.join(SCRIPTS_DIR, "train_credit_risk_model.py"),
            },
            "train_data": {
                "type":   "uri_file",
                "source": "${{parent.steps.data_prep.outputs.processed_parquet}}",
            },
        },
        "outputs": {
            "model_dir": {
                "type":        "uri_folder",
                "destination": "${{parent.outputs.model_dir}}",
            },
        },
        "command":     credit_risk_training_component.command,
        "environment": AML_ENVIRONMENT,
        "mlflow": {
            "experiment":      MLFLOW_EXPERIMENT,
            "tracked_metrics": [
                "cv_mean_auc", "cv_std_auc", "cv_gini",
                "test_auc", "test_gini", "test_ks",
                "precision", "recall", "f1",
            ],
            "note": (
                "MLFLOW_TRACKING_URI is injected automatically by Azure ML "
                "at job runtime; no explicit set_tracking_uri() is needed "
                "inside the component script."
            ),
        },
    }

    # ── Step 3: register_model ─────────────────────────────────────────────────
    # mlflow.register_model() is called with the run URI produced by step 2,
    # pushing the trained artifact to the Azure ML Model Registry.
    step_register_model = {
        "display_name": "Register Model in Azure ML Model Registry",
        "description": (
            f"Call mlflow.register_model() with the MLflow run URI from "
            f"train_model to push the XGBoost artifact to the Azure ML "
            f"Model Registry as '{AML_MODEL_NAME}', then transition the "
            f"version to Staging via MlflowClient."
        ),
        "inputs": {
            "model_dir": {
                "type":   "uri_folder",
                "source": "${{parent.steps.train_model.outputs.model_dir}}",
            },
            "mlflow_run_id": {
                "type":   "string",
                "source": "${{parent.steps.train_model.mlflow_run_id}}",
            },
        },
        "outputs": {
            "registered_model_version": {"type": "string"},
        },
        "environment": AML_ENVIRONMENT,
        "command": (
            "python -c \""
            "import mlflow; "
            "from mlflow.tracking import MlflowClient; "
            "mv = mlflow.register_model("
            "    model_uri=\\\"runs:/${{inputs.mlflow_run_id}}/model\\\", "
            f"    name=\\\"{AML_MODEL_NAME}\\\"); "
            f"MlflowClient().transition_model_version_stage("
            f"    name=\\\"{AML_MODEL_NAME}\\\", "
            "    version=mv.version, "
            "    stage=\\\"Staging\\\")"
            "\""
        ),
        "registry_target": {
            "model_name":       AML_MODEL_NAME,
            "transition_stage": "Staging",
        },
    }

    return {
        "pipeline_name": "credit_risk_azure_pipeline",
        "pipeline_description": (
            "End-to-end credit risk classification: data preparation "
            "→ XGBoost training + MLflow tracking "
            "→ Azure ML Model Registry registration."
        ),
        "mlflow_experiment":   MLFLOW_EXPERIMENT,
        "default_compute":     "cpu-cluster",
        "default_environment": AML_ENVIRONMENT,

        # Reusable component registered in the workspace
        "component_registry": {
            "credit_risk_training": {
                "display_name": credit_risk_training_component.display_name,
                "description":  credit_risk_training_component.description,
                "type":         "command",
                "inputs": {
                    "script":     {"type": "uri_file"},
                    "train_data": {"type": "uri_file"},
                },
                "outputs": {
                    "model_dir": {"type": "uri_folder"},
                },
                "command":     credit_risk_training_component.command,
                "environment": AML_ENVIRONMENT,
            },
        },

        # Pipeline input / output interface
        "pipeline_inputs": {
            "scored_parquet": {
                "type":        "uri_folder",
                "description": (
                    "PySpark txn_scored Parquet directory "
                    "(data asset: aml_txn_scored)"
                ),
            },
        },
        "pipeline_outputs": {
            "model_dir": {
                "type":        "uri_folder",
                "description": "Trained XGBoost model artifact",
            },
        },

        # Ordered step graph
        "pipeline_steps": {
            "1_data_prep":      step_data_prep,
            "2_train_model":    step_train_model,
            "3_register_model": step_register_model,
        },
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if LOCAL_DEV_MODE:
    # ── LOCAL mode: serialise and print — no Azure connection required ─────────
    pipeline_def = build_pipeline()
    print(yaml.dump(pipeline_def, default_flow_style=False, sort_keys=False))
    print(
        "Pipeline defined. Set LOCAL_DEV_MODE=False and configure "
        "AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME "
        "to submit to Azure ML."
    )

else:
    # ── LIVE mode: authenticate, create MLClient, submit ──────────────────────
    from azure.ai.ml.dsl import pipeline as aml_pipeline  # lazy import

    # Authenticate — picks up az login session, managed identity, or
    # AZURE_CLIENT_ID / AZURE_CLIENT_SECRET / AZURE_TENANT_ID env vars.
    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
    )

    # Inline components for the two non-reusable steps
    _data_prep_component = command(
        name="data_prep",
        display_name="Data Preparation",
        inputs={"scored_parquet": Input(type="uri_folder")},
        outputs={"processed_parquet": Output(type="uri_folder")},
        command=(
            "python scripts/run_typology_detection.py"
            " --scored-path ${{inputs.scored_parquet}}"
            " --output-path ${{outputs.processed_parquet}}"
        ),
        environment=AML_ENVIRONMENT,
    )

    _register_component = command(
        name="register_model",
        display_name="Register Model in Azure ML Model Registry",
        inputs={"model_dir": Input(type="uri_folder")},
        outputs={"registered_model_version": Output(type="string")},
        command=(
            "python -c \""
            "import mlflow, os; "
            "from mlflow.tracking import MlflowClient; "
            "run_id = os.environ.get('MLFLOW_RUN_ID', ''); "
            f"mv = mlflow.register_model(model_uri=f'runs:/{{run_id}}/model', name='{AML_MODEL_NAME}'); "
            f"MlflowClient().transition_model_version_stage(name='{AML_MODEL_NAME}', version=mv.version, stage='Staging')"
            "\""
        ),
        environment=AML_ENVIRONMENT,
    )

    @aml_pipeline(
        name="credit_risk_azure_pipeline",
        description=(
            "End-to-end credit risk classification: data preparation "
            "→ XGBoost training + MLflow tracking "
            "→ Azure ML Model Registry registration."
        ),
        experiment_name=MLFLOW_EXPERIMENT,
    )
    def credit_risk_pipeline(
        scored_parquet: Input(type="uri_folder"),  # type: ignore[valid-type]
    ):
        data_prep_step = _data_prep_component(scored_parquet=scored_parquet)

        train_step = credit_risk_training_component(
            script=Input(
                path=os.path.join(SCRIPTS_DIR, "train_credit_risk_model.py"),
                type="uri_file",
            ),
            train_data=data_prep_step.outputs.processed_parquet,
        )

        _register_component(model_dir=train_step.outputs.model_dir)

        return {"model_dir": train_step.outputs.model_dir}

    pipeline_job = credit_risk_pipeline(
        scored_parquet=Input(path="azureml:aml_txn_scored:1", type="uri_folder")
    )
    pipeline_job.settings.default_compute = "cpu-cluster"

    returned_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name=MLFLOW_EXPERIMENT,
    )
    print(f"Pipeline submitted: {returned_job.name}")
    print(f"Studio URL:         {returned_job.studio_url}")
