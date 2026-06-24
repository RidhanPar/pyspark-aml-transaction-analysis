"""
Step 4 – Credit Risk ML Model: XGBoost binary classifier on AML-scored transactions.

Input:  data/processed/txn_scored/   (Parquet, written by run_typology_detection.py)
Output: output/shap_summary.png
        output/lime_explanations.html
        MLflow experiment "credit_risk_classification" + Model Registry "credit_risk_xgboost"
"""
import glob
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import mlflow
import mlflow.xgboost
import shap
import xgboost as xgb
from mlflow.tracking import MlflowClient
from scipy.stats import ks_2samp
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
SCORED_PATH = os.environ.get("AML_SCORED_PATH", "/opt/airflow/data/processed/txn_scored")
OUTPUT_DIR  = os.environ.get("AML_OUTPUT_DIR",  "/opt/airflow/output")
MLFLOW_URI  = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

FEATURES = [
    "rolling_7d_amount",
    "rolling_7d_count",
    "rolling_7d_avg",
    "amount_vs_7d_avg_ratio",
    "seconds_since_last_txn",
    "is_weekend",
    "is_offhours",
    "involves_high_risk_country",
    "cumulative_amount",
    "cumulative_count",
]

LABEL_THRESHOLD = 25
MODEL_NAME      = "credit_risk_xgboost"
N_FOLDS         = 5
TEST_SIZE       = 0.20
RANDOM_STATE    = 42

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_parquet_dir(path: str) -> pd.DataFrame:
    """Concatenate all Parquet part-files produced by a PySpark coalesce write."""
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {path!r}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Return float feature matrix; booleans cast to 0/1, NaN preserved for XGBoost."""
    X = df[FEATURES].copy()
    for col in ("is_weekend", "is_offhours", "involves_high_risk_country"):
        X[col] = X[col].astype(float)
    return X.astype(float).values


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS = max separation between positive and negative score CDFs."""
    stat, _ = ks_2samp(y_score[y_true == 1], y_score[y_true == 0])
    return float(stat)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading scored transactions from {SCORED_PATH!r} …")
    df = load_parquet_dir(SCORED_PATH)
    print(f"  {len(df):,} rows · {df.shape[1]} columns")

    X_all = prepare_features(df)
    y_all = (df["risk_score"] >= LABEL_THRESHOLD).astype(int).values
    print(f"  Label=1 (high-risk): {y_all.sum():,} ({y_all.mean():.1%})")

    # ── Train / test split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=TEST_SIZE,
        stratify=y_all,
        random_state=RANDOM_STATE,
    )

    # ── Stratified 5-fold cross-validation ────────────────────────────────────
    print(f"\n{N_FOLDS}-fold stratified cross-validation …")
    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_aucs = []
    for fold, (tr, va) in enumerate(skf.split(X_train, y_train), 1):
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(X_train[tr], y_train[tr])
        auc = roc_auc_score(y_train[va], m.predict_proba(X_train[va])[:, 1])
        cv_aucs.append(auc)
        print(f"  Fold {fold}  AUC = {auc:.4f}")

    cv_mean = float(np.mean(cv_aucs))
    cv_std  = float(np.std(cv_aucs))
    cv_gini = 2 * cv_mean - 1
    print(f"\n  CV  AUC  = {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  CV  Gini = {cv_gini:.4f}")

    # ── Final model on full training set ──────────────────────────────────────
    print("\nTraining final model on full train set …")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    test_auc  = float(roc_auc_score(y_test, y_prob))
    test_gini = 2 * test_auc - 1
    test_ks   = ks_statistic(y_test, y_prob)
    prec      = float(precision_score(y_test, y_pred, zero_division=0))
    rec       = float(recall_score(y_test, y_pred, zero_division=0))
    f1        = float(f1_score(y_test, y_pred, zero_division=0))
    cm        = confusion_matrix(y_test, y_pred)

    print(f"\n  Test AUC   = {test_auc:.4f}")
    print(f"  Test Gini  = {test_gini:.4f}")
    print(f"  Test KS    = {test_ks:.4f}")
    print(f"  Precision  = {prec:.4f}")
    print(f"  Recall     = {rec:.4f}")
    print(f"  F1         = {f1:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print()
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

    # ── SHAP beeswarm summary plot ─────────────────────────────────────────────
    print("Generating SHAP summary plot …")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # Handle both SHAP < 0.40 (list per class) and >= 0.40 (single array)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap.summary_plot(sv, X_test, feature_names=FEATURES, show=False, plot_size=(10, 6))
    shap_path = os.path.join(OUTPUT_DIR, "shap_summary.png")
    plt.savefig(shap_path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  Saved → {shap_path}")

    # ── LIME explanations for top-3 highest-risk predictions ──────────────────
    print("Generating LIME explanations …")
    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=FEATURES,
        class_names=["Low Risk", "High Risk"],
        mode="classification",
        random_state=RANDOM_STATE,
    )
    top3_idx = np.argsort(y_prob)[-3:][::-1]

    html_parts = [
        "<html><body>",
        "<h1>LIME Explanations – Top-3 Highest-Risk Predictions</h1>",
    ]
    for rank, idx in enumerate(top3_idx, 1):
        exp = lime_exp.explain_instance(
            X_test[idx], model.predict_proba, num_features=len(FEATURES)
        )
        html_parts.append(f"<h2>Rank {rank} &nbsp;·&nbsp; P(High Risk) = {y_prob[idx]:.4f}</h2>")
        html_parts.append(exp.as_html())
        html_parts.append("<hr/>")
    html_parts.append("</body></html>")

    lime_path = os.path.join(OUTPUT_DIR, "lime_explanations.html")
    with open(lime_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(html_parts))
    print(f"  Saved → {lime_path}")

    # ── MLflow logging ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("credit_risk_classification")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "n_estimators":     XGB_PARAMS["n_estimators"],
            "max_depth":        XGB_PARAMS["max_depth"],
            "learning_rate":    XGB_PARAMS["learning_rate"],
            "subsample":        XGB_PARAMS["subsample"],
            "colsample_bytree": XGB_PARAMS["colsample_bytree"],
            "n_folds":          N_FOLDS,
            "test_size":        TEST_SIZE,
            "label_threshold":  LABEL_THRESHOLD,
            "features":         ",".join(FEATURES),
        })
        mlflow.log_metrics({
            "cv_mean_auc": cv_mean,
            "cv_std_auc":  cv_std,
            "cv_gini":     cv_gini,
            "test_auc":    test_auc,
            "test_gini":   test_gini,
            "test_ks":     test_ks,
            "precision":   prec,
            "recall":      rec,
            "f1":          f1,
        })
        mlflow.log_artifact(shap_path, artifact_path="plots")
        mlflow.log_artifact(lime_path, artifact_path="explanations")
        mlflow.xgboost.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    print(f"\nMLflow run_id: {run_id}")

    # ── MLflow Model Registry ──────────────────────────────────────────────────
    client = MlflowClient()
    try:
        client.create_registered_model(MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        pass  # model already exists in registry

    mv = client.create_model_version(
        name=MODEL_NAME,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
    )
    client.set_model_version_tag(MODEL_NAME, mv.version, "auc",  f"{test_auc:.4f}")
    client.set_model_version_tag(MODEL_NAME, mv.version, "gini", f"{test_gini:.4f}")
    client.set_model_version_tag(MODEL_NAME, mv.version, "ks",   f"{test_ks:.4f}")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"Registered '{MODEL_NAME}' v{mv.version} → Staging")


if __name__ == "__main__":
    main()
