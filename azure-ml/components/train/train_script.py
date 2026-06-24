"""
Azure ML training component script – wraps train_credit_risk_model.py logic
for Azure ML component I/O.

Inputs  (via CLI args):
  --scored_parquet_path   URI folder: PySpark txn_scored Parquet directory
Outputs (via CLI args):
  --model_dir             URI folder: xgboost_model.json + features.json
  --metrics_json          URI file:   JSON with all CV and test-set metrics
  --shap_plot             URI file:   SHAP beeswarm summary PNG
  --lime_html             URI file:   LIME explanations HTML

Azure ML automatically sets MLFLOW_TRACKING_URI when running on compute,
so mlflow.log_* calls are directed to the Azure ML experiment server.
"""
import argparse
import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

import lime
import lime.lime_tabular
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
N_FOLDS         = 5
TEST_SIZE        = 0.20
RANDOM_STATE     = 42

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


def load_parquet_dir(path: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {path!r}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].copy()
    for col in ("is_weekend", "is_offhours", "involves_high_risk_country"):
        X[col] = X[col].astype(float)
    return X.astype(float).values


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    stat, _ = ks_2samp(y_score[y_true == 1], y_score[y_true == 0])
    return float(stat)


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit risk XGBoost model")
    parser.add_argument("--scored_parquet_path", required=True)
    parser.add_argument("--model_dir",           required=True)
    parser.add_argument("--metrics_json",        required=True)
    parser.add_argument("--shap_plot",           required=True)
    parser.add_argument("--lime_html",           required=True)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    for p in (args.metrics_json, args.shap_plot, args.lime_html):
        _ensure_parent(p)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading data from {args.scored_parquet_path!r} …")
    df = load_parquet_dir(args.scored_parquet_path)
    print(f"  {len(df):,} rows · {df.shape[1]} columns")

    X_all = prepare_features(df)
    y_all = (df["risk_score"] >= LABEL_THRESHOLD).astype(int).values
    print(f"  Label=1 (high-risk): {y_all.sum():,} ({y_all.mean():.1%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
    )

    # ── Stratified 5-fold CV ───────────────────────────────────────────────────
    print(f"\n{N_FOLDS}-fold stratified CV …")
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

    # ── Final model ────────────────────────────────────────────────────────────
    print("\nTraining final model …")
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
    cm        = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "cv_mean_auc":     cv_mean,
        "cv_std_auc":      cv_std,
        "cv_gini":         2 * cv_mean - 1,
        "test_auc":        test_auc,
        "test_gini":       test_gini,
        "test_ks":         test_ks,
        "precision":       prec,
        "recall":          rec,
        "f1":              f1,
        "confusion_matrix": cm,
        "n_train":         int(len(X_train)),
        "n_test":          int(len(X_test)),
        "pos_rate_train":  float(y_train.mean()),
    }

    print(f"\n  Test AUC={test_auc:.4f}  Gini={test_gini:.4f}  KS={test_ks:.4f}")
    print(f"  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

    # ── MLflow logging (Azure ML auto-configures tracking URI on compute) ──────
    with mlflow.start_run():
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
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")

    # ── Save model artifact for downstream evaluate component ─────────────────
    model.save_model(os.path.join(args.model_dir, "xgboost_model.json"))
    with open(os.path.join(args.model_dir, "features.json"), "w") as fh:
        json.dump({"features": FEATURES, "label_threshold": LABEL_THRESHOLD}, fh, indent=2)
    print(f"Model saved → {args.model_dir}")

    # ── Save metrics JSON ──────────────────────────────────────────────────────
    with open(args.metrics_json, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Metrics saved → {args.metrics_json}")

    # ── SHAP beeswarm plot ─────────────────────────────────────────────────────
    print("Generating SHAP summary …")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(sv, X_test, feature_names=FEATURES, show=False, plot_size=(10, 6))
    plt.savefig(args.shap_plot, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"SHAP plot saved → {args.shap_plot}")

    # ── LIME explanations (top-3 highest-risk predictions) ────────────────────
    print("Generating LIME explanations …")
    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=FEATURES,
        class_names=["Low Risk", "High Risk"],
        mode="classification",
        random_state=RANDOM_STATE,
    )
    top3 = np.argsort(y_prob)[-3:][::-1]
    html_parts = [
        "<html><body>",
        "<h1>LIME Explanations – Top-3 Highest-Risk Predictions</h1>",
    ]
    for rank, idx in enumerate(top3, 1):
        exp = lime_exp.explain_instance(
            X_test[idx], model.predict_proba, num_features=len(FEATURES)
        )
        html_parts.append(
            f"<h2>Rank {rank} &nbsp;·&nbsp; P(High Risk) = {y_prob[idx]:.4f}</h2>"
        )
        html_parts.append(exp.as_html())
        html_parts.append("<hr/>")
    html_parts.append("</body></html>")
    with open(args.lime_html, "w", encoding="utf-8") as fh:
        fh.write("\n".join(html_parts))
    print(f"LIME HTML saved → {args.lime_html}")

    print("\nTraining component complete.")


if __name__ == "__main__":
    main()
