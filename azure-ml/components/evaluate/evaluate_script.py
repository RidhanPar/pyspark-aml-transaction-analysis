"""
Azure ML evaluation component script.

Loads the trained XGBoost model from the train component's model_dir output,
re-evaluates on the held-out test set (same 80/20 stratified split + seed as
train_script.py), logs all metrics to the Azure ML experiment via MLflow, and
writes a JSON evaluation report.

Inputs  (via CLI args):
  --model_dir        URI folder: xgboost_model.json + features.json (from train component)
  --scored_parquet   URI folder: PySpark txn_scored Parquet directory
Outputs (via CLI args):
  --eval_report      URI file:   JSON evaluation report with all test-set metrics
"""
import argparse
import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import ks_2samp
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Must match train_script.py exactly so the test split is identical.
TEST_SIZE    = 0.20
RANDOM_STATE = 42


def load_parquet_dir(path: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {path!r}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate credit risk XGBoost model")
    parser.add_argument("--model_dir",      required=True, help="Folder with xgboost_model.json")
    parser.add_argument("--scored_parquet", required=True, help="txn_scored Parquet folder")
    parser.add_argument("--eval_report",    required=True, help="Output path for JSON eval report")
    args = parser.parse_args()

    parent = os.path.dirname(args.eval_report)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # ── Load model + feature metadata ─────────────────────────────────────────
    print(f"Loading model from {args.model_dir!r} …")
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(args.model_dir, "xgboost_model.json"))

    with open(os.path.join(args.model_dir, "features.json")) as fh:
        meta = json.load(fh)
    features         = meta["features"]
    label_threshold  = meta["label_threshold"]

    # ── Load data and reproduce the same test split ────────────────────────────
    print(f"Loading data from {args.scored_parquet!r} …")
    df = load_parquet_dir(args.scored_parquet)

    X = df[features].copy()
    for col in ("is_weekend", "is_offhours", "involves_high_risk_country"):
        X[col] = X[col].astype(float)
    X_all = X.astype(float).values
    y_all = (df["risk_score"] >= label_threshold).astype(int).values

    _, X_test, _, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc   = float(roc_auc_score(y_test, y_prob))
    gini  = 2 * auc - 1
    ks, _ = ks_2samp(y_prob[y_test == 1], y_prob[y_test == 0])
    prec  = float(precision_score(y_test, y_pred, zero_division=0))
    rec   = float(recall_score(y_test, y_pred, zero_division=0))
    f1    = float(f1_score(y_test, y_pred, zero_division=0))
    cm    = confusion_matrix(y_test, y_pred).tolist()

    report = {
        "test_auc":         auc,
        "test_gini":        gini,
        "test_ks":          float(ks),
        "precision":        prec,
        "recall":           rec,
        "f1":               f1,
        "confusion_matrix": cm,
        "n_test":           int(len(y_test)),
        "pos_rate_test":    float(y_test.mean()),
    }

    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))
    print(f"AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}  F1={f1:.4f}")

    # ── Log metrics to Azure ML experiment (via MLflow auto-tracking) ─────────
    with mlflow.start_run():
        for k, v in report.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

    # ── Write evaluation report ────────────────────────────────────────────────
    with open(args.eval_report, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Evaluation report saved → {args.eval_report}")


if __name__ == "__main__":
    main()
