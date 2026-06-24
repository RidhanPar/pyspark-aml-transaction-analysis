"""
PySpark AML Transaction Monitoring – Interactive Dashboard
Generates 2,000 synthetic transactions in-browser.
No PySpark, no data files, no cloud credentials required.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
import xgboost as xgb
from scipy.stats import ks_2samp
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AML Transaction Monitor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Pipeline constants (mirror run_typology_detection.py exactly) ──────────────
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
HIGH_RISK = {"CY", "MT", "PAN", "BVI", "SCH"}
WEIGHTS = {
    "flag_structuring":       30,
    "flag_high_velocity":     25,
    "flag_high_risk_country": 20,
    "flag_amount_spike":      15,
    "flag_rapid_succession":  10,
}
LABEL_THRESHOLD = 25
RANDOM_STATE    = 42
TIER_ORDER      = ["HIGH", "MEDIUM", "LOW", "NONE"]
TIER_COLORS     = {
    "HIGH":   "#ef4444",
    "MEDIUM": "#f97316",
    "LOW":    "#eab308",
    "NONE":   "#6b7280",
}


# ── Synthetic data generation (PySpark logic translated to pandas) ─────────────
@st.cache_data(show_spinner="Simulating AML pipeline output …")
def build_scored_transactions(
    n_transactions: int = 2_000,
    n_customers: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    N   = n_transactions

    cids = [f"CUST_{i:04d}" for i in range(n_customers)]
    cust = rng.choice(cids, size=N)

    base = pd.Timestamp("2024-01-01")
    ts   = pd.to_datetime(
        base.value + rng.integers(0, 90 * 24 * 3600 * int(1e9), N)
    )

    # Amounts: log-normal with injected structuring candidates
    amounts = np.exp(rng.normal(6.5, 1.2, N)).clip(10, 50_000)
    sm = rng.random(N) < 0.04
    amounts[sm] = rng.uniform(9_000, 9_999.99, int(sm.sum()))

    countries = ["DE", "FR", "NL", "GB", "US", "CY", "MT", "PAN", "BVI", "SCH"]
    cw        = [.25, .20, .15, .12, .10, .04, .04, .04, .03, .03]
    orig      = rng.choice(countries, N, p=cw)
    bene      = rng.choice(countries, N, p=cw)

    channels = ["Online", "ATM", "Branch", "Wire", "Mobile"]
    cw2      = [.35, .25, .15, .15, .10]

    df = pd.DataFrame({
        "transaction_id": [f"TXN_{i:06d}" for i in range(N)],
        "customer_id":    cust,
        "timestamp":      ts,
        "amount":         amounts,
        "currency":       "EUR",
        "channel":        rng.choice(channels, N, p=cw2),
        "originator_country":  orig,
        "beneficiary_country": bene,
        "involves_high_risk_country": np.array(
            [o in HIGH_RISK or b in HIGH_RISK for o, b in zip(orig, bene)]
        ),
        "is_weekend":  pd.DatetimeIndex(ts).dayofweek >= 5,
        "is_offhours": (pd.DatetimeIndex(ts).hour < 6)
                     | (pd.DatetimeIndex(ts).hour >= 22),
    })
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    # Rolling 7-day window features – mirrors PySpark rangeBetween(-7*86400, 0)
    parts = []
    for _, grp in df.groupby("customer_id", sort=False):
        grp = grp.set_index("timestamp").sort_index()
        r7  = grp["amount"].rolling("7D")
        grp["rolling_7d_amount"]      = r7.sum()
        grp["rolling_7d_count"]       = r7.count()
        grp["rolling_7d_avg"]         = r7.mean()
        grp["cumulative_amount"]      = grp["amount"].cumsum()
        grp["cumulative_count"]       = np.arange(1, len(grp) + 1, dtype=float)
        grp["seconds_since_last_txn"] = (
            grp.index.to_series().diff().dt.total_seconds()
        )
        parts.append(grp.reset_index())
    df = pd.concat(parts, ignore_index=True)

    df["amount_vs_7d_avg_ratio"] = (
        df["amount"] / df["rolling_7d_avg"].replace(0.0, np.nan)
    )

    # Typology flags
    df["flag_structuring"] = (
        df["amount"].between(9_000, 9_999.99) & (df["rolling_7d_count"] >= 3)
    )
    df["flag_high_velocity"] = (
        (df["rolling_7d_count"] >= 5) & (df["rolling_7d_amount"] >= 20_000)
    )
    df["flag_high_risk_country"] = df["involves_high_risk_country"].astype(bool)
    df["flag_amount_spike"] = (
        (df["amount_vs_7d_avg_ratio"] >= 3.0)
        & df["amount_vs_7d_avg_ratio"].notna()
    )
    df["flag_rapid_succession"] = (
        (df["seconds_since_last_txn"] <= 300)
        & df["seconds_since_last_txn"].notna()
    )

    df["flag_count"] = sum(df[f].astype(int) for f in WEIGHTS)
    df["risk_score"] = sum(df[f].astype(int) * w for f, w in WEIGHTS.items())

    df["risk_tier"] = "NONE"
    df.loc[df["risk_score"] >  0,  "risk_tier"] = "LOW"
    df.loc[df["risk_score"] >= 25, "risk_tier"] = "MEDIUM"
    df.loc[df["risk_score"] >= 50, "risk_tier"] = "HIGH"

    return df


# ── Model training + explainability ───────────────────────────────────────────
@st.cache_data(show_spinner="Training XGBoost credit risk model …")
def train_credit_risk_model(df: pd.DataFrame) -> dict:
    X = df[FEATURES].copy()
    for col in ("is_weekend", "is_offhours", "involves_high_risk_country"):
        X[col] = X[col].astype(float)
    X = X.astype(float).values
    y = (df["risk_score"] >= LABEL_THRESHOLD).astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc   = float(roc_auc_score(y_te, y_prob))
    ks, _ = ks_2samp(y_prob[y_te == 1], y_prob[y_te == 0])
    fpr, tpr, _ = roc_curve(y_te, y_prob)

    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(X_te)
    sv          = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    mean_shap   = np.abs(sv).mean(axis=0)

    return dict(
        auc=auc,
        gini=float(2 * auc - 1),
        ks=float(ks),
        precision=float(precision_score(y_te, y_pred, zero_division=0)),
        recall=float(recall_score(y_te, y_pred, zero_division=0)),
        f1=float(f1_score(y_te, y_pred, zero_division=0)),
        fpr=fpr, tpr=tpr,
        cm=confusion_matrix(y_te, y_pred),
        mean_shap=mean_shap,
        n_train=int(len(X_tr)),
        n_test=int(len(X_te)),
    )


# ── Tab 1: Pipeline Overview ───────────────────────────────────────────────────
def render_overview(df: pd.DataFrame) -> None:
    n_total  = len(df)
    n_alerts = int((df["risk_score"] >= LABEL_THRESHOLD).sum())
    n_high   = int((df["risk_tier"] == "HIGH").sum())
    n_cust   = df["customer_id"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions",     f"{n_total:,}")
    c2.metric("Flagged Alerts (≥ 25)",  f"{n_alerts:,}",
              f"{n_alerts / n_total:.1%} alert rate")
    c3.metric("HIGH Risk Transactions", f"{n_high:,}",
              f"{n_high / n_total:.1%}")
    c4.metric("Unique Customers",       f"{n_cust:,}")

    st.markdown("---")
    st.subheader("Pipeline Steps")
    steps = pd.DataFrame({
        "Step": ["1", "2", "3", "4"],
        "Script": [
            "ingest_transactions.py",
            "run_typology_detection.py",
            "export_to_parquet.py",
            "train_credit_risk_model.py",
        ],
        "Description": [
            "Generate 10 k synthetic transactions, run DQ checks (nulls / dupes / negatives), write raw Parquet",
            "Rolling-window features, 5 AML typology flags, weighted risk scoring (HIGH / MEDIUM / LOW / NONE)",
            "Filter alert queue (score ≥ 25), export alert_queue/ and customer_risk_profiles/ Parquet",
            "XGBoost binary classifier, SHAP + LIME explainability, MLflow logging, Azure ML pipeline",
        ],
        "Orchestrator": ["Airflow DAG", "Airflow DAG", "Airflow DAG", "Airflow DAG"],
    })
    st.dataframe(steps, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("AML Typologies Detected")
    typo = pd.DataFrame({
        "Typology":   ["Structuring", "High-Velocity Layering", "High-Risk Country", "Amount Spike", "Rapid Succession"],
        "Rule":       [
            "Amount 9,000–9,999 AND ≥ 3 txns in 7 days",
            "≥ 5 txns AND ≥ €20 k volume in 7 days",
            "Originator or beneficiary in CY / MT / PAN / BVI / SCH",
            "Current amount ≥ 3× customer 7-day average",
            "< 5 minutes since previous transaction",
        ],
        "Weight": [30, 25, 20, 15, 10],
    })
    st.dataframe(typo, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Tech Stack")
    badges = [
        ("PySpark 3.5",       "#E25A1C"),
        ("Apache Airflow 2.9","#017CEE"),
        ("XGBoost",           "#189AB4"),
        ("MLflow",            "#0194E2"),
        ("Azure ML SDK v2",   "#0078D4"),
        ("SHAP",              "#FF7514"),
        ("LIME",              "#6B21A8"),
        ("BigQuery",          "#4285F4"),
        ("Docker",            "#2496ED"),
        ("Streamlit",         "#FF4B4B"),
    ]
    st.markdown(
        " &nbsp; ".join(
            f'<span style="background:{c};color:white;padding:5px 12px;'
            f'border-radius:14px;font-size:13px;font-weight:600;">{t}</span>'
            for t, c in badges
        ),
        unsafe_allow_html=True,
    )


# ── Tab 2: AML Monitoring ──────────────────────────────────────────────────────
def render_monitoring(df: pd.DataFrame) -> None:
    col_l, col_r = st.columns([4, 6])

    # Risk tier donut
    with col_l:
        st.subheader("Risk Tier Distribution")
        tier_vc = df["risk_tier"].value_counts().reindex(TIER_ORDER, fill_value=0)
        fig_pie = go.Figure(go.Pie(
            labels=tier_vc.index.tolist(),
            values=tier_vc.values.tolist(),
            hole=0.52,
            marker_colors=[TIER_COLORS[t] for t in tier_vc.index],
            textinfo="label+percent",
            hovertemplate="%{label}<br>%{value:,} transactions<extra></extra>",
        ))
        fig_pie.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=320,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Typology flag counts
    with col_r:
        st.subheader("Typology Flag Breakdown")
        flag_data = {
            "Structuring":       int(df["flag_structuring"].sum()),
            "High Velocity":     int(df["flag_high_velocity"].sum()),
            "High-Risk Country": int(df["flag_high_risk_country"].sum()),
            "Amount Spike":      int(df["flag_amount_spike"].sum()),
            "Rapid Succession":  int(df["flag_rapid_succession"].sum()),
        }
        flag_df = (
            pd.DataFrame({"Typology": list(flag_data), "Count": list(flag_data.values())})
            .sort_values("Count")
        )
        fig_bar = px.bar(
            flag_df, x="Count", y="Typology", orientation="h",
            color="Count", color_continuous_scale="Oranges",
            text="Count",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            coloraxis_showscale=False,
            yaxis_title="",
            xaxis_title="Transactions flagged",
            margin=dict(t=10, b=10, l=10, r=50),
            height=320,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # Daily alert volume time-series
    st.subheader("Daily Alert Volume")
    alert_df          = df[df["risk_score"] >= LABEL_THRESHOLD].copy()
    alert_df["date"]  = alert_df["timestamp"].dt.normalize()
    daily = (
        alert_df.groupby(["date", "risk_tier"])
        .size()
        .reset_index(name="count")
    )
    fig_ts = px.bar(
        daily, x="date", y="count", color="risk_tier",
        color_discrete_map=TIER_COLORS,
        labels={"date": "Date", "count": "Alerts", "risk_tier": "Risk Tier"},
        category_orders={"risk_tier": TIER_ORDER},
        barmode="stack",
    )
    fig_ts.update_layout(
        legend_title_text="Risk Tier",
        margin=dict(t=10, b=10),
        height=300,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")

    # Risk score histogram
    st.subheader("Risk Score Distribution")
    fig_hist = px.histogram(
        df[df["risk_score"] > 0],
        x="risk_score", nbins=40,
        color="risk_tier",
        color_discrete_map=TIER_COLORS,
        labels={"risk_score": "Risk Score", "count": "Transactions"},
        category_orders={"risk_tier": TIER_ORDER},
        opacity=0.85,
    )
    fig_hist.add_vline(
        x=LABEL_THRESHOLD, line_dash="dash", line_color="white",
        annotation_text="Alert threshold (25)",
        annotation_position="top right",
    )
    fig_hist.update_layout(
        bargap=0.05,
        legend_title_text="Risk Tier",
        margin=dict(t=10, b=10),
        height=300,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # Top flagged transactions table
    st.subheader("Top Flagged Transactions (highest risk score)")
    cols_show = [
        "transaction_id", "customer_id", "timestamp", "amount", "channel",
        "originator_country", "beneficiary_country",
        "risk_score", "risk_tier", "flag_count",
    ]
    alert_table = (
        df[df["risk_score"] >= LABEL_THRESHOLD][cols_show]
        .sort_values(["risk_score", "amount"], ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    alert_table["amount"]    = alert_table["amount"].map("€{:,.2f}".format)
    alert_table["timestamp"] = alert_table["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(alert_table, hide_index=True, use_container_width=True)


# ── Tab 3: Credit Risk Model ───────────────────────────────────────────────────
def render_model(df: pd.DataFrame, m: dict) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("ROC-AUC",   f"{m['auc']:.4f}")
    c2.metric("Gini",      f"{m['gini']:.4f}")
    c3.metric("KS Stat",   f"{m['ks']:.4f}")
    c4.metric("Precision", f"{m['precision']:.4f}")
    c5.metric("Recall",    f"{m['recall']:.4f}")
    c6.metric("F1 Score",  f"{m['f1']:.4f}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("ROC Curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=m["fpr"], y=m["tpr"], mode="lines",
            name=f"XGBoost  (AUC = {m['auc']:.4f})",
            line=dict(color="#14b8a6", width=2.5),
            fill="tozeroy", fillcolor="rgba(20,184,166,0.08)",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random classifier",
            line=dict(color="#94a3b8", width=1.5, dash="dash"),
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.42, y=0.1),
            margin=dict(t=10, b=40, l=40, r=10),
            height=370,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_r:
        st.subheader("SHAP Feature Importance  (mean |SHAP value|)")
        shap_df = (
            pd.DataFrame({"Feature": FEATURES, "MeanSHAP": m["mean_shap"]})
            .sort_values("MeanSHAP", ascending=True)
        )
        fig_shap = px.bar(
            shap_df, x="MeanSHAP", y="Feature", orientation="h",
            color="MeanSHAP", color_continuous_scale="Teal",
            text=shap_df["MeanSHAP"].map("{:.4f}".format),
        )
        fig_shap.update_traces(textposition="outside")
        fig_shap.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Mean |SHAP value|",
            yaxis_title="",
            margin=dict(t=10, b=40, l=10, r=60),
            height=370,
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("---")
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.subheader("Confusion Matrix")
        cm_labels = ["Low Risk (0)", "High Risk (1)"]
        fig_cm = px.imshow(
            m["cm"],
            x=cm_labels, y=cm_labels,
            text_auto=True,
            color_continuous_scale="RdYlGn",
            aspect="auto",
        )
        fig_cm.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            coloraxis_showscale=False,
            margin=dict(t=10, b=40, l=50, r=10),
            height=320,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_r2:
        st.subheader("Model Configuration")
        cfg = {
            "Algorithm":        "XGBoost  binary:logistic",
            "n_estimators":     "300",
            "max_depth":        "5",
            "learning_rate":    "0.05",
            "subsample":        "0.8",
            "colsample_bytree": "0.8",
            "CV strategy":      "Stratified 5-fold",
            "Test split":       "80 / 20 (stratified)",
            "Label threshold":  f"risk_score ≥ {LABEL_THRESHOLD}",
            "Explainability":   "SHAP TreeExplainer + LIME",
            "MLflow tracking":  "Local SQLite + Azure ML",
            "Train samples":    f"{m['n_train']:,}",
            "Test samples":     f"{m['n_test']:,}",
        }
        cfg_df = pd.DataFrame.from_dict(
            cfg, orient="index", columns=["Value"]
        )
        cfg_df.index.name = "Parameter"
        st.dataframe(cfg_df, use_container_width=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🏦 AML Transaction Monitor")
        st.caption("PySpark · Airflow · XGBoost · Azure ML")
        st.markdown("---")
        st.markdown(
            "**Author:** Ridhan Parvendhan  \n"
            "[![GitHub](https://img.shields.io/badge/GitHub-RidhanPar-181717?"
            "logo=github&style=flat)](https://github.com/RidhanPar/"
            "pyspark-aml-transaction-analysis)"
        )
        st.markdown("---")
        st.markdown(
            "**About**  \n"
            "End-to-end Anti-Money Laundering post-transaction analysis "
            "pipeline — PySpark feature engineering, Airflow orchestration, "
            "five AML typology rules, XGBoost credit risk classification with "
            "SHAP / LIME explainability, MLflow tracking, and an Azure ML SDK "
            "v2 cloud pipeline."
        )
        st.markdown("---")
        st.info(
            "⚡ All data shown is **synthetic** — generated deterministically "
            "from a fixed seed. No real transaction data is used."
        )
        st.markdown("---")
        st.caption("Simulation period: Jan – Mar 2024")


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    render_sidebar()

    st.title("AML Transaction Monitoring Dashboard")
    st.caption(
        "End-to-end pipeline demo  ·  PySpark  ·  Apache Airflow  ·  "
        "XGBoost  ·  MLflow  ·  Azure ML  ·  SHAP"
    )
    st.markdown("---")

    df = build_scored_transactions()
    m  = train_credit_risk_model(df)

    tab1, tab2, tab3 = st.tabs([
        "📊  Pipeline Overview",
        "🚨  AML Monitoring",
        "🤖  Credit Risk Model",
    ])
    with tab1:
        render_overview(df)
    with tab2:
        render_monitoring(df)
    with tab3:
        render_model(df, m)


main()
