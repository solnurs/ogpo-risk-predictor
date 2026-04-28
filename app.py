"""
OGPO Claim Risk Predictor — Freedom Insurance Kazakhstan
Streamlit application for Week 4 deployment.

Run:  streamlit run app.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OGPO Claim Risk Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load artifacts ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    pipeline      = joblib.load(BASE_DIR / "model.pkl")
    explainer     = joblib.load(BASE_DIR / "shap_explainer.pkl")
    feat_names    = joblib.load(BASE_DIR / "feature_names.pkl")   # SHAP/post-proc names
    feat_medians  = joblib.load(BASE_DIR / "feature_medians.pkl") # raw feature medians (pd.Series)
    return pipeline, explainer, feat_names, feat_medians

pipeline, explainer, feat_names, feat_medians = load_artifacts()

# ─── Helper: build input row ────────────────────────────────────────────────────

REGION_LABELS = {
    1:  "Almaty",       2:  "Astana",       3:  "Shymkent",
    4:  "Karaganda",    5:  "Aktobe",       6:  "Atyrau",
    7:  "Kostanay",     8:  "Pavlodar",     9:  "Semey",
    10: "Oskemen",      11: "Taraz",        12: "Kyzylorda",
    13: "Oral",         14: "Petropavl",    15: "Aktau",
    16: "Taldykorgan",  17: "Zhezkazgan",   18: "Balqash",
    19: "Temirtau",     20: "Other",
}

VEHICLE_TYPE_MAP = {
    "Passenger car": 1,
    "Truck":         2,
    "Motorcycle":    3,
    "Bus":           4,
}


def build_input_row(
    bonus_malus: float,
    experience_years: float,
    n_drivers: int,
    engine_power: float,
    engine_volume: float,
    car_year: int,
    region_id: int,
    vehicle_type: int,
    month: int,
    termination_ratio: float,
    feat_medians: pd.Series,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame in the same feature space as X_train.
    Start from medians, then override user-supplied fields and derived features.

    IMPORTANT scale conversions:
      - bonus_malus: OGPO coefficient 0.50-2.45 → training data stores INTEGER CLASS 0-13
        where class 0 = coef 2.45 (most malus/risky), class 13 = coef 0.50 (most bonus/safest)
        Formula: class = round((2.45 - ogpo_coef) * 13 / 1.95)
      - engine_volume: user inputs Liters → training data stores in CC (×1000)
    """
    row = feat_medians.copy().to_dict()

    # ── Scale conversions ─────────────────────────────────────────────────────
    current_year = 2025  # dataset reference year
    car_age_binary = int((current_year - car_year) > 7)
    # BM: OGPO coefficient → integer class 0-13 (0=riskiest, 13=safest)
    bm_class = int(round((2.45 - bonus_malus) * 13.0 / 1.95))
    bm_class = max(0, min(13, bm_class))
    engine_volume_cc = engine_volume * 1000.0  # Liters → CC

    for key, val in {
        "bonus_malus_mean":      float(bm_class),
        "bonus_malus_max":       float(bm_class),
        "bonus_malus_std":       0.0,
        "experience_year_mean":  experience_years,
        "experience_year_max":   experience_years,
        "experience_year_std":   0.0,
        "n_drivers":             float(n_drivers),
        "engine_power":          engine_power,
        "engine_volume":         engine_volume_cc,
        "car_year":              float(car_year),
        "region_id":             float(region_id),
        "vehicle_type_id":       float(vehicle_type),
        "month":                 float(month),
        "car_age_binary":        float(car_age_binary),
        "termination_ratio":     termination_ratio,
    }.items():
        if key in row:
            row[key] = val

    # ── Derived / interaction features ────────────────────────────────────────
    if "quarter" in row:
        row["quarter"] = float((month - 1) // 3 + 1)
    if "day_of_year" in row:
        row["day_of_year"] = float(month * 30)
    if "is_winter" in row:
        row["is_winter"] = float(month in [12, 1, 2])
    if "month_sin" in row:
        row["month_sin"] = float(np.sin(2 * np.pi * month / 12))
    if "month_cos" in row:
        row["month_cos"] = float(np.cos(2 * np.pi * month / 12))
    # termination_ratio cannot be derived from user inputs (requires premium data)
    # — kept at training median, which is the correct fallback
    if "power_density" in row:
        row["power_density"] = engine_power / (engine_volume_cc + 1e-5)
    if "bm_car_age" in row:
        row["bm_car_age"] = float(bm_class) * car_age_binary
    if "experience_sq" in row:
        row["experience_sq"] = experience_years ** 2

    return pd.DataFrame([row])


def predict_and_explain(input_df: pd.DataFrame):
    """Return (prob_claim, shap_df) where shap_df has columns ['feature','value','shap']."""
    prob = pipeline.predict_proba(input_df)[0, 1]

    # Transform for SHAP
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(input_df)
    X_df = pd.DataFrame(X_transformed, columns=feat_names)

    sv = explainer(X_df)
    # For binary classification: take positive class
    if len(sv.shape) == 3:
        sv_pos = sv[:, :, 1]
    else:
        sv_pos = sv

    shap_series = pd.Series(sv_pos.values[0], index=feat_names, name="shap")
    feature_vals = pd.Series(X_df.iloc[0].values, index=feat_names, name="value")

    shap_df = pd.DataFrame({
        "feature": feat_names,
        "value":   feature_vals.values,
        "shap":    shap_series.values,
    }).sort_values("shap", key=abs, ascending=False)

    return prob, shap_df


# ─── UI: Sidebar inputs ─────────────────────────────────────────────────────────
st.sidebar.title("Policy Inputs")
st.sidebar.markdown("Enter the policy details below:")

# 14 discrete OGPO BM classes: class 0=coef 2.45 (worst) → class 13=coef 0.50 (safest)
_BM_OPTIONS = {
    "0.50  (K10 — max discount, safest)": 0.50,
    "0.65  (K9)": 0.65,
    "0.80  (K8)": 0.80,
    "0.95  (K7 — slight discount)": 0.95,
    "1.10  (K0 — neutral)": 1.10,
    "1.25  (M1 — slight surcharge)": 1.25,
    "1.40  (M1+)": 1.40,
    "1.55  (M2)": 1.55,
    "1.70  (M2+)": 1.70,
    "1.85  (M3)": 1.85,
    "2.00  (M3+)": 2.00,
    "2.15  (M3++)": 2.15,
    "2.30  (M3+++)": 2.30,
    "2.45  (M3 max — highest surcharge, riskiest)": 2.45,
}
_bm_keys = list(_BM_OPTIONS.keys())
bm_label = st.sidebar.selectbox(
    "Bonus-Malus coefficient",
    options=_bm_keys,
    index=3,   # default: K7 = 0.95 (≈ training median class 10)
    help="OGPO classes: K = bonus (discount, safe driver). M = malus (surcharge, at-fault history). K10=0.50 is the safest.",
)
bonus_malus = _BM_OPTIONS[bm_label]

experience_years = st.sidebar.slider(
    "Driving experience (years)",
    min_value=0, max_value=50, value=10,
)

n_drivers = st.sidebar.slider(
    "Number of drivers on policy",
    min_value=1, max_value=10, value=1,
)

engine_power = st.sidebar.slider(
    "Engine power (hp)",
    min_value=50, max_value=400, value=120,
)

engine_volume = st.sidebar.slider(
    "Engine volume (L)",
    min_value=0.8, max_value=6.0, value=1.6, step=0.1,
)

car_year = st.sidebar.slider(
    "Car manufacturing year",
    min_value=1990, max_value=2024, value=2015,
)

region_label = st.sidebar.selectbox(
    "Region",
    options=list(REGION_LABELS.values()),
    index=0,
)
region_id = [k for k, v in REGION_LABELS.items() if v == region_label][0]

vehicle_type_label = st.sidebar.selectbox(
    "Vehicle type",
    options=list(VEHICLE_TYPE_MAP.keys()),
    index=0,
)
vehicle_type = VEHICLE_TYPE_MAP[vehicle_type_label]

month = st.sidebar.slider(
    "Policy start month",
    min_value=1, max_value=12, value=6,
)

termination_ratio = st.sidebar.slider(
    "Early termination ratio",
    min_value=0.00, max_value=0.95, value=0.00, step=0.05,
    help="0 = policy ran its full term (most common). >0 = policy was cancelled early and partial premium was refunded. "
         "Note: the model associates full-term policies (0) with higher claim likelihood — early termination often means the driver switched insurer without a claim.",
)

predict_btn = st.sidebar.button("Predict Risk", type="primary", use_container_width=True)

# ─── UI: Main panel ─────────────────────────────────────────────────────────────
st.title("OGPO Claim Risk Predictor")
st.markdown(
    "**Freedom Insurance Kazakhstan** — Predict the claim probability for an OGPO "
    "(compulsory motor insurance) policy using a tuned HistGradientBoosting model "
    "with SHAP explanations."
)

if not predict_btn:
    st.info("Adjust the policy parameters in the sidebar and click **Predict Risk**.")
    st.stop()

# ─── Prediction ─────────────────────────────────────────────────────────────────
with st.spinner("Running model…"):
    input_df = build_input_row(
        bonus_malus, experience_years, n_drivers,
        engine_power, engine_volume, car_year,
        region_id, vehicle_type, month,
        termination_ratio,
        feat_medians,
    )
    prob, shap_df = predict_and_explain(input_df)

prob_pct = prob * 100

# Risk category thresholds (calibrated for class_weight='balanced' model output range 7-71%)
# LOW < 30%  |  MEDIUM 30-55%  |  HIGH > 55%
if prob_pct < 30.0:
    risk_label  = "LOW"
    risk_color  = "green"
    bar_color   = "#28a745"
elif prob_pct < 55.0:
    risk_label  = "MEDIUM"
    risk_color  = "orange"
    bar_color   = "#fd7e14"
else:
    risk_label  = "HIGH"
    risk_color  = "red"
    bar_color   = "#dc3545"

# ─── Metrics row ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Claim Probability",
        value=f"{prob_pct:.2f}%",
    )

with col2:
    st.markdown(
        f"<h3 style='color:{risk_color}; margin-top:0.3rem;'>Risk: {risk_label}</h3>",
        unsafe_allow_html=True,
    )

with col3:
    # Gauge bar
    st.markdown("**Risk gauge (0% – 70%+)**")
    gauge_pct = min(prob_pct / 70.0, 1.0)
    st.progress(gauge_pct)
    st.caption(f"{prob_pct:.2f}% (threshold: 30% LOW / 55% HIGH)")

st.divider()

# ─── SHAP explanation ────────────────────────────────────────────────────────────
st.subheader("SHAP Explanation — Top Feature Contributions")

top_n = 10
top_shap = shap_df.head(top_n)

# Waterfall-style bar chart
fig, ax = plt.subplots(figsize=(9, 4))
colors = ["#dc3545" if v > 0 else "#28a745" for v in top_shap["shap"]]
bars = ax.barh(top_shap["feature"][::-1], top_shap["shap"][::-1],
               color=colors[::-1], edgecolor="black", alpha=0.85)
ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax.set_xlabel("SHAP value (impact on log-odds of claim)")
ax.set_title(f"Top {top_n} Features — Local Explanation\n"
             f"Base rate ≈ 1.95%  |  Predicted P(claim) = {prob_pct:.2f}%",
             fontsize=10, fontweight="bold")
for bar, (_, row_s) in zip(bars, top_shap[::-1].iterrows()):
    offset = 0.0002 if row_s["shap"] >= 0 else -0.0002
    ha = "left" if row_s["shap"] >= 0 else "right"
    ax.text(row_s["shap"] + offset, bar.get_y() + bar.get_height() / 2,
            f'{row_s["shap"]:+.4f}', va="center", ha=ha, fontsize=8)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ─── Plain-language interpretation ──────────────────────────────────────────────
top_pos = shap_df[shap_df["shap"] > 0].iloc[0] if (shap_df["shap"] > 0).any() else None
top_neg = shap_df[shap_df["shap"] < 0].iloc[0] if (shap_df["shap"] < 0).any() else None

if top_pos is not None:
    st.info(
        f"**Primary risk factor:** `{top_pos['feature']}` "
        f"(SHAP = {top_pos['shap']:+.4f}) — this feature **increases** the claim probability "
        f"most for this policy."
    )
if top_neg is not None:
    st.success(
        f"**Primary protective factor:** `{top_neg['feature']}` "
        f"(SHAP = {top_neg['shap']:+.4f}) — this feature **reduces** the claim probability "
        f"most for this policy."
    )

# ─── Full SHAP table (expandable) ───────────────────────────────────────────────
with st.expander("Full SHAP feature table"):
    shap_display = shap_df.copy()
    shap_display["shap"] = shap_display["shap"].map("{:+.5f}".format)
    shap_display["value"] = shap_display["value"].map("{:.4f}".format)
    st.dataframe(shap_display, use_container_width=True, height=300)

# ─── Input summary (expandable) ─────────────────────────────────────────────────
with st.expander("Input summary"):
    st.json({
        "bonus_malus_ogpo_coef":   bonus_malus,
        "bonus_malus_class":       int(round((2.45 - bonus_malus) * 13.0 / 1.95)),
        "experience_years":        experience_years,
        "n_drivers":               n_drivers,
        "engine_power_hp":         engine_power,
        "engine_volume_L":         engine_volume,
        "engine_volume_cc":        int(engine_volume * 1000),
        "car_year":                car_year,
        "region":                  region_label,
        "vehicle_type":            vehicle_type_label,
        "month":                   month,
        "termination_ratio":       termination_ratio,
        "car_age_binary":          int((2025 - car_year) > 7),
        "power_density":           round(engine_power / (engine_volume * 1000 + 1e-5), 4),
        "bm_car_age":              round(int(round((2.45 - bonus_malus) * 13.0 / 1.95)) * int((2025 - car_year) > 7), 3),
        "experience_sq":           round(experience_years ** 2, 1),
    })

st.caption(
    "Model: HistGradientBoostingClassifier (tuned) | "
    "CV strategy: StratifiedKFold(5) | "
    "Primary metric: PR-AUC | "
    "Dataset: KBTU Risk Management Case Competition 2025 — Freedom Insurance OGPO"
)
