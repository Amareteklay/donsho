# pages/5_Results.py

import streamlit as st

# core data + viz API
from src.viz    import load_data, FIGURES, TABLES, plot_lagged_coefficients, plot_trend_panel, plot_baseline_forest
# baseline spec & fitters
from model_selection import build_spec, fit_poisson, fit_negbin

st.set_page_config(page_title="Model results", layout="wide")
st.title("📊 Results")

# ── 1) Load and prepare the default panel ────────────────────────────────
df = load_data()

# ── 2) Fit recommended baselines ────────────────────────────────────────
# Poisson FE + linear year‐trend
df_spec, y_base, X_base, formula_base = build_spec(
    df,
    outcome_mode="count",
    pred_mode="count",
    add_year_trend=True,
    fe={"Continent": True},
)
poisson_res = fit_poisson(df_spec, formula_base)
# Negative‐Binomial FE + linear year‐trend (for overdispersion check)
negbin_res  = fit_negbin(df_spec, formula_base)

st.subheader("📌 Baseline: Poisson FE + Year Trend")
st.write(poisson_res.summary())

st.subheader("📌 Baseline: Negative-Binomial FE + Year Trend")
st.write(negbin_res.summary())

# ── 3) Blueprint Figures ────────────────────────────────────────────────
st.header("📈 Figures")

# Section: Trend Plot
st.subheader("Trend plot")
trend_chart = plot_trend_panel(df)
st.altair_chart(trend_chart, use_container_width=True)

# Section: Baseline Forest Plot
st.subheader("Baseline forest plot")
baseline_chart = plot_baseline_forest(df)

if hasattr(baseline_chart, "to_dict"):
    st.altair_chart(baseline_chart, use_container_width=True)
else:
    st.pyplot(baseline_chart, use_container_width=True)


st.subheader("📌 Lagged coefficients")
st.pyplot(plot_lagged_coefficients(df, max_lag=5))

st.header("📋 Tables")
for name, table_fn in TABLES.items():
    st.subheader(name)
    try:
        html = table_fn(df, model=poisson_res)
    except TypeError:
        html = table_fn(df)
    st.markdown(html, unsafe_allow_html=True)
