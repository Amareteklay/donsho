# pages/5_Results.py

import streamlit as st

# core data + viz API
from src.viz    import load_data, FIGURES, TABLES, plot_lagged_coefficients
# baseline spec & fitters
from src.models import build_spec, fit_poisson, fit_negbin

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

for name, plot_fn in FIGURES.items():
    st.subheader(name)
    try:
        chart = plot_fn(df, model=poisson_res)
    except TypeError:
        chart = plot_fn(df)
    if hasattr(chart, "to_dict"):  # Altair chart
        st.altair_chart(chart, use_container_width=True)
    else:                           # Matplotlib Figure
        st.pyplot(chart, use_container_width=True)

st.subheader("📌 Lagged coefficients")
st.pyplot(plot_lagged_coefficients(df, max_lag=5))

# ── 4) Blueprint Tables ────────────────────────────────────────────────
st.header("📋 Tables")
for name, table_fn in TABLES.items():
    st.subheader(name)
    try:
        html = table_fn(df, model=poisson_res)
    except TypeError:
        html = table_fn(df)
    st.markdown(html, unsafe_allow_html=True)
