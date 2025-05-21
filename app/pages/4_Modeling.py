# pages/4_Modeling.py
# ------------------------------------------------------------
# Interactive model exploration with persistent comparison
# ------------------------------------------------------------
import streamlit as st
import pandas as pd

from z_legacy.data_loader   import load_raw_csv
from z_legacy.prep          import preprocess_counts
from feature_engineering      import build_panel, add_lags
from src.config        import MAX_LAG as DEFAULT_MAX_LAG

# helpers in utils.py
from src.utils         import (
    prepare_spec, add_binary, add_year_trend, make_formula, reg_table
)

# wrappers in models.py
from model_selection        import (
    fit_poisson, fit_negbin, fit_logit,
    fit_panel_re, fit_mixed_effects,
    evaluate_model, train_test_split_ts, filter_model_data
)

# ------------------------------------------------------------
# 0 ▸ Title & session state
# ------------------------------------------------------------
st.title("📈 Infectious-disease shock models")

if "stored_models" not in st.session_state:
    st.session_state["stored_models"] = []   # list of dicts

# ------------------------------------------------------------
# 1 ▸ Load & preprocess
# ------------------------------------------------------------
panel = build_panel(preprocess_counts(load_raw_csv()))
continents = sorted(panel["Continent"].dropna().unique())
shock_cols = [c for c in panel.columns
              if c not in ("Country_name", "Continent",
                           "Year", "Infectious_disease")]

# model maps split by outcome type
COUNT_MODELS  = {
    "Poisson (FE)"           : "poisson",
    "Negative Binomial (FE)" : "negbin",
    "Random effects (panel)" : "panel_re",
}
BIN_MODELS    = {
    "Logit (FE)"                      : "logit",
    "Mixed effects (random intercept)": "mixed",
}

# ------------------------------------------------------------
# 2 ▸ UI – specification
# ------------------------------------------------------------
spec = st.expander("🔧 Model specification", expanded=True)
with spec:
    # geography
    scope = st.radio("Geographic scope",
                     ["Global (all continents)", "Selected continent(s)"])
    sel_conts = (continents if scope == "Global (all continents)"
                 else st.multiselect("Choose continent(s)", continents, [continents[0]]))

    st.divider()

    # outcome + FE
    c1, c2 = st.columns(2)
    with c1:
        outcome_mode = st.radio("Outcome variable", ["count", "binary"])
        add_trend    = st.checkbox("Add linear year trend", value=True)
    with c2:
        fe_continent = st.checkbox("Continent fixed effects", value=True)
        fe_country   = st.checkbox("Country fixed effects",  value=False)

    st.divider()

    # predictors
    chosen_shocks = st.multiselect("Shock categories (predictors)",
                                   shock_cols, shock_cols)
    binary_shocks = st.multiselect(
        "Convert selected shocks to **binary** 0/1",
        chosen_shocks, help="Leave empty to keep counts."
    )

    st.divider()

    # lags & model families
    c3, c4 = st.columns(2)
    with c3:
        max_lag = st.slider("Maximum lag (years)", 0, 10,
                            DEFAULT_MAX_LAG, step=1)
    with c4:
        model_choices = COUNT_MODELS if outcome_mode == "count" else BIN_MODELS
        model_label   = st.selectbox("Model family", list(model_choices.keys()))

# ------------------------------------------------------------
# 3 ▸ Build modelling frame
# ------------------------------------------------------------
panel_scoped = panel.loc[panel["Continent"].isin(sel_conts)].copy()

df, y, X = prepare_spec(panel_scoped,
                        outcome_mode=outcome_mode,
                        pred_mode="count",
                        shock_cols=chosen_shocks)

# lags (on count vars only)
if max_lag:
    lag_base = [c for c in X if not c.endswith("_bin")]
    df = add_lags(df, lag_base, max_lag)
    X += [f"{c}_lag{l}" for c in lag_base for l in range(1, max_lag+1)]

# binary conversion
if binary_shocks:
    df = add_binary(df, binary_shocks)
    X = [f"{c}_bin" if c in binary_shocks else c for c in X]

# year trend
if add_trend:
    df = add_year_trend(df)

# formula
formula = make_formula(
    y, X, year_trend=add_trend,
    fe={"Country_name": fe_country, "Continent": fe_continent}
)

df, X = filter_model_data(df, y, X)
if not X:
    formula = f"{y} ~ 1"

train, test = train_test_split_ts(df, "Year", 0.2)

# ------------------------------------------------------------
# 4 ▸ Fit current model if user clicks
# ------------------------------------------------------------
def fit_current_model():
    code = model_choices[model_label]

    if code == "poisson":
        res = fit_poisson(train, formula)
    elif code == "negbin":
        res = fit_negbin(train, formula)
    elif code == "logit":
        res = fit_logit(train, formula)
    elif code == "panel_re":
        res = fit_panel_re(train, y, X + (["Year_trend"] if add_trend else []))
    elif code == "mixed":
        res = fit_mixed_effects(train, formula, groups="Country_name")
    else:
        return

    m_dict = {
        "label": f"{model_label} ({outcome_mode})",
        "result": res,
        "metrics": evaluate_model(res, test, y),
        "outcome_mode": outcome_mode,
    }
    st.session_state.stored_models.append(m_dict)

st.button("➕ Add model to comparison", on_click=fit_current_model)
st.button("🗑️ Clear stored models",
          on_click=lambda: st.session_state.update(stored_models=[]))

# ------------------------------------------------------------
# 5 ▸ Display stored models
# ------------------------------------------------------------
models = st.session_state.stored_models

if not models:
    st.info("No models stored yet. Fit a spec and click *Add model to comparison*.")
    st.stop()

# individual summaries
for m in models:
    with st.expander(f"📄 {m['label']} summary"):
        st.write(m["result"].summary())

# regression table
st.subheader("Stacked regression table")
st.markdown(
    reg_table([m["result"] for m in models],
              [m["label"] for m in models],
              output="html"),
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────
# 6 ▸ Hold-out metrics (separate for count vs binary models)
# ─────────────────────────────────────────────────────────────
def metrics_df(mode: str) -> pd.DataFrame:
    """Return a DataFrame of metrics for stored models with the given outcome mode."""
    rows = [m["metrics"] | {"Model": m["label"]}
            for m in models if m["outcome_mode"] == mode]
    return pd.DataFrame(rows).set_index("Model")

# count models
df_count = metrics_df("count")
if not df_count.empty:
    st.subheader("Hold-out metrics (count models)")
    st.bar_chart(df_count[["RMSE", "R2"]])

# binary models
df_bin = metrics_df("binary")
if not df_bin.empty:
    st.subheader("Hold-out metrics (binary models)")
    st.bar_chart(df_bin[["Accuracy", "ROC_AUC"]])
