# 5_Results.py
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from src import data_loader, prep, features, viz
from src.config import SAVED_MODELS_DIR   # defined in the modelling page :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

st.set_page_config(page_title="Model results", layout="wide")
st.title("📊 Results")

# ── 1) pick a saved model ──────────────────────────────────────────────
model_files = sorted(SAVED_MODELS_DIR.glob("*.pkl"))
mdl_path    = st.sidebar.selectbox("Choose a saved model", model_files)

if not mdl_path:
    st.info("👈 Select a model file in the sidebar to continue.")
    st.stop()

with open(mdl_path, "rb") as fh:
    # the custom classes (PoissonGLMM, LogisticGLM, …) are pickle-able
    model = pickle.load(fh)

st.success(f"Loaded **{mdl_path.name}**")

# ── 2) choose / build an evaluation panel ──────────────────────────────
interims = sorted(Path("data/02_interim").glob("panel_*.parquet"))
panel_fn = st.sidebar.selectbox("Evaluation dataset", interims)

# fall back to raw-counts workflow if no interim chosen
if panel_fn:
    test = data_loader.load_interim(panel_fn.name)
else:
    raw  = prep.add_continent(prep.clean_counts(data_loader.load_raw_counts()))
    test = features.build_panel(raw)

# ── 3) generate predictions ────────────────────────────────────────────
# If the model was fitted on a subset of predictors, keep only those cols
predictors = [c for c in model.exog_names if c != "Intercept"]
missing    = set(predictors) - set(test.columns)
if missing:
    st.warning(f"Test set is missing predictors: {missing}. "
               "Those columns will be filled with zero.")
    for col in missing:
        test[col] = 0

y_true = test["Infectious_disease"]
y_pred = model.predict(test[predictors])

# attach predictions so viz.plot_residuals can re-use them if desired
test["pred"] = y_pred

# ── 4) plots ───────────────────────────────────────────────────────────
st.subheader("Model coefficients")
viz.plot_coefficients(model)                         # uses params & bse

st.subheader("Residual diagnostics")
viz.plot_residuals(model,
                   test[predictors],                # X matrix
                   y_true,                         # y_true
                   y_pred)                         # (pre-computed)
