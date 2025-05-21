import streamlit as st, src
import pandas as pd
from src.config import YEAR_MIN, YEAR_MAX, RARE_THRESHOLD
from src.data_processing import get_processed_data, add_continent
from src.feature_engineering import (
    pivot_shock_categories,
    build_event_panel,
    build_full_panel
)
from src.model_selection import (
    generate_model_grid,
    run_model_selection
)
from src import visualization

st.set_page_config(page_title="EPPS Shocks", layout="centered")
st.title("🌍 EPPS Shocks & Disease Outbreak Explorer")

don_path = "data/01_raw/DONdatabase.csv"
shock_path = "data/01_raw/Shocks_Database_counts.csv"

don_df, shocks_df = get_processed_data(don_path, shock_path)
st.write("Processed data")
st.dataframe(don_df, use_container_width=True)

st.dataframe(shocks_df, use_container_width=True)

st.write("Feature Engineering")
event_panel = build_event_panel(
    shocks_df,
    don_df=don_df,
    max_lag=5
)

full_panel = build_full_panel(shocks_df, don_df, max_lag=5)


st.write(f"Event Panel: {event_panel.shape[0]} rows, {event_panel.shape[1]} columns")
st.dataframe(event_panel, use_container_width=True)

st.write(f"Full Panel: {full_panel.shape[0]} rows, {full_panel.shape[1]} columns")
st.dataframe(full_panel, use_container_width=True)

st.header("📊 Model Selection (Binary DV)")

# Prepare binary outcome
full_panel["Infectious_disease_binary"] = (full_panel["Infectious_disease"] > 0).astype(int)

# Select shocks with lags to test
shock_candidates = [c for c in full_panel.columns if "_lag" in c]

# Define fixed effect and scope options
fe_options = ["", "C(Year)", "C(Continent)", "C(Year) + C(Country)"]
scope_options = ["global", "Africa", "Asia", "Europe", "Americas"]

# Show user-selected config
st.write("✅ Number of predictor columns selected:", len(shock_candidates))
st.write("🔍 Sample predictors:", shock_candidates[:5])


# Run model selection
if st.button("🔎 Run model selection grid (logit)"):
    with st.spinner("Fitting models..."):
        model_grid = generate_model_grid(
            shock_candidates,
            fixed_effects=fe_options,
            scopes=scope_options,
            max_predictors_per_model=2
        )
        results_df = run_model_selection(
            df=full_panel,
            grid_df=model_grid,
            dv="Infectious_disease_binary"
        )

        # Store and export
        results_df.to_csv("data/04_outputs/streamlit_model_results.csv", index=False)
        st.session_state["model_results_df"] = results_df
        st.success(f"✅ Model selection completed: {len(results_df)} models estimated.")

# Display results if available
if "model_results_df" in st.session_state:
    st.subheader("📋 Model Selection Results")

    df = st.session_state["model_results_df"]
    df = df.dropna(subset=["AUC", "LogLoss"])  # Clean broken rows

    if df.empty:
        st.warning("⚠️ No valid models with AUC/LogLoss were computed.")
    else:
        # --- Controls ---
        sort_by = st.selectbox("Sort models by", options=["AUC", "LogLoss"])
        top_n = st.slider("Number of models to show", 5, 100, 10)
        ascending = (sort_by == "LogLoss")
        top_models = df.sort_values(sort_by, ascending=ascending).head(top_n)

        # --- Table ---
        st.dataframe(top_models, use_container_width=True)

        # --- Inspect one ---
        selected_model = st.selectbox("Inspect model formula", top_models["ModelID"])
        st.code(top_models[top_models["ModelID"] == selected_model]["Formula"].values[0])

        # --- 📊 Plot 1: AUC distribution ---
        st.subheader("📈 AUC Distribution (All Models)")
        st.bar_chart(df["AUC"].dropna().round(2).value_counts().sort_index())

        # --- 📊 Plot 2: AUC by Scope ---
        st.subheader("🌍 AUC by Continent/Scope")
        auc_by_scope = df.groupby("Scope")["AUC"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        st.dataframe(auc_by_scope)

        # --- 📊 Plot 3: Most Common Predictors in Top Models ---
        st.subheader("🧠 Top Predictors in Top Models")
        import collections
        pred_counts = collections.Counter()
        for preds in top_models["Predictors"]:
            pred_counts.update(preds)
        pred_df = pd.DataFrame(pred_counts.items(), columns=["Predictor", "Frequency"]).sort_values("Frequency", ascending=False)
        st.bar_chart(pred_df.set_index("Predictor"))

        # --- 🪄 Best model ---
        st.subheader("🌟 Best Model Summary")
        best_model = df.sort_values("AUC", ascending=False).iloc[0]
        st.markdown(f"**Best AUC:** {best_model['AUC']:.3f} (Model {best_model['ModelID']})")
        st.code(best_model["Formula"])


if "model_results_df" in st.session_state:
    visualization.app(full_panel, st.session_state["model_results_df"])
else:
    st.warning("Please run model selection first.")



st.markdown("""
In this dashboard we:

1. Inspect the raw shock database  
2. Decide preprocessing parameters (rare-to-binary, lag length …)  
3. Fit Poisson / NB / GEE models step-by-step  
4. Compare predictions to observations
""")
