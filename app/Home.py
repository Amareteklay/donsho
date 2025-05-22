import streamlit as st
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

# ─────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="EPPS Shocks", layout="centered")
st.title("🌍 EPPS Shocks & Disease Outbreak Explorer")

don_path = "data/01_raw/DONdatabase.csv"
shock_path = "data/01_raw/Shocks_Database_counts.csv"

don_df, shocks_df = get_processed_data(don_path, shock_path)

st.header("🧹 Raw Data")
st.dataframe(don_df, use_container_width=True)
st.dataframe(shocks_df, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Build panels
# ─────────────────────────────────────────────────────────────
st.header("🛠 Feature Engineering")

event_panel = build_event_panel(shocks_df, don_df=don_df, max_lag=5)
full_panel = build_full_panel(shocks_df, don_df, max_lag=5)
full_panel["Infectious_disease_binary"] = (full_panel["Infectious_disease"] > 0).astype(int)

st.write(f"📊 Event Panel: {event_panel.shape}")
st.dataframe(event_panel, use_container_width=True)

st.write(f"📊 Full Panel: {full_panel.shape}")
st.dataframe(full_panel, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Model Selection
# ─────────────────────────────────────────────────────────────
st.header("🧪 Model Selection (Logistic Regression)")

shock_candidates = [c for c in full_panel.columns if "_lag" in c]
fe_options = ["", "C(Year)", "C(Continent)", "C(Year) + C(Country)"]
scope_options = ["global", "Africa", "Asia", "Europe", "Americas"]

st.write("✅ Number of predictor columns selected:", len(shock_candidates))
st.write("🔍 Sample predictors:", shock_candidates[:5])

if st.button("🔎 Run model selection grid"):
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
        results_df.to_csv("data/04_outputs/streamlit_model_results.csv", index=False)
        st.session_state["model_results_df"] = results_df
        st.success(f"✅ {len(results_df)} models estimated and saved.")

# ─────────────────────────────────────────────────────────────
# Model Summary + Visual Diagnostics
# ─────────────────────────────────────────────────────────────
if "model_results_df" in st.session_state:
    st.header("📋 Model Selection Results")

    df = st.session_state["model_results_df"].dropna(subset=["AUC", "LogLoss"])
    if df.empty:
        st.warning("⚠️ No valid models with AUC/LogLoss were computed.")
    else:
        sort_by = st.selectbox("Sort models by", options=["AUC", "LogLoss"])
        top_n = st.slider("Number of models to show", 5, 100, 10)
        ascending = (sort_by == "LogLoss")
        top_models = df.sort_values(sort_by, ascending=ascending).head(top_n)
        st.dataframe(top_models, use_container_width=True)

        selected_model = st.selectbox("Inspect model formula", top_models["ModelID"])
        st.code(top_models[top_models["ModelID"] == selected_model]["Formula"].values[0])

        # ─────────────────────────────────────────────────────────────
        # Full Visual Diagnostics
        # ─────────────────────────────────────────────────────────────
        st.header("🖼️ Visual Diagnostics")

        # 1. Histograms of base shock variables (non-lagged)
        st.subheader("📊 Shock Variable Distributions")
        base_shocks = [col for col in full_panel.columns if "_l" not in col and col not in [
            "Year", "DON_year", "Country", "Continent", "Infectious_disease", "Infectious_disease_binary", "CasesTotal", "Deaths"]]
        for fig in visualization.plot_histograms(full_panel, base_shocks):
            st.pyplot(fig)

        # 2. Zero Proportions
        st.subheader("🧮 Proportion of Zeros")
        zero_df = visualization.plot_zero_proportions(full_panel, base_shocks)
        st.dataframe(zero_df)

        # 3. Outcome Balance
        st.subheader("✅ Outcome Balance (Binary DV)")
        st.pyplot(visualization.plot_outcome_balance(full_panel))

        # 4. Country-Year Density
        st.subheader("🌐 Country-Year Data Density")
        st.pyplot(visualization.plot_country_year_density(full_panel))

        # 5. Correlation of Lagged Predictors
        st.subheader("🔗 Lagged Predictor Correlation Matrix")
        lagged_cols = [col for col in full_panel.columns if "_l" in col]
        st.pyplot(visualization.plot_correlation_matrix(full_panel, lagged_cols))

        # 6. AUC Distribution
        st.subheader("📈 AUC Distribution Across Models")
        st.pyplot(visualization.plot_auc_distribution(df))

        # 7. AUC by Region
        st.subheader("🌍 AUC by Region")
        st.dataframe(visualization.summarize_auc_by_scope(df))

        # 8. Top Predictors in Best Models
        st.subheader("🧠 Most Common Predictors (Top Models)")
        top_n = st.slider("Top N Models", 5, 100, 25)
        freq_df = visualization.count_predictor_frequency(df, top_n=top_n)
        st.pyplot(visualization.plot_predictor_frequencies(freq_df))
else:
    st.warning("Please run model selection first.")

# ─────────────────────────────────────────────────────────────
# Model Grid Comparison Section
# ─────────────────────────────────────────────────────────────
st.header("🔍 Compare Multiple Model Grids")

# Initialize or update stored model grids
if "model_grids" not in st.session_state:
    st.session_state["model_grids"] = {}

if "model_results_df" in st.session_state:
    current_name = st.text_input("Name this model grid (for comparison)", value="Default Run")
    if st.button("📌 Save current model grid"):
        st.session_state["model_grids"][current_name] = st.session_state["model_results_df"]
        st.success(f"Saved model grid as '{current_name}'")

# Select and compare saved grids
grid_names = list(st.session_state["model_grids"].keys())
selected_grids = st.multiselect("Select model grids to compare", grid_names, default=grid_names[:2])

if len(selected_grids) >= 2:
    st.subheader("📈 AUC Distribution Across Grids")

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()
    for name in selected_grids:
        df = st.session_state["model_grids"][name]
        sns.histplot(df["AUC"], label=name, kde=False, bins=20, ax=ax, alpha=0.5)
    ax.legend()
    ax.set_title("AUC Comparison Across Model Grids")
    st.pyplot(fig)

    st.subheader("🧠 Top Predictors (Frequency in Top N)")
    top_n = st.slider("Top N Models", 5, 100, 25)
    for name in selected_grids:
        df = st.session_state["model_grids"][name]
        st.markdown(f"**{name}**")
        freq_df = visualization.count_predictor_frequency(df, top_n=top_n)
        st.pyplot(visualization.plot_predictor_frequencies(freq_df))

    st.subheader("🌟 Best Model Per Grid")
    for name in selected_grids:
        df = st.session_state["model_grids"][name]
        best = df.sort_values("AUC", ascending=False).iloc[0]
        st.markdown(f"**{name}** — AUC: `{best['AUC']:.3f}`")
        st.code(best["Formula"])

st.header("📊 Shock × Continent → Outbreak Interaction")

selected_shock = st.selectbox("Select a shock variable", [c for c in full_panel.columns if "_lag" in c])
plot_type = st.radio("Choose interaction plot type", ["Facet", "Grouped Bar", "Line"])

if plot_type == "Facet":
    g = visualization.plot_interaction_facet(full_panel, selected_shock)
    st.pyplot(g.fig)
elif plot_type == "Grouped Bar":
    st.pyplot(visualization.plot_interaction_grouped_bar(full_panel, selected_shock))
else:
    st.pyplot(visualization.plot_interaction_line(full_panel, selected_shock))

# ─────────────────────────────────────────────────────────────
st.markdown("📍 *Playground for the EPPS Shocks & Disease Outbreak Explorer.*")
