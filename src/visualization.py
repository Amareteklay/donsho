import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections


def show_variable_distributions(full_panel: pd.DataFrame):
    st.subheader("📊 Variable Distribution: Shock Counts")
    count_cols = [col for col in full_panel.columns if "_lag" not in col and full_panel[col].dtype in [int, float] and col not in ["Year", "CasesTotal", "Deaths", "Infectious_disease", "Infectious_disease_binary"]]
    
    for col in count_cols:
        fig, ax = plt.subplots()
        sns.histplot(full_panel[col], bins=30, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    st.subheader("🧮 Proportion of Zeros")
    zero_stats = {col: (full_panel[col] == 0).mean() for col in count_cols}
    zero_df = pd.DataFrame.from_dict(zero_stats, orient='index', columns=["% Zeros"]).sort_values("% Zeros", ascending=False)
    st.dataframe(zero_df)

    st.subheader("🌐 Data Density by Country-Year")
    density = full_panel.groupby("Country")["Year"].nunique().sort_values(ascending=False)
    st.bar_chart(density)

    st.subheader("✅ Outcome Balance (Binary DV)")
    st.bar_chart(full_panel["Infectious_disease_binary"].value_counts().sort_index())


def show_correlation_matrix(full_panel: pd.DataFrame):
    st.subheader("🔗 Correlation Between Shock Variables")
    shock_cols = [c for c in full_panel.columns if c not in ["Country", "Continent", "Year", "DON_year", "Infectious_disease", "Infectious_disease_binary", "CasesTotal", "Deaths"] and full_panel[c].dtype in [int, float]]
    corr = full_panel[shock_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    ax.set_title("Shock Correlation Matrix")
    st.pyplot(fig)


def show_predictor_vs_dv_distributions(full_panel: pd.DataFrame):
    st.subheader("🔍 Shock Distributions by Outbreak Presence")

    predictors = [col for col in full_panel.columns if "_lag" in col and full_panel[col].dtype in [int, float]]

    dv = "Infectious_disease_binary"

    selected_predictor = st.selectbox("Choose a predictor to inspect", predictors)

    fig, ax = plt.subplots()
    sns.boxplot(data=full_panel, x=dv, y=selected_predictor)
    ax.set_title(f"{selected_predictor} by Outbreak Presence (0/1)")
    st.pyplot(fig)

    st.subheader("📉 Mean Shock Level by Outcome")
    grouped = full_panel.groupby(dv)[selected_predictor].mean().reset_index()
    st.bar_chart(grouped.set_index(dv))

    st.subheader("📈 Smoothed Relationships for All Predictors")

    fit_type = st.radio("Choose smoothing method", ["Logistic Regression", "LOESS (Lowess)", "Linear"], horizontal=True)
    predictors = [col for col in full_panel.columns if "_lag" in col and full_panel[col].dtype in [int, float]]
    dv = "Infectious_disease_binary"

    for predictor in predictors:
        fig, ax = plt.subplots()

        try:
            if fit_type == "Logistic Regression":
                sns.regplot(data=full_panel, x=predictor, y=dv,
                            logistic=True, scatter_kws={"s": 10, "alpha": 0.5}, ax=ax)
            elif fit_type == "LOESS (Lowess)":
                sns.regplot(data=full_panel, x=predictor, y=dv,
                            lowess=True, scatter_kws={"s": 10, "alpha": 0.5}, ax=ax)
            else:
                sns.regplot(data=full_panel, x=predictor, y=dv,
                            scatter_kws={"s": 10, "alpha": 0.5}, ax=ax)

            ax.set_title(f"{predictor} vs Probability of Outbreak")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"⚠️ Could not plot {predictor}: {e}")




def show_model_selection_summary(results_df: pd.DataFrame):
    st.subheader("📈 Model Performance (AUC)")

    results_df = results_df.dropna(subset=["AUC", "LogLoss"])

    st.bar_chart(results_df["AUC"].round(2).value_counts().sort_index())

    st.subheader("🌍 AUC by Region")
    auc_by_scope = results_df.groupby("Scope")["AUC"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    st.dataframe(auc_by_scope)

    st.subheader("🧠 Most Common Predictors in Top Models")
    top_n = st.slider("Top N Models to Analyze", 5, 200, 25)
    top_models = results_df.sort_values("AUC", ascending=False).head(top_n)

    pred_counts = collections.Counter()
    for preds in top_models["Predictors"]:
        pred_counts.update(preds)
    pred_df = pd.DataFrame(pred_counts.items(), columns=["Predictor", "Frequency"]).sort_values("Frequency", ascending=False)

    st.bar_chart(pred_df.set_index("Predictor"))

    st.subheader("🌟 Best Model Overview")
    best_model = results_df.sort_values("AUC", ascending=False).iloc[0]
    st.markdown(f"**Best AUC:** {best_model['AUC']:.3f} (Model {best_model['ModelID']})")
    st.code(best_model["Formula"])


def app(full_panel: pd.DataFrame, results_df: pd.DataFrame):
    st.title("📊 EPPs & Shocks: Visual Diagnostics")
    st.markdown("Use the tabs below to explore variable patterns and model selection insights.")

    tab1, tab2, tab3, tab4 = st.tabs(["Variable Inspection", "Correlation", "Model Selection", "Relationships"])

    with tab1:
        show_variable_distributions(full_panel)

    with tab2:
        show_correlation_matrix(full_panel)

    with tab3:
        show_model_selection_summary(results_df)
    
    with tab4:
        show_predictor_vs_dv_distributions(full_panel)
