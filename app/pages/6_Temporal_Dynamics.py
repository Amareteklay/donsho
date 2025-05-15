# app/pages/6_Temporal_Dynamics.py
import streamlit as st
import pandas as pd
from src.features import build_event_panel, build_lagged_shock_features
from src.models import fit_lag_model, fit_logit_lagged_shock
from src.viz import plot_lag_irrs, plot_lag_effects_logit

st.title("Temporal Dynamics: Shock Effects Relative to Outbreaks")

df = pd.read_parquet("data/02_interim/clean.parquet")

max_lag = st.slider("Lag window", 1, 10, 5)
panel = build_event_panel(df, max_lag=max_lag)

st.dataframe(panel)

model = fit_lag_model(panel)
st.write(model.summary())

fig = plot_lag_irrs(model, title=f"Lag Effects on Disease Outbreaks (±{max_lag} years)")
st.pyplot(fig)

st.subheader("Lagged Shock Effects")

shock = "CLIMATIC"


def get_valid_anchors(panel_df, max_lag=5):
    all_years = panel_df[["Country_name", "Continent", "Year"]].drop_duplicates()

    # Anchor on years that have full ±lag range available
    valid_years = []
    for _, row in all_years.iterrows():
        country, year = row["Country_name"], row["Year"]
        for lag in range(-max_lag, max_lag + 1):
            if not ((panel_df["Country_name"] == country) & 
                    (panel_df["Year"] == year + lag)).any():
                break
        else:
            valid_years.append((country, year))

    valid_df = pd.DataFrame(valid_years, columns=["Country_name", "Year"])
    return valid_df


def build_lagged_shock_dataset(panel_df, shock: str, max_lag=5):
    anchors = get_valid_anchors(panel_df, max_lag=max_lag)
    rows = []

    for _, row in anchors.iterrows():
        country = row["Country_name"]
        anchor_year = row["Year"]

        record = {
            "Country_name": country,
            "Year": anchor_year
        }

        # Add lagged shock values
        for lag in range(-max_lag, max_lag + 1):
            lag_year = anchor_year + lag
            val = panel_df.loc[
                (panel_df["Country_name"] == country) &
                (panel_df["Year"] == lag_year),
                shock
            ]
            if not val.empty:
                label = f"{shock}_lag_m{abs(lag)}" if lag < 0 else (f"{shock}_lag_p{lag}" if lag > 0 else f"{shock}_lag_0")
                record[label] = val.values[0]

        # Add outcome (did a disease occur this year?)
        outcome = panel_df.loc[
            (panel_df["Country_name"] == country) &
            (panel_df["Year"] == anchor_year),
            "Infectious_disease"
        ]
        record["Infectious_disease"] = int(outcome.sum() > 0)

        rows.append(record)
        result = pd.DataFrame(rows)
        
    return result


lagged_df = build_lagged_shock_dataset(panel, shock=shock, max_lag=max_lag)
st.write("Lagged Shock Dataset")
st.write("Number of rows:", len(lagged_df))
st.dataframe(lagged_df)
model, lag_vars = fit_logit_lagged_shock(lagged_df, shock)
st.write(model.summary())
fig = plot_lag_effects_logit(model, lag_vars, shock)
st.pyplot(fig)

st.subheader("Multiple Shock Effects")

def build_lagged_shock_dataset_multi(panel_df, shocks: list[str], max_lag=5):
    anchors = get_valid_anchors(panel_df, max_lag=max_lag)
    rows = []

    for _, row in anchors.iterrows():
        country = row["Country_name"]
        anchor_year = row["Year"]
        record = {
            "Country_name": country,
            "Year": anchor_year
        }

        # Add lagged shock values for each shock
        for shock in shocks:
            for lag in range(-max_lag, max_lag + 1):
                lag_year = anchor_year + lag
                val = panel_df.loc[
                    (panel_df["Country_name"] == country) &
                    (panel_df["Year"] == lag_year),
                    shock
                ]
                if not val.empty:
                    label = f"{shock}_lag_m{abs(lag)}" if lag < 0 else (f"{shock}_lag_p{lag}" if lag > 0 else f"{shock}_lag_0")
                    record[label] = val.values[0]

        # Add outcome (did a disease occur this year?)
        outcome = panel_df.loc[
            (panel_df["Country_name"] == country) &
            (panel_df["Year"] == anchor_year),
            "Infectious_disease"
        ]
        record["Infectious_disease"] = int(outcome.sum() > 0)

        rows.append(record)
        new_result = pd.DataFrame(rows)

    return new_result

def fit_joint_logit_lag_model(df, shocks: list[str]):
    import statsmodels.formula.api as smf

    lagged_vars = []
    for shock in shocks:
        lagged_vars.extend([col for col in df.columns if col.startswith(f"{shock}_lag_")])

    formula = "Infectious_disease ~ " + " + ".join(lagged_vars)
    model = smf.logit(formula=formula, data=df).fit()
    return model, lagged_vars
def plot_joint_lag_effects(model, lagged_vars):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = []
    for var in lagged_vars:
        shock_name, lag_part = var.split("_lag_")
        if lag_part.startswith("m"):
            year_rel = -int(lag_part[1:])
        elif lag_part.startswith("p"):
            year_rel = int(lag_part[1:])
        else:
            year_rel = 0

        coef = model.params[var]
        conf_int = model.conf_int().loc[var]
        or_value = np.exp(coef)
        lower = np.exp(conf_int[0])
        upper = np.exp(conf_int[1])

        data.append({
            "Shock": shock_name,
            "Year_rel": year_rel,
            "OR": or_value,
            "lower": lower,
            "upper": upper
        })

    df_plot = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_plot, x="Year_rel", y="OR", hue="Shock", marker="o")
    for _, group in df_plot.groupby("Shock"):
        plt.fill_between(group["Year_rel"], group["lower"], group["upper"], alpha=0.2)

    plt.axhline(1.0, linestyle="--", color="gray")
    plt.title("Lagged Shock Effects on Disease Outbreaks (Odds Ratios)")
    plt.xlabel("Year relative to outbreak")
    plt.ylabel("Odds Ratio (95% CI)")
    plt.legend(title="Shock Type")
    plt.tight_layout()
    return plt.gcf()

shock_list = ["CLIMATIC", "CONFLICTS", "ECONOMIC"]
lagged_df = build_lagged_shock_dataset_multi(panel, shocks=shock_list, max_lag=5)

model, lag_vars = fit_joint_logit_lag_model(lagged_df, shocks=shock_list)
st.write(model.summary())
fig = plot_joint_lag_effects(model, lag_vars)
st.pyplot(fig)

