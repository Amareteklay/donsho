import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from src.config import YEAR_MIN, YEAR_MAX


def plot_histograms(df: pd.DataFrame, columns: list[str]) -> list[plt.Figure]:
    """Plot histogram for each column in `columns`."""
    figures = []
    for col in columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=30, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        figures.append(fig)
    return figures


def plot_zero_proportions(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a DataFrame with proportion of zeros for each column."""
    return pd.DataFrame({
        "Variable": columns,
        "% Zeros": [(df[col] == 0).mean() for col in columns]
    }).sort_values("% Zeros", ascending=False)


def plot_outcome_balance(df: pd.DataFrame, outcome_col: str = "Infectious_disease_binary") -> plt.Figure:
    """Plot class balance for the binary outcome."""
    fig, ax = plt.subplots()
    df[outcome_col].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title(f"Distribution of {outcome_col}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    return fig


def plot_country_year_density(df: pd.DataFrame) -> plt.Figure:
    """Plot number of years per country (data density)."""
    fig, ax = plt.subplots()
    density = df.groupby("Country")["Year"].nunique().sort_values(ascending=False)
    density.plot(kind="bar", ax=ax)
    ax.set_ylabel("Number of years")
    ax.set_title("Data Density by Country")
    return fig


def plot_correlation_matrix(df: pd.DataFrame, columns: list[str]) -> plt.Figure:
    """Plot correlation heatmap for selected columns."""
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Correlation Matrix")
    return fig


def plot_binary_predictor_vs_outcome(df: pd.DataFrame, predictor: str, outcome: str = "Infectious_disease_binary") -> list[plt.Figure]:
    """Plot bar chart and count chart for binary predictor vs binary outcome."""
    grouped = df.groupby(predictor)[outcome].mean().reset_index()
    fig1, ax1 = plt.subplots()
    sns.barplot(data=grouped, x=predictor, y=outcome, ax=ax1)
    ax1.set_ylabel("P(Outbreak)")
    ax1.set_title(f"Outbreak Probability by {predictor}")

    fig2, ax2 = plt.subplots()
    counts = pd.crosstab(df[predictor], df[outcome])
    counts.plot(kind="bar", stacked=True, ax=ax2)
    ax2.set_title(f"Outbreak Counts by {predictor}")
    ax2.set_ylabel("Count")

    return [fig1, fig2]


def plot_continuous_predictor_vs_outcome(df: pd.DataFrame, predictor: str, outcome: str = "Infectious_disease_binary", fit_type: str = "logit") -> plt.Figure:
    """Plot smoothed relationship between continuous predictor and binary outcome."""
    fig, ax = plt.subplots()
    scatter_kws = {"s": 10, "alpha": 0.5}
    
    if fit_type == "logit":
        sns.regplot(data=df, x=predictor, y=outcome, logistic=True, scatter_kws=scatter_kws, ax=ax)
    elif fit_type == "lowess":
        sns.regplot(data=df, x=predictor, y=outcome, lowess=True, scatter_kws=scatter_kws, ax=ax)
    else:
        sns.regplot(data=df, x=predictor, y=outcome, scatter_kws=scatter_kws, ax=ax)

    ax.set_title(f"{predictor} vs P(Outbreak) [{fit_type}]")
    return fig


def plot_auc_distribution(df: pd.DataFrame) -> plt.Figure:
    """Plot distribution of AUCs."""
    fig, ax = plt.subplots()
    df["AUC"].round(2).value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Model AUC Distribution")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Number of models")
    return fig


def summarize_auc_by_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean AUC by Scope."""
    return df.groupby("Scope")["AUC"].agg(["mean", "count"]).sort_values("mean", ascending=False)


def count_predictor_frequency(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Return frequency of predictors in top N models by AUC."""
    top_models = df.sort_values("AUC", ascending=False).head(top_n)
    counter = collections.Counter()
    for preds in top_models["Predictors"]:
        counter.update(preds)
    return pd.DataFrame(counter.items(), columns=["Predictor", "Frequency"]).sort_values("Frequency", ascending=False)


def plot_predictor_frequencies(freq_df: pd.DataFrame) -> plt.Figure:
    """Plot frequency of predictors in top models."""
    fig, ax = plt.subplots()
    freq_df.set_index("Predictor").plot(kind="bar", legend=False, ax=ax)
    ax.set_title("Predictor Frequency in Top Models")
    ax.set_ylabel("Count")
    return fig

def plot_interaction_facet(df: pd.DataFrame, shock: str, dv: str = "Infectious_disease_binary", group: str = "Continent") -> sns.FacetGrid:
    """Facet bar plot of shock vs outcome, faceted by group."""
    g = sns.catplot(data=df, x=shock, y=dv, col=group, kind="bar", ci=None)
    g.fig.suptitle(f"{shock} × {group} → {dv}", y=1.02)
    return g


def plot_interaction_grouped_bar(df: pd.DataFrame, shock: str, dv: str = "Infectious_disease_binary", group: str = "Continent") -> plt.Figure:
    """Grouped bar plot (mean DV by shock × group)."""
    grouped = df.groupby([shock, group])[dv].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=grouped, x=shock, y=dv, hue=group, ax=ax)
    ax.set_title(f"{shock} × {group} → {dv}")
    return fig


def plot_interaction_line(df: pd.DataFrame, shock: str, dv: str = "Infectious_disease_binary", group: str = "Continent") -> plt.Figure:
    """Line plot: shock on x, DV on y, colored by group."""
    grouped = df.groupby([shock, group])[dv].mean().reset_index()
    fig, ax = plt.subplots()
    sns.lineplot(data=grouped, x=shock, y=dv, hue=group, marker="o", ax=ax)
    ax.set_title(f"{shock} × {group} → {dv}")
    return fig

