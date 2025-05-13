# ──────────────────────────────────────────────────────────────
# src/viz.py
# Re-usable plotting helpers for EPPs & Shocks
# ──────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Sequence

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── global styling (neutral for dark/light IDEs) ──────────────
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.autolayout": True,
                     "axes.titlesize": "x-large"})

# ──────────────────────────────────────────────────────────────
# Basic distributions & relationships
# ──────────────────────────────────────────────────────────────
def plot_histogram(df: pd.DataFrame,
                   col: str,
                   *,
                   bins: int = 30,
                   **kwargs) -> None:
    """Histogram (optionally with KDE) of *col*."""
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not in DataFrame")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[col].dropna(), bins=bins, ax=ax, **kwargs)
    ax.set(title=f"Histogram of {col}", xlabel=col, ylabel="Frequency")
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame,
                            *,
                            annot: bool = False) -> None:
    """Lower-triangle Pearson correlation heat-map for numeric cols."""
    num_df = df.select_dtypes("number")
    if num_df.empty:
        raise ValueError("DataFrame has no numeric columns")

    corr = num_df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, ax=ax, annot=annot,
                cmap="coolwarm", fmt=".2f",
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation matrix")
    plt.show()

# ──────────────────────────────────────────────────────────────
# Time-series helper
# ──────────────────────────────────────────────────────────────
def interactive_ts_plot(df: pd.DataFrame,
                         date_col: str,
                         value_col: str,
                         *,
                         title: str | None = None) -> alt.Chart:
    """Simple Altair line chart with pan/zoom + tooltips."""
    chart = (alt.Chart(df)
               .mark_line(point=True)
               .encode(x=alt.X(date_col, title="Date"),
                       y=alt.Y(value_col, title=value_col, tooltip=True),
                       tooltip=[date_col, value_col])
               .interactive())
    return chart if title is None else chart.properties(title=title)

# ──────────────────────────────────────────────────────────────
# Model diagnostics
# ──────────────────────────────────────────────────────────────
def plot_coefficients(model: Any,
                      *,
                      feature_names: Sequence[str] | None = None,
                      top_n: int | None = None) -> None:
    """Horizontal bar-plot of coefficients with 1 s.e. bars."""
    if not hasattr(model, "params"):
        raise AttributeError("Model lacks `.params` attribute")

    coefs = model.params.copy()
    errs  = getattr(model, "bse",
                    pd.Series(index=coefs.index, data=np.nan))

    if feature_names is not None:
        if len(feature_names) != len(coefs):
            raise ValueError("'feature_names' len mismatch")
        coefs.index = errs.index = feature_names

    df_plot = (pd.DataFrame({"coef": coefs, "err": errs})
                 .assign(abs_coef=lambda d: d.coef.abs())
                 .sort_values("abs_coef", ascending=False))
    if top_n:
        df_plot = df_plot.head(top_n)
    df_plot = df_plot.sort_values("coef")

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4*len(df_plot))))
    sns.barplot(data=df_plot, x="coef", y=df_plot.index, ax=ax,
                orient="h", xerr=df_plot["err"], errorbar=None)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set(title="Model coefficients", xlabel="Estimate", ylabel="Feature")
    plt.show()


def plot_residuals(model: Any,
                   X: pd.DataFrame,
                   y_true: pd.Series,
                   *,
                   lowess: bool = True) -> None:
    """Residual-vs-predicted scatter with optional LOWESS trend."""
    if not hasattr(model, "predict"):
        raise AttributeError("Model lacks `.predict` method")

    y_pred = model.predict(X)
    resid  = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.residplot(x=y_pred, y=resid, lowess=lowess, ax=ax,
                  scatter_kws={"alpha": .6, "s": 40})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual plot")
    plt.show()
