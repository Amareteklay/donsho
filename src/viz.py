# src/viz.py
# ──────────────────────────────────────────────────────────────
# Re-usable plotting helpers for EPPs & Shocks (Streamlit & Quarto)
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
from typing import Any, Callable

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_loader import load_raw_csv
from .prep        import preprocess_counts
from .features    import build_panel, add_lags
from .models      import (
    build_spec,
    fit_poisson,
    fit_negbin,
    fit_logit,
    fit_panel_re,
)
from .utils       import reg_table, prepare_spec, add_year_trend, make_formula

# ── global styling ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.autolayout": True,
    "axes.titlesize"   : "x-large"
})

def load_data() -> pd.DataFrame:
    """
    Convenience: raw CSV → preprocessing → panel build.
    Returns a wide panel with columns including:
      Country_name, Continent, Year, Infectious_disease, CLIMATIC, …
    """
    raw = load_raw_csv()
    pp  = preprocess_counts(raw)
    return build_panel(pp)


# ──────────────────────────────────────────────────────────────
# Blueprint Figure 1: Trend plot by Continent ─────────────────
def plot_trend_panel(df: pd.DataFrame) -> alt.Chart:
    """
    Smoothed and normalized shock trends by continent and category.
    Each count is divided by the max for its (shock, continent) group.
    """
    shock_cols = [
        "CLIMATIC", "CONFLICTS", "ECOLOGICAL",
        "ECONOMIC", "GEOPHYSICAL", "TECHNOLOGICAL"
    ]
    missing = set(shock_cols) - set(df.columns)
    if missing:
        raise KeyError(f"Missing expected shock columns: {missing}")

    df_long = df.melt(
        id_vars=["Year", "Continent"],
        value_vars=shock_cols,
        var_name="Shock category",
        value_name="Count"
    )
    df_long["date"] = pd.to_datetime(df_long["Year"], format="%Y")

    # normalize counts within each (shock, continent)
    df_long["Normalized"] = (
        df_long
        .groupby(["Shock category", "Continent"])["Count"]
        .transform(lambda x: x / x.max() if x.max() > 0 else 0)
    )

    base = alt.Chart(df_long).encode(
        x=alt.X("date:T", title="Year"),
        y=alt.Y("Normalized:Q", title="Normalized count", scale=alt.Scale(zero=False)),
        color=alt.Color("Continent:N", legend=alt.Legend(title="Continent"))
    ).properties(
        width=500,
        height=300
    )

    line = base.mark_line().transform_loess(
        "date", "Normalized", groupby=["Continent", "Shock category"], bandwidth=0.5
    )

    return (
        line
        .facet(
            row=alt.Row("Shock category:N", title=None)
        )
        .resolve_scale(y="independent")
        .properties(title="Normalized smoothed shock trends by category and continent")
    )



# ──────────────────────────────────────────────────────────────
# Blueprint Figure 2: Forest plot of baseline models ───────────
def plot_baseline_forest(df: pd.DataFrame) -> plt.Figure:
    """
    Coefficient forest plot comparing:
      • Poisson FE + trend
      • NegBin FE + trend
      • Poisson random-effects (by Continent)
    """
    # 1) Fit the three models (FE + RE on Continent)
    df_fe, y, X, formula = build_spec(
        df, outcome_mode="count", pred_mode="count",
        add_year_trend=True, fe={"Continent": True}
    )
    res1 = fit_poisson(df_fe, formula)
    res2 = fit_negbin(df_fe, formula)
    df_re = df_fe.set_index(["Continent", "Year"])
    res3 = fit_panel_re(df_re, y=y, X=X)

    # 2) Stack estimates
    records = []
    for res, label in [
        (res1, "Poisson FE"),
        (res2, "NegBin FE"),
        (res3, "Poisson RE"),
    ]:
        ci = res.conf_int()
        records.append(pd.DataFrame({
            "Term": res.params.index,
            "Estimate": res.params.values,
            "Lower": ci.iloc[:, 0].values,
            "Upper": ci.iloc[:, 1].values,
            "Model": label,
        }))
    tab = pd.concat(records, ignore_index=True)

    # 3) Plot with aligned y-axis
    palette = sns.color_palette("colorblind", n_colors=3)
    models = tab["Model"].unique()
    terms = tab["Term"].unique()
    tab["Term"] = pd.Categorical(tab["Term"], categories=terms[::-1], ordered=True)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.5 * len(terms))))
    for i, model in enumerate(models):
        sub = tab[tab["Model"] == model]
        ypos = np.arange(len(sub))
        ax.errorbar(
            sub["Estimate"], ypos,
            xerr=[sub["Estimate"] - sub["Lower"], sub["Upper"] - sub["Estimate"]],
            fmt="o", label=model, color=palette[i],
            capsize=4, elinewidth=2, markersize=6
        )

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(terms)))
    ax.set_yticklabels(terms[::-1], fontsize=12)
    ax.set_xlabel("Coefficient estimate", fontsize=13)
    ax.set_title("Baseline model comparison: FE vs. RE (Continent)", fontsize=14)
    ax.tick_params(axis="x", labelsize=11)
    ax.legend(title="Model", loc="center left", bbox_to_anchor=(1.02, 0.5))
    sns.despine()
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Register blueprint figures here
# ──────────────────────────────────────────────────────────────
FIGURES: dict[str, Callable[[pd.DataFrame], plt.Figure | alt.Chart]] = {
    "Trend plot": plot_trend_panel,
    "Baseline forest plot": plot_baseline_forest,
}


# ──────────────────────────────────────────────────────────────
# Blueprint Tables: Key regressions & robustness
# ──────────────────────────────────────────────────────────────
def table_baseline(df: pd.DataFrame) -> str:
    """
    Tab 1: Poisson FE, NegBin FE, Poisson RE regression table.
    Returns HTML string for Streamlit.
    """
    df_fe, y, X, formula = build_spec(
        df,
        outcome_mode="count",
        pred_mode="count",
        add_year_trend=True,
        fe={"Continent": True},
    )
    res1 = fit_poisson(df_fe, formula)
    res2 = fit_negbin(df_fe, formula)
    df_re = df_fe.set_index(["Continent", "Year"])
    res3 = fit_panel_re(df_re, y=y, X=X)

    # Patch linearmodels object to behave like statsmodels for summary_col
    if not hasattr(res3, "bse") and hasattr(res3, "std_errors"):
        res3.bse = res3.std_errors
    if not hasattr(res3, "tvalues") and hasattr(res3, "tstats"):
        res3.tvalues = res3.tstats
    if not hasattr(res3.model, "exog_names"):
        res3.model.exog_names = list(res3.params.index)
    if not hasattr(res3.model, "endog_names"):
        res3.model.endog_names = y

    return reg_table(
        [res1, res2, res3],
        ["Poisson FE", "NegBin FE", "Poisson RE"],
        output="html",
    )

def table_binary_robust(df: pd.DataFrame) -> str:
    """
    Tab 2: Logit FE on binary outcome.
    Returns HTML string for Streamlit.
    """
    df2, y2, X2, formula2 = build_spec(
        df,
        outcome_mode="binary",
        pred_mode="count",
        add_year_trend=True,
        fe={"Continent": True},
    )
    res = fit_logit(df2, formula2)
    return reg_table([res], ["Logit FE"], output="html")


TABLES: dict[str, Callable[[pd.DataFrame], str]] = {
    "Baseline regression (Tab 1)": table_baseline,
    "Binary logit robustness (Tab 2)": table_binary_robust,
}

def plot_lagged_coefficients(df: pd.DataFrame, max_lag: int = 5) -> plt.Figure:
    """
    Plot coefficient paths across lags for each shock category (Poisson FE).
    """
    base_cols = ["CLIMATIC", "CONFLICTS", "ECOLOGICAL", "ECONOMIC", "GEOPHYSICAL", "TECHNOLOGICAL"]
    df_spec, y, X0 = prepare_spec(df, outcome_mode="count", pred_mode="count", shock_cols=base_cols)

    # add lags
    df_spec = add_lags(df_spec, base_cols, max_lag)
    lagged_cols = [f"{c}_lag{l}" for c in base_cols for l in range(1, max_lag + 1)]
    X = X0 + lagged_cols

    df_spec = add_year_trend(df_spec)
    formula = make_formula(y, X, year_trend=True, fe={"Continent": True})
    res = fit_poisson(df_spec, formula, cluster=None)

    # collect coefficient estimates + CI
    coefs = res.params
    ci = res.conf_int()
    df_plot = []
    for base in base_cols:
        for lag in range(1, max_lag + 1):
            name = f"{base}_lag{lag}"
            if name in coefs.index:
                df_plot.append({
                    "Shock": base,
                    "Lag": lag,
                    "Coef": coefs[name],
                    "Lower": ci.loc[name][0],
                    "Upper": ci.loc[name][1],
                })
    tab = pd.DataFrame(df_plot)

    # plot
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("Set2", n_colors=len(base_cols))
    sns.lineplot(data=tab, x="Lag", y="Coef", hue="Shock", marker="o", ax=ax, palette=palette)

    for i, shock in enumerate(tab["Shock"].unique()):
        sub = tab[tab["Shock"] == shock]
        ax.fill_between(
            sub["Lag"],
            sub["Lower"],
            sub["Upper"],
            alpha=0.2,
            color=palette[i]
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Lagged coefficient paths (Poisson FE)", fontsize=14)
    ax.set_xlabel("Lag (years)", fontsize=12)
    ax.set_ylabel("Coefficient", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)

    # Smaller legend, outside
    leg = ax.legend(
        title="Shock category",
        title_fontsize=10,
        fontsize=9,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    sns.despine()
    fig.tight_layout()
    return fig

def plot_lag_irrs(model, title="IRRs by Relative Year"):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Get conf_int with proper term names
    conf_int = model.conf_int()
    conf_int.columns = ['lower', 'upper']
    conf_int = conf_int.reset_index().rename(columns={'index': 'term'})

    # Get coef terms
    coefs = model.params.reset_index()
    coefs.columns = ['term', 'estimate']

    # Merge safely
    df_plot = pd.merge(coefs, conf_int, on='term')

    # 🔍 Filter only the Year_rel terms — find the correct prefix
    df_plot = df_plot[df_plot['term'].str.contains('Year_rel')]

    # Extract numeric Year_rel value
    df_plot['Year_rel'] = df_plot['term'].str.extract(r'\[T\.(-?\d+)\]')
    df_plot['Year_rel'] = pd.to_numeric(df_plot['Year_rel'], errors='coerce')

    # Drop rows with missing extracted year
    df_plot = df_plot.dropna(subset=['Year_rel', 'estimate', 'lower', 'upper'])

    # Plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_plot, x='Year_rel', y='estimate', marker='o')
    plt.fill_between(df_plot['Year_rel'], df_plot['lower'], df_plot['upper'], alpha=0.2)
    plt.axhline(0, linestyle='--', color='gray')
    plt.title(title)
    plt.xlabel("Years relative to outbreak")
    plt.ylabel("Log IRR (coefficient)")
    plt.tight_layout()

    return plt.gcf()


def plot_lag_effects_logit(model, lag_vars, shock):

    coefs = model.params[lag_vars]
    conf = model.conf_int().loc[lag_vars]

    # ✅ Convert 'm1' → -1, 'p1' → 1, '0' → 0
    def parse_lag(var):
        part = var.split("_lag_")[-1]
        if part.startswith("m"):
            return -int(part[1:])
        elif part.startswith("p"):
            return int(part[1:])
        else:
            return int(part)

    lags = [parse_lag(v) for v in lag_vars]

    df_plot = pd.DataFrame({
        "Year_rel": lags,
        "OR": np.exp(coefs.values),
        "lower": np.exp(conf[0].values),
        "upper": np.exp(conf[1].values)
    }).sort_values("Year_rel")

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_plot, x="Year_rel", y="OR", marker="o")
    plt.fill_between(df_plot["Year_rel"], df_plot["lower"], df_plot["upper"], alpha=0.2)
    plt.axhline(1.0, linestyle="--", color="gray")
    plt.title(f"Odds Ratios of {shock} by Lag (Logistic Regression)")
    plt.xlabel("Years before/after outbreak")
    plt.ylabel("Odds Ratio (95% CI)")
    plt.tight_layout()

    return plt.gcf()


__all__ = ["load_data", "FIGURES", "TABLES"]
