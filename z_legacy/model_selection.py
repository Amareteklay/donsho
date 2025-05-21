"""
Models helpers for shock panel analysis
--------------------------------------
This module is intentionally *stateless*: every function takes explicit
inputs and returns explicit outputs, so you can reuse it from Streamlit,
Quarto, or plain scripts.

Key ideas
~~~~~~~~~
* Data preparation (count → binary, year trend, FE dummies, …) lives in
  utils.py.  This module *imports* utils but never mutates the original
  panel.
* All fitting helpers expect a ready‑to‑use dataframe plus a Patsy
  formula string.  We supply small convenience wrappers to build both.

Dependencies
~~~~~~~~~~~~
numpy, pandas, statsmodels, linearmodels, scikit‑learn
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects

import utils

# ---------------------------------------------------------------------------
# 0 ▸ train / test split (time order) ----------------------------------------
# ---------------------------------------------------------------------------


def train_test_split_ts(
    df: pd.DataFrame, date_col: str, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(date_col)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]


# ---------------------------------------------------------------------------
# 1 ▸ spec builder -----------------------------------------------------------
# ---------------------------------------------------------------------------


def build_spec(
    panel: pd.DataFrame,
    *,
    outcome_mode: str = "count",
    pred_mode: str = "count",
    add_year_trend: bool = True,
    fe: Mapping[str, bool] | None = None,
    suffix: str = "_bin",
) -> Tuple[pd.DataFrame, str, list[str], str]:
    """
    Convenience wrapper around utils.prepare_spec + utils.add_year_trend +
    utils.make_formula.

    Returns
    -------
    df
        A *copy* of the panel with new columns as requested.
    y
        Name of the outcome column.
    X
        List of predictor column names.
    formula
        Patsy formula string ready for statsmodels.
    """
    df, y, X = utils.prepare_spec(
        panel, outcome_mode=outcome_mode, pred_mode=pred_mode, suffix=suffix
    )

    if add_year_trend:
        df = utils.add_year_trend(df)

    formula = utils.make_formula(y, X, year_trend=add_year_trend, fe=fe)
    return df, y, X, formula


# ---------------------------------------------------------------------------
# 2 ▸ model fit helpers ------------------------------------------------------
# ---------------------------------------------------------------------------


def fit_glm(
    df: pd.DataFrame,
    formula: str,
    family: sm.families.Family,
    cluster: str | None = "Country_name",
):
    """Generic GLM with optional clustered SE."""
    mod = smf.glm(formula=formula, data=df, family=family)
    if cluster:
        return mod.fit(cov_type="cluster", cov_kwds={"groups": df[cluster]})
    return mod.fit()


def fit_poisson(df: pd.DataFrame, formula: str, cluster: str | None = "Country_name"):
    return fit_glm(df, formula, sm.families.Poisson(), cluster)


def fit_negbin(df: pd.DataFrame, formula: str, cluster: str | None = "Country_name"):
    return fit_glm(df, formula, sm.families.NegativeBinomial(), cluster)


def fit_logit(df: pd.DataFrame, formula: str, cluster: str | None = "Country_name"):
    mod = smf.logit(formula=formula, data=df)
    if cluster:
        return mod.fit(
            cov_type="cluster",
            cov_kwds={"groups": df[cluster]},
            method="newton",
            maxiter=100,
        )
    return mod.fit(method="newton", maxiter=100)


def fit_mixed_effects(
    df: pd.DataFrame, formula: str, groups: str = "Country_name"
):
    """Random‑intercept linear (Gaussian) mixed model."""
    return smf.mixedlm(formula, df, groups=df[groups]).fit()


def fit_panel_re(
    df: pd.DataFrame, y: str, X: Sequence[str], entity: str = "Country_name"
):
    """
    Classical random effects (BLP) for continuous / count outcomes.

    Parameters
    ----------
    df
        Must be indexed by [entity, time] MultiIndex.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("df must have a MultiIndex [entity, time]")
    exog = sm.add_constant(df[list(X)])
    return RandomEffects(df[y], exog).fit()


# ---------------------------------------------------------------------------
# 3 ▸ evaluation -------------------------------------------------------------
# ---------------------------------------------------------------------------


def evaluate_regression(preds: np.ndarray, y: pd.Series) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, preds))),
        "R2": float(r2_score(y, preds)),
    }


def evaluate_classification(
    preds_prob: np.ndarray, y: pd.Series, threshold: float = 0.5
) -> Dict[str, float]:
    preds = (preds_prob >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y, preds)),
        "ROC_AUC": float(roc_auc_score(y, preds_prob)),
    }


def evaluate_model(
    model: Any,
    df: pd.DataFrame,
    y: str,
    *,
    kind: str = "auto",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Universal evaluation: decides regression vs classification based on dtype
    unless `kind` is supplied explicitly.
    """
    if kind == "auto":
        kind = "classification" if df[y].dropna().isin([0, 1]).all() else "regression"

    preds = model.predict(df)

    if kind == "classification":
        return evaluate_classification(preds, df[y], threshold=threshold)
    return evaluate_regression(preds, df[y])


# ---------------------------------------------------------------------------
# 4 ▸ sklearn grid search (optional) -----------------------------------------
# ---------------------------------------------------------------------------


def tune_hyperparameters(
    estimator, param_grid: dict, X: pd.DataFrame, y: pd.Series, cv: int = 5
):
    grid = GridSearchCV(estimator, param_grid, cv=cv)
    grid.fit(X, y)
    return grid.best_estimator_


# ---------------------------------------------------------------------------
# 5 ▸ data hygiene -----------------------------------------------------------
# ---------------------------------------------------------------------------


def filter_model_data(
    df: pd.DataFrame, outcome: str, predictors: Sequence[str]
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Drop rows with NaN/Inf in outcome or predictors; drop zero‑variance cols.
    """
    df = df.copy()
    cols = [outcome, *predictors]
    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=cols)
        .reset_index(drop=True)
    )

    zero_var = [col for col in predictors if df[col].std() == 0]
    keep = [col for col in predictors if col not in zero_var]
    return df.drop(columns=zero_var, errors="ignore"), keep

def fit_lag_model(panel_df):
    from model_selection import fit_logit
    from src.utils import make_formula
    import pandas as pd

    df = panel_df.copy()
    df["Infectious_disease"] = (df["Infectious_disease"] > 0).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    df["Year_rel"] = df["Year_rel"].astype(int)

    formula = "Infectious_disease ~ C(Year_rel) + C(Continent)"
    model = fit_logit(df, formula=formula, cluster="Country_name")

    return model


def fit_logit_lagged_shock(df: pd.DataFrame, shock: str):
    import statsmodels.formula.api as smf
    df["Infectious_disease"] = (df["Infectious_disease"] > 0).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    lag_vars = [col for col in df.columns if col.startswith(f"{shock}_lag_")]
    formula = "Infectious_disease ~ " + " + ".join(lag_vars)

    model = smf.logit(formula=formula, data=df).fit()
    return model, lag_vars

