from __future__ import annotations
from typing import Iterable, Mapping, Callable, Sequence
from statsmodels.iolib.summary2 import summary_col
import numpy as np
import pandas as pd


def add_binary(df: pd.DataFrame, cols: Iterable[str], *,
               suffix: str = "_bin") -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[f"{c}{suffix}"] = (out[c] > 0).astype("uint8")
    return out


def add_year_trend(df: pd.DataFrame, *, col: str = "Year",
                   new: str = "Year_trend") -> pd.DataFrame:
    out = df.copy()
    out[new] = df[col] - df[col].min()
    return out


def prepare_spec(
    panel: pd.DataFrame,
    *,
    outcome_mode: str = "count",      
    pred_mode: str = "count",        
    shock_cols: Iterable[str] | None = None,
    suffix: str = "_bin",
) -> tuple[pd.DataFrame, str, list[str]]:
    if shock_cols is None:
        shock_cols = [c for c in panel.columns
                      if c not in ("Country_name", "Continent", "Year", "Infectious_disease")]

    df = panel.copy()

    # ▸ outcome
    if outcome_mode == "binary":
        df["Infectious_disease_bin"] = (df["Infectious_disease"] > 0).astype("uint8")
        y_col = "Infectious_disease_bin"
    else:
        y_col = "Infectious_disease"

    # ▸ predictors
    if pred_mode == "binary":
        df = add_binary(df, shock_cols, suffix=suffix)
        X_cols = [f"{c}{suffix}" for c in shock_cols]
    else:
        X_cols = list(shock_cols)

    return df, y_col, X_cols

def make_formula(
    y: str,
    X_cols: Iterable[str],
    *,
    year_trend: bool = False,
    fe: Mapping[str, bool] | None = None,  # e.g. {"Country_name": True, "Continent": False}
) -> str:
    """
    Returns a Patsy / statsmodels formula string.
    """
    fe = fe or {}
    items = list(X_cols)

    if year_trend:
        items.append("Year_trend")

    # add fixed-effect dummies the Patsy way: C(var)
    for v, is_fe in fe.items():
        if is_fe:
            items.append(f"C({v})")

    rhs = " + ".join(items) if items else "1"
    return f"{y} ~ {rhs}"


def reg_table(
    results: Sequence,                   # list of fitted models
    model_names: Sequence[str] | None = None,
    stars: bool = True,
    info: Mapping[str, Callable] | None = None,
    float_fmt: str = "%.3f",
    output: str = "dataframe",           # "dataframe" | "latex" | "html"
):
    """
    Build a regression table like Stata's `esttab`.

    Parameters
    ----------
    results      list of statsmodels results objects
    model_names  column headers, e.g. ["(1)", "(2)", "(3)"]
    stars        add significance stars *, **, ***
    info         extra rows:  {"N": lambda r: f"{int(r.nobs):,}"}
    float_fmt    python float format for coeffs / SEs
    output       return type

    Returns
    -------
    pd.DataFrame | str
    """
    if model_names is None:
        model_names = [f"({i+1})" for i in range(len(results))]

    info_dict = {k: (v if callable(v) else (lambda _: v))
                 for k, v in (info or {}).items()}

    sm_table = summary_col(
        results,
        stars=stars,
        model_names=model_names,
        info_dict=info_dict,
        float_format=float_fmt,
        regressor_order=None,    # keep original order
    )

    if output == "latex":
        return sm_table.as_latex()
    if output == "html":
        return sm_table.as_html()
    return pd.read_html(sm_table.as_html(), header=0, index_col=0)[0]