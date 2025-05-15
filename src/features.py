from __future__ import annotations

from typing import List, Sequence
import numpy as np
import pandas as pd
import country_converter as coco

from .config import MAX_LAG

def _ensure_continent(df: pd.DataFrame) -> pd.DataFrame:
    if "Continent" in df.columns:
        return df

    cc = coco.convert(df["Country_name"].unique(), 
                       src="name_short", to="Continent", not_found=None)
    mapping = dict(zip(df["Country_name"].unique(), cc))

    out = df.copy()
    out["Continent"] = out["Country_name"].map(mapping)
    return out


def pivot_categories(df: pd.DataFrame, *, value_col: str = "count") -> pd.DataFrame:
    df = _ensure_continent(df)

    wide = (
        df
        .pivot_table(index=["Country_name", "Continent", "Year"],
                     columns="Shock_category",
                     values=value_col,
                     aggfunc="sum",
                     fill_value=0)
        .rename_axis(None, axis=1) 
        .reset_index()
        .sort_values(["Country_name", "Year"], ignore_index=True)
    )

    # ensure integer dtype when the source column was integer‑like
    numeric_cols = wide.columns.difference(["Country_name", "Continent", "Year"])
    wide[numeric_cols] = wide[numeric_cols].astype(int)
    return wide


def add_lags(panel: pd.DataFrame,
             cols: Sequence[str],
             max_lag: int = MAX_LAG) -> pd.DataFrame:
    if not cols or max_lag < 1:
        return panel.copy()

    out = panel.sort_values(["Country_name", "Year"], ignore_index=True).copy()

    for col in cols:
        grp = out.groupby("Country_name", sort=False)[col]
        for l in range(1, max_lag + 1):
            out[f"{col}_lag{l}"] = grp.shift(l)

    return out


def build_panel(df: pd.DataFrame,
                outcome_shock: str = "Infectious disease") -> pd.DataFrame:
    df = _ensure_continent(df)

    # 1 ▸ outcome (dependent variable)
    dv = (
        df.loc[df["Shock_type"] == outcome_shock]
          .groupby(["Country_name", "Continent", "Year"], as_index=False)["count"]
          .sum()
          .rename(columns={"count": "Infectious_disease"})
    )

    # 2 ▸ predictors
    pred = df.loc[df["Shock_type"] != outcome_shock]
    panel = pivot_categories(pred)

    # 3 ▸ merge predictors + outcome, fill missing with 0 and keep ints
    panel = panel.merge(dv, on=["Country_name", "Continent", "Year"], how="left")
    panel["Infectious_disease"] = panel["Infectious_disease"].fillna(0).astype(int)

    return panel


def build_event_panel(
    df: pd.DataFrame,
    *,
    outcome_shock: str = "Infectious disease",
    max_lag: int = MAX_LAG
) -> pd.DataFrame:
    df = _ensure_continent(df)

    # 1 ▸ disease counts per country‑year
    dv = (
        df.loc[df["Shock_type"] == outcome_shock]
          .groupby(["Country_name", "Continent", "Year"], as_index=False)["count"]
          .sum()
          .rename(columns={"count": "n_disease"})
    )

    # 2 ▸ outbreak events (n_disease > 0)
    events = dv.query("n_disease > 0").rename(columns={"Year": "DON_year"})
    events = events.drop(columns="n_disease").assign(key=1)

    # 3 ▸ relative year grid
    lags = pd.DataFrame({"Year_rel": np.arange(-max_lag, max_lag + 1), "key": 1})

    # 4 ▸ Cartesian join events × lags → event grid
    evt_grid = (events.merge(lags, on="key", how="outer")  # keep all events even if dv empty
                       .drop(columns="key"))
    evt_grid["Year"] = evt_grid["DON_year"] + evt_grid["Year_rel"]

    # 5 ▸ predictors for every year in the window (excluding outcome_shock rows)
    pred = df.loc[df["Shock_type"] != outcome_shock]
    wide = pivot_categories(pred)

    panel = (evt_grid
             .merge(wide, on=["Country_name", "Continent", "Year"], how="left")
             .fillna(0))

    # 6 ▸ outcome counts attached only to the *event* year (Year_rel == 0)
    dv_rename = dv.rename(columns={"Year": "DON_year", "n_disease": "Infectious_disease"})
    
    panel = panel.merge(
    dv.rename(columns={"Year": "Year", "n_disease": "Infectious_disease"}),
    on=["Country_name", "Continent", "Year"],
    how="left"
)
    panel["Infectious_disease"] = panel["Infectious_disease"].fillna(0).astype(int)


    # 7 ▸ tidy column order
    meta_cols = ["Country_name", "Continent", "DON_year", "Year", "Year_rel"]
    shock_cols = sorted(c for c in panel.columns if c not in meta_cols + ["Infectious_disease"])
    return panel[meta_cols + shock_cols + ["Infectious_disease"]]


def build_lagged_shock_features(panel_df, shock: str, max_lag=5):
    df = panel_df.copy()
    df = df[df["Year_rel"].between(-max_lag, max_lag)]
    df = df[["Country_name", "Continent", "DON_year", "Year_rel", shock, "Infectious_disease"]].copy()

    # Rename Year_rel for safe variable names
    def safe_lag(x):
        return f"{shock}_lag_m{abs(x)}" if x < 0 else (f"{shock}_lag_p{x}" if x > 0 else f"{shock}_lag_0")

    df["lag_col"] = df["Year_rel"].apply(safe_lag)
    df = df.pivot_table(index=["Country_name", "Continent", "DON_year"], 
                        columns="lag_col", values=shock).reset_index()

    # Add outcome at Year_rel = 0
    outcome = panel_df.query("Year_rel == 0")[["Country_name", "Continent", "DON_year", "Infectious_disease"]]
    result = pd.merge(outcome, df, on=["Country_name", "Continent", "DON_year"])
    result["Infectious_disease"] = (result["Infectious_disease"] > 0).astype(int)

    return result


def build_balanced_lagged_df(panel_df, shock: str, max_lag=5):
    df = panel_df.copy()
    df = df[df["Year_rel"].between(-max_lag, max_lag)]
    
    # Rename lags safely
    def safe_lag(x):
        return f"{shock}_lag_m{abs(x)}" if x < 0 else (f"{shock}_lag_p{x}" if x > 0 else f"{shock}_lag_0")

    df["lag_col"] = df["Year_rel"].apply(safe_lag)

    # Pivot into wide format with lagged shock values
    df_wide = df.pivot_table(index=["Country_name", "Continent", "DON_year"], 
                              columns="lag_col", values=shock).reset_index()

    # Grab the outcome for every DON_year (can be 1 or 0)
    outcome_df = df[df["Year_rel"] == 0][["Country_name", "Continent", "DON_year", "Infectious_disease"]].drop_duplicates()
    outcome_df["Infectious_disease"] = (outcome_df["Infectious_disease"] > 0).astype(int)

    result = pd.merge(outcome_df, df_wide, on=["Country_name", "Continent", "DON_year"])

    return result
