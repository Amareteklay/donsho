from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
import country_converter as coco
from .config import MAX_LAG


def pivot_shock_categories(df: pd.DataFrame) -> pd.DataFrame:
    wide_df = (
        df
        .pivot_table(index=["Country", "Continent", "Year"],
                     columns="Shock_category",
                     values="count",
                     aggfunc="sum",
                     fill_value=0)
        .rename_axis(None, axis=1)
        .reset_index()
        .sort_values(["Country", "Year"], ignore_index=True)
    )
    numeric_cols = wide_df.columns.difference(["Country", "Continent", "Year"])
    for col in numeric_cols:
        wide_df[col] = pd.to_numeric(wide_df[col], errors='coerce').fillna(0).astype(int)
    return wide_df


def add_leads_and_lags(
    df: pd.DataFrame,
    cols: Sequence[str],
    max_lag: int,
    group_col: str = "Country",
    time_col: str = "Year"
) -> pd.DataFrame:
    df = df.sort_values([group_col, time_col]).copy()
    for col in cols:
        grp = df.groupby(group_col)[col]
        for l in range(1, max_lag + 1):
            df[f"{col}_lag{l}"] = grp.shift(l)
            df[f"{col}_lead{l}"] = grp.shift(-l)
    return df


def get_disease_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["Shock_type"] == "Infectious disease"]
          .groupby(["Country", "Continent", "Year"], as_index=False)["count"]
          .sum()
          .rename(columns={"count": "Infectious_disease"})
    )


def get_event_grid(dv: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    events = dv.query("Infectious_disease > 0").rename(columns={"Year": "DON_year"})
    events = events.drop(columns="Infectious_disease").assign(key=1)
    lags = pd.DataFrame({"Year_rel": np.arange(-max_lag, max_lag + 1), "key": 1})
    evt_grid = events.merge(lags, on="key").drop(columns="key")
    evt_grid["Year"] = evt_grid["DON_year"] + evt_grid["Year_rel"]
    return evt_grid


def get_predictors_panel(df: pd.DataFrame) -> pd.DataFrame:
    return pivot_shock_categories(df[df["Shock_type"] != "Infectious disease"])


def merge_event_data(evt_grid, predictors, dv, don_df):
    panel = evt_grid.merge(predictors, on=["Country", "Continent", "Year"], how="left").fillna(0)

    # Convert sparse predictors to binary
    for col in ["ECOLOGICAL", "GEOPHYSICAL"]:
        if col in panel.columns:
            panel[col] = (panel[col] > 0).astype(int)

    # Add disease label
    dv_rename = dv.rename(columns={"Year": "DON_year"})
    panel = panel.merge(dv_rename, on=["Country", "Continent", "DON_year"], how="left")

    # Add DON data
    don_slim = (
        don_df
        .groupby(["Country", "Year"], as_index=False)[["CasesTotal", "Deaths"]]
        .sum()
        .rename(columns={"Year": "DON_year"})
    )
    panel = panel.merge(don_slim, on=["Country", "DON_year"], how="left")

    panel["Infectious_disease"] = panel["Infectious_disease"].fillna(0).astype(int)
    panel["CasesTotal"] = panel["CasesTotal"].fillna(0).astype(int)
    panel["Deaths"] = panel["Deaths"].fillna(0).astype(int)

    return panel


def reorder_panel_columns(panel: pd.DataFrame) -> pd.DataFrame:
    has_don_year = "DON_year" in panel.columns
    has_year_rel = "Year_rel" in panel.columns

    meta = ["Country", "Continent"]
    if has_don_year:
        meta.append("DON_year")
    meta.append("Year")

    outcome_block = ["Infectious_disease", "CasesTotal", "Deaths"]
    if has_year_rel:
        outcome_block.append("Year_rel")

    predictors = sorted(
        c for c in panel.columns if c not in meta + outcome_block
    )

    return panel[meta + outcome_block + predictors]


def build_event_panel(
    df: pd.DataFrame,
    *,
    don_df: pd.DataFrame,
    max_lag: int = MAX_LAG
) -> pd.DataFrame:
    dv = get_disease_counts(df)
    evt_grid = get_event_grid(dv, max_lag)
    predictors = get_predictors_panel(df)
    panel = merge_event_data(evt_grid, predictors, dv, don_df)
    panel = reorder_panel_columns(panel)

    cols_to_lag = [
        col for col in panel.columns
        if col not in ["Country", "Continent", "DON_year", "Year", "Infectious_disease", "CasesTotal", "Deaths", "Year_rel"]
    ]
    panel = add_leads_and_lags(panel, cols=cols_to_lag, max_lag=max_lag)
    return panel


def build_full_panel(
    shocks_df: pd.DataFrame,
    don_df: pd.DataFrame,
    max_lag: int = MAX_LAG
) -> pd.DataFrame:
    full_index = shocks_df[["Country", "Continent", "Year"]].drop_duplicates()
    predictors = get_predictors_panel(shocks_df)

    # Convert sparse predictors to binary (in-place)
    for col in ["ECOLOGICAL", "GEOPHYSICAL"]:
        if col in predictors.columns:
            predictors[col] = (predictors[col] > 0).astype(int)

    panel = full_index.merge(predictors, on=["Country", "Continent", "Year"], how="left").fillna(0)

    dv = get_disease_counts(shocks_df)
    panel = panel.merge(dv, on=["Country", "Continent", "Year"], how="left")
    panel["Infectious_disease"] = panel["Infectious_disease"].fillna(0).astype(int)

    don_slim = (
        don_df
        .groupby(["Country", "Year"], as_index=False)[["CasesTotal", "Deaths"]]
        .sum()
    )
    panel = panel.merge(don_slim, on=["Country", "Year"], how="left")
    panel["CasesTotal"] = panel["CasesTotal"].fillna(0).astype(int)
    panel["Deaths"] = panel["Deaths"].fillna(0).astype(int)

    predictors_to_lag = sorted(
        col for col in panel.columns
        if col not in ["Country", "Continent", "Year", "Infectious_disease", "CasesTotal", "Deaths"]
    )
    panel = reorder_panel_columns(panel)
    panel = add_leads_and_lags(panel, cols=predictors_to_lag, max_lag=max_lag)

    return panel
