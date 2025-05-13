"""
Feature‑engineering helpers
───────────────────────────
• pivot_categories : wide panel of shock categories (country × year)
• add_lags         : distributed‑lag columns
• build_panel      : ready‑to‑model panel (predictors + outcome)
• build_event_panel: ±MAX_LAG window around every outbreak year

All helpers are pure functions and never mutate the input dataframe.
They also try to keep integer dtypes wherever possible so that later
transformations (e.g. log, binary) behave as expected in downstream code
(VS Code + PowerShell friendly).  
"""
from __future__ import annotations

from typing import List, Sequence
import numpy as np
import pandas as pd
import country_converter as coco

from .config import MAX_LAG

# ──────────────────────────────────────────────────────────────────────
# Internal utilities
# ----------------------------------------------------------------------

def _ensure_continent(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with a **Continent** column.

    The conversion is performed only when the column is missing so the
    function is effectively idempotent.
    """
    if "Continent" in df.columns:
        return df

    cc = coco.convert(df["Country_name"].unique(),  # type: ignore[arg-type]
                       src="name_short", to="Continent", not_found=None)
    mapping = dict(zip(df["Country_name"].unique(), cc))

    out = df.copy()
    out["Continent"] = out["Country_name"].map(mapping)
    return out


# ──────────────────────────────────────────────────────────────────────
# Public helpers
# ----------------------------------------------------------------------

def pivot_categories(df: pd.DataFrame, *, value_col: str = "count") -> pd.DataFrame:
    """Wide **country‑year** panel with one column per *Shock_category*.

    Parameters
    ----------
    df : tidy counts with columns  *Country_name*, *Year*, *Shock_category*,
         and the *value_col* (default **count**).
    value_col : the column that contains the numeric values to be summed.
    """
    df = _ensure_continent(df)

    wide = (
        df
        .pivot_table(index=["Country_name", "Continent", "Year"],
                     columns="Shock_category",
                     values=value_col,
                     aggfunc="sum",
                     fill_value=0)
        .rename_axis(None, axis=1)  # drop the pivoted level name
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
    """Append lagged versions of *cols* up to **max_lag** years.

    The panel is first sorted by *Country_name* and *Year* to guarantee
    that :pyfunc:`pandas.Series.shift` operates on a monotonically
    increasing time axis.
    """
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
    """Return a *wide* modelling panel with predictors + outcome.

    Predictors   – summed annual counts for every *Shock_category*
    Outcome      – annual counts of **outcome_shock** (default *Infectious disease*)
    """
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
    """±*max_lag* window around every *outcome_shock* outbreak.

    The resulting dataframe follows the *event‑study* convention with the
    following key columns:

    * **DON_year**   – calendar year of the outbreak (DON = date of notification)
    * **Year**       – calendar year inside the window
    * **Year_rel**   – relative year index (−max_lag … +max_lag)
    * **Infectious_disease** – counts in the *outbreak* year only (0 elsewhere)
    """
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
    panel = panel.merge(dv_rename, on=["Country_name", "Continent", "DON_year"], how="left")
    panel["Infectious_disease"] = panel.apply(
        lambda r: r["Infectious_disease"] if r["Year_rel"] == 0 else 0,
        axis=1,
    ).fillna(0).astype(int)

    # 7 ▸ tidy column order
    meta_cols = ["Country_name", "Continent", "DON_year", "Year", "Year_rel"]
    shock_cols = sorted(c for c in panel.columns if c not in meta_cols + ["Infectious_disease"])
    return panel[meta_cols + shock_cols + ["Infectious_disease"]]
