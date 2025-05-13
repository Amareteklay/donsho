# src/prep.py

import pandas as pd
import country_converter as coco
from src.config import YEAR_MIN, YEAR_MAX, RARE_THRESHOLD

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names: strip, replace spaces/hyphens with underscores.
    """
    cols = (
        df.columns
          .str.strip()
          .str.replace(r"[ \-]+", "_", regex=True)
    )
    df = df.copy()
    df.columns = cols
    return df

def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove completely duplicated rows."""
    return df.drop_duplicates()

def filter_years_and_required(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep rows in the defined year range and drop those missing a shock category.
    """
    return df.query("@YEAR_MIN <= Year <= @YEAR_MAX").dropna(subset=["Shock_category"])

def fix_country_typos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply known corrections to country names.
    """
    df = df.copy()
    df["Country_name"] = df["Country_name"].str.replace("TÃ¼rkiye", "Türkiye", regex=False)
    return df

def add_shock_comb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a combined shock label: first 5 chars of category + shock type.
    """
    df = df.copy()
    df["Shock_comb"] = df["Shock_category"].str[:5] + ":" + df["Shock_type"]
    return df

def add_continent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map 'country_name' to a 'continent' using the country_converter package.
    """
    df = df.copy()
    countries = df["Country_name"].unique()
    continents = coco.convert(
        names=countries,
        src="name_short",
        to="Continent",
        not_found=None
    )
    mapping = dict(zip(countries, continents))
    df["Continent"] = df["Country_name"].map(mapping)
    return df

def rare_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert rare shock types (below RARE_THRESHOLD) into binary 0/1 for 'count'.
    """
    df = df.copy()
    rare_types = df["Shock_type"].value_counts()
    rare = rare_types[rare_types < RARE_THRESHOLD].index
    mask = df["Shock_type"].isin(rare)
    df.loc[mask, "count"] = (df.loc[mask, "count"] > 0).astype(int)
    return df

def preprocess_counts(df: pd.DataFrame, save_path: str | None = None) -> pd.DataFrame:
    """
    Full pipeline from raw → cleaned. Optionally save result to disk.
    """
    df = (
        df
        .pipe(clean_column_names)
        .pipe(drop_exact_duplicates)
        .pipe(filter_years_and_required)
        .pipe(fix_country_typos)
        .pipe(add_shock_comb)
        .pipe(add_continent)
        .pipe(rare_to_binary)
    )
    if save_path:
        from .data_loader import save_interim
        save_interim(df, save_path)
    return df

def validate_counts_format(df: pd.DataFrame) -> None:
    """
    Raise if required columns are missing or malformed.
    Use before modeling to catch pipeline errors.
    """
    required = {"Country_name", "Shock_type", "Shock_category", "count", "Year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
