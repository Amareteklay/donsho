import pandas as pd
import streamlit as st
import country_converter as coco
from src.config import YEAR_MIN, YEAR_MAX


def load_data(don_path, shocks_path):
    """Load DON and Shocks datasets from CSV files."""
    don_df = pd.read_csv(don_path)
    shocks_df = pd.read_csv(shocks_path)
    return don_df, shocks_df


def prepare_don_data(don_df: pd.DataFrame) -> pd.DataFrame:
    """Clean DON dataset and aggregate repeated disease outbreaks by Country-Year."""

    # Parse dates and extract year
    don_df['ReportDate'] = pd.to_datetime(don_df['ReportDate'], errors='coerce')
    don_df['Year'] = don_df['ReportDate'].dt.year

    # Clean and convert numeric columns
    don_df['Deaths'] = don_df['Deaths'].astype(str).str.replace('>', '', regex=False).str.strip()
    don_df['Deaths'] = pd.to_numeric(don_df['Deaths'], errors='coerce')
    don_df['CasesTotal'] = pd.to_numeric(don_df['CasesTotal'], errors='coerce')

    # Keep only relevant columns
    don_df = don_df[['Country', 'DiseaseLevel1', 'Year', 'CasesTotal', 'Deaths']]

    # Aggregate by Country, Year, Disease
    don_df = (
        don_df
        .groupby(['Country', 'Year', 'DiseaseLevel1'], as_index=False)
        .agg({
            'CasesTotal': 'sum',
            'Deaths': 'sum'
        })
    )

    return don_df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip, replace spaces/hyphens with underscores."""
    cols = (
        df.columns
          .str.strip()
          .str.replace(r"[ \-]+", "_", regex=True)
    )
    df = df.copy()
    df.columns = cols
    df = df.rename(columns={"Country_name": "Country"})
    return df

def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove completely duplicated rows."""
    return df.drop_duplicates()

def filter_years_and_required(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows in the defined year range and drop those missing a shock category."""
    return df.query("@YEAR_MIN <= Year <= @YEAR_MAX").dropna(subset=["Shock_category"])

def fix_country_typos(df: pd.DataFrame) -> pd.DataFrame:
    """Apply known corrections to country names."""
    df = df.copy()
    df["Country"] = df["Country"].str.replace("TÃ¼rkiye", "Türkiye", regex=False)
    return df

def add_continent(df: pd.DataFrame) -> pd.DataFrame:
    """Map 'country' to a 'continent' using the country_converter package."""
    df = df.copy()
    countries = df["Country"].unique()
    continents = coco.convert(
        names=countries,
        src="name_short",
        to="Continent",
        not_found=None
    )
    mapping = dict(zip(countries, continents))
    df["Continent"] = df["Country"].map(mapping)
    return df

def prepare_shocks_data(shocks_df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline for shock data."""
    shocks_df = (
        shocks_df
        .pipe(clean_column_names)
        .pipe(drop_exact_duplicates)
        .pipe(filter_years_and_required)
        .pipe(fix_country_typos)
        .pipe(add_continent)
    )
    shocks_agg = shocks_df.groupby(['Country', 'Continent', 'Year', 'Shock_category', 'Shock_type']).agg({'count': 'sum'}).reset_index()
    return shocks_agg

def get_processed_data(don_path, shocks_path):
    """Pipeline to load, process, and merge DON and shocks data."""
    don_df, shocks_df = load_data(don_path, shocks_path)
    don_df = prepare_don_data(don_df)
    shocks_df = prepare_shocks_data(shocks_df)
    return don_df, shocks_df