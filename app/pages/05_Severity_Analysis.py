import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import country_converter as coco

st.set_page_config(page_title="Severity Analysis", layout="wide")
st.title("📌 Severity Analysis")

def load_data(don_path, shocks_path):
    don_df = pd.read_csv(don_path)
    shocks_df = pd.read_csv(shocks_path)
    return don_df, shocks_df

def prepare_don_data(don_df):
    don_df['ReportDate'] = pd.to_datetime(don_df['ReportDate'], errors='coerce')
    don_df['Year'] = don_df['ReportDate'].dt.year
    don_df['Outbreak'] = 1
    don_df['Deaths'] = don_df['Deaths'].astype(str).str.replace('>', '', regex=False).str.strip()
    don_df['Deaths'] = pd.to_numeric(don_df['Deaths'], errors='coerce')
    don_df = don_df[['Country', 'DiseaseLevel1', 'Year', 'CasesTotal', 'Deaths']]
    don_df.dropna(subset=['Country', 'Year'], inplace=True)

    # Collapse by country-year
    don_agg = don_df.groupby(['Country', 'Year'], as_index=False).agg({
        'CasesTotal': 'sum',
        'Deaths': 'sum',
        'DiseaseLevel1': lambda x: ', '.join(set(x.dropna())),
    })
    don_agg['Outbreak'] = 1

    st.write(f"DON data: {don_agg.shape[0]} rows, {don_agg.shape[1]} columns")
    st.dataframe(don_agg.head(100), use_container_width=True)
    return don_agg

def _ensure_continent(df):
    if "Continent" in df.columns:
        return df

    df = df.copy()
    if "Country_name" in df.columns:
        df["Country_name"] = df["Country_name"].str.replace("TÃ¼rkiye", "Türkiye", regex=False)
    elif "Country" in df.columns:
        df["Country"] = df["Country"].str.replace("TÃ¼rkiye", "Türkiye", regex=False)

    cc = coco.convert(df["Country_name"].unique(), src="name_short", to="Continent", not_found=None)
    mapping = dict(zip(df["Country_name"].unique(), cc))
    df["Continent"] = df["Country_name"].map(mapping)
    return df


def pivot_shocks(shocks_df):
    shocks_df.columns = shocks_df.columns.str.replace(' ', '_')
    shocks_df = _ensure_continent(shocks_df)
    shocks_df.rename(columns={'Country_name': 'Country'}, inplace=True)
    shocks_df.dropna(subset=['Country', 'Year', 'Shock_category'], inplace=True)

    # Pivot categories to wide format
    wide = shocks_df.pivot_table(
        index=['Country', 'Continent', 'Year'],
        columns='Shock_category',
        values='count',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    wide.columns.name = None
    st.write(f"Pivoted shocks data: {wide.shape[0]} rows, {wide.shape[1]} columns")
    st.dataframe(wide.head(100), use_container_width=True)
    return wide

def merge_data(don_df, shocks_wide):
    merged = pd.merge(shocks_wide, don_df, on=['Country', 'Year'], how='left')

    # Ensure all years and countries are preserved
    merged['Outbreak'] = merged['Outbreak'].fillna(0).astype(int)
    merged['CasesTotal'] = merged['CasesTotal'].fillna(0)
    merged['Deaths'] = merged['Deaths'].fillna(0)

    # Check if Continent got dropped or lost
    if 'Continent' not in merged.columns:
        merged = _ensure_continent(merged)

    st.write(f"Merged panel: {merged.shape[0]} rows, {merged.shape[1]} columns")
    st.dataframe(merged.head(100), use_container_width=True)
    return merged

@st.cache_data
def load_and_clean_population(path):
    df_raw = pd.read_csv(path)
    df_long = df_raw.melt(
        id_vars=['Country Name'],
        value_vars=[str(year) for year in range(1960, 2024)],
        var_name='Year',
        value_name='Population'
    )
    df_long.rename(columns={'Country Name': 'Country'}, inplace=True)
    df_long.dropna(subset=['Population'], inplace=True)
    df_long['Year'] = df_long['Year'].astype(int)
    df_long['Population'] = pd.to_numeric(df_long['Population'], errors='coerce')
    df_long.dropna(subset=['Population'], inplace=True)
    st.write(f"Population data: {df_long.shape[0]} rows")
    return df_long[['Country', 'Year', 'Population']]

# === PIPELINE ===
don_df, shocks_df = load_data(
    'data/01_raw/DONdatabase.csv',
    'data/01_raw/Shocks_Database_counts.csv'
)

don_prepared = prepare_don_data(don_df)
shocks_wide = pivot_shocks(shocks_df)
merged_df = merge_data(don_prepared, shocks_wide)

population_df = load_and_clean_population('data/01_raw/population.csv')
st.header("🌍 Cleaned Population Data")
st.dataframe(population_df.head(10), use_container_width=True)

st.header("📊 Merged Data Preview")
st.dataframe(merged_df.head(100), use_container_width=True)


def build_combined_severity_panel(don_df, shocks_df, population_df):
    """Build a full severity panel with shocks, DON, and population-adjusted metrics."""
    # Clean shocks data
    shocks_df = shocks_df.copy()
    shocks_df.columns = shocks_df.columns.str.replace(" ", "_")
    shocks_df["Country_name"] = shocks_df["Country_name"].str.replace("TÃ¼rkiye", "Türkiye", regex=False)
    shocks_df = _ensure_continent(shocks_df)

    # 1 ▸ Infectious disease counts from shocks
    dv = (
        shocks_df[shocks_df["Shock_type"] == "Infectious disease"]
        .groupby(["Country_name", "Continent", "Year"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "Infectious_disease"})
    )

    # 2 ▸ Other shocks pivoted wide
    pred = shocks_df[shocks_df["Shock_type"] != "Infectious disease"]
    pred_wide = (
        pred.pivot_table(
            index=["Country_name", "Continent", "Year"],
            columns="Shock_category",
            values="count",
            aggfunc="sum",
            fill_value=0
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    # 3 ▸ Merge predictors and outcome
    shock_panel = pd.merge(pred_wide, dv, on=["Country_name", "Continent", "Year"], how="left")
    shock_panel["Infectious_disease"] = shock_panel["Infectious_disease"].fillna(0).astype(int)

    # 4 ▸ Clean and aggregate DON data
    don_df = don_df.copy()
    don_df["ReportDate"] = pd.to_datetime(don_df["ReportDate"], errors="coerce")
    don_df["Year"] = don_df["ReportDate"].dt.year
    don_df["Deaths"] = pd.to_numeric(don_df["Deaths"].astype(str).str.replace(">", "").str.strip(), errors="coerce")
    don_df["CasesTotal"] = pd.to_numeric(don_df["CasesTotal"], errors="coerce")
    don_df.dropna(subset=["Country", "Year"], inplace=True)

    don_agg = (
        don_df.groupby(["Country", "Year"], as_index=False)
        .agg({
            "CasesTotal": "sum",
            "Deaths": "sum",
            "DiseaseLevel1": lambda x: ", ".join(sorted(set(x.dropna())))
        })
    )

    # 5 ▸ Merge DON data into shock panel
    combined = pd.merge(
        shock_panel,
        don_agg,
        left_on=["Country_name", "Year"],
        right_on=["Country", "Year"],
        how="left"
    )
    combined.drop(columns=["Country"], inplace=True)

    # 6 ▸ Create balanced full panel (Country_name × Year 1990–2019)
    min_year = 1990
    max_year = 2019
    all_countries = set(shock_panel["Country_name"]).union(set(don_agg["Country"]))
    full_panel = pd.DataFrame([(c, y) for c in all_countries for y in range(min_year, max_year + 1)],
                              columns=["Country_name", "Year"])

    combined = pd.merge(full_panel, combined, on=["Country_name", "Year"], how="left")

    # 7 ▸ Restore continent info if lost
    combined = _ensure_continent(combined)

    # 8 ▸ Fill missing values
    shock_columns = [col for col in pred_wide.columns if col not in ["Country_name", "Continent", "Year"]]
    for col in shock_columns:
        combined[col] = combined[col].fillna(0).astype(int)
    combined["Infectious_disease"] = combined["Infectious_disease"].fillna(0).astype(int)
    combined["CasesTotal"] = combined["CasesTotal"].fillna(0)
    combined["Deaths"] = combined["Deaths"].fillna(0)

    # 9 ▸ Merge population
    population_df = population_df.rename(columns={"Country": "Country_name"})
    population_df["Country_name"] = population_df["Country_name"].str.replace("TÃ¼rkiye", "Türkiye", regex=False)
    combined = combined.merge(population_df, on=["Country_name", "Year"], how="left")

    # 10 ▸ Calculate severity metrics
    combined["cases_per_100k"] = (combined["CasesTotal"] / combined["Population"]) * 100000
    combined["deaths_per_100k"] = (combined["Deaths"] / combined["Population"]) * 100000

    return combined


# Load data
don_df = pd.read_csv("data/01_raw/DONdatabase.csv")
shocks_df = pd.read_csv("data/01_raw/Shocks_Database_counts.csv")
population_df = load_and_clean_population("data/01_raw/population.csv")

# Build full panel
severity_df = build_combined_severity_panel(don_df, shocks_df, population_df)

# Display
st.header("🧬 Full Severity Panel: DON + Infectious Counts + Predictors")
st.write(f"{severity_df.shape[0]} rows × {severity_df.shape[1]} columns")
st.dataframe(severity_df.head(50), use_container_width=True)

def plot_severity(severity_df, continent):
    """Plot population-adjusted severity for selected region (continent or global)."""
    if continent == "Global":
        df_geo = severity_df.copy()
        title = "Global"
    else:
        df_geo = severity_df[severity_df['Continent'] == continent].copy()
        title = continent

    # Aggregate per year
    yearly = (
        df_geo.groupby("Year", as_index=False)
        .agg({
            "CasesTotal": "sum",
            "Deaths": "sum",
            "Population": "sum"
        })
    )
    yearly["cases_per_100k"] = (yearly["CasesTotal"] / yearly["Population"]) * 100000
    yearly["deaths_per_100k"] = (yearly["Deaths"] / yearly["Population"]) * 100000

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly["Year"], yearly["cases_per_100k"], marker='o', label="Cases per 100k")
    ax.plot(yearly["Year"], yearly["deaths_per_100k"], marker='x', label="Deaths per 100k")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count per 100k population")
    ax.set_title(f"Outbreak Severity in {title}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig



geo_options = ['Global'] + sorted(severity_df['Continent'].dropna().unique().tolist())
continent = st.selectbox("Select a continent or Global view:", geo_options)
fig = plot_severity(severity_df.rename(columns={"Continent": "Continent"}), continent)
st.pyplot(fig)
