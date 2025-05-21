import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import country_converter as coco
import statsmodels.api as sm

st.set_page_config(page_title="Data Overview", layout="wide")
st.title("📚 Data Overview")
shock_df = pd.read_csv("data/01_raw/Shocks_Database_counts.csv")
shock_df.columns = shock_df.columns.str.replace(' ', '_')

rem_shock = ["Impact","Oil spill","Radiation","Extrasystemic conflict"]
time_span = [7013, 7019, 9619]
time_span_val = time_span[2]
shock_df = shock_df[shock_df["Shock_category"].notnull()]
shock_df["Country_name"] = shock_df["Country_name"].str.replace("TÃ¼rkiye", "Türkiye", regex=False)

@st.cache_data
def add_continent_column(country_names):
    return coco.convert(names=country_names, src="name_short", to="continent")
shock_df["Continent"] = add_continent_column(shock_df["Country_name"].tolist())
shock_df = shock_df[~shock_df["Shock_type"].isin(rem_shock)]

if time_span_val == 9619:
    shock_df = shock_df[(shock_df["Year"] > 1990) & (shock_df["Year"] < 2020)]
shock_df["Shock_comb"] = shock_df["Shock_category"].str[:5] + ":" + shock_df["Shock_type"]
shock_don_df = shock_df[shock_df["Shock_comb"] != "ECOLO:Infectious disease"]

def _plot_series(series, series_name, series_index=0):
    palette = list(sns.palettes.mpl_palette('Dark2'))
    counted = (series['Year']
               .value_counts()
               .reset_index(name='counts')
               .rename({'index': 'Year'}, axis=1)
               .sort_values('Year', ascending=True))
    xs = counted['Year']
    ys = counted['counts']
    plt.plot(xs, ys, label=series_name, color=palette[series_index % len(palette)])

fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
df_sorted = shock_don_df.sort_values('Year', ascending=True)

for i, (series_name, series) in enumerate(df_sorted.groupby('Continent')):
    _plot_series(series, series_name, i)

fig.legend(title='Continent', bbox_to_anchor=(1, 1), loc='upper left')
sns.despine(fig=fig, ax=ax)
plt.xlabel('Year')
plt.ylabel('Shock count')
st.pyplot(fig)

st.subheader("Panel data approach")
df_shocks = pd.read_csv('data/01_raw/Shocks_Database_counts.csv')
df_shocks = df_shocks.rename(columns={'Country name': 'Country', 'Shock category': 'Shock_category', 'Shock type': 'Shock_type'})
df_shocks['Shock_Category_Type'] = df_shocks['Shock_category'] + ' - ' + df_shocks['Shock_type']
df_shocks['Year'] = pd.to_datetime(df_shocks['Year'], format='%Y')
shock_counts = df_shocks.groupby(['Country', 'Year']).size().reset_index(name='shock_count')
shock_counts['Year'] = shock_counts['Year'].dt.year

st.subheader("Shocks data")
st.dataframe(shock_counts)
st.subheader("DON data")

df_don = pd.read_csv('data/01_raw/DONdatabase.csv')
df_don = df_don[['DONid', 'ReportDate', 'DiseaseLevel1', 'Country', 'ISO', 'CasesTotal', 'Deaths']]
df_don['Year'] = pd.to_datetime(df_don['ReportDate']).dt.year
disease_counts = df_don.groupby(['Country', 'Year']).size().reset_index(name='disease_count')

st.subheader("Combined data")
df = pd.merge(disease_counts, shock_counts, on=['Country', 'Year'], how='inner')
df = df.sort_values(['Country', 'Year'])
for lag in range(1, 6):
    df[f'shock_count_lag{lag}'] = df.groupby('Country')['shock_count'].shift(lag)

st.dataframe(df)
st.subheader("Poisson model")
X = df[['shock_count', 'shock_count_lag1', 'shock_count_lag2', 'shock_count_lag3', 'shock_count_lag4', 'shock_count_lag5']]
X = sm.add_constant(X)
y = df['disease_count']
X_clean = X.dropna()
y_clean = y[X_clean.index]

poisson_model = sm.GLM(y_clean, X_clean, family=sm.families.Poisson()).fit()
st.subheader("Poisson model summary")
st.write(poisson_model.summary())

