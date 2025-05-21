### Code that I might need to reuse

st.subheader("Panel data approach")

df_shocks = pd.read_csv('data/Shocks_Database_counts.csv')

# Rename the 'Country name' column to 'Country'
df_shocks = df_shocks.rename(columns={'Country name': 'Country', 'Shock category': 'Shock_category', 'Shock type': 'Shock_type'})

# Create a new column 'Shock_Category_Type'
df_shocks['Shock_Category_Type'] = df_shocks['Shock_category'] + ' - ' + df_shocks['Shock_type']

# Convert 'Year' column to datetime format
df_shocks['Year'] = pd.to_datetime(df_shocks['Year'], format='%Y')

shock_counts = df_shocks.groupby(['Country', 'Year']).size().reset_index(name='shock_count')

# Convert 'Year' column in shock_counts to int
shock_counts['Year'] = shock_counts['Year'].dt.year

st.subheader("Shocks data")
st.dataframe(shock_counts)

st.subheader("DON data")

df_don = pd.read_csv('data/DONdatabase.csv')

df_don = df_don[['DONid', 'ReportDate', 'DiseaseLevel1', 'Country', 'ISO', 'CasesTotal', 'Deaths']]

df_don['Year'] = pd.to_datetime(df_don['ReportDate']).dt.year

disease_counts = df_don.groupby(['Country', 'Year']).size().reset_index(name='disease_count')

st.subheader("Combined data")

# Now perform the merge
df = pd.merge(disease_counts, shock_counts, on=['Country', 'Year'], how='inner')

# Ensure data is sorted by country and year
df = df.sort_values(['Country', 'Year'])  # or 'Year' if you renamed it

# Create lagged shock variables for up to 5 years
for lag in range(1, 6):
    df[f'shock_count_lag{lag}'] = df.groupby('Country')['shock_count'].shift(lag)

st.dataframe(df)

st.subheader("Poisson model")

# Specify the independent variables (including a constant)
X = df[['shock_count', 'shock_count_lag1', 'shock_count_lag2', 'shock_count_lag3', 'shock_count_lag4', 'shock_count_lag5']]
X = sm.add_constant(X)
y = df['disease_count']

# Drop rows with any missing values in X
X_clean = X.dropna()
y_clean = y[X_clean.index]  # align y with cleaned X

poisson_model = sm.GLM(y_clean, X_clean, family=sm.families.Poisson()).fit()

st.subheader("Poisson model summary")
st.write(poisson_model.summary())



#@st.cache_data
#def add_continent_column(country_names):
#    return coco.convert(names=country_names, src="name_short", to="continent")

#shock_df["Continent"] = add_continent_column(shock_df["Country_name"].tolist())