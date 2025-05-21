import scipy.stats as stats
import pandas as pd

def lag_correlation_analysis(df):
    """Calculate correlation between lagged shocks (1 to 5 years) and outbreak occurrence."""
    
    # Extract relevant columns for correlation analysis
    lag_cols = [f'Shock_t-{i}' for i in range(1, 6)]
    
    # Compute Pearson correlations between lagged shocks and outbreaks
    correlations = {}
    for lag in lag_cols:
        if lag in df.columns:  # Ensure the lag column exists
            corr, p_value = stats.pearsonr(df[lag].fillna(0), df['Deaths'])
            correlations[lag] = {'Correlation': corr, 'P-value': p_value}
    
    # Convert results to a DataFrame for easy visualization
    corr_df = pd.DataFrame.from_dict(correlations, orient='index').reset_index()
    corr_df.rename(columns={'index': 'Lag Variable'}, inplace=True)
    
    return corr_df

def cooccurrence_analysis(df):
    """Calculate percentage of co-occurrences of shocks and disease outbreaks."""
    total_outbreaks = df[['Country', 'Year']].drop_duplicates().shape[0]
    cooccur_df = df.groupby('Shock_category').size().reset_index(name='cooccurrence_count')
    cooccur_df['percentage'] = (cooccur_df['cooccurrence_count'] / total_outbreaks) * 100
    return cooccur_df.sort_values(by='percentage', ascending=False)

def regional_variation(df, region_df):
    """Analyze regional variations."""
    merged_df = pd.merge(df, region_df, on='Country', how='left')
    regional_stats = merged_df.groupby('Region').agg({
        'lag_years': ['mean', 'median'],
        'CasesTotal': 'sum',
        'Deaths': 'sum'
    }).reset_index()
    regional_stats.columns = ['Region', 'Mean Lag', 'Median Lag', 'Total Cases', 'Total Deaths']
    return regional_stats

def severity_analysis(df, population_df):
    """Calculate outbreak severity adjusted for population."""
    df_severity = df.merge(population_df, on=['Country', 'Year'], how='left')
    df_severity['CasesTotal'] = pd.to_numeric(df_severity['CasesTotal'], errors='coerce').fillna(0)
    df_severity['Deaths'] = pd.to_numeric(df_severity['Deaths'], errors='coerce').fillna(0)

    df_severity['cases_per_100k'] = (df_severity['CasesTotal'] / df_severity['Population']) * 100000
    df_severity['deaths_per_100k'] = (df_severity['Deaths'] / df_severity['Population']) * 100000
    return df_severity
