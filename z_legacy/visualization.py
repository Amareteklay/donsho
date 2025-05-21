import matplotlib.pyplot as plt

def plot_lag_correlation(corr_df):
    """Plot correlation between lagged shocks and outbreaks."""
    
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract values
    x_values = corr_df['Correlation']
    y_labels = corr_df['Lag Variable']
    xerr_values = 1.96 * corr_df['P-value']  # Approximate confidence interval using p-value
    
    # Plot correlation values with error bars
    ax.errorbar(
        x=x_values,
        y=y_labels,
        xerr=xerr_values,
        fmt='o',
        ecolor='gray',
        capsize=4,
        color='black'
    )
    # Reference line at 0 (no correlation)
    ax.axvline(x=0, color='black', linestyle='--')

    # Labels and title
    ax.set_xlabel("Correlation with Outbreak")
    ax.set_ylabel("Lagged Shock Variable")
    ax.set_title("Correlation Between Lagged Shocks and Outbreaks")
    ax.grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout()
    return fig

def plot_cooccurrence(cooccur_df):
    """Plot co-occurrence percentages."""
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(cooccur_df['Shock_category'], cooccur_df['percentage'])
    ax.set_xlabel('Percentage of Co-occurrence (%)')
    ax.set_ylabel('Shock_category')
    ax.set_title('Co-occurrence of Shocks and Disease Outbreaks')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    fig.tight_layout()
    return fig

def plot_regional_variation(regional_df):
    """Plot regional variation in lag."""
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(regional_df['Region'], regional_df['Mean Lag'])
    ax.set_xlabel('Region')
    ax.set_ylabel('Mean Lag (years)')
    ax.set_title('Regional Variation in Lag between Shocks and Outbreaks')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig

def plot_severity(severity_df, country):
    """Plot severity (cases and deaths per 100k) for a specific country."""
    df_country = severity_df[severity_df['Country'] == country]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df_country['Year'], df_country['cases_per_100k'], marker='o', label='Cases per 100k')
    ax.plot(df_country['Year'], df_country['deaths_per_100k'], marker='x', label='Deaths per 100k')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count per 100k population')
    ax.set_title(f'Severity of Outbreaks in {country}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    return fig
