import pandas as pd

def show_head(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the first n rows of the DataFrame."""
    return df.head(n)

def show_info(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of dtypes, non-null counts, and null stats for each column."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null_count": df.count(),
        "null_count": df.isna().sum(),
        "pct_null": df.isna().mean() * 100
    })

def show_description(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics for numeric and object columns."""
    return df.describe(include="all")
