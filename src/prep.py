

def clean_counts(df):
    """
    Cleans the counts dataframe by removing unnecessary columns and renaming others.
    
    Parameters:
    df (pd.DataFrame): The raw counts dataframe.
    
    Returns:
    pd.DataFrame: The cleaned counts dataframe.
    """
    # Drop unnecessary columns
    df = df.drop(columns=["Unnamed: 0", "Unnamed: 1", "Unnamed: 2"])
    
    # Rename columns
    df.columns = ["Country", "Year", "Count"]
    
    # Convert 'Year' to string type
    df["Year"] = df["Year"].astype(str)
    
    return df