import pandas as pd
#from src import prep

def load_and_clean_counts(path="data/01_raw/Shocks_Database_counts.csv"):
    df = pd.read_csv(path)
    return df
