import argparse
import pandas as pd
from src.data_loader import load_and_clean_counts

def main(lag, thr):
    df = load_and_clean_counts()
    # your old prep logic here, using src.prep & src.features...
    df.to_parquet(f"data/02_interim/panel_lag{lag}_thr{thr}.parquet")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lag", type=int, default=5)
    p.add_argument("--thr", type=int, default=10)
    args = p.parse_args()
    main(args.lag, args.thr)
