# src/data_loader.py

from pathlib import Path
from functools import lru_cache
import pandas as pd
from src.config import RAW_COUNTS_PATH

@lru_cache(maxsize=1)
def load_raw_csv(path: str | Path = None) -> pd.DataFrame:
    """
    Load the raw counts CSV file as a DataFrame.
    This is cached to avoid redundant I/O.
    """
    csv_path = Path(path) if path else Path(RAW_COUNTS_PATH)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).parent.parent / csv_path
    return pd.read_csv(csv_path)

def save_interim(df: pd.DataFrame, path: str | Path) -> None:
    """
    Save a processed panel to disk as a Parquet file.
    Creates parent directories if needed.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

def load_interim(filename: str | Path) -> pd.DataFrame:
    """
    Load a saved interim panel from disk.
    Accepts either relative name (inside `data/02_interim`) or full path.
    """
    path = Path(filename)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / "data" / "02_interim" / path
    return pd.read_parquet(path)

def list_interim_panels() -> list[str]:
    """
    Return a list of all available panel files in the interim directory.
    """
    interim_dir = Path(__file__).parent.parent / "data" / "02_interim"
    return [p.name for p in interim_dir.glob("panel_*.parquet")]