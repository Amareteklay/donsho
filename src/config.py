from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent     # project root
DATA_RAW      = ROOT / "data" / "01_raw"
DATA_INTERIM  = ROOT / "data" / "02_interim"

# New: point directly at your raw counts CSV
RAW_COUNTS_PATH = DATA_RAW / "Shocks_Database_counts.csv"

YEAR_MIN, YEAR_MAX = 1990, 2019
RARE_THRESHOLD     = 10         
MAX_LAG            = 5
SEED               = 42

SAVED_MODELS_DIR = ROOT / "saved_models"
