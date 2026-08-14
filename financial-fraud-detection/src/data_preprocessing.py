from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"Time", "Amount", "Class"}

def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.dropna().copy()
