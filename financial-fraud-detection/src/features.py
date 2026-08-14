import numpy as np
import pandas as pd

TARGET = "Class"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features."""
    out = df.copy()
    if "Time" in out.columns:
        out["Hour"] = (out["Time"] / 3600.0) % 24
    if "Amount" in out.columns:
        out["AmountLog"] = np.log1p(out["Amount"])
    return out

def split_features_target(df: pd.DataFrame):
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}'")
    return engineer_features(df.drop(columns=[TARGET])), df[TARGET].astype(int)
