"""
Shared data helpers for the glacier-wide discharge analysis (used by recommended_approach.ipynb).

    AWS_PROFILE=kopah pixi run -e local jupyter lab

load_monthly()   reads the glacier-wide table (discharge plus forcings) from S3, averages it to
                 monthly means, and caches the result in data/ so later runs work offline.
slow_component() is the centered 12-month running mean used to remove the long-term trend.
"""
import os

import pandas as pd

from config import S3_BUCKET, GLACIER_NAME, TARGET_COL, storage_options

RUNOFF = "Basin-Averaged Meltwater Runoff (mmWE/day)"
AIRT = "Air Temperature (C)"
LOCAL_CACHE = "data/monthly_non_spatial.parquet"


def load_monthly() -> pd.DataFrame:
    """Monthly means of the glacier-wide (non-spatial) table, indexed by month start. Cached locally."""
    if os.path.exists(LOCAL_CACHE):
        return pd.read_parquet(LOCAL_CACHE)
    raw = pd.read_parquet(f"{S3_BUCKET}/cjense/data/testmodel/{GLACIER_NAME}_non_spatial.parquet",
                          storage_options=storage_options)
    raw = raw.reset_index().drop(columns=["index"], errors="ignore")
    raw["time"] = pd.to_datetime(raw["time"]).dt.normalize()
    df = raw.resample("MS", on="time").mean()
    os.makedirs(os.path.dirname(LOCAL_CACHE), exist_ok=True)
    df.to_parquet(LOCAL_CACHE)
    return df


def slow_component(y: pd.Series) -> pd.Series:
    """Centered 12-month running mean (NaN within 6 months of either end of y)."""
    return y.rolling(12, center=True, min_periods=12).mean()
