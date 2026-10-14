"""Data cleaning: duplicates, gaps, outliers, missing values."""
import numpy as np
import pandas as pd

from src import config as C


def load_raw() -> pd.DataFrame:
    if not C.RAW_FILE.exists():
        from src.data_generator import save_raw
        save_raw()
    return pd.read_csv(C.RAW_FILE, parse_dates=[C.TIMESTAMP_COL])


def clean(df: pd.DataFrame):
    """Return (clean_dataframe_indexed_by_time, report_dict)."""
    rep = {"rows_raw": len(df)}

    df = df.sort_values(C.TIMESTAMP_COL)
    before = len(df)
    df = df.drop_duplicates(subset=C.TIMESTAMP_COL).set_index(C.TIMESTAMP_COL)
    rep["duplicates_removed"] = before - len(df)

    # Make the time axis complete: one row for every hour
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", name=C.TIMESTAMP_COL)
    rep["missing_hours_added"] = len(full) - len(df)
    df = df.reindex(full)

    # Outliers: compare each value with the typical value for that weekday+hour
    typical = df[C.TARGET_COL].groupby([df.index.dayofweek, df.index.hour]).transform("median")
    ratio = df[C.TARGET_COL] / typical
    bad = (ratio > 2.0) | (ratio < 0.3)
    rep["outliers_removed"] = int(bad.sum())
    df.loc[bad, C.TARGET_COL] = np.nan

    # Fill every gap by interpolating between neighbouring hours
    rep["gaps_interpolated"] = int(df[C.TARGET_COL].isna().sum())
    for col in (C.TARGET_COL, C.TEMP_COL):
        df[col] = df[col].interpolate(method="time", limit_direction="both")
    df[C.TARGET_COL] = df[C.TARGET_COL].clip(lower=0)

    rep["rows_clean"] = len(df)
    rep["start"], rep["end"] = str(df.index.min()), str(df.index.max())
    return df, rep


def run() -> dict:
    C.ensure_dirs()
    df, rep = clean(load_raw())
    df.to_csv(C.CLEAN_FILE)
    return rep


def load_clean() -> pd.DataFrame:
    return pd.read_csv(C.CLEAN_FILE, index_col=0, parse_dates=True)


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k:22s} {v}")
