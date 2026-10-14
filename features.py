"""Feature engineering.

Day-ahead rule: every lag is >= 24 hours, so a model can forecast tomorrow
using only data that really exists today (no cheating / data leakage).
"""
import numpy as np
import pandas as pd

from src import config as C

FEATURE_COLUMNS = [
    "hour", "dayofweek", "month", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "temperature_c", "cooling_deg", "heating_deg",
    "lag_24", "lag_48", "lag_168",
    "roll_mean_24", "roll_std_24", "roll_mean_168",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out["hour"], out["dayofweek"], out["month"] = idx.hour, idx.dayofweek, idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_holiday"] = np.array([(m, d) in C.HOLIDAYS for m, d in zip(idx.month, idx.day)]).astype(int)

    # Cyclical encoding: hour 23 and hour 0 are neighbours, not opposites
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * idx.hour / 24), np.cos(2 * np.pi * idx.hour / 24)
    out["doy_sin"], out["doy_cos"] = np.sin(2 * np.pi * idx.dayofyear / 365.25), np.cos(2 * np.pi * idx.dayofyear / 365.25)

    # Weather: energy rises when it is too hot (cooling) or too cold (heating)
    out["cooling_deg"] = np.maximum(out[C.TEMP_COL] - 22, 0)
    out["heating_deg"] = np.maximum(14 - out[C.TEMP_COL], 0)

    y = out[C.TARGET_COL]
    out["lag_24"], out["lag_48"], out["lag_168"] = y.shift(24), y.shift(48), y.shift(168)
    past = y.shift(24)
    out["roll_mean_24"] = past.rolling(24).mean()
    out["roll_std_24"] = past.rolling(24).std()
    out["roll_mean_168"] = past.rolling(168).mean()
    return out


def build_training_set(df: pd.DataFrame) -> pd.DataFrame:
    feats = add_features(df)
    return feats.dropna(subset=FEATURE_COLUMNS + [C.TARGET_COL])


if __name__ == "__main__":
    from src.preprocess import load_clean
    f = build_training_set(load_clean())
    print(f[FEATURE_COLUMNS + [C.TARGET_COL]].head())
    print("shape:", f.shape)
