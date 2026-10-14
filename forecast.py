"""Forecast future demand hour by hour (recursive multi-step forecasting)."""
import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.features import FEATURE_COLUMNS, add_features
from src.preprocess import load_clean


def load_bundle() -> dict:
    return joblib.load(C.MODEL_FILE)


def future_temperature(hist: pd.DataFrame, future_idx: pd.DatetimeIndex) -> np.ndarray:
    """Stand-in for a weather-service forecast: same hour one year earlier."""
    ref = future_idx - pd.DateOffset(years=1)
    temps = hist[C.TEMP_COL].reindex(ref).values
    fallback = hist.groupby(hist.index.hour)[C.TEMP_COL].mean().reindex(future_idx.hour).values
    return np.where(np.isnan(temps), fallback, temps)


def recursive_forecast(bundle: dict, hist: pd.DataFrame, horizon: int = C.FORECAST_HOURS) -> pd.DataFrame:
    model = bundle["model"]
    work = hist[[C.TARGET_COL, C.TEMP_COL]].tail(300).copy()
    future_idx = pd.date_range(work.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    temps = future_temperature(hist, future_idx)

    values = []
    for ts, t in zip(future_idx, temps):
        work.loc[ts] = [np.nan, t]                      # add the future hour
        row = add_features(work.tail(260)).iloc[[-1]][FEATURE_COLUMNS]
        pred = max(float(model.predict(row)[0]), 0.0)
        work.loc[ts, C.TARGET_COL] = pred               # feed the prediction back in as history
        values.append(pred)

    band = 1.96 * bundle["residual_std"]
    out = pd.DataFrame({"forecast_kwh": values, "temperature_c": temps}, index=future_idx)
    out["lower_kwh"] = (out["forecast_kwh"] - band).clip(lower=0)
    out["upper_kwh"] = out["forecast_kwh"] + band
    out.index.name = C.TIMESTAMP_COL
    return out.round(2)


def run(horizon: int = C.FORECAST_HOURS) -> pd.DataFrame:
    fc = recursive_forecast(load_bundle(), load_clean(), horizon)
    fc.to_csv(C.FORECAST_FILE)
    return fc


if __name__ == "__main__":
    print(run().head(24))
