"""Creates a realistic simulated hourly energy dataset (a manufacturing campus).

Why simulate? Real utility data is private. This generator bakes in the same
patterns real meters show: daily double peak, weekends, holidays, weather
sensitivity (cooling/heating), slow growth, plus messy data problems
(duplicates, gaps, spikes) so the cleaning step has real work to do.
"""
import numpy as np
import pandas as pd

from src import config as C


def generate(start=C.START_DATE, end=C.END_DATE, seed=C.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h")
    n = len(idx)
    hour, dow, doy = idx.hour.values, idx.dayofweek.values, idx.dayofyear.values

    # --- Temperature: season + day/night swing + day-to-day weather + noise
    season = 24 + 9 * np.sin(2 * np.pi * (doy - 110) / 365.25)
    daily = 5 * np.sin(2 * np.pi * (hour - 9) / 24)
    weather = np.repeat(rng.normal(0, 2.0, n // 24 + 2), 24)[:n]
    temp = season + daily + weather + rng.normal(0, 0.8, n)

    # --- Demand: base load x daily shape x calendar effects + weather load
    profile = 0.65 + 0.25 * np.exp(-((hour - 9) ** 2) / 8) + 0.35 * np.exp(-((hour - 19) ** 2) / 6)
    weekend = np.where(dow >= 5, 0.82, 1.0)
    holiday = np.array([0.78 if (m, d) in C.HOLIDAYS else 1.0 for m, d in zip(idx.month, idx.day)])
    growth = 1 + 0.04 * np.arange(n) / n
    weather_load = (18 * np.maximum(temp - 22, 0) + 12 * np.maximum(14 - temp, 0)) * profile
    energy = (800 * profile * weekend * holiday * growth + weather_load) * (1 + rng.normal(0, 0.03, n))

    df = pd.DataFrame({C.TIMESTAMP_COL: idx, C.TARGET_COL: energy.round(2), C.TEMP_COL: temp.round(2)})

    # --- Messy real-world problems
    spikes = rng.choice(n, 40, replace=False)
    df.loc[spikes, C.TARGET_COL] *= rng.uniform(2.6, 3.5, 40)           # sensor spikes
    df.loc[rng.choice(n, 60, replace=False), C.TARGET_COL] = np.nan     # missing readings
    df.loc[rng.choice(n, 15, replace=False), C.TEMP_COL] = np.nan       # missing weather
    df = df.drop(df.index[5000:5006])                                   # 6-hour meter outage
    df = pd.concat([df, df.sample(10, random_state=seed)]).sort_values(C.TIMESTAMP_COL)  # duplicates
    return df.reset_index(drop=True)


def save_raw(overwrite=False) -> pd.DataFrame:
    C.ensure_dirs()
    if C.RAW_FILE.exists() and not overwrite:
        return pd.read_csv(C.RAW_FILE, parse_dates=[C.TIMESTAMP_COL])
    df = generate()
    df.to_csv(C.RAW_FILE, index=False)
    return df


if __name__ == "__main__":
    d = save_raw(overwrite=True)
    print(f"Saved {len(d):,} rows to {C.RAW_FILE}")
    print(d.head())
