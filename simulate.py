"""Virtual business simulation: what is the forecast worth in money?

Scenario: a plant buys electricity one day ahead using a forecast.
  * Under-forecast -> missing energy is bought at spot price (1.6x)
  * Over-forecast  -> surplus is resold cheaply (0.5x)
We replay the 60 unseen test days with a naive plan vs the AI plan.
Then we show peak shaving: moving 10% of load out of forecast peak hours.
"""
import json

import numpy as np
import pandas as pd

from src import config as C


def tariff(index: pd.DatetimeIndex) -> np.ndarray:
    h = index.hour
    return np.where(h.isin(C.PEAK_HOURS), C.TARIFF["peak"],
                    np.where(h.isin(C.OFFPEAK_HOURS), C.TARIFF["offpeak"], C.TARIFF["shoulder"]))


def procurement_cost(actual, forecast, price) -> float:
    short = np.maximum(actual - forecast, 0)
    surplus = np.maximum(forecast - actual, 0)
    return float(np.sum(forecast * price + short * price * C.SHORTFALL_MULT - surplus * price * C.SURPLUS_RESALE))


def run() -> dict:
    p = pd.read_csv(C.TEST_PRED_FILE, index_col=0, parse_dates=True)
    price = tariff(p.index)
    actual = p["actual"].values

    cost_perfect = procurement_cost(actual, actual, price)
    cost_naive = procurement_cost(actual, p["seasonal_naive"].values, price)
    cost_ai = procurement_cost(actual, p["predicted"].values, price)

    # Peak shaving using the AI forecast to find tomorrow's peak hours
    df = pd.DataFrame({"actual": actual, "forecast": p["predicted"].values, "price": price}, index=p.index)
    shifted = df["actual"].copy()
    for _, g in df.groupby(df.index.date):
        if len(g) < 24:
            continue
        peak = g["forecast"].nlargest(4).index
        cheap = g.sort_values(["price", "forecast"]).index[:4]
        moved = g.loc[peak, "actual"] * C.SHIFT_FRACTION
        shifted.loc[peak] -= moved
        shifted.loc[cheap] += moved.sum() / 4
    bill_before = float((df["actual"] * df["price"]).sum())
    bill_after = float((shifted * df["price"]).sum())

    report = {
        "days_simulated": int(len(p) / 24),
        "cost_perfect_inr": round(cost_perfect, 2),
        "cost_naive_inr": round(cost_naive, 2),
        "cost_ai_inr": round(cost_ai, 2),
        "forecasting_savings_inr": round(cost_naive - cost_ai, 2),
        "forecasting_savings_pct": round((cost_naive - cost_ai) / cost_naive * 100, 2),
        "peak_demand_before_kwh": round(float(df["actual"].max()), 1),
        "peak_demand_after_kwh": round(float(shifted.max()), 1),
        "peak_shaving_savings_inr": round(bill_before - bill_after, 2),
        "peak_shaving_savings_pct": round((bill_before - bill_after) / bill_before * 100, 2),
    }
    C.SIM_FILE.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
