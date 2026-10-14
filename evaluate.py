"""Evaluation metrics and the model scoreboard."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config as C

MODEL_NAMES = {
    "seasonal_naive": "Seasonal Naive (baseline)",
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


def metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),   # typical error in kWh (punishes big misses)
        "MAE": float(mean_absolute_error(y_true, y_pred)),            # average error in kWh
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, y_pred)),                        # 1.0 = perfect, 0 = no better than the average
    }


def run() -> pd.DataFrame:
    preds = pd.read_csv(C.TEST_PRED_FILE, index_col=0, parse_dates=True)
    rows = [{"model": name, **metrics(preds["actual"], preds[col])}
            for col, name in MODEL_NAMES.items() if col in preds]
    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    table.to_csv(C.METRICS_FILE, index=False)
    return table


if __name__ == "__main__":
    print(run().round(3).to_string(index=False))
