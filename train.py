"""Train several models, keep the best one."""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C
from src.evaluate import MODEL_NAMES, metrics
from src.features import FEATURE_COLUMNS, build_training_set
from src.preprocess import load_clean


def time_split(feat: pd.DataFrame):
    """Chronological split: train on the past, test on the most recent days."""
    cutoff = feat.index.max() - pd.Timedelta(days=C.TEST_DAYS)
    return feat[feat.index <= cutoff], feat[feat.index > cutoff]


def get_models() -> dict:
    return {
        "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=3,
                                               n_jobs=-1, random_state=C.RANDOM_STATE),
        "gradient_boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                                          random_state=C.RANDOM_STATE),
    }


def run() -> dict:
    C.ensure_dirs()
    feat = build_training_set(load_clean())
    feat.to_csv(C.FEATURE_FILE)
    train, test = time_split(feat)
    X_tr, y_tr = train[FEATURE_COLUMNS], train[C.TARGET_COL]
    X_te, y_te = test[FEATURE_COLUMNS], test[C.TARGET_COL]

    preds = pd.DataFrame({"actual": y_te, "temperature_c": test[C.TEMP_COL],
                          "seasonal_naive": test["lag_168"]})   # baseline: same hour last week
    fitted, log = {}, []
    for key, model in get_models().items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        preds[key] = model.predict(X_te)
        fitted[key] = model
        m = metrics(y_te, preds[key])
        log.append(f"{MODEL_NAMES[key]:<22} RMSE={m['RMSE']:.1f}  R2={m['R2']:.4f}  ({time.time() - t0:.1f}s)")

    best = min(fitted, key=lambda k: metrics(y_te, preds[k])["RMSE"])
    preds["predicted"] = preds[best]
    preds.to_csv(C.TEST_PRED_FILE)

    imp = permutation_importance(fitted[best], X_te, y_te, n_repeats=3, random_state=C.RANDOM_STATE, n_jobs=-1)
    pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": imp.importances_mean}) \
        .sort_values("importance", ascending=False).to_csv(C.IMPORTANCE_FILE, index=False)

    resid_std = float(np.std(y_te - preds[best]))
    joblib.dump({"model": fitted[best], "name": MODEL_NAMES[best], "key": best,
                 "features": FEATURE_COLUMNS, "residual_std": resid_std}, C.MODEL_FILE)
    return {"best": MODEL_NAMES[best], "log": log, "train_rows": len(train), "test_rows": len(test)}


if __name__ == "__main__":
    r = run()
    print("\n".join(r["log"]))
    print("Best:", r["best"])
