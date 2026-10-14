"""Flask API + dashboard. Run:  python -m api.app   then open http://127.0.0.1:8000"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from src import config as C
from src.forecast import load_bundle, recursive_forecast
from src.preprocess import load_clean

app = Flask(__name__, static_folder="static", static_url_path="/static")
_cache = {}
MAX_HOURS = 336


def _state():
    """Load model + data once, reuse for every request."""
    if not _cache:
        if not C.MODEL_FILE.exists():
            raise FileNotFoundError("Model not found. Run: python runall.py")
        _cache["bundle"] = load_bundle()
        _cache["hist"] = load_clean()
    return _cache


def _rows(df: pd.DataFrame) -> list:
    out = df.reset_index()
    first = out.columns[0]
    out[first] = out[first].dt.strftime("%Y-%m-%dT%H:%M:%S")   # plain text timestamps for JSON
    return out.to_dict(orient="records")


def _hours(default: int) -> int:
    try:
        return max(1, min(int(request.args.get("hours", default)), MAX_HOURS))
    except ValueError:
        return default


@app.errorhandler(FileNotFoundError)
def _missing(e):
    return jsonify(error=str(e)), 503


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", model_ready=C.MODEL_FILE.exists())


@app.get("/api/summary")
def summary():
    s = _state()
    metrics = pd.read_csv(C.METRICS_FILE).round(3).to_dict(orient="records")
    sim = json.loads(C.SIM_FILE.read_text())
    return jsonify(model=s["bundle"]["name"], residual_std_kwh=round(s["bundle"]["residual_std"], 2),
                   data_start=str(s["hist"].index.min()), data_end=str(s["hist"].index.max()),
                   metrics=metrics, simulation=sim)


@app.get("/api/history")
def history():
    h = _state()["hist"].tail(_hours(168))[[C.TARGET_COL]]
    return jsonify(_rows(h))


@app.get("/api/forecast")
def forecast():
    s = _state()
    return jsonify(_rows(recursive_forecast(s["bundle"], s["hist"], _hours(48))))


@app.get("/api/backtest")
def backtest():
    p = pd.read_csv(C.TEST_PRED_FILE, index_col=0, parse_dates=True).tail(_hours(168))
    return jsonify(_rows(p[["actual", "predicted"]].round(2)))


@app.get("/api/feature-importance")
def importance():
    return jsonify(pd.read_csv(C.IMPORTANCE_FILE).head(10).round(4).to_dict(orient="records"))


@app.get("/api/predict")
def predict():
    """Example: /api/predict?datetime=2025-01-03T18:00"""
    s = _state()
    try:
        ts = pd.Timestamp(request.args["datetime"]).floor("h")
    except (KeyError, ValueError):
        return jsonify(error="Pass a valid ?datetime=YYYY-MM-DDTHH:MM"), 400
    start = s["hist"].index.max() + pd.Timedelta(hours=1)
    hours = int((ts - start) / pd.Timedelta(hours=1)) + 1
    if hours < 1 or hours > MAX_HOURS:
        return jsonify(error=f"datetime must be between {start} and {start + pd.Timedelta(hours=MAX_HOURS - 1)}"), 400
    row = recursive_forecast(s["bundle"], s["hist"], hours).loc[ts]
    return jsonify(datetime=str(ts), predicted_kwh=row["forecast_kwh"], lower_kwh=row["lower_kwh"],
                   upper_kwh=row["upper_kwh"], temperature_c=row["temperature_c"])


def run_server(host="127.0.0.1", port=8000):
    print(f"\n  Energy Forecast API running at http://{host}:{port}  (Ctrl+C to stop)\n")
    try:
        from waitress import serve
        serve(app, host=host, port=port)
    except ImportError:
        app.run(host=host, port=port)


if __name__ == "__main__":
    run_server(os.environ.get("HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"),
               int(os.environ.get("PORT", 8000)))
