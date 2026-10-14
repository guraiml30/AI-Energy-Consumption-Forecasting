"""Central settings. Change values here instead of hunting through the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
IMAGES = ROOT / "images"

RAW_FILE = DATA_RAW / "energy_data.csv"
CLEAN_FILE = DATA_PROCESSED / "energy_clean.csv"
FEATURE_FILE = DATA_PROCESSED / "energy_features.csv"
MODEL_FILE = MODELS / "best_model.joblib"
TEST_PRED_FILE = OUTPUTS / "test_predictions.csv"
METRICS_FILE = OUTPUTS / "metrics.csv"
IMPORTANCE_FILE = OUTPUTS / "feature_importance.csv"
FORECAST_FILE = OUTPUTS / "forecast.csv"
SIM_FILE = OUTPUTS / "simulation_report.json"

# ---- Column names (change these if you plug in a real dataset) ----
TIMESTAMP_COL = "timestamp"
TARGET_COL = "energy_kwh"
TEMP_COL = "temperature_c"

# ---- Simulated dataset ----
START_DATE = "2023-01-01 00:00"
END_DATE = "2024-12-31 23:00"
RANDOM_STATE = 42
HOLIDAYS = {(1, 1), (1, 26), (8, 15), (10, 2), (12, 25)}  # (month, day)

# ---- Modelling ----
TEST_DAYS = 60          # last 60 days are kept aside for testing
FORECAST_HOURS = 168    # default forecast horizon = 7 days

# ---- Business simulation ----
TARIFF = {"peak": 9.0, "shoulder": 6.0, "offpeak": 4.0}  # Rs (INR) per kWh
PEAK_HOURS = list(range(17, 22))
OFFPEAK_HOURS = list(range(0, 6)) + [22, 23]
SHORTFALL_MULT = 1.6     # buying missing energy on the spot market costs 1.6x
SURPLUS_RESALE = 0.5     # selling unused energy back returns only 0.5x
SHIFT_FRACTION = 0.10    # share of peak-hour load that can be moved (EV charging, chillers...)

# ---- Theme shared by charts and the API dashboard ----
THEME = {
    "night": "#0B1620", "panel": "#112230", "line": "#223645",
    "text": "#E8F0F5", "muted": "#8CA3B2",
    "amber": "#F5A524", "teal": "#2DD4BF", "coral": "#FF6B5B",
}


def ensure_dirs():
    for p in (DATA_RAW, DATA_PROCESSED, MODELS, OUTPUTS, IMAGES):
        p.mkdir(parents=True, exist_ok=True)
