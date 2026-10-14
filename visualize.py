"""Charts saved to images/. Colors match the API dashboard (see config.THEME)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src import config as C
from src.preprocess import load_clean

T = C.THEME


def _style():
    plt.rcParams.update({
        "figure.facecolor": T["night"], "axes.facecolor": T["panel"], "savefig.facecolor": T["night"],
        "axes.edgecolor": T["line"], "axes.labelcolor": T["muted"], "text.color": T["text"],
        "xtick.color": T["muted"], "ytick.color": T["muted"], "grid.color": T["line"],
        "axes.grid": True, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.facecolor": T["panel"], "legend.edgecolor": T["line"],
    })


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(C.IMAGES / name, dpi=150)
    plt.close(fig)


def run() -> list:
    C.ensure_dirs()
    _style()
    clean = load_clean()
    preds = pd.read_csv(C.TEST_PRED_FILE, index_col=0, parse_dates=True)
    metrics = pd.read_csv(C.METRICS_FILE)
    imp = pd.read_csv(C.IMPORTANCE_FILE).head(10).iloc[::-1]
    fc = pd.read_csv(C.FORECAST_FILE, index_col=0, parse_dates=True)
    sim = json.loads(C.SIM_FILE.read_text())

    # 1. Dataset overview
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(clean.index, clean[C.TARGET_COL], color=T["amber"], lw=0.4)
    ax[0].set_title("Hourly energy demand (kWh)")
    ax[1].plot(clean.index, clean[C.TEMP_COL], color=T["teal"], lw=0.4)
    ax[1].set_title("Temperature (°C)")
    _save(fig, "01_dataset_overview.png")

    # 2. Demand pattern heatmap
    pivot = clean.groupby([clean.index.dayofweek, clean.index.hour])[C.TARGET_COL].mean().unstack()
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    sns.heatmap(pivot, cmap=LinearSegmentedColormap.from_list("e", [T["panel"], T["amber"]]),
                ax=ax, cbar_kws={"label": "avg kWh"}, linewidths=0.3, linecolor=T["night"])
    ax.set_title("When does demand peak? (weekday x hour)")
    ax.set_xlabel("Hour of day"); ax.grid(False)
    _save(fig, "02_hourly_heatmap.png")

    # 3. Actual vs predicted (last 7 test days)
    last = preds.tail(168)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(last.index, last["actual"], color=T["amber"], lw=1.8, label="Actual")
    ax.plot(last.index, last["predicted"], color=T["teal"], lw=1.8, ls="--", label="Predicted")
    ax.set_title("Actual vs predicted demand, last 7 days of the test set")
    ax.set_ylabel("kWh"); ax.legend()
    _save(fig, "03_actual_vs_predicted.png")

    # 4. Model comparison
    m = metrics.sort_values("RMSE", ascending=False)
    colors = [T["teal"] if i == len(m) - 1 else T["muted"] for i in range(len(m))]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.barh(m["model"], m["RMSE"], color=colors)
    for y, (v, r2) in enumerate(zip(m["RMSE"], m["R2"])):
        ax.text(v + 0.5, y, f"{v:.1f} kWh  (R² {r2:.3f})", va="center", fontsize=9)
    ax.set_title("Model comparison: lower RMSE is better"); ax.set_xlim(0, m["RMSE"].max() * 1.35)
    _save(fig, "04_model_comparison.png")

    # 5. Feature importance
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(imp["feature"], imp["importance"], color=T["amber"])
    ax.set_title("What drives the forecast? (permutation importance)")
    _save(fig, "05_feature_importance.png")

    # 6. Future forecast
    hist = clean.tail(168)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hist.index, hist[C.TARGET_COL], color=T["amber"], lw=1.6, label="History")
    ax.plot(fc.index, fc["forecast_kwh"], color=T["teal"], lw=1.8, label="Forecast")
    ax.fill_between(fc.index, fc["lower_kwh"], fc["upper_kwh"], color=T["teal"], alpha=0.15, label="95% range")
    ax.axvline(fc.index[0], color=T["coral"], lw=1, ls=":")
    ax.set_title("7-day demand forecast"); ax.set_ylabel("kWh"); ax.legend(ncol=3)
    _save(fig, "06_forecast_7day.png")

    # 7. Residuals
    res = preds["actual"] - preds["predicted"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(res, bins=40, color=T["teal"], alpha=0.9)
    ax.axvline(0, color=T["coral"], lw=1.2)
    ax.set_title("Forecast errors (actual minus predicted)"); ax.set_xlabel("kWh")
    _save(fig, "07_residuals.png")

    # 8. Business value
    labels = ["Perfect\nforecast", "AI forecast", "Naive\nforecast"]
    vals = [sim["cost_perfect_inr"], sim["cost_ai_inr"], sim["cost_naive_inr"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, vals, color=[T["muted"], T["teal"], T["coral"]])
    ax.set_ylim(min(vals) * 0.98, max(vals) * 1.01)
    for i, v in enumerate(vals):
        ax.text(i, v, f"₹{v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Energy bill over {sim['days_simulated']} simulated days (₹)")
    _save(fig, "08_simulation_savings.png")
    return sorted(p.name for p in C.IMAGES.glob("*.png"))


if __name__ == "__main__":
    print(run())
