"""One command to run everything:  python runall.py

Steps: data -> clean -> train -> evaluate -> forecast -> simulate -> charts -> API
Options:
  --no-api        run the pipeline only
  --fresh         regenerate the simulated dataset
  --port 8000     API port        --host 127.0.0.1   API host
  --open          open the dashboard in your browser
"""
import argparse
import sys
import threading
import time
import webbrowser

from src import config as C


def step_data(fresh):
    from src.data_generator import save_raw
    df = save_raw(overwrite=fresh)
    return f"{len(df):,} raw rows -> {C.RAW_FILE.name}"


def step_clean(_):
    from src.preprocess import run
    r = run()
    return (f"{r['duplicates_removed']} duplicates, {r['missing_hours_added']} missing hours, "
            f"{r['outliers_removed']} outliers fixed -> {r['rows_clean']:,} clean rows")


def step_train(_):
    from src.train import run
    r = run()
    return "\n".join("      " + line for line in r["log"]).lstrip() + f"\n      Best model: {r['best']}"


def step_evaluate(_):
    from src.evaluate import run
    t = run().iloc[0]
    return f"{t['model']}: RMSE {t['RMSE']:.1f} kWh, R2 {t['R2']:.3f}, MAPE {t['MAPE_%']:.2f}%"


def step_forecast(_):
    from src.forecast import run
    fc = run()
    return f"{len(fc)} hours forecast ({fc.index[0]:%d %b} to {fc.index[-1]:%d %b %Y})"


def step_simulate(_):
    from src.simulate import run
    r = run()
    return (f"forecasting saves {r['forecasting_savings_pct']}%, "
            f"peak shaving saves {r['peak_shaving_savings_pct']}% (Rs {r['peak_shaving_savings_inr']:,.0f})")


def step_visualize(_):
    from src.visualize import run
    return f"{len(run())} charts saved to images/"


STEPS = [("Generate dataset", step_data), ("Clean data", step_clean), ("Train models", step_train),
         ("Evaluate", step_evaluate), ("Forecast", step_forecast), ("Business simulation", step_simulate),
         ("Create charts", step_visualize)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    C.ensure_dirs()
    print("\nAI-Powered Energy Consumption Forecasting System\n" + "=" * 50)
    start = time.time()
    for i, (name, fn) in enumerate(STEPS, 1):
        t0 = time.time()
        try:
            msg = fn(a.fresh)
        except Exception as exc:
            print(f"[{i}/{len(STEPS)}] {name} FAILED: {exc}")
            sys.exit(1)
        print(f"[{i}/{len(STEPS)}] {name}  ({time.time() - t0:.1f}s)\n      {msg}")
    print("=" * 50 + f"\nPipeline finished in {time.time() - start:.0f}s. Charts: images/  Reports: outputs/")

    if a.no_api:
        return
    from api.app import run_server
    if a.open:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{a.host}:{a.port}")).start()
    run_server(a.host, a.port)


if __name__ == "__main__":
    main()
