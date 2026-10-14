"""Run a single stage:  python main.py <stage>
Stages: data | clean | train | evaluate | forecast | simulate | plots | api
(For everything at once use: python runall.py)"""
import sys

STAGES = {
    "data": lambda: __import__("src.data_generator", fromlist=["x"]).save_raw(overwrite=True),
    "clean": lambda: print(__import__("src.preprocess", fromlist=["x"]).run()),
    "train": lambda: print("\n".join(__import__("src.train", fromlist=["x"]).run()["log"])),
    "evaluate": lambda: print(__import__("src.evaluate", fromlist=["x"]).run().round(3).to_string(index=False)),
    "forecast": lambda: print(__import__("src.forecast", fromlist=["x"]).run().head(24)),
    "simulate": lambda: print(__import__("src.simulate", fromlist=["x"]).run()),
    "plots": lambda: print(__import__("src.visualize", fromlist=["x"]).run()),
    "api": lambda: __import__("api.app", fromlist=["x"]).run_server(),
}

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage not in STAGES:
        sys.exit(__doc__)
    STAGES[stage]()
