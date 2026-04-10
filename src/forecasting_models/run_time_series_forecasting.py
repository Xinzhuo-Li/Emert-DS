"""Run time-series forecasting models and generate comparison artifacts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from output_visualization.plots import save_holdout_comparison_plot


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"


def read_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["fiscal_year"] = frame["fiscal_year"].astype(int)
    frame["total_medicaid_expenditures_billions"] = pd.to_numeric(
        frame["total_medicaid_expenditures_billions"], errors="coerce"
    )
    return frame


def read_holdout(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["fiscal_year"] = frame["fiscal_year"].astype(int)
    frame["actual_billions"] = pd.to_numeric(frame["actual"], errors="coerce") / 1_000_000_000
    frame["predicted_billions"] = pd.to_numeric(frame["predicted"], errors="coerce") / 1_000_000_000
    return frame


def merge_metrics() -> None:
    baseline = pd.read_csv(TABLES_DIR / "baseline_model_metrics.csv")
    baseline["model_family"] = "baseline"
    baseline["model_details"] = ""

    time_series = pd.read_csv(TABLES_DIR / "time_series_model_metrics.csv")
    time_series["model_family"] = "time_series"

    combined = pd.concat([baseline, time_series], ignore_index=True)
    combined = combined[
        [
            "geography",
            "model_family",
            "model",
            "model_details",
            "train_end_year",
            "test_start_year",
            "test_end_year",
            "mae",
            "rmse",
            "mape",
        ]
    ]
    combined.to_csv(TABLES_DIR / "model_comparison_metrics.csv", index=False)


def run_training_subprocess() -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".matplotlib-cache"))
    subprocess.run(
        [sys.executable, str(SRC_DIR / "forecasting_models" / "time_series_models.py")],
        check=True,
        env=env,
    )


def generate_figures() -> Path:
    history_map = {
        "Michigan": read_history(PROCESSED_DIR / "medicaid_michigan_timeseries.csv"),
        "United States": read_history(PROCESSED_DIR / "medicaid_national_timeseries.csv"),
    }
    holdout = read_holdout(TABLES_DIR / "time_series_holdout_predictions.csv")

    phase_dir = FIGURES_DIR / "phase_06_time_series_forecasting"
    for geography in holdout["geography"].unique():
        for model_name in holdout.loc[holdout["geography"] == geography, "model"].unique():
            holdout_slice = holdout[
                (holdout["geography"] == geography) & (holdout["model"] == model_name)
            ].copy()
            history = history_map[geography]
            save_holdout_comparison_plot(
                history,
                holdout_slice,
                geography,
                model_name,
                phase_dir / f"{geography.lower().replace(' ', '_')}_{model_name}_holdout.png",
            )
    return phase_dir


def main() -> None:
    # Prophet/cmdstan can destabilize the interpreter after training in some
    # environments, so we isolate model fitting in a child process.
    run_training_subprocess()
    phase_dir = generate_figures()
    merge_metrics()

    print("Created time-series forecasting figures:")
    print("-", phase_dir / "michigan_arima_holdout.png")
    print("-", phase_dir / "michigan_holt_winters_holdout.png")
    print("-", phase_dir / "michigan_prophet_holdout.png")
    print("-", phase_dir / "united_states_arima_holdout.png")
    print("-", phase_dir / "united_states_holt_winters_holdout.png")
    print("-", phase_dir / "united_states_prophet_holdout.png")
    print("-", TABLES_DIR / "model_comparison_metrics.csv")


if __name__ == "__main__":
    main()
