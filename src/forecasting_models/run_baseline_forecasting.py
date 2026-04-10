"""Run baseline forecasting models and generate comparison plots."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from forecasting_models.baseline_models import main as run_baselines
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


def main() -> None:
    run_baselines()

    history_map = {
        "Michigan": read_history(PROCESSED_DIR / "medicaid_michigan_timeseries.csv"),
        "United States": read_history(PROCESSED_DIR / "medicaid_national_timeseries.csv"),
    }
    holdout = read_holdout(TABLES_DIR / "baseline_holdout_predictions.csv")

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
                FIGURES_DIR / f"{geography.lower().replace(' ', '_')}_{model_name}_holdout.png",
            )

    print("Created baseline forecasting figures:")
    print("-", FIGURES_DIR / "michigan_linear_regression_holdout.png")
    print("-", FIGURES_DIR / "michigan_polynomial_regression_degree_2_holdout.png")
    print("-", FIGURES_DIR / "united_states_linear_regression_holdout.png")
    print("-", FIGURES_DIR / "united_states_polynomial_regression_degree_2_holdout.png")


if __name__ == "__main__":
    main()
