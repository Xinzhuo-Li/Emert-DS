"""Run model evaluation, model selection, and final projection packaging."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from output_visualization.export_results import export_csv
from output_visualization.plots import save_metric_comparison_plot


ROOT_DIR = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT_DIR / "outputs" / "tables"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"


def select_best_models(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics["mape"] = pd.to_numeric(metrics["mape"], errors="coerce")
    selected = (
        metrics.sort_values(["geography", "mape"])
        .groupby("geography", as_index=False)
        .first()
    )
    selected["selection_reason"] = "Selected as lowest-MAPE model on the 2023-2025 holdout window."
    return selected


def build_final_projection_table(
    selected_models: pd.DataFrame,
    baseline_forecasts: pd.DataFrame,
    time_series_forecasts: pd.DataFrame,
) -> pd.DataFrame:
    forecast_rows = []
    for _, row in selected_models.iterrows():
        geography = row["geography"]
        model = row["model"]
        if row["model_family"] == "baseline":
            source = baseline_forecasts
        else:
            source = time_series_forecasts

        chosen = source[(source["geography"] == geography) & (source["model"] == model)].copy()
        chosen["selected_model"] = model
        chosen["model_family"] = row["model_family"]
        chosen["projection_note"] = row["selection_reason"]
        forecast_rows.append(chosen)

    final_projection = pd.concat(forecast_rows, ignore_index=True)
    final_projection = final_projection[
        ["geography", "model_family", "selected_model", "fiscal_year", "forecast", "projection_note"]
    ]
    return final_projection


def main() -> None:
    comparison_metrics = pd.read_csv(TABLES_DIR / "model_comparison_metrics.csv")
    baseline_forecasts = pd.read_csv(TABLES_DIR / "baseline_future_forecasts.csv")
    time_series_forecasts = pd.read_csv(TABLES_DIR / "time_series_future_forecasts.csv")

    selected_models = select_best_models(comparison_metrics)
    final_projection = build_final_projection_table(
        selected_models,
        baseline_forecasts,
        time_series_forecasts,
    )

    export_csv(selected_models, TABLES_DIR / "selected_models.csv")
    export_csv(final_projection, TABLES_DIR / "final_projection_table.csv")

    phase_dir = FIGURES_DIR / "phase_07_model_evaluation"
    for geography in comparison_metrics["geography"].unique():
        save_metric_comparison_plot(
            comparison_metrics,
            geography,
            "mape",
            phase_dir / f"{geography.lower().replace(' ', '_')}_model_mape_comparison.png",
        )

    print("Created evaluation outputs:")
    print("-", TABLES_DIR / "selected_models.csv")
    print("-", TABLES_DIR / "final_projection_table.csv")
    print("-", phase_dir / "michigan_model_mape_comparison.png")
    print("-", phase_dir / "united_states_model_mape_comparison.png")


if __name__ == "__main__":
    main()
