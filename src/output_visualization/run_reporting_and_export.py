"""Create final reporting and export artifacts for submission."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from output_visualization.export_results import export_csv, export_html_table
from output_visualization.plots import save_final_projection_plot


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"


def load_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["fiscal_year"] = frame["fiscal_year"].astype(int)
    frame["total_medicaid_expenditures"] = pd.to_numeric(
        frame["total_medicaid_expenditures"], errors="coerce"
    )
    frame["total_medicaid_expenditures_billions"] = pd.to_numeric(
        frame["total_medicaid_expenditures_billions"], errors="coerce"
    )
    return frame


def prophet_history_to_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": [f"{int(year)}-12-31" for year in frame["fiscal_year"]],
            "y": frame["total_medicaid_expenditures"].astype(float).tolist(),
        }
    )


def parse_prophet_config(model_details: str) -> tuple[float, int]:
    changepoint_prior_scale = 0.2
    n_changepoints = 5
    for part in str(model_details).split(";"):
        cleaned = part.strip()
        if cleaned.startswith("changepoint_prior_scale="):
            changepoint_prior_scale = float(cleaned.split("=", 1)[1])
        elif cleaned.startswith("n_changepoints="):
            n_changepoints = int(cleaned.split("=", 1)[1])
    return changepoint_prior_scale, n_changepoints


def fit_final_prophet(
    history: pd.DataFrame,
    changepoint_prior_scale: float = 0.2,
    n_changepoints: int = 5,
) -> Prophet:
    from prophet import Prophet

    effective_n_changepoints = min(n_changepoints, max(0, len(history) - 2))
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,
        changepoint_prior_scale=changepoint_prior_scale,
        n_changepoints=effective_n_changepoints,
    )
    model.fit(prophet_history_to_frame(history))
    return model


def build_forecast_table(
    geography: str,
    history: pd.DataFrame,
    changepoint_prior_scale: float = 0.2,
    n_changepoints: int = 5,
) -> pd.DataFrame:
    model = fit_final_prophet(
        history,
        changepoint_prior_scale=changepoint_prior_scale,
        n_changepoints=n_changepoints,
    )
    future = model.make_future_dataframe(periods=10, freq="YE")
    forecast = model.predict(future).tail(10).copy()

    output = pd.DataFrame(
        {
            "geography": geography,
            "fiscal_year": pd.to_datetime(forecast["ds"]).dt.year.astype(int),
            "forecast": forecast["yhat"].astype(float),
            "lower_bound": forecast["yhat_lower"].astype(float),
            "upper_bound": forecast["yhat_upper"].astype(float),
        }
    )
    output["forecast_billions"] = output["forecast"] / 1_000_000_000
    output["lower_bound_billions"] = output["lower_bound"] / 1_000_000_000
    output["upper_bound_billions"] = output["upper_bound"] / 1_000_000_000
    return output


def build_summary_frame(selected_models: pd.DataFrame, final_forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, selection in selected_models.iterrows():
        geography = selection["geography"]
        projection = final_forecasts[final_forecasts["geography"] == geography].copy()
        first_year = projection.iloc[0]
        last_year = projection.iloc[-1]
        rows.append(
            {
                "geography": geography,
                "selected_model": selection["model"],
                "model_family": selection["model_family"],
                "holdout_mape": selection["mape"],
                "forecast_start_year": int(first_year["fiscal_year"]),
                "forecast_end_year": int(last_year["fiscal_year"]),
                "forecast_start_billions": round(float(first_year["forecast_billions"]), 2),
                "forecast_end_billions": round(float(last_year["forecast_billions"]), 2),
            }
        )
    return pd.DataFrame(rows)


def train_only() -> None:
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".matplotlib-cache"))

    selected_models = pd.read_csv(TABLES_DIR / "selected_models.csv")
    selected_lookup = {
        row["geography"]: row for _, row in selected_models.iterrows()
    }
    history_map = {
        "Michigan": load_history(PROCESSED_DIR / "medicaid_michigan_timeseries.csv"),
        "United States": load_history(PROCESSED_DIR / "medicaid_national_timeseries.csv"),
    }

    final_forecasts = []
    for geography, history in history_map.items():
        selection = selected_lookup[geography]
        changepoint_prior_scale, n_changepoints = parse_prophet_config(
            selection.get("model_details", "")
        )
        forecast_table = build_forecast_table(
            geography,
            history,
            changepoint_prior_scale=changepoint_prior_scale,
            n_changepoints=n_changepoints,
        )
        final_forecasts.append(forecast_table)

    final_projection_with_intervals = pd.concat(final_forecasts, ignore_index=True)
    export_csv(
        final_projection_with_intervals,
        TABLES_DIR / "phase_08_reporting_and_export" / "final_projection_with_intervals.csv",
    )


def run_training_subprocess() -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".matplotlib-cache"))
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--train-only"],
        check=True,
        env=env,
    )


def main() -> None:
    run_training_subprocess()

    selected_models = pd.read_csv(TABLES_DIR / "selected_models.csv")
    final_projection_with_intervals = pd.read_csv(
        TABLES_DIR / "phase_08_reporting_and_export" / "final_projection_with_intervals.csv"
    )
    summary_frame = build_summary_frame(selected_models, final_projection_with_intervals)
    phase_dir = FIGURES_DIR / "phase_08_reporting_and_export"

    history_map = {
        "Michigan": load_history(PROCESSED_DIR / "medicaid_michigan_timeseries.csv"),
        "United States": load_history(PROCESSED_DIR / "medicaid_national_timeseries.csv"),
    }
    for geography, history in history_map.items():
        forecast_table = final_projection_with_intervals[
            final_projection_with_intervals["geography"] == geography
        ].copy()
        save_final_projection_plot(
            history,
            forecast_table,
            geography,
            phase_dir / f"{geography.lower().replace(' ', '_')}_final_projection.png",
        )

    export_csv(
        summary_frame,
        TABLES_DIR / "phase_08_reporting_and_export" / "final_results_summary.csv",
    )
    export_html_table(
        final_projection_with_intervals,
        TABLES_DIR / "phase_08_reporting_and_export" / "final_projection_with_intervals.html",
        "Final Projection With 95 Percent Intervals",
    )
    export_html_table(
        summary_frame,
        TABLES_DIR / "phase_08_reporting_and_export" / "final_results_summary.html",
        "Final Results Summary",
    )

    print("Created final reporting and export outputs:")
    print("-", TABLES_DIR / "phase_08_reporting_and_export" / "final_projection_with_intervals.csv")
    print("-", TABLES_DIR / "phase_08_reporting_and_export" / "final_results_summary.csv")
    print("-", TABLES_DIR / "phase_08_reporting_and_export" / "final_projection_with_intervals.html")
    print("-", TABLES_DIR / "phase_08_reporting_and_export" / "final_results_summary.html")
    print("-", phase_dir / "michigan_final_projection.png")
    print("-", phase_dir / "united_states_final_projection.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    if args.train_only:
        train_only()
    else:
        main()
