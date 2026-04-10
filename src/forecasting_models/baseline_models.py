"""Baseline forecasting models for annual Medicaid expenditure series."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from model_evaluation.metrics import mae, mape, rmse, train_test_split_time_ordered


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"

HOLDOUT_SIZE = 3
FORECAST_HORIZON = 10


def read_series(path: Path) -> list[dict[str, float | int | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "fiscal_year": int(row["fiscal_year"]),
                    "geography": row["geography"],
                    "total_medicaid_expenditures": float(row["total_medicaid_expenditures"]),
                }
            )
        return rows


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    coeffs = np.polyfit(x, y, deg=1)
    return coeffs


def fit_polynomial_regression(x: np.ndarray, y: np.ndarray, degree: int = 2) -> np.ndarray:
    coeffs = np.polyfit(x, y, deg=degree)
    return coeffs


def predict(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    polynomial = np.poly1d(coeffs)
    return polynomial(x)


def evaluate_model(
    geography: str,
    model_name: str,
    rows: list[dict[str, float | int | str]],
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    train_rows, test_rows = train_test_split_time_ordered(rows, HOLDOUT_SIZE)
    train_x = np.array([row["fiscal_year"] for row in train_rows], dtype=float)
    train_y = np.array([row["total_medicaid_expenditures"] for row in train_rows], dtype=float)
    test_x = np.array([row["fiscal_year"] for row in test_rows], dtype=float)
    test_y = np.array([row["total_medicaid_expenditures"] for row in test_rows], dtype=float)

    if model_name == "linear_regression":
        coeffs = fit_linear_regression(train_x, train_y)
    elif model_name == "polynomial_regression_degree_2":
        coeffs = fit_polynomial_regression(train_x, train_y, degree=2)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    test_predictions = predict(coeffs, test_x)
    metric_row = {
        "geography": geography,
        "model": model_name,
        "train_end_year": str(int(train_rows[-1]["fiscal_year"])),
        "test_start_year": str(int(test_rows[0]["fiscal_year"])),
        "test_end_year": str(int(test_rows[-1]["fiscal_year"])),
        "mae": f"{mae(test_y, test_predictions):.2f}",
        "rmse": f"{rmse(test_y, test_predictions):.2f}",
        "mape": f"{mape(test_y, test_predictions):.2f}",
    }

    holdout_rows = []
    for row, prediction in zip(test_rows, test_predictions):
        holdout_rows.append(
            {
                "geography": geography,
                "model": model_name,
                "fiscal_year": str(int(row["fiscal_year"])),
                "actual": f"{float(row['total_medicaid_expenditures']):.2f}",
                "predicted": f"{float(prediction):.2f}",
                "error": f"{float(row['total_medicaid_expenditures']) - float(prediction):.2f}",
            }
        )

    full_x = np.array([row["fiscal_year"] for row in rows], dtype=float)
    full_y = np.array([row["total_medicaid_expenditures"] for row in rows], dtype=float)
    if model_name == "linear_regression":
        full_coeffs = fit_linear_regression(full_x, full_y)
    else:
        full_coeffs = fit_polynomial_regression(full_x, full_y, degree=2)

    final_year = int(rows[-1]["fiscal_year"])
    future_years = np.array(list(range(final_year + 1, final_year + FORECAST_HORIZON + 1)), dtype=float)
    future_predictions = predict(full_coeffs, future_years)
    forecast_rows = []
    for year, prediction in zip(future_years, future_predictions):
        forecast_rows.append(
            {
                "geography": geography,
                "model": model_name,
                "fiscal_year": str(int(year)),
                "forecast": f"{float(prediction):.2f}",
            }
        )

    return metric_row, holdout_rows, forecast_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    michigan_rows = read_series(PROCESSED_DIR / "medicaid_michigan_timeseries.csv")
    national_rows = read_series(PROCESSED_DIR / "medicaid_national_timeseries.csv")

    metrics_rows: list[dict[str, str]] = []
    holdout_rows: list[dict[str, str]] = []
    forecast_rows: list[dict[str, str]] = []

    for geography, rows in (
        ("Michigan", michigan_rows),
        ("United States", national_rows),
    ):
        for model_name in ("linear_regression", "polynomial_regression_degree_2"):
            metric_row, holdout_part, forecast_part = evaluate_model(geography, model_name, rows)
            metrics_rows.append(metric_row)
            holdout_rows.extend(holdout_part)
            forecast_rows.extend(forecast_part)

    write_csv(
        TABLES_DIR / "baseline_model_metrics.csv",
        metrics_rows,
        ["geography", "model", "train_end_year", "test_start_year", "test_end_year", "mae", "rmse", "mape"],
    )
    write_csv(
        TABLES_DIR / "baseline_holdout_predictions.csv",
        holdout_rows,
        ["geography", "model", "fiscal_year", "actual", "predicted", "error"],
    )
    write_csv(
        TABLES_DIR / "baseline_future_forecasts.csv",
        forecast_rows,
        ["geography", "model", "fiscal_year", "forecast"],
    )

    print("Created baseline forecasting outputs:")
    print("-", TABLES_DIR / "baseline_model_metrics.csv")
    print("-", TABLES_DIR / "baseline_holdout_predictions.csv")
    print("-", TABLES_DIR / "baseline_future_forecasts.csv")


if __name__ == "__main__":
    main()
"""Baseline forecasting model utilities.

This module is reserved for simple forecasting baselines such as:
- linear regression trend models
- polynomial regression trend models

Implementation will be added in Phase 5.
"""

