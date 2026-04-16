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
POLYNOMIAL_CANDIDATE_DEGREES = (2, 3, 4)
POLYNOMIAL_COMPLEXITY_TOLERANCE = 1.0


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


def rolling_validation_mape(
    rows: list[dict[str, float | int | str]], degree: int
) -> float:
    train_rows, _ = train_test_split_time_ordered(rows, HOLDOUT_SIZE)
    errors: list[float] = []

    # Use expanding-window one-step-ahead validation inside the training window.
    # This keeps the final 2023-2025 holdout untouched for model comparison.
    for end in range(degree + 2, len(train_rows)):
        fit_rows = train_rows[:end]
        validation_row = train_rows[end]
        fit_x = np.array([row["fiscal_year"] for row in fit_rows], dtype=float)
        fit_y = np.array([row["total_medicaid_expenditures"] for row in fit_rows], dtype=float)
        validation_x = np.array([validation_row["fiscal_year"]], dtype=float)
        validation_y = float(validation_row["total_medicaid_expenditures"])

        coeffs = fit_polynomial_regression(fit_x, fit_y, degree=degree)
        prediction = float(predict(coeffs, validation_x)[0])
        errors.append(abs((validation_y - prediction) / validation_y) * 100)

    return sum(errors) / len(errors)


def select_polynomial_degree(
    geography: str, rows: list[dict[str, float | int | str]]
) -> tuple[int, list[dict[str, str]]]:
    tuning_rows: list[dict[str, str]] = []
    candidate_scores: list[tuple[int, float]] = []

    for degree in POLYNOMIAL_CANDIDATE_DEGREES:
        score = rolling_validation_mape(rows, degree)
        candidate_scores.append((degree, score))

    best_score = min(score for _, score in candidate_scores)
    eligible = [
        (degree, score)
        for degree, score in candidate_scores
        if score <= best_score + POLYNOMIAL_COMPLEXITY_TOLERANCE
    ]
    selected_degree, selected_score = min(eligible, key=lambda item: item[0])

    for degree, score in candidate_scores:
        tuning_rows.append(
            {
                "geography": geography,
                "candidate_degree": str(degree),
                "rolling_cv_mape": f"{score:.2f}",
                "selected_degree": str(selected_degree),
                "selected": "yes" if degree == selected_degree else "no",
                "selection_rule": (
                    "Lowest rolling CV MAPE within 1.0 MAPE point of the best score; "
                    "choose the simplest degree among eligible candidates."
                ),
            }
        )

    return selected_degree, tuning_rows


def evaluate_model(
    geography: str,
    model_name: str,
    rows: list[dict[str, float | int | str]],
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    train_rows, test_rows = train_test_split_time_ordered(rows, HOLDOUT_SIZE)
    train_x = np.array([row["fiscal_year"] for row in train_rows], dtype=float)
    train_y = np.array([row["total_medicaid_expenditures"] for row in train_rows], dtype=float)
    test_x = np.array([row["fiscal_year"] for row in test_rows], dtype=float)
    test_y = np.array([row["total_medicaid_expenditures"] for row in test_rows], dtype=float)

    model_details = ""
    tuning_rows: list[dict[str, str]] = []
    if model_name == "linear_regression":
        coeffs = fit_linear_regression(train_x, train_y)
        evaluated_model_name = model_name
    elif model_name == "polynomial_regression":
        selected_degree, tuning_rows = select_polynomial_degree(geography, rows)
        coeffs = fit_polynomial_regression(train_x, train_y, degree=selected_degree)
        evaluated_model_name = f"polynomial_regression_degree_{selected_degree}"
        selected_summary = next(row for row in tuning_rows if row["selected"] == "yes")
        model_details = (
            f"selected_degree={selected_degree}; "
            f"rolling_cv_mape={selected_summary['rolling_cv_mape']}; "
            "selection=rolling_cv_with_parsimony"
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    test_predictions = predict(coeffs, test_x)
    metric_row = {
        "geography": geography,
        "model": evaluated_model_name,
        "model_details": model_details,
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
                "model": evaluated_model_name,
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
        full_coeffs = fit_polynomial_regression(full_x, full_y, degree=selected_degree)

    final_year = int(rows[-1]["fiscal_year"])
    future_years = np.array(list(range(final_year + 1, final_year + FORECAST_HORIZON + 1)), dtype=float)
    future_predictions = predict(full_coeffs, future_years)
    forecast_rows = []
    for year, prediction in zip(future_years, future_predictions):
        forecast_rows.append(
            {
                "geography": geography,
                "model": evaluated_model_name,
                "fiscal_year": str(int(year)),
                "forecast": f"{float(prediction):.2f}",
            }
        )

    return metric_row, holdout_rows, forecast_rows, tuning_rows


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
    tuning_rows: list[dict[str, str]] = []

    for geography, rows in (
        ("Michigan", michigan_rows),
        ("United States", national_rows),
    ):
        for model_name in ("linear_regression", "polynomial_regression"):
            metric_row, holdout_part, forecast_part, tuning_part = evaluate_model(
                geography, model_name, rows
            )
            metrics_rows.append(metric_row)
            holdout_rows.extend(holdout_part)
            forecast_rows.extend(forecast_part)
            tuning_rows.extend(tuning_part)

    write_csv(
        TABLES_DIR / "baseline_model_metrics.csv",
        metrics_rows,
        [
            "geography",
            "model",
            "model_details",
            "train_end_year",
            "test_start_year",
            "test_end_year",
            "mae",
            "rmse",
            "mape",
        ],
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
    write_csv(
        TABLES_DIR / "baseline_polynomial_tuning_summary.csv",
        tuning_rows,
        [
            "geography",
            "candidate_degree",
            "rolling_cv_mape",
            "selected_degree",
            "selected",
            "selection_rule",
        ],
    )

    print("Created baseline forecasting outputs:")
    print("-", TABLES_DIR / "baseline_model_metrics.csv")
    print("-", TABLES_DIR / "baseline_holdout_predictions.csv")
    print("-", TABLES_DIR / "baseline_future_forecasts.csv")
    print("-", TABLES_DIR / "baseline_polynomial_tuning_summary.csv")


if __name__ == "__main__":
    main()
"""Baseline forecasting model utilities.

This module is reserved for simple forecasting baselines such as:
- linear regression trend models
- polynomial regression trend models

Implementation will be added in Phase 5.
"""

