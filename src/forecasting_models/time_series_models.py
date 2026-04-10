"""Advanced time-series forecasting model utilities."""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_best_arima(train_y: np.ndarray) -> tuple[object, tuple[int, int, int]]:
    best_result = None
    best_order = None
    best_aic = float("inf")
    warnings.filterwarnings("ignore")

    for p in range(0, 3):
        for d in range(0, 3):
            for q in range(0, 3):
                if p == d == q == 0:
                    continue
                try:
                    result = ARIMA(train_y, order=(p, d, q)).fit()
                except Exception:
                    continue
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_result = result
                    best_order = (p, d, q)

    if best_result is None or best_order is None:
        raise RuntimeError("Unable to fit any ARIMA configuration.")
    return best_result, best_order


def fit_best_holt_winters(train_y: np.ndarray) -> tuple[object, str]:
    candidates: list[tuple[object, str, float]] = []
    warnings.filterwarnings("ignore")

    try:
        fit = SimpleExpSmoothing(train_y).fit()
        candidates.append((fit, "simple_exponential_smoothing", fit.sse))
    except Exception:
        pass

    for damped in (False, True):
        try:
            fit = ExponentialSmoothing(
                train_y,
                trend="add",
                seasonal=None,
                damped_trend=damped,
            ).fit(optimized=True)
            label = "holt_winters_additive_damped" if damped else "holt_winters_additive"
            candidates.append((fit, label, fit.sse))
        except Exception:
            continue

    if not candidates:
        raise RuntimeError("Unable to fit any exponential smoothing configuration.")

    best_fit, best_label, _ = min(candidates, key=lambda item: item[2])
    return best_fit, best_label


def prophet_frame(rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": [f"{int(row['fiscal_year'])}-12-31" for row in rows],
            "y": [float(row["total_medicaid_expenditures"]) for row in rows],
        }
    )


def fit_prophet_model(train_rows: list[dict[str, float | int | str]]) -> Prophet:
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.2,
        n_changepoints=5,
    )
    model.fit(prophet_frame(train_rows))
    return model


def evaluate_arima(
    geography: str,
    rows: list[dict[str, float | int | str]],
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    train_rows, test_rows = train_test_split_time_ordered(rows, HOLDOUT_SIZE)
    train_y = np.array([row["total_medicaid_expenditures"] for row in train_rows], dtype=float)
    test_y = np.array([row["total_medicaid_expenditures"] for row in test_rows], dtype=float)

    model, order = fit_best_arima(train_y)
    predictions = model.forecast(steps=HOLDOUT_SIZE)
    metric_row = {
        "geography": geography,
        "model": "arima",
        "model_details": f"order={order}",
        "train_end_year": str(int(train_rows[-1]["fiscal_year"])),
        "test_start_year": str(int(test_rows[0]["fiscal_year"])),
        "test_end_year": str(int(test_rows[-1]["fiscal_year"])),
        "mae": f"{mae(test_y, predictions):.2f}",
        "rmse": f"{rmse(test_y, predictions):.2f}",
        "mape": f"{mape(test_y, predictions):.2f}",
    }
    holdout_rows = [
        {
            "geography": geography,
            "model": "arima",
            "fiscal_year": str(int(row["fiscal_year"])),
            "actual": f"{float(row['total_medicaid_expenditures']):.2f}",
            "predicted": f"{float(prediction):.2f}",
            "error": f"{float(row['total_medicaid_expenditures']) - float(prediction):.2f}",
        }
        for row, prediction in zip(test_rows, predictions)
    ]

    full_y = np.array([row["total_medicaid_expenditures"] for row in rows], dtype=float)
    full_model, full_order = fit_best_arima(full_y)
    future_predictions = full_model.forecast(steps=FORECAST_HORIZON)
    final_year = int(rows[-1]["fiscal_year"])
    forecast_rows = [
        {
            "geography": geography,
            "model": "arima",
            "fiscal_year": str(final_year + step + 1),
            "forecast": f"{float(prediction):.2f}",
        }
        for step, prediction in enumerate(future_predictions)
    ]
    note = {
        "geography": geography,
        "model_category": "ARIMA / SARIMA",
        "model_used": "arima",
        "note": f"Annual data does not support meaningful seasonal estimation, so non-seasonal ARIMA with selected order {full_order} was used in place of SARIMA.",
    }
    return metric_row, holdout_rows, forecast_rows, note


def evaluate_holt_winters(
    geography: str,
    rows: list[dict[str, float | int | str]],
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    train_rows, test_rows = train_test_split_time_ordered(rows, HOLDOUT_SIZE)
    train_y = np.array([row["total_medicaid_expenditures"] for row in train_rows], dtype=float)
    test_y = np.array([row["total_medicaid_expenditures"] for row in test_rows], dtype=float)

    model, label = fit_best_holt_winters(train_y)
    predictions = model.forecast(HOLDOUT_SIZE)
    metric_row = {
        "geography": geography,
        "model": "holt_winters",
        "model_details": label,
        "train_end_year": str(int(train_rows[-1]["fiscal_year"])),
        "test_start_year": str(int(test_rows[0]["fiscal_year"])),
        "test_end_year": str(int(test_rows[-1]["fiscal_year"])),
        "mae": f"{mae(test_y, predictions):.2f}",
        "rmse": f"{rmse(test_y, predictions):.2f}",
        "mape": f"{mape(test_y, predictions):.2f}",
    }
    holdout_rows = [
        {
            "geography": geography,
            "model": "holt_winters",
            "fiscal_year": str(int(row["fiscal_year"])),
            "actual": f"{float(row['total_medicaid_expenditures']):.2f}",
            "predicted": f"{float(prediction):.2f}",
            "error": f"{float(row['total_medicaid_expenditures']) - float(prediction):.2f}",
        }
        for row, prediction in zip(test_rows, predictions)
    ]

    full_y = np.array([row["total_medicaid_expenditures"] for row in rows], dtype=float)
    full_model, full_label = fit_best_holt_winters(full_y)
    future_predictions = full_model.forecast(FORECAST_HORIZON)
    final_year = int(rows[-1]["fiscal_year"])
    forecast_rows = [
        {
            "geography": geography,
            "model": "holt_winters",
            "fiscal_year": str(final_year + step + 1),
            "forecast": f"{float(prediction):.2f}",
        }
        for step, prediction in enumerate(future_predictions)
    ]
    metric_row["model_details"] = full_label
    return metric_row, holdout_rows, forecast_rows


def evaluate_prophet(
    geography: str,
    rows: list[dict[str, float | int | str]],
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    train_rows, test_rows = train_test_split_time_ordered(rows, HOLDOUT_SIZE)
    model = fit_prophet_model(train_rows)
    future = model.make_future_dataframe(periods=HOLDOUT_SIZE, freq="YE")
    forecast = model.predict(future)
    predictions = forecast["yhat"].tail(HOLDOUT_SIZE).to_numpy()
    test_y = np.array([row["total_medicaid_expenditures"] for row in test_rows], dtype=float)

    metric_row = {
        "geography": geography,
        "model": "prophet",
        "model_details": "linear_growth_no_seasonality",
        "train_end_year": str(int(train_rows[-1]["fiscal_year"])),
        "test_start_year": str(int(test_rows[0]["fiscal_year"])),
        "test_end_year": str(int(test_rows[-1]["fiscal_year"])),
        "mae": f"{mae(test_y, predictions):.2f}",
        "rmse": f"{rmse(test_y, predictions):.2f}",
        "mape": f"{mape(test_y, predictions):.2f}",
    }
    holdout_rows = [
        {
            "geography": geography,
            "model": "prophet",
            "fiscal_year": str(int(row["fiscal_year"])),
            "actual": f"{float(row['total_medicaid_expenditures']):.2f}",
            "predicted": f"{float(prediction):.2f}",
            "error": f"{float(row['total_medicaid_expenditures']) - float(prediction):.2f}",
        }
        for row, prediction in zip(test_rows, predictions)
    ]

    full_model = fit_prophet_model(rows)
    final_year = int(rows[-1]["fiscal_year"])
    future = full_model.make_future_dataframe(periods=FORECAST_HORIZON, freq="YE")
    forecast = full_model.predict(future)
    future_predictions = forecast["yhat"].tail(FORECAST_HORIZON).to_numpy()
    forecast_rows = [
        {
            "geography": geography,
            "model": "prophet",
            "fiscal_year": str(final_year + step + 1),
            "forecast": f"{float(prediction):.2f}",
        }
        for step, prediction in enumerate(future_predictions)
    ]
    return metric_row, holdout_rows, forecast_rows


def main() -> None:
    michigan_rows = read_series(PROCESSED_DIR / "medicaid_michigan_timeseries.csv")
    national_rows = read_series(PROCESSED_DIR / "medicaid_national_timeseries.csv")

    metrics_rows: list[dict[str, str]] = []
    holdout_rows: list[dict[str, str]] = []
    forecast_rows: list[dict[str, str]] = []
    notes_rows: list[dict[str, str]] = []

    for geography, rows in (
        ("Michigan", michigan_rows),
        ("United States", national_rows),
    ):
        arima_metric, arima_holdout, arima_forecast, arima_note = evaluate_arima(geography, rows)
        hw_metric, hw_holdout, hw_forecast = evaluate_holt_winters(geography, rows)
        prophet_metric, prophet_holdout, prophet_forecast = evaluate_prophet(geography, rows)

        metrics_rows.extend([arima_metric, hw_metric, prophet_metric])
        holdout_rows.extend(arima_holdout + hw_holdout + prophet_holdout)
        forecast_rows.extend(arima_forecast + hw_forecast + prophet_forecast)
        notes_rows.append(arima_note)

    write_csv(
        TABLES_DIR / "time_series_model_metrics.csv",
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
        TABLES_DIR / "time_series_holdout_predictions.csv",
        holdout_rows,
        ["geography", "model", "fiscal_year", "actual", "predicted", "error"],
    )
    write_csv(
        TABLES_DIR / "time_series_future_forecasts.csv",
        forecast_rows,
        ["geography", "model", "fiscal_year", "forecast"],
    )
    write_csv(
        TABLES_DIR / "time_series_model_notes.csv",
        notes_rows,
        ["geography", "model_category", "model_used", "note"],
    )

    print("Created time-series forecasting outputs:")
    print("-", TABLES_DIR / "time_series_model_metrics.csv")
    print("-", TABLES_DIR / "time_series_holdout_predictions.csv")
    print("-", TABLES_DIR / "time_series_future_forecasts.csv")
    print("-", TABLES_DIR / "time_series_model_notes.csv")


if __name__ == "__main__":
    main()

