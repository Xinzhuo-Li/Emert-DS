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
ROLLING_VALIDATION_MIN_POINTS = 6
ARIMA_CANDIDATES = [(p, d, q) for p in range(0, 3) for d in range(0, 3) for q in range(0, 3) if not (p == d == q == 0)]
HOLT_CANDIDATES = (
    "simple_exponential_smoothing",
    "holt_winters_additive",
    "holt_winters_additive_damped",
)
PROPHET_CANDIDATES = (
    {"changepoint_prior_scale": 0.05, "n_changepoints": 3},
    {"changepoint_prior_scale": 0.05, "n_changepoints": 5},
    {"changepoint_prior_scale": 0.20, "n_changepoints": 3},
    {"changepoint_prior_scale": 0.20, "n_changepoints": 5},
    {"changepoint_prior_scale": 0.50, "n_changepoints": 3},
    {"changepoint_prior_scale": 0.50, "n_changepoints": 5},
)


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


def fit_arima(train_y: np.ndarray, order: tuple[int, int, int]) -> object:
    warnings.filterwarnings("ignore")
    return ARIMA(train_y, order=order).fit()


def fit_holt_model(train_y: np.ndarray, label: str) -> object:
    warnings.filterwarnings("ignore")
    if label == "simple_exponential_smoothing":
        return SimpleExpSmoothing(train_y).fit()
    if label == "holt_winters_additive":
        return ExponentialSmoothing(
            train_y,
            trend="add",
            seasonal=None,
            damped_trend=False,
        ).fit(optimized=True)
    if label == "holt_winters_additive_damped":
        return ExponentialSmoothing(
            train_y,
            trend="add",
            seasonal=None,
            damped_trend=True,
        ).fit(optimized=True)
    raise ValueError(f"Unsupported Holt-Winters candidate: {label}")


def prophet_frame(rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": [f"{int(row['fiscal_year'])}-12-31" for row in rows],
            "y": [float(row["total_medicaid_expenditures"]) for row in rows],
        }
    )


def fit_prophet_model(
    train_rows: list[dict[str, float | int | str]],
    changepoint_prior_scale: float = 0.2,
    n_changepoints: int = 5,
) -> Prophet:
    effective_n_changepoints = min(n_changepoints, max(0, len(train_rows) - 2))
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        n_changepoints=effective_n_changepoints,
    )
    model.fit(prophet_frame(train_rows))
    return model


def rolling_validation_mape(
    rows: list[dict[str, float | int | str]],
    forecast_fn,
    min_points: int = ROLLING_VALIDATION_MIN_POINTS,
) -> float:
    train_rows, _ = train_test_split_time_ordered(rows, HOLDOUT_SIZE)
    errors: list[float] = []
    for end in range(min_points, len(train_rows)):
        fit_rows = train_rows[:end]
        validation_row = train_rows[end]
        prediction = float(forecast_fn(fit_rows))
        actual = float(validation_row["total_medicaid_expenditures"])
        errors.append(abs((actual - prediction) / actual) * 100)
    return sum(errors) / len(errors)


def arima_one_step_forecast(
    fit_rows: list[dict[str, float | int | str]], order: tuple[int, int, int]
) -> float:
    fit_y = np.array([row["total_medicaid_expenditures"] for row in fit_rows], dtype=float)
    model = fit_arima(fit_y, order)
    return float(model.forecast(steps=1)[0])


def fit_best_arima(
    geography: str, rows: list[dict[str, float | int | str]]
) -> tuple[tuple[int, int, int], list[dict[str, str]]]:
    train_rows, _ = train_test_split_time_ordered(rows, HOLDOUT_SIZE)
    train_y = np.array([row["total_medicaid_expenditures"] for row in train_rows], dtype=float)
    candidate_rows: list[dict[str, str]] = []
    scored_candidates: list[tuple[tuple[int, int, int], float, float]] = []

    for order in ARIMA_CANDIDATES:
        try:
            rolling_mape = rolling_validation_mape(
                rows, lambda fit_rows, order=order: arima_one_step_forecast(fit_rows, order)
            )
            fit_result = fit_arima(train_y, order)
        except Exception:
            continue
        scored_candidates.append((order, rolling_mape, float(fit_result.aic)))

    if not scored_candidates:
        raise RuntimeError("Unable to fit any ARIMA configuration.")

    selected_order, _, _ = min(scored_candidates, key=lambda item: (item[1], sum(item[0]), item[2]))

    for order, rolling_mape, fit_stat in scored_candidates:
        candidate_rows.append(
            {
                "geography": geography,
                "model": "arima",
                "candidate_config": f"order={order}",
                "rolling_cv_mape": f"{rolling_mape:.2f}",
                "fit_stat_name": "aic",
                "fit_stat_value": f"{fit_stat:.2f}",
                "selected": "yes" if order == selected_order else "no",
                "selection_rule": "Lowest rolling CV MAPE; ties broken by lower complexity and lower AIC.",
            }
        )

    return selected_order, candidate_rows


def holt_one_step_forecast(
    fit_rows: list[dict[str, float | int | str]], label: str
) -> float:
    fit_y = np.array([row["total_medicaid_expenditures"] for row in fit_rows], dtype=float)
    model = fit_holt_model(fit_y, label)
    return float(model.forecast(1)[0])


def fit_best_holt_winters(
    geography: str, rows: list[dict[str, float | int | str]]
) -> tuple[str, list[dict[str, str]]]:
    train_rows, _ = train_test_split_time_ordered(rows, HOLDOUT_SIZE)
    train_y = np.array([row["total_medicaid_expenditures"] for row in train_rows], dtype=float)
    candidate_rows: list[dict[str, str]] = []
    scored_candidates: list[tuple[str, float, float, int]] = []
    complexity_rank = {
        "simple_exponential_smoothing": 1,
        "holt_winters_additive": 2,
        "holt_winters_additive_damped": 3,
    }

    for label in HOLT_CANDIDATES:
        try:
            rolling_mape = rolling_validation_mape(
                rows, lambda fit_rows, label=label: holt_one_step_forecast(fit_rows, label)
            )
            fit_model = fit_holt_model(train_y, label)
        except Exception:
            continue
        scored_candidates.append((label, rolling_mape, float(fit_model.sse), complexity_rank[label]))

    if not scored_candidates:
        raise RuntimeError("Unable to fit any exponential smoothing configuration.")

    selected_label, _, _, _ = min(scored_candidates, key=lambda item: (item[1], item[3], item[2]))

    for label, rolling_mape, fit_stat, _ in scored_candidates:
        candidate_rows.append(
            {
                "geography": geography,
                "model": "holt_winters",
                "candidate_config": label,
                "rolling_cv_mape": f"{rolling_mape:.2f}",
                "fit_stat_name": "sse",
                "fit_stat_value": f"{fit_stat:.2f}",
                "selected": "yes" if label == selected_label else "no",
                "selection_rule": "Lowest rolling CV MAPE; ties broken by simpler smoothing structure and lower SSE.",
            }
        )

    return selected_label, candidate_rows


def prophet_one_step_forecast(
    fit_rows: list[dict[str, float | int | str]],
    changepoint_prior_scale: float,
    n_changepoints: int,
) -> float:
    model = fit_prophet_model(
        fit_rows,
        changepoint_prior_scale=changepoint_prior_scale,
        n_changepoints=n_changepoints,
    )
    future = model.make_future_dataframe(periods=1, freq="YE")
    return float(model.predict(future)["yhat"].iloc[-1])


def fit_best_prophet(
    geography: str, rows: list[dict[str, float | int | str]]
) -> tuple[dict[str, float | int], list[dict[str, str]]]:
    candidate_rows: list[dict[str, str]] = []
    scored_candidates: list[tuple[dict[str, float | int], float, tuple[float, int]]] = []

    for config in PROPHET_CANDIDATES:
        try:
            rolling_mape = rolling_validation_mape(
                rows,
                lambda fit_rows, config=config: prophet_one_step_forecast(
                    fit_rows,
                    changepoint_prior_scale=float(config["changepoint_prior_scale"]),
                    n_changepoints=int(config["n_changepoints"]),
                ),
            )
        except Exception:
            continue
        scored_candidates.append(
            (
                config,
                rolling_mape,
                (float(config["changepoint_prior_scale"]), int(config["n_changepoints"])),
            )
        )

    if not scored_candidates:
        raise RuntimeError("Unable to fit any Prophet configuration.")

    selected_config, _, _ = min(scored_candidates, key=lambda item: (item[1], item[2][0], item[2][1]))

    for config, rolling_mape, _ in scored_candidates:
        candidate_rows.append(
            {
                "geography": geography,
                "model": "prophet",
                "candidate_config": (
                    f"changepoint_prior_scale={float(config['changepoint_prior_scale']):.2f}; "
                    f"n_changepoints={int(config['n_changepoints'])}"
                ),
                "rolling_cv_mape": f"{rolling_mape:.2f}",
                "fit_stat_name": "",
                "fit_stat_value": "",
                "selected": "yes" if config == selected_config else "no",
                "selection_rule": "Lowest rolling CV MAPE; ties broken by lower changepoint_prior_scale and fewer changepoints.",
            }
        )

    return selected_config, candidate_rows


def evaluate_arima(
    geography: str,
    rows: list[dict[str, float | int | str]],
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]], dict[str, str], list[dict[str, str]]]:
    train_rows, test_rows = train_test_split_time_ordered(rows, HOLDOUT_SIZE)
    train_y = np.array([row["total_medicaid_expenditures"] for row in train_rows], dtype=float)
    test_y = np.array([row["total_medicaid_expenditures"] for row in test_rows], dtype=float)

    order, tuning_rows = fit_best_arima(geography, rows)
    selected_summary = next(row for row in tuning_rows if row["selected"] == "yes")
    model = fit_arima(train_y, order)
    predictions = model.forecast(steps=HOLDOUT_SIZE)
    metric_row = {
        "geography": geography,
        "model": "arima",
        "model_details": f"order={order}; rolling_cv_mape={selected_summary['rolling_cv_mape']}",
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
    full_model = fit_arima(full_y, order)
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
        "note": f"Annual data does not support meaningful seasonal estimation, so non-seasonal ARIMA with selected order {order} was used in place of SARIMA.",
    }
    return metric_row, holdout_rows, forecast_rows, note, tuning_rows


def evaluate_holt_winters(
    geography: str,
    rows: list[dict[str, float | int | str]],
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    train_rows, test_rows = train_test_split_time_ordered(rows, HOLDOUT_SIZE)
    train_y = np.array([row["total_medicaid_expenditures"] for row in train_rows], dtype=float)
    test_y = np.array([row["total_medicaid_expenditures"] for row in test_rows], dtype=float)

    label, tuning_rows = fit_best_holt_winters(geography, rows)
    selected_summary = next(row for row in tuning_rows if row["selected"] == "yes")
    model = fit_holt_model(train_y, label)
    predictions = model.forecast(HOLDOUT_SIZE)
    metric_row = {
        "geography": geography,
        "model": "holt_winters",
        "model_details": f"{label}; rolling_cv_mape={selected_summary['rolling_cv_mape']}",
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
    full_model = fit_holt_model(full_y, label)
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
    return metric_row, holdout_rows, forecast_rows, tuning_rows


def evaluate_prophet(
    geography: str,
    rows: list[dict[str, float | int | str]],
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    train_rows, test_rows = train_test_split_time_ordered(rows, HOLDOUT_SIZE)
    selected_config, tuning_rows = fit_best_prophet(geography, rows)
    selected_summary = next(row for row in tuning_rows if row["selected"] == "yes")
    model = fit_prophet_model(
        train_rows,
        changepoint_prior_scale=float(selected_config["changepoint_prior_scale"]),
        n_changepoints=int(selected_config["n_changepoints"]),
    )
    future = model.make_future_dataframe(periods=HOLDOUT_SIZE, freq="YE")
    forecast = model.predict(future)
    predictions = forecast["yhat"].tail(HOLDOUT_SIZE).to_numpy()
    test_y = np.array([row["total_medicaid_expenditures"] for row in test_rows], dtype=float)

    metric_row = {
        "geography": geography,
        "model": "prophet",
        "model_details": (
            "linear_growth_no_seasonality; "
            f"changepoint_prior_scale={float(selected_config['changepoint_prior_scale']):.2f}; "
            f"n_changepoints={int(selected_config['n_changepoints'])}; "
            f"rolling_cv_mape={selected_summary['rolling_cv_mape']}"
        ),
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

    full_model = fit_prophet_model(
        rows,
        changepoint_prior_scale=float(selected_config["changepoint_prior_scale"]),
        n_changepoints=int(selected_config["n_changepoints"]),
    )
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
    return metric_row, holdout_rows, forecast_rows, tuning_rows


def main() -> None:
    michigan_rows = read_series(PROCESSED_DIR / "medicaid_michigan_timeseries.csv")
    national_rows = read_series(PROCESSED_DIR / "medicaid_national_timeseries.csv")

    metrics_rows: list[dict[str, str]] = []
    holdout_rows: list[dict[str, str]] = []
    forecast_rows: list[dict[str, str]] = []
    notes_rows: list[dict[str, str]] = []
    tuning_rows: list[dict[str, str]] = []

    for geography, rows in (
        ("Michigan", michigan_rows),
        ("United States", national_rows),
    ):
        arima_metric, arima_holdout, arima_forecast, arima_note, arima_tuning = evaluate_arima(
            geography, rows
        )
        hw_metric, hw_holdout, hw_forecast, hw_tuning = evaluate_holt_winters(geography, rows)
        prophet_metric, prophet_holdout, prophet_forecast, prophet_tuning = evaluate_prophet(
            geography, rows
        )

        metrics_rows.extend([arima_metric, hw_metric, prophet_metric])
        holdout_rows.extend(arima_holdout + hw_holdout + prophet_holdout)
        forecast_rows.extend(arima_forecast + hw_forecast + prophet_forecast)
        notes_rows.append(arima_note)
        tuning_rows.extend(arima_tuning + hw_tuning + prophet_tuning)

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
    write_csv(
        TABLES_DIR / "time_series_tuning_summary.csv",
        tuning_rows,
        [
            "geography",
            "model",
            "candidate_config",
            "rolling_cv_mape",
            "fit_stat_name",
            "fit_stat_value",
            "selected",
            "selection_rule",
        ],
    )

    print("Created time-series forecasting outputs:")
    print("-", TABLES_DIR / "time_series_model_metrics.csv")
    print("-", TABLES_DIR / "time_series_holdout_predictions.csv")
    print("-", TABLES_DIR / "time_series_future_forecasts.csv")
    print("-", TABLES_DIR / "time_series_model_notes.csv")
    print("-", TABLES_DIR / "time_series_tuning_summary.csv")


if __name__ == "__main__":
    main()

