"""Model evaluation metrics and time-aware split helpers."""

from __future__ import annotations

import math
from typing import Iterable


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_list = list(actual)
    predicted_list = list(predicted)
    return sum(abs(a - p) for a, p in zip(actual_list, predicted_list)) / len(actual_list)


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_list = list(actual)
    predicted_list = list(predicted)
    mse = sum((a - p) ** 2 for a, p in zip(actual_list, predicted_list)) / len(actual_list)
    return math.sqrt(mse)


def mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_list = list(actual)
    predicted_list = list(predicted)
    return (
        sum(abs((a - p) / a) for a, p in zip(actual_list, predicted_list) if a != 0)
        / len(actual_list)
        * 100
    )


def train_test_split_time_ordered[T](rows: list[T], test_size: int) -> tuple[list[T], list[T]]:
    if test_size <= 0 or test_size >= len(rows):
        raise ValueError("test_size must be between 1 and len(rows) - 1")
    return rows[:-test_size], rows[-test_size:]
"""Model evaluation metrics and validation helpers.

This module will contain shared evaluation logic such as:
- MAE
- RMSE
- MAPE
- holdout and time-aware validation helpers
"""

