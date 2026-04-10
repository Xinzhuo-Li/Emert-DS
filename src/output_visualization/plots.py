"""Reusable plotting helpers for project outputs."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd


def save_holdout_comparison_plot(
    history: pd.DataFrame,
    holdout: pd.DataFrame,
    geography: str,
    model_name: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(
        history["fiscal_year"],
        history["total_medicaid_expenditures_billions"],
        marker="o",
        linewidth=2.2,
        label="Observed History",
        color="#1f77b4",
    )
    plt.plot(
        holdout["fiscal_year"],
        holdout["actual_billions"],
        marker="o",
        linewidth=2.2,
        label="Holdout Actual",
        color="#2ca02c",
    )
    plt.plot(
        holdout["fiscal_year"],
        holdout["predicted_billions"],
        marker="o",
        linewidth=2.2,
        linestyle="--",
        label="Holdout Predicted",
        color="#d62728",
    )
    plt.title(f"{geography} Holdout Forecast Comparison: {model_name}")
    plt.xlabel("Fiscal Year")
    plt.ylabel("Total Medicaid Expenditures (Billions USD)")
    plt.grid(True, alpha=0.3)
    plt.xticks(sorted(history["fiscal_year"].tolist()))
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_metric_comparison_plot(
    metrics: pd.DataFrame,
    geography: str,
    metric_name: str,
    output_path: Path,
) -> None:
    subset = metrics[metrics["geography"] == geography].copy()
    subset[metric_name] = pd.to_numeric(subset[metric_name], errors="coerce")
    subset = subset.sort_values(metric_name)

    plt.figure(figsize=(10, 5))
    plt.bar(subset["model"], subset[metric_name], color="#4c78a8")
    plt.title(f"{geography} Model Comparison: {metric_name.upper()}")
    plt.xlabel("Model")
    plt.ylabel(metric_name.upper())
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_final_projection_plot(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    geography: str,
    output_path: Path,
) -> None:
    history = history.copy()
    forecast = forecast.copy()
    history["fiscal_year"] = pd.to_numeric(history["fiscal_year"], errors="coerce")
    history["total_medicaid_expenditures_billions"] = pd.to_numeric(
        history["total_medicaid_expenditures_billions"], errors="coerce"
    )
    forecast["fiscal_year"] = pd.to_numeric(forecast["fiscal_year"], errors="coerce")
    forecast["forecast_billions"] = pd.to_numeric(forecast["forecast_billions"], errors="coerce")
    forecast["lower_bound_billions"] = pd.to_numeric(forecast["lower_bound_billions"], errors="coerce")
    forecast["upper_bound_billions"] = pd.to_numeric(forecast["upper_bound_billions"], errors="coerce")

    plt.figure(figsize=(10, 5))
    plt.plot(
        history["fiscal_year"],
        history["total_medicaid_expenditures_billions"],
        marker="o",
        linewidth=2.2,
        label="Historical",
        color="#1f77b4",
    )
    plt.plot(
        forecast["fiscal_year"],
        forecast["forecast_billions"],
        marker="o",
        linewidth=2.2,
        linestyle="--",
        label="Forecast",
        color="#d62728",
    )
    if forecast["lower_bound_billions"].notna().all() and forecast["upper_bound_billions"].notna().all():
        plt.fill_between(
            forecast["fiscal_year"],
            forecast["lower_bound_billions"],
            forecast["upper_bound_billions"],
            color="#d62728",
            alpha=0.15,
            label="95% Interval",
        )
    plt.title(f"{geography} Historical and Projected Medicaid Expenditures")
    plt.xlabel("Fiscal Year")
    plt.ylabel("Total Medicaid Expenditures (Billions USD)")
    plt.grid(True, alpha=0.3)
    plt.xticks(
        sorted(
            set(history["fiscal_year"].dropna().astype(int).tolist())
            | set(forecast["fiscal_year"].dropna().astype(int).tolist())
        ),
        rotation=45,
    )
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

