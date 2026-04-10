"""Generate EDA plots and summary statistics for Medicaid expenditure series."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"


def read_timeseries(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["fiscal_year"] = frame["fiscal_year"].astype(int)
    frame["total_medicaid_expenditures"] = pd.to_numeric(
        frame["total_medicaid_expenditures"], errors="coerce"
    )
    frame["total_medicaid_expenditures_billions"] = pd.to_numeric(
        frame["total_medicaid_expenditures_billions"], errors="coerce"
    )
    frame["yoy_growth_pct"] = pd.to_numeric(frame["yoy_growth_pct"], errors="coerce")
    return frame.sort_values("fiscal_year")


def save_line_plot(
    frame: pd.DataFrame,
    y_col: str,
    title: str,
    y_label: str,
    output_path: Path,
    color: str,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(
        frame["fiscal_year"],
        frame[y_col],
        marker="o",
        linewidth=2.2,
        color=color,
    )
    plt.title(title)
    plt.xlabel("Fiscal Year")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.xticks(frame["fiscal_year"])
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_growth_comparison_plot(
    michigan: pd.DataFrame, national: pd.DataFrame, output_path: Path
) -> None:
    michigan_growth = michigan.dropna(subset=["yoy_growth_pct"])
    national_growth = national.dropna(subset=["yoy_growth_pct"])

    plt.figure(figsize=(10, 5))
    plt.plot(
        michigan_growth["fiscal_year"],
        michigan_growth["yoy_growth_pct"],
        marker="o",
        linewidth=2.2,
        label="Michigan",
        color="#1f77b4",
    )
    plt.plot(
        national_growth["fiscal_year"],
        national_growth["yoy_growth_pct"],
        marker="o",
        linewidth=2.2,
        label="United States",
        color="#ff7f0e",
    )
    plt.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    plt.title("Year-over-Year Growth Comparison: Michigan vs United States")
    plt.xlabel("Fiscal Year")
    plt.ylabel("YoY Growth (%)")
    plt.grid(True, alpha=0.3)
    plt.xticks(michigan_growth["fiscal_year"])
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_trend_residual_frame(frame: pd.DataFrame) -> pd.DataFrame:
    component_frame = frame.copy()
    component_frame["trend_billions"] = (
        component_frame["total_medicaid_expenditures_billions"]
        .rolling(window=3, center=True, min_periods=1)
        .mean()
    )
    component_frame["residual_billions"] = (
        component_frame["total_medicaid_expenditures_billions"]
        - component_frame["trend_billions"]
    )
    return component_frame


def save_trend_residual_plot(
    frame: pd.DataFrame, geography: str, output_path: Path, color: str
) -> None:
    component_frame = build_trend_residual_frame(frame)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(
        component_frame["fiscal_year"],
        component_frame["total_medicaid_expenditures_billions"],
        marker="o",
        linewidth=2.2,
        label="Observed",
        color=color,
    )
    axes[0].plot(
        component_frame["fiscal_year"],
        component_frame["trend_billions"],
        linewidth=2,
        label="3-Year Rolling Trend",
        color="black",
    )
    axes[0].set_title(f"{geography} Trend and Residual Assessment")
    axes[0].set_ylabel("Expenditures (Billions USD)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].bar(
        component_frame["fiscal_year"],
        component_frame["residual_billions"],
        color=color,
        alpha=0.75,
    )
    axes[1].axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Fiscal Year")
    axes[1].set_ylabel("Residual (Billions USD)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(component_frame["fiscal_year"])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def summarize_series(frame: pd.DataFrame, geography: str) -> dict[str, str]:
    start = frame.iloc[0]
    end = frame.iloc[-1]
    yoy = frame["yoy_growth_pct"].dropna()
    return {
        "geography": geography,
        "start_year": str(int(start["fiscal_year"])),
        "end_year": str(int(end["fiscal_year"])),
        "start_expenditure": f"{start['total_medicaid_expenditures']:.2f}",
        "end_expenditure": f"{end['total_medicaid_expenditures']:.2f}",
        "absolute_change": f"{(end['total_medicaid_expenditures'] - start['total_medicaid_expenditures']):.2f}",
        "percent_change": f"{((end['total_medicaid_expenditures'] - start['total_medicaid_expenditures']) / start['total_medicaid_expenditures']) * 100:.2f}",
        "min_yoy_growth_pct": f"{yoy.min():.2f}",
        "max_yoy_growth_pct": f"{yoy.max():.2f}",
        "mean_yoy_growth_pct": f"{yoy.mean():.2f}",
    }


def write_summary(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "geography",
                "start_year",
                "end_year",
                "start_expenditure",
                "end_expenditure",
                "absolute_change",
                "percent_change",
                "min_yoy_growth_pct",
                "max_yoy_growth_pct",
                "mean_yoy_growth_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_decomposition_note(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "geography",
                "data_frequency",
                "seasonality_estimable",
                "trend_method",
                "residual_definition",
                "note",
            ],
        )
        writer.writeheader()
        note = {
            "data_frequency": "annual",
            "seasonality_estimable": "no",
            "trend_method": "3-year centered rolling average",
            "residual_definition": "observed minus rolling trend",
            "note": "With annual data and only 13 observations, within-year seasonality cannot be meaningfully estimated. The decomposition step therefore focuses on trend and residual assessment only.",
        }
        writer.writerow({"geography": "Michigan", **note})
        writer.writerow({"geography": "United States", **note})


def main() -> None:
    michigan = read_timeseries(PROCESSED_DIR / "medicaid_michigan_timeseries.csv")
    national = read_timeseries(PROCESSED_DIR / "medicaid_national_timeseries.csv")

    save_line_plot(
        michigan,
        "total_medicaid_expenditures_billions",
        "Michigan Medicaid Expenditures Over Time",
        "Total Medicaid Expenditures (Billions USD)",
        FIGURES_DIR / "michigan_expenditure_trend.png",
        "#1f77b4",
    )
    save_line_plot(
        national,
        "total_medicaid_expenditures_billions",
        "United States Medicaid Expenditures Over Time",
        "Total Medicaid Expenditures (Billions USD)",
        FIGURES_DIR / "national_expenditure_trend.png",
        "#ff7f0e",
    )
    save_line_plot(
        michigan.dropna(subset=["yoy_growth_pct"]),
        "yoy_growth_pct",
        "Michigan Medicaid Expenditure Year-over-Year Growth",
        "YoY Growth (%)",
        FIGURES_DIR / "michigan_yoy_growth.png",
        "#2ca02c",
    )
    save_line_plot(
        national.dropna(subset=["yoy_growth_pct"]),
        "yoy_growth_pct",
        "United States Medicaid Expenditure Year-over-Year Growth",
        "YoY Growth (%)",
        FIGURES_DIR / "national_yoy_growth.png",
        "#d62728",
    )
    save_growth_comparison_plot(
        michigan,
        national,
        FIGURES_DIR / "michigan_vs_national_yoy_growth.png",
    )
    save_trend_residual_plot(
        michigan,
        "Michigan",
        FIGURES_DIR / "michigan_trend_residual_assessment.png",
        "#1f77b4",
    )
    save_trend_residual_plot(
        national,
        "United States",
        FIGURES_DIR / "national_trend_residual_assessment.png",
        "#ff7f0e",
    )

    summary_rows = [
        summarize_series(michigan, "Michigan"),
        summarize_series(national, "United States"),
    ]
    write_summary(summary_rows, TABLES_DIR / "eda_summary_statistics.csv")
    write_decomposition_note(TABLES_DIR / "decomposition_assessment.csv")

    print("Created EDA outputs:")
    print("-", FIGURES_DIR / "michigan_expenditure_trend.png")
    print("-", FIGURES_DIR / "national_expenditure_trend.png")
    print("-", FIGURES_DIR / "michigan_yoy_growth.png")
    print("-", FIGURES_DIR / "national_yoy_growth.png")
    print("-", FIGURES_DIR / "michigan_vs_national_yoy_growth.png")
    print("-", FIGURES_DIR / "michigan_trend_residual_assessment.png")
    print("-", FIGURES_DIR / "national_trend_residual_assessment.png")
    print("-", TABLES_DIR / "eda_summary_statistics.csv")
    print("-", TABLES_DIR / "decomposition_assessment.csv")


if __name__ == "__main__":
    main()
