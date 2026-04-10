"""Prepare analysis-ready datasets for EDA and forecasting."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

STATE_COLUMNS = [
    "fiscal_year",
    "jurisdiction",
    "total_medicaid_expenditures",
    "mfcu_grant_expenditures",
    "total_recoveries",
    "staff_on_board",
]

BALANCED_COLUMNS = [
    "fiscal_year",
    "jurisdiction",
    "has_observed_data",
    "total_medicaid_expenditures",
    "mfcu_grant_expenditures",
    "total_recoveries",
    "staff_on_board",
]

TIMESERIES_COLUMNS = [
    "fiscal_year",
    "geography",
    "total_medicaid_expenditures",
    "total_medicaid_expenditures_billions",
    "yoy_growth_pct",
    "mfcu_grant_expenditures",
    "total_recoveries",
    "staff_on_board",
]

QUALITY_COLUMNS = [
    "dataset",
    "min_year",
    "max_year",
    "row_count",
    "missing_total_medicaid_expenditures",
    "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def to_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_decimal(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def build_timeseries_row(
    row: dict[str, str], geography: str, previous_value: float | None
) -> tuple[dict[str, str], float | None]:
    current_value = to_float(row.get("total_medicaid_expenditures", ""))
    yoy_growth = None
    if previous_value not in (None, 0) and current_value is not None:
        yoy_growth = ((current_value - previous_value) / previous_value) * 100

    timeseries_row = {
        "fiscal_year": row["fiscal_year"],
        "geography": geography,
        "total_medicaid_expenditures": row.get("total_medicaid_expenditures", ""),
        "total_medicaid_expenditures_billions": format_decimal(
            current_value / 1_000_000_000 if current_value is not None else None
        ),
        "yoy_growth_pct": format_decimal(yoy_growth),
        "mfcu_grant_expenditures": row.get("mfcu_grant_expenditures", ""),
        "total_recoveries": row.get("total_recoveries", ""),
        "staff_on_board": row.get("staff_on_board", ""),
    }
    return timeseries_row, current_value


def main() -> None:
    jurisdiction_rows = read_csv(PROCESSED_DIR / "medicaid_jurisdiction_level.csv")
    national_rows = read_csv(PROCESSED_DIR / "medicaid_national_totals.csv")

    state_rows = [row for row in jurisdiction_rows if row["jurisdiction_type"] == "state"]
    state_names = sorted({row["jurisdiction"] for row in state_rows})
    years = sorted({int(row["fiscal_year"]) for row in jurisdiction_rows})

    states_only_observed = [
        {column: row.get(column, "") for column in STATE_COLUMNS}
        for row in sorted(state_rows, key=lambda row: (int(row["fiscal_year"]), row["jurisdiction"]))
    ]

    observed_lookup = {
        (int(row["fiscal_year"]), row["jurisdiction"]): row for row in state_rows
    }
    states_only_balanced: list[dict[str, str]] = []
    for year in years:
        for state in state_names:
            observed = observed_lookup.get((year, state))
            states_only_balanced.append(
                {
                    "fiscal_year": str(year),
                    "jurisdiction": state,
                    "has_observed_data": "yes" if observed else "no",
                    "total_medicaid_expenditures": observed.get("total_medicaid_expenditures", "") if observed else "",
                    "mfcu_grant_expenditures": observed.get("mfcu_grant_expenditures", "") if observed else "",
                    "total_recoveries": observed.get("total_recoveries", "") if observed else "",
                    "staff_on_board": observed.get("staff_on_board", "") if observed else "",
                }
            )

    michigan_rows = sorted(
        [row for row in state_rows if row["jurisdiction"] == "Michigan"],
        key=lambda row: int(row["fiscal_year"]),
    )
    national_rows_sorted = sorted(national_rows, key=lambda row: int(row["fiscal_year"]))

    michigan_timeseries: list[dict[str, str]] = []
    previous = None
    for row in michigan_rows:
        output_row, previous = build_timeseries_row(row, "Michigan", previous)
        michigan_timeseries.append(output_row)

    national_timeseries: list[dict[str, str]] = []
    previous = None
    for row in national_rows_sorted:
        output_row, previous = build_timeseries_row(row, "United States", previous)
        national_timeseries.append(output_row)

    eda_ready = sorted(
        michigan_timeseries + national_timeseries,
        key=lambda row: (row["geography"], int(row["fiscal_year"])),
    )

    quality_rows = [
        {
            "dataset": "medicaid_states_only_observed",
            "min_year": str(min(years)),
            "max_year": str(max(years)),
            "row_count": str(len(states_only_observed)),
            "missing_total_medicaid_expenditures": str(
                sum(1 for row in states_only_observed if row["total_medicaid_expenditures"] == "")
            ),
            "notes": "Observed state rows only; North Dakota is absent before 2020.",
        },
        {
            "dataset": "medicaid_states_only_balanced",
            "min_year": str(min(years)),
            "max_year": str(max(years)),
            "row_count": str(len(states_only_balanced)),
            "missing_total_medicaid_expenditures": str(
                sum(1 for row in states_only_balanced if row["total_medicaid_expenditures"] == "")
            ),
            "notes": "Balanced 50-state panel with placeholder rows for missing state-year combinations.",
        },
        {
            "dataset": "medicaid_michigan_timeseries",
            "min_year": michigan_timeseries[0]["fiscal_year"],
            "max_year": michigan_timeseries[-1]["fiscal_year"],
            "row_count": str(len(michigan_timeseries)),
            "missing_total_medicaid_expenditures": str(
                sum(1 for row in michigan_timeseries if row["total_medicaid_expenditures"] == "")
            ),
            "notes": "Michigan annual series for direct state-level EDA and forecasting.",
        },
        {
            "dataset": "medicaid_national_timeseries",
            "min_year": national_timeseries[0]["fiscal_year"],
            "max_year": national_timeseries[-1]["fiscal_year"],
            "row_count": str(len(national_timeseries)),
            "missing_total_medicaid_expenditures": str(
                sum(1 for row in national_timeseries if row["total_medicaid_expenditures"] == "")
            ),
            "notes": "National annual series taken from workbook total rows.",
        },
        {
            "dataset": "medicaid_eda_ready",
            "min_year": str(min(years)),
            "max_year": str(max(years)),
            "row_count": str(len(eda_ready)),
            "missing_total_medicaid_expenditures": str(
                sum(1 for row in eda_ready if row["total_medicaid_expenditures"] == "")
            ),
            "notes": "Combined Michigan and United States series with year-over-year growth metrics.",
        },
    ]

    write_csv(PROCESSED_DIR / "medicaid_states_only_observed.csv", states_only_observed, STATE_COLUMNS)
    write_csv(PROCESSED_DIR / "medicaid_states_only_balanced.csv", states_only_balanced, BALANCED_COLUMNS)
    write_csv(PROCESSED_DIR / "medicaid_michigan_timeseries.csv", michigan_timeseries, TIMESERIES_COLUMNS)
    write_csv(PROCESSED_DIR / "medicaid_national_timeseries.csv", national_timeseries, TIMESERIES_COLUMNS)
    write_csv(PROCESSED_DIR / "medicaid_eda_ready.csv", eda_ready, TIMESERIES_COLUMNS)
    write_csv(PROCESSED_DIR / "eda_data_quality_summary.csv", quality_rows, QUALITY_COLUMNS)

    print("Created EDA-ready datasets:")
    print("-", PROCESSED_DIR / "medicaid_states_only_observed.csv")
    print("-", PROCESSED_DIR / "medicaid_states_only_balanced.csv")
    print("-", PROCESSED_DIR / "medicaid_michigan_timeseries.csv")
    print("-", PROCESSED_DIR / "medicaid_national_timeseries.csv")
    print("-", PROCESSED_DIR / "medicaid_eda_ready.csv")
    print("-", PROCESSED_DIR / "eda_data_quality_summary.csv")


if __name__ == "__main__":
    main()
