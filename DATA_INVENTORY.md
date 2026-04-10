# Data Inventory

## Raw File Location
All annual Excel files have been moved into `data/raw/`.

## Files Present
- `FY_2013_MFCU_Statistical_Chart.xlsx`
- `FY_2014_MFCU_Statistical_Chart.xlsx`
- `FY_2015_MFCU_Statistical_Chart.xlsx`
- `FY_2016_MFCU_Statistical_Chart.xlsx`
- `FY_2017_MFCU_Statistical_Chart.xlsx`
- `FY_2018_MFCU_Statistical_Chart.xlsx`
- `FY_2019_MFCU_Statistical_Chart.xlsx`
- `FY_2020_MFCU_Statistical_Chart.xlsx`
- `FY_2021_MFCU_Statistical_Chart.xlsx`
- `FY_2022_MFCU_Statistical_Chart.xlsx`
- `FY_2023_MFCU_Statistical_Chart.xlsx`
- `FY_2024_MFCU_Statistical_Chart.xlsx`
- `FY_2025_MFCU_Statistical_Chart.xlsx`

## Common Workbook Structure
- Most workbooks contain a single sheet.
- Sheet names vary slightly by year:
  - `Statistical Chart`
  - `Sheet0`
  - `FY 2024 MFCU Statistical Chart`
  - `MFCU Statistical Chart`
- The first non-empty row is the report title, for example `FY 2023 MFCU Statistical Chart`.
- The second non-empty row is the column header row.
- The data rows are state-level observations.
- The last few rows are footnotes and metadata, not data records.

## Common Header Pattern
The core columns are highly consistent across years. After full-column extraction, the recurring fields are:

- `State`
- `Total Investigations`
- `Fraud Investigations`
- `Abuse/Neglect Investigations`
- `Total Indictments`
- `Fraud Indictments`
- `Abuse/Neglect Indictments`
- `Total Convictions`
- `Fraud Convictions`
- `Abuse/Neglect Convictions`
- `Civil Settlements and Judgments`
- `Total Recoveries`
- `Total Criminal Recoveries`
- `Total Civil Recoveries` or split civil recovery fields
- `Civil Recoveries Global`
- `Civil Recoveries Other`
- `MFCU Grant Expenditures`
- `Total Medicaid Expenditures`
- `Staff On Board`

## Year-to-Year Variations
- `FY_2019` uses `Sheet0` as the sheet name and appends different footnote numbers in headers such as `State1` and `Total Recoveries3`.
- `FY_2020`, `FY_2024`, and `FY_2025` show state names in uppercase.
- Some years use `Abuse/neglect Convictions` instead of `Abuse/Neglect Convictions`.
- Non-empty row counts vary slightly:
  - 57 rows in `FY_2013` and `FY_2014`
  - 59 rows in `FY_2015` to `FY_2018`
  - 62 rows in `FY_2019` to `FY_2025`

## Important Observation
These files are `MFCU Statistical Chart` reports, but they do contain a usable expenditure field: `Total Medicaid Expenditures`.

This means the current data can support the core project objective of building a historical expenditure series. In addition, the files also include related financial fields such as `MFCU Grant Expenditures` and `Total Recoveries`, which may be useful for contextual analysis later.

The workbook total row also provides a national annual aggregate, which can support the national forecasting component of the project.

## Recommended Next Step
Continue with data cleaning and validation:

1. validate cleaned outputs in `data/processed/`
2. confirm jurisdiction coverage changes by year
3. decide whether territories should be modeled separately from states
4. prepare the final cleaned time series used for EDA and forecasting
