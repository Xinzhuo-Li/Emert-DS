# Medicaid Expenditure Projection Model

## Overview
This project builds a data-driven forecasting pipeline for Medicaid expenditure projections. It uses historical spending data to estimate future expenditures for:

- the user's current state
- the United States overall

The expected output is a 10-year forward projection with confidence intervals, visual comparison of historical and forecasted values, and a model comparison summary for policy and budget planning use cases.

## Quick Access
- Final report PDF: `report_and_visualization/medicaid_expenditure_projection_report.pdf`
- Final report HTML: `report_and_visualization/index.html`
- Final packaged presentation assets: `report_and_visualization/`
- Core outputs: `outputs/`

## Project Goals
- Ingest and clean historical Medicaid expenditure data from Excel files
- Explore long-term spending trends, growth rates, and possible anomalies
- Train and compare multiple forecasting approaches
- Select the best-performing model using standard forecast metrics
- Produce a 10-year projection table and supporting charts

## Expected Input Data
The project expects structured Excel data with fields similar to:

- `Fiscal Year` or `Period`
- `Expenditure Amount`
- optional geography field such as `State`, `Region`, or `National`

## Current Source Files
The workspace already includes annual Excel files covering fiscal years 2013 through 2025:

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

## Planned Modeling Approaches
- Linear Regression
- Polynomial Regression
- ARIMA / SARIMA
- Exponential Smoothing (Holt-Winters)
- Prophet

## Evaluation Metrics
- `MAE`
- `RMSE`
- `MAPE`

## Project Structure
```text
emrts/
  data/
    raw/              # Original Excel source files
    processed/        # Cleaned and transformed datasets
  src/
    data_ingestion_preprocessing/
    exploratory_data_analysis/
    forecasting_models/
    model_evaluation/
    output_visualization/
  outputs/
    figures/          # Charts and plots
    tables/           # Forecast tables and exported summaries
  README.md
  IMPLEMENTATION_PLAN.md
  requirements.txt
  .gitignore
```

## Planned Workflow
1. Load raw Medicaid expenditure data from Excel.
2. Standardize date, amount, and geographic fields.
3. Perform exploratory analysis on trends and annual growth.
4. Create train/test splits for time-series evaluation.
5. Compare baseline and advanced forecasting models.
6. Generate 10-year forecasts with 95 percent confidence intervals.
7. Export plots, summary tables, and model comparison outputs.

## Setup
Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Order
Run the project in this order:

```bash
python3 src/data_ingestion_preprocessing/ingest_raw_data.py
python3 src/data_ingestion_preprocessing/prepare_analysis_data.py
python3 src/exploratory_data_analysis/eda.py
python3 src/forecasting_models/run_baseline_forecasting.py
python3 src/forecasting_models/run_time_series_forecasting.py
python3 src/output_visualization/run_model_evaluation.py
python3 src/output_visualization/run_reporting_and_export.py
```

## Final Selected Models
- `Michigan`: `Prophet`
- `United States`: `Prophet`

## Key Final Outputs
- Final model selection: `outputs/tables/selected_models.csv`
- Final 2026-2035 projection table: `outputs/tables/final_projection_table.csv`
- Final projection table with 95 percent intervals: `outputs/tables/phase_08_reporting_and_export/final_projection_with_intervals.csv`
- Final summary table: `outputs/tables/phase_08_reporting_and_export/final_results_summary.csv`
- Final projection figures:
  - `outputs/figures/phase_08_reporting_and_export/michigan_final_projection.png`
  - `outputs/figures/phase_08_reporting_and_export/united_states_final_projection.png`
- Final HTML exports:
  - `outputs/tables/phase_08_reporting_and_export/final_projection_with_intervals.html`
  - `outputs/tables/phase_08_reporting_and_export/final_results_summary.html`

## Initial Deliverables
- Forecast-ready cleaned dataset
- Model comparison results
- Projection table for years 2026-2035
- Historical vs. forecast visualization
- Reproducible project documentation

## Notes
- The project description mentions both 10 years and 13 years of historical data. The implementation should use all reliable historical data available, with a minimum target of 10 years.
- The currently observed files provide 13 years of annual source coverage from 2013 to 2025.
- State-level forecasting depends on confirming which state's data is available in the source files.
