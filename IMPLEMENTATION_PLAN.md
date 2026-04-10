# Implementation Plan

## Phase 1: Project Setup
Goal: Create a clean and reproducible project foundation.

Tasks:
- finalize repository structure
- inventory the existing annual Excel files for 2013-2025
- move or mirror source Excel files into `data/raw/`
- create Python virtual environment
- install required libraries
- confirm available geography levels such as state and national data

Outputs:
- organized folders
- dependency list
- documented project scope and assumptions

## Phase 2: Data Understanding and Ingestion
Goal: Inspect the source files and define a reliable ingestion process.

Tasks:
- identify all input files and worksheets
- inspect column names, data types, and year coverage
- map source columns to standard fields
- build a repeatable Excel loading script
- document data quality issues and missing values

Outputs:
- raw data inventory
- data dictionary
- first-pass ingestion script

## Phase 3: Data Cleaning and Transformation
Goal: Produce a clean time-series dataset for forecasting.

Tasks:
- standardize fiscal year fields
- coerce expenditure values to numeric format
- remove or flag invalid rows
- handle missing years and duplicate records
- prepare one clean dataset for state-level analysis
- prepare one clean dataset for national analysis

Outputs:
- cleaned datasets in `data/processed/`
- documented cleaning rules

## Phase 4: Exploratory Data Analysis
Goal: Understand historical behavior before modeling.

Tasks:
- plot expenditure over time
- compute year-over-year growth
- identify structural breaks and outliers
- compare state and national growth patterns
- determine whether annual seasonality is relevant based on data frequency

Outputs:
- EDA notebook or script
- charts stored in `outputs/figures/`
- summary observations for modeling decisions

## Phase 5: Baseline Forecasting
Goal: Establish simple benchmark models.

Tasks:
- build linear regression trend model
- build polynomial regression trend model
- generate holdout forecasts
- record benchmark metrics

Outputs:
- baseline forecast results
- benchmark metric table

## Phase 6: Time-Series Forecasting
Goal: Build and compare stronger forecasting models.

Tasks:
- fit ARIMA or SARIMA where appropriate
- fit Holt-Winters exponential smoothing
- fit Prophet if dependency support is available
- tune model parameters with a validation strategy that respects time order

Outputs:
- trained model comparison set
- forecast performance table

## Phase 7: Model Selection and Projection
Goal: Produce final 10-year projections with uncertainty estimates.

Tasks:
- select the best model for each geography
- retrain selected models on full historical data
- generate forecasts for years 2026-2035
- compute 95 percent confidence intervals

Outputs:
- final projection tables in `outputs/tables/`
- final model selection summary

## Phase 8: Reporting and Export
Goal: Package the results for submission and reuse.

Tasks:
- export projection tables to CSV or Excel
- generate publication-ready charts
- write model comparison summary
- update README with exact run instructions
- optionally add a notebook or Streamlit dashboard

Outputs:
- final visuals
- submission-ready documentation
- optional interactive dashboard

## Risks and Decisions to Resolve
- Confirm the exact source dataset referenced by the project description.
- Confirm whether the data is annual, quarterly, or monthly.
- Confirm the state to forecast if multiple states are present.
- Confirm whether `Prophet` is allowed in the environment and needed for the final comparison.
- Confirm whether output must include both PDF and HTML or if HTML plus image exports is sufficient.

## Suggested Immediate Next Steps
1. Consolidate the existing `FY_2013` to `FY_2025` Excel files into `data/raw/`.
2. Inspect the source columns, sheet names, and year coverage.
3. Build the first ingestion script and a simple data profile report.
