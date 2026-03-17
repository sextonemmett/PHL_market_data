# Repo Summary

This repo currently focuses on downloading, validating, combining, and analyzing two IEMOP RTD datasets:

- `MP`: RTD market clearing price
- `MP_RESERVE`: RTD reserve market clearing price

The main pipeline is:

1. Download daily CSVs from IEMOP
2. Run per-file QC during download
3. Concatenate all daily files into one combined Parquet
4. Generate QC, analysis, and resource-name reports from the combined Parquets

## Download + Combine

The two downloader scripts are:

- [`scripts/download_mp.py`](/Users/emmett/Desktop/Github/PHL_market_data/scripts/download_mp.py)
- [`scripts/download_mp_reserve.py`](/Users/emmett/Desktop/Github/PHL_market_data/scripts/download_mp_reserve.py)

Both scripts:

- accept either start/end URLs or explicit date ranges
- generate all in-between daily IEMOP download URLs
- download raw CSVs into `data/<dataset>/raw`
- write a QC manifest into `data/<dataset>/qc`
- normalize `RUN_TIME` and `TIME_INTERVAL`
- write one combined Parquet into `data/<dataset>/combined`

Current combined outputs:

- MP: [`data/mp/combined/MP_20251216_20260316.parquet`](/Users/emmett/Desktop/Github/PHL_market_data/data/mp/combined/MP_20251216_20260316.parquet)
- Reserve: [`data/mp_reserve/combined/MP_RESERVE_20251216_20260316.parquet`](/Users/emmett/Desktop/Github/PHL_market_data/data/mp_reserve/combined/MP_RESERVE_20251216_20260316.parquet)

Current QC manifests:

- MP: [`data/mp/qc/mp_qc_20251216_20260316.csv`](/Users/emmett/Desktop/Github/PHL_market_data/data/mp/qc/mp_qc_20251216_20260316.csv)
- Reserve: [`data/mp_reserve/qc/mp_reserve_qc_20251216_20260316.csv`](/Users/emmett/Desktop/Github/PHL_market_data/data/mp_reserve/qc/mp_reserve_qc_20251216_20260316.csv)

Typical rerun commands:

```bash
/opt/homebrew/bin/uv run python scripts/download_mp.py
/opt/homebrew/bin/uv run python scripts/download_mp_reserve.py
```

## Combined DataFrame Structure

Both combined Parquets use the same column layout:

- `RUN_TIME`: `datetime64[ns]`
- `MKT_TYPE`: `category`
- `TIME_INTERVAL`: `datetime64[ns]`
- `REGION_NAME`: `category`
- `RESOURCE_NAME`: `category`
- `RESOURCE_TYPE`: `category`
- `COMMODITY_TYPE`: `category`
- `MARGINAL_PRICE`: `float64`

Practical meaning:

- `RUN_TIME`: dispatch run timestamp
- `TIME_INTERVAL`: market interval timestamp
- `REGION_NAME`: one of `CLUZ`, `CVIS`, `CMIN`
- `RESOURCE_NAME`: resource identifier from IEMOP
- `COMMODITY_TYPE`:
  - MP: only `En`
  - Reserve: `Dr`, `Fr`, `Rd`, `Ru`
- `MARGINAL_PRICE`: price in `PHP/MWh`

## Current Dataset Contents

### MP

Source:

- [`data/mp/combined/MP_20251216_20260316.parquet`](/Users/emmett/Desktop/Github/PHL_market_data/data/mp/combined/MP_20251216_20260316.parquet)

Current contents:

- rows: `71,460`
- date range: `2025-12-16 00:00:00` to `2026-03-16 23:55:00`
- regions: `CLUZ`, `CVIS`, `CMIN`
- unique resources: `259`
- commodity types: `En`
- price range: about `-10,000.024` to `32,000.0096 PHP/MWh`

### Reserve

Source:

- [`data/mp_reserve/combined/MP_RESERVE_20251216_20260316.parquet`](/Users/emmett/Desktop/Github/PHL_market_data/data/mp_reserve/combined/MP_RESERVE_20251216_20260316.parquet)

Current contents:

- rows: `314,160`
- date range: `2025-12-16 00:00:00` to `2026-03-16 23:55:00`
- regions: `CLUZ`, `CVIS`, `CMIN`
- unique resources: `182`
- commodity types:
  - `Dr`: Delayed Contingency Raise
  - `Fr`: Fast Contingency Raise
  - `Rd`: Regulation Down
  - `Ru`: Regulation Up
- price range: `0` to `25,000 PHP/MWh`

## Reports + Analysis Scripts

QC and analysis scripts:

- [`scripts/generate_qc_report.py`](/Users/emmett/Desktop/Github/PHL_market_data/scripts/generate_qc_report.py)
- [`scripts/generate_analysis_figures.py`](/Users/emmett/Desktop/Github/PHL_market_data/scripts/generate_analysis_figures.py)
- [`scripts/generate_resource_name_analysis.py`](/Users/emmett/Desktop/Github/PHL_market_data/scripts/generate_resource_name_analysis.py)

Current report folders:

- QC: [`reports/qc`](/Users/emmett/Desktop/Github/PHL_market_data/reports/qc)
- General analysis: [`reports/analysis`](/Users/emmett/Desktop/Github/PHL_market_data/reports/analysis)
- Resource-name analysis: [`reports/analysis/resources`](/Users/emmett/Desktop/Github/PHL_market_data/reports/analysis/resources)

Typical rerun commands:

```bash
/opt/homebrew/bin/uv run python scripts/generate_qc_report.py
/opt/homebrew/bin/uv run python scripts/generate_analysis_figures.py
/opt/homebrew/bin/uv run python scripts/generate_resource_name_analysis.py
```

## Notes

- The raw daily CSVs live under `data/mp/raw` and `data/mp_reserve/raw`.
- The combined artifacts are Parquet, not CSV.
- The current analysis assumes “collapsed resource name” means strict last-underscore stripping, for example:
  - `01MAGAT_U04` -> `01MAGAT`
- The current resource-analysis outputs compare both full resource names and collapsed prefixes.
