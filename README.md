# PHL Market Data Handover

This repository downloads, validates, reshapes, and analyzes Philippine electricity market data published by IEMOP. The codebase is opinionated: it is not a generic market-data toolkit, it is a working research pipeline centered on a specific set of RTD-era inputs, a pair of derived panel datasets, and a collection of regression and visualization outputs under `regressions/`.

This document is written as a hand-over for the next person to inherit the repo. It is intentionally detailed and includes both the intended workflow and the current on-disk state as of `2026-04-08`.

## What This Repo Does

At a high level, the repository has three layers:

1. Download and QC raw IEMOP files into `data/<dataset>/raw`, `data/<dataset>/qc`, and `data/<dataset>/combined`.
2. Build two cleaned panel datasets in `data/panels/` by joining price, demand/control, congestion, and HVDC data.
3. Produce HTML, CSV, and PNG outputs in `regressions/` for descriptive work and econometric analysis.

The most important scripts are:

- `scripts/download_market_data.py`
- `scripts/build_rtd_panels.py`
- `scripts/run_panel_regressions.py`
- `scripts/run_evening_spike_visual_report.py`

If you only remember four files, remember those.

## Environment And Dependencies

The project uses Python and `uv`.

Minimal setup:

```bash
uv sync
```

Runtime dependencies from `pyproject.toml`:

- `matplotlib`
- `numpy`
- `pandas`
- `pyarrow`
- `statsmodels`

The declared Python requirement is `>=3.9`.

There is no formal test suite in the repo right now. Validation is mostly done through QC manifests plus spot checks in downstream outputs.

## Repo Structure

Top-level layout:

```text
.
├── data/
│   ├── mp/
│   ├── mp_reserve/
│   ├── panels/
│   ├── rtd/
│   ├── rtdcv/
│   ├── rtdhs/
│   └── rtdreg/
├── regressions/
├── scripts/
├── pyproject.toml
├── uv.lock
├── LICENSE
└── AGENT.MD
```

What each major path means:

- `data/mp/`: RTD market clearing price downloads, QC, and combined parquet.
- `data/mp_reserve/`: RTD reserve market clearing price downloads, QC, and combined parquet.
- `data/rtd/`: hourly RTD zip downloads, extracted hourly CSVs, QC, and a region-level `LMP_SMP` parquet.
- `data/rtdcv/`: RTD congestion records plus the equipment-to-island mapping CSV used in panel construction.
- `data/rtdhs/`: RTD HVDC schedules and congestion flags.
- `data/rtdreg/`: RTD regional summaries used for demand and control variables.
- `data/panels/`: the two main derived panel datasets used in regressions.
- `regressions/`: all current analytical outputs, including coefficient exports, HTML reports, and PNG figures.
- `scripts/`: everything operational. There is no package module layer; the scripts are the pipeline.
- `AGENT.MD`: an older repo summary. Useful for historical context, but it is no longer a full description of the active repo state.

## Pipeline Overview

### 1. Raw download and file-level QC

`scripts/download_market_data.py` is the top-level orchestrator. It pulls:

- `RTDREG`
- `MP`
- `MP_RESERVE`
- `RTD`
- `RTDCV`
- `RTDHS`

The script handles two kinds of sources:

- Daily CSV datasets through the generic machinery in `scripts/rtd_download_core.py`.
- Hourly zipped RTD files through custom logic in `scripts/download_market_data.py`.

For daily datasets, the generic flow is:

1. Build IEMOP URLs from a date range.
2. Download raw CSVs to `data/<dataset>/raw/`.
3. Validate headers, EOF markers, timestamps, numeric columns, and interval expectations.
4. Write a QC manifest CSV to `data/<dataset>/qc/`.
5. Concatenate non-error files into a combined parquet in `data/<dataset>/combined/`.

For hourly RTD:

1. Download hourly ZIP files to `data/rtd/raw/`.
2. Extract one CSV per ZIP into the same folder.
3. Validate each hourly CSV and aggregate to unique `TIME_INTERVAL x REGION_NAME` `LMP_SMP`.
4. Write the RTD QC manifest to `data/rtd/qc/`.
5. Write the combined region-level parquet to `data/rtd/combined/`.

### 2. Panel construction

`scripts/build_rtd_panels.py` reads the latest combined files and creates:

- `RTD_DIRECT_PAIR_PANEL_*.parquet`
- `RTD_ISLAND_SYSTEM_PANEL_*.parquet`

Inputs:

- `data/rtd/combined/RTD_LMP_SMP_*.parquet`
- `data/rtdreg/combined/RTDREG_*.parquet`
- `data/rtdcv/combined/RTDCV_*.parquet`
- `data/rtdcv/rtd_congestion_resources_with_island_group.csv`
- `data/rtdhs/combined/RTDHS_*.parquet`

### 3. Regression and reporting layer

Most analysis scripts pull the latest matching parquet or CSV automatically and write to `regressions/`.

The pattern to know is `latest_matching_file(...)`: many scripts pick the artifact with the most recent filename token, not necessarily the artifact from the most recent successful end-to-end run. That is convenient, but it also means mixed vintage windows can coexist.

## Data Sources And Current Snapshot

This section describes the actual current artifacts in the repo as of `2026-04-08`.

### Important cross-cutting timing note

There are two different notions of time throughout the repo:

- `RUN_TIME`: when the market run occurred.
- `TIME_INTERVAL`: the market interval the row refers to.

For most daily datasets, `TIME_INTERVAL` is in 5-minute increments.

For some datasets, the latest `TIME_INTERVAL` in a daily combined file extends into the next calendar day. For example:

- `RTDREG` and `RTDHS` files through `2026-03-24` have `TIME_INTERVAL` values up to `2026-03-25 00:00:00`.
- `MP_RESERVE` files through `2026-03-31` have `TIME_INTERVAL` values up to `2026-04-01 00:00:00`.

That is not a bug in the combined parquet naming. It reflects how the source files encode intervals.

### `MP` (`data/mp/`)

Purpose:

- Resource-level RTD market clearing price data.

Raw inventory currently present:

- `99` raw CSVs.
- Raw filename range: `MP_20251216.csv` through `MP_20260324.csv`.

Latest combined parquet:

- `data/mp/combined/MP_20251224_20260324.parquet`

Current contents:

- Rows: `70,300`
- `RUN_TIME` range: `2025-12-24 00:00:00` to `2026-03-24 23:45:00`
- `TIME_INTERVAL` range: `2025-12-24 00:05:00` to `2026-03-24 23:50:00`
- Unique `TIME_INTERVAL`s: `25,062`
- Regions: `CLUZ`, `CVIS`, `CMIN`
- Unique resources: `258`
- Commodity types: `En`
- `MARGINAL_PRICE` range: `-10000.024` to `32000.0096`

QC state of latest manifest:

- Manifest: `data/mp/qc/mp_qc_20251224_20260324.csv`
- Files in manifest: `91`
- Status counts: `91 warning`, `0 ok`, `0 error`

What those warnings mean:

- Every day in the latest MP window is flagged.
- The dominant warnings are `unexpected_time_interval_count:<n>` and `missing_region_name_values:<n>`.
- This means the MP data is materially incomplete at the interval/region level relative to the repo's expected 288-interval daily template.

Practical implication:

- Treat MP as usable but noisy/incomplete.
- Do not assume MP has full 5-minute coverage even when a raw file exists.
- Do not assume MP is a clean substitute for the RTD `LMP_SMP` regional series.

### `MP_RESERVE` (`data/mp_reserve/`)

Purpose:

- Resource-level RTD reserve market clearing prices.

Raw inventory currently present:

- `90` raw CSVs.
- Raw filename range: `MP_RESERVE_20260101.csv` through `MP_RESERVE_20260331.csv`.

Latest combined parquet:

- `data/mp_reserve/combined/MP_RESERVE_20260101_20260331.parquet`

Current contents:

- Rows: `310,776`
- `RUN_TIME` range: `2026-01-01 00:00:00` to `2026-03-31 23:55:00`
- `TIME_INTERVAL` range: `2026-01-01 00:05:00` to `2026-04-01 00:00:00`
- Unique `TIME_INTERVAL`s: `25,814`
- Regions: `CLUZ`, `CVIS`, `CMIN`
- Unique resources: `185`
- Commodity types: `Dr`, `Fr`, `Rd`, `Ru`
- `MARGINAL_PRICE` range: `0.0` to `25000.0`

Reserve shorthand used in this repo:

- `Dr`: delayed contingency raise
- `Fr`: fast contingency raise
- `Rd`: regulation down
- `Ru`: regulation up

QC state of latest manifest:

- Manifest: `data/mp_reserve/qc/mp_reserve_qc_20260101_20260401.csv`
- Files in manifest: `91`
- Status counts: `85 ok`, `5 warning`, `1 error`

Known issue in the latest reserve run:

- `2026-04-01` is an `HTTP 404: Not Found` in the manifest.
- The latest combined parquet still ends at the `20260331` file window.
- The latest QC manifest and latest combined parquet are therefore not the same "run window".

Known interval anomalies in reserve data:

- Warning dates: `2026-01-12`, `2026-02-13`, `2026-02-15`, `2026-02-23`, `2026-03-15`
- Most are single 10-minute gaps or one larger gap.
- The largest observed reserve `TIME_INTERVAL` hole in the latest combined parquet is `2026-02-15 00:00:00` to `2026-02-15 07:05:00`.

### `RTD` region-level `LMP_SMP` (`data/rtd/`)

Purpose:

- Hourly RTD source files aggregated to one regional `LMP_SMP` per `TIME_INTERVAL x REGION_NAME`.
- This is the key regional price input for panel construction.

Raw inventory currently present:

- `4,368` files in `data/rtd/raw/`
- That count includes both ZIP and extracted CSV files.
- Operationally, this corresponds to `2,184` hourly source hours from `2025-12-24 00:00` through `2026-03-24 23:00`.

Latest combined parquet:

- `data/rtd/combined/RTD_LMP_SMP_202512240000_202603242300.parquet`

Current contents:

- Rows: `78,558`
- `TIME_INTERVAL` range: `2025-12-23 23:05:00` to `2026-03-24 23:00:00`
- Unique `TIME_INTERVAL`s: `26,186`
- Regions: `CLUZ`, `CVIS`, `CMIN`
- `LMP_SMP` range: `-11012.1172` to `127567.0017`

QC state of latest manifest:

- Manifest: `data/rtd/qc/rtd_lmp_smp_qc_202512240000_202603242300.csv`
- Files in manifest: `2,184`
- Status counts: `2,183 ok`, `1 warning`, `0 error`

Known issue in the latest RTD run:

- One hourly file is flagged `empty_data_file`: `202602232200`

Observed cadence anomalies in RTD `TIME_INTERVAL`:

- `2026-01-12 14:35:00` -> `2026-01-12 14:45:00`
- `2026-02-13 02:10:00` -> `2026-02-13 02:20:00`
- `2026-02-15 08:00:00` -> `2026-02-15 08:10:00`
- `2026-02-23 21:00:00` -> `2026-02-23 22:35:00`
- `2026-03-15 07:00:00` -> `2026-03-15 07:10:00`

Those same anomalies also propagate into `RTDHS`, `RTDREG`, and the derived panel files.

### `RTDCV` (`data/rtdcv/`)

Purpose:

- RTD congestion events at the equipment level.

Raw inventory currently present:

- `99` raw CSVs.
- Raw filename range: `RTDCV_20251216.csv` through `RTDCV_20260324.csv`.

Latest combined parquet:

- `data/rtdcv/combined/RTDCV_20251224_20260324.parquet`

Current contents:

- Rows: `3,978`
- `RUN_TIME` range: `2025-12-24 14:10:00` to `2026-03-24 18:40:00`
- `TIME_INTERVAL` range: `2025-12-24 14:15:00` to `2026-03-24 18:45:00`
- Unique `TIME_INTERVAL`s: `3,481`
- `CONGEST_TYPE`s: `BASE CASE`, `CONTINGENCY`
- `RUN_TYPE`: `SCHED_RUN`
- Unique `EQUIPMENT_NAME`s: `22`

QC state of latest manifest:

- Manifest: `data/rtdcv/qc/rtdcv_qc_20251224_20260324.csv`
- Files in manifest: `91`
- Status counts: `61 ok`, `30 warning`, `0 error`

Important interpretation of RTDCV warnings:

- Every RTDCV warning in the latest manifest is `empty_file`.
- In practice, that often means "no congestion records listed for the day", not necessarily a download failure.
- Downstream code treats RTDCV as a sparse event source and can handle empty days.

Important supporting file:

- `data/rtdcv/rtd_congestion_resources_with_island_group.csv`

This mapping file is required to turn `EQUIPMENT_NAME` values into island-level congestion indicators in `build_rtd_panels.py`. If it is missing, panel construction will fail when RTDCV has non-empty rows.

### `RTDHS` (`data/rtdhs/`)

Purpose:

- RTD HVDC schedules and link congestion flags.

Raw inventory currently present:

- `91` raw CSVs.
- Raw filename range: `RTDHS_20251224.csv` through `RTDHS_20260324.csv`.

Latest combined parquet:

- `data/rtdhs/combined/RTDHS_20251224_20260324.parquet`

Current contents:

- Rows: `52,372`
- `RUN_TIME` range: `2025-12-24 00:00:00` to `2026-03-24 23:55:00`
- `TIME_INTERVAL` range: `2025-12-24 00:05:00` to `2026-03-25 00:00:00`
- Unique `TIME_INTERVAL`s: `26,186`
- `HVDC_NAME`s: `MINVIS1`, `VISLUZ1`

QC state of latest manifest:

- Manifest: `data/rtdhs/qc/rtdhs_qc_20251224_20260324.csv`
- Files in manifest: `91`
- Status counts: `86 ok`, `5 warning`, `0 error`

The warning dates match the RTD/RTDREG interval anomalies:

- `2026-01-12`
- `2026-02-13`
- `2026-02-15`
- `2026-02-23`
- `2026-03-15`

### `RTDREG` (`data/rtdreg/`)

Purpose:

- Regional summaries used for demand and control variables.

Raw inventory currently present:

- `91` raw CSVs.
- Raw filename range: `RTDREG_20251224.csv` through `RTDREG_20260324.csv`.

Latest combined parquet:

- `data/rtdreg/combined/RTDREG_20251224_20260324.parquet`

Current contents:

- Rows: `392,790`
- `RUN_TIME` range: `2025-12-24 00:00:00` to `2026-03-24 23:55:00`
- `TIME_INTERVAL` range: `2025-12-24 00:05:00` to `2026-03-25 00:00:00`
- Unique `TIME_INTERVAL`s: `26,186`
- Regions: `CLUZ`, `CVIS`, `CMIN`
- Commodity types: `En`, `Dr`, `Fr`, `Rd`, `Ru`

QC state of latest manifest:

- Manifest: `data/rtdreg/qc/rtdreg_qc_20251224_20260324.csv`
- Files in manifest: `91`
- Status counts: `86 ok`, `5 warning`, `0 error`

The same five interval-anomaly dates appear here as in `RTDHS`.

## Current Combined Schemas

### Daily combined parquet layout

The daily datasets created through `rtd_download_core.py` use the source schema minus the trailing blank column in IEMOP CSVs.

`MP` and `MP_RESERVE` columns:

- `RUN_TIME`
- `MKT_TYPE`
- `TIME_INTERVAL`
- `REGION_NAME`
- `RESOURCE_NAME`
- `RESOURCE_TYPE`
- `COMMODITY_TYPE`
- `MARGINAL_PRICE`

`RTDCV` columns:

- `RUN_TIME`
- `MKT_TYPE`
- `TIME_INTERVAL`
- `CONGEST_TYPE`
- `RUN_TYPE`
- `EQUIPMENT_NAME`
- `STATION_NAME`
- `VOLTAGE_LEVEL`
- `BINDING_LIMIT`
- `MW_FLOW`
- `OVERLOAD_MW`
- `PCT_MW`

`RTDHS` columns:

- `RUN_TIME`
- `MKT_TYPE`
- `TIME_INTERVAL`
- `HVDC_NAME`
- `CONGESTION_FLAG`
- `FLOW_FROM`
- `FLOW_TO`
- `OVERLOAD_MW`

`RTDREG` columns:

- `RUN_TIME`
- `MKT_TYPE`
- `TIME_INTERVAL`
- `REGION_NAME`
- `COMMODITY_TYPE`
- `MKT_REQT`
- `LOAD_BID`
- `LOAD_CURTAILED`
- `LOSSES`
- `GENERATION`
- `MKT_IMPORT`
- `MKT_EXPORT`

### RTD combined parquet layout

The RTD combined parquet is intentionally much narrower:

- `TIME_INTERVAL`
- `REGION_NAME`
- `LMP_SMP`

That file is already aggregated to unique region-level prices.

## Derived Panel Datasets

### `RTD_DIRECT_PAIR_PANEL`

Latest file:

- `data/panels/RTD_DIRECT_PAIR_PANEL_202512240005_202603242300.parquet`

Current contents:

- Rows: `52,348`
- Unique intervals: `26,174`
- Two pairs per interval:
  - `CLUZ_CVIS` with link `VISLUZ1`
  - `CVIS_CMIN` with link `MINVIS1`

Key columns:

- `time_interval`
- `pair_key`
- `island_1`
- `island_2`
- `price_1`
- `price_2`
- `dep_abs_price_gap`
- `equip_cong_any_1`
- `equip_overload_any_1`
- `equip_cong_any_2`
- `equip_overload_any_2`
- `link_name`
- `link_congested_any`
- `losses_1`, `losses_2`, `losses_total`
- `generation_1`, `generation_2`, `generation_total`
- `mkt_import_1`, `mkt_import_2`, `mkt_import_total`
- `mkt_export_1`, `mkt_export_2`, `mkt_export_total`
- `fe_day`

Definitions:

- `dep_abs_price_gap` = `abs(price_1 - price_2)`
- `link_congested_any` is pulled from the pair-specific HVDC link.
- `equip_cong_any_*` and `equip_overload_any_*` come from RTDCV after mapping equipment names to island groups.
- Controls come from `RTDREG` `COMMODITY_TYPE == 'En'`.

### `RTD_ISLAND_SYSTEM_PANEL`

Latest file:

- `data/panels/RTD_ISLAND_SYSTEM_PANEL_202512240005_202603242300.parquet`

Current contents:

- Rows: `78,522`
- Unique intervals: `26,174`
- Three islands per interval: `CLUZ`, `CVIS`, `CMIN`

Key columns:

- `time_interval`
- `island_code`
- `price_island`
- `price_sys_dw`
- `dep_price_minus_sys`
- `interlink_congested_any`
- `equip_cong_any`
- `equip_overload_any`
- `losses_island`, `losses_total`
- `generation_island`, `generation_total`
- `mkt_import_island`, `mkt_import_total`
- `mkt_export_island`, `mkt_export_total`
- `fe_day`

Definitions:

- `price_sys_dw` is the demand-weighted average island price at each interval.
- `dep_price_minus_sys` = `abs(price_island - price_sys_dw)`
- `interlink_congested_any` is whether any link incident to the focal island is congested.
  - Luzon uses `VISLUZ1`
  - Mindanao uses `MINVIS1`
  - Visayas uses `max(VISLUZ1, MINVIS1)`

### Panel construction assumptions you should know

- `build_rtd_panels.py` requires full three-island coverage in the RTD regional price source. If RTD lacks one region at an interval, that interval is rejected.
- Only `RTDREG` rows with `COMMODITY_TYPE == 'En'` are used for demand and controls.
- Island-level congestion is binary and built via `max()` within `TIME_INTERVAL x island`.
- HVDC congestion is binary and built via `CONGESTION_FLAG == 'Y'`.
- Fixed effects are represented as simple day strings in `fe_day`.

## Main Analytical Outputs

The `regressions/` directory currently contains the main presentation artifacts. The filenames are the de facto index of the analysis work.

### Core panel regression outputs

- `panel_regression_tables.html`
  - Produced by `scripts/run_panel_regressions.py`
  - Main retained PPML regression tables for direct-pair and island-vs-system panels
- `panel_regression_coefficients.csv`
  - Produced by `scripts/run_panel_regressions.py`
  - Tidy coefficient export used by some diagnostics
- `large_effect_diagnostics.html`
  - Produced by `scripts/run_large_effect_diagnostics.py`
  - Follow-up diagnostics for very large reported PPML effects
- `large_effect_diagnostics.csv`
  - Companion CSV for the large-effect diagnostics

### Direct-pair descriptive and OLS outputs

- `direct_pair_price_difference_histograms.png`
  - Produced by `scripts/run_pair_price_difference_histograms.py`
  - Signed price-difference histograms by congestion regime
- `direct_pair_ols_clean_base_visual_full_sample.png`
- `direct_pair_ols_clean_base_visual_winsor_99.png`
- `direct_pair_ols_clean_base_coefficients.csv`
  - Produced by `scripts/run_pair_gap_ols_clean_base_visual.py`
  - Pair-specific OLS scaled to a "clean base" price level
- `direct_pair_ols_seasonality_fe_visual.png`
- `direct_pair_ols_seasonality_fe_coefficients.csv`
  - Produced by `scripts/run_pair_gap_ols_seasonality_fe_visual.py`
  - OLS variant with month, ISO-week, and day-of-week FE
- `direct_pair_sign_flip_diagnostics.html`
- `direct_pair_sign_flip_diagnostics.md`
  - Produced by `scripts/run_direct_pair_sign_flip_diagnostics.py`
  - Investigates sign flips between level and elasticity-style specifications

### Luzon-Visayas focused outputs

- `luzon_visayas_ols_progressive.html`
- `luzon_visayas_ols_progressive_coefficients.csv`
  - Produced by `scripts/run_luzon_visayas_ols_progressive.py`
  - Progressive OLS on the Luzon-Visayas absolute gap with link congestion and demand
- `luzon_visayas_price_pooled_ols.html`
- `luzon_visayas_price_pooled_ols_coefficients.csv`
  - Produced by `scripts/run_luzon_visayas_price_pooled_ols.py`
  - Separate-panel OLS for Luzon and Visayas prices under pooled/split specifications
- `luzon_visayas_price_split_ols.html`
- `luzon_visayas_price_split_ols_coefficients.csv`
  - Produced by `scripts/run_luzon_visayas_price_split_ols.py`
  - Link-congested vs link-uncongested split regressions
- `luzon_visayas_targeted_congestion_ols.html`
- `luzon_visayas_targeted_congestion_ols_coefficients.csv`
  - Produced by `scripts/run_luzon_visayas_targeted_congestion_ols.py`
  - Targeted congestion regressions for the Luzon-Visayas pair

### Evening spike and reserve visuals

- `evening_spike_visual_report.html`
- `evening_spike_visual_15min.csv`
  - Produced by `scripts/run_evening_spike_visual_report.py`
  - Main descriptive HTML report plus the merged 15-minute analysis frame
- `evening_spike_evening_ecdf_by_congestion.png`
- `evening_spike_intraday_profile_by_congestion.png`
- `evening_spike_congestion_summary.csv`
  - Produced by `scripts/run_evening_spike_congestion_visuals.py`
  - Congested vs uncongested evening spike comparisons
- `mp_fast_raise_15min_quantiles.png`
- `mp_fast_raise_15min_quantiles.csv`
  - Produced by `scripts/run_mp_fast_raise_15min_quantiles_png.py`
  - 15-minute quantile profiles for market price and fast contingency raise reserve price
- `mp_fast_raise_hourly_quantiles.html`
- `mp_fast_raise_hourly_quantiles.csv`
  - Existing hourly-style output in `regressions/`; there is no matching script in the current tracked repo snapshot

### Island congestion share visuals

- `island_equipment_congestion_intraday_share.png`
- `island_equipment_overload_intraday_share.png`
  - Produced by `scripts/run_island_equipment_congestion_intraday_png.py`
  - 15-minute intraday shares of congestion and overload by island

Important git-state note:

- `scripts/run_island_equipment_congestion_intraday_png.py` exists in the current working tree but was untracked in git when this README was written.
- If you commit this README, either also commit that script or remove/update the references above.

## Definitions And Modeling Assumptions

### Geographic codes

The repo consistently uses:

- `CLUZ` = Luzon
- `CVIS` = Visayas
- `CMIN` = Mindanao

### Link structure

Only two direct inter-island links are modeled:

- `VISLUZ1` for Luzon-Visayas
- `MINVIS1` for Visayas-Mindanao

There is no direct Luzon-Mindanao link. When that comparison appears in descriptive work, it is a non-direct pair.

### Event indicators

Congestion indicators are binary:

- `link_congested_any` or `interlink_congested_any`
- `equip_cong_any`
- `equip_overload_any`

The panel layer generally uses `max()` across rows within an interval, so these are "did anything happen in this interval?" flags, not counts or intensities.

### Demand and control variables

Panel controls come from `RTDREG` `COMMODITY_TYPE == 'En'` only.

The main controls are:

- `LOSSES`
- `GENERATION`
- `MKT_IMPORT`
- `MKT_EXPORT`

For some regressions, reported effects for continuous controls are scaled to `+100 MW`.

### Evening spike construction

The evening-spike report uses a derived 15-minute frame built from MP, reserve, HVDC, and RTDREG data.

Important definitions in that layer:

- Binning is fixed at `15` minutes for the downstream PNG scripts.
- `evening_window_flag` is `16:00 <= clock < 19:00`.
- For each island, `baseline_<region>` is the same-day median price outside that evening window.
- `spike_<region>` = `price_<region> - baseline_<region>`.

This is a same-day relative spike measure, not a model residual and not a deviation from a global average.

### PPML vs OLS

The main retained regression script is `scripts/run_panel_regressions.py`, which uses PPML.

Other scripts use OLS for diagnostics, intuition-building, or presentation variants.

Do not mix interpretations casually:

- PPML outputs in the main tables are transformed and reported as percent effects.
- Several OLS scripts report level effects or percent-of-clean-base effects.
- `large_effect_diagnostics.py` exists because some transformed PPML effects can become very large and need support/tail checks.

### Scarcity thresholds used in diagnostics

Some diagnostics define scarcity-like extreme prices using:

- upper cutoff: `30000`
- lower cutoff: `-9000`

Those thresholds are not used to clean the core datasets. They only appear in diagnostic scripts.

## Known Data Completeness And Consistency Issues

These are the most important caveats for a successor.

### 1. Raw inventories and latest combined outputs are not perfectly aligned

Examples:

- `data/mp/raw/` starts at `2025-12-16`, but the latest combined MP parquet starts at `2025-12-24`.
- `data/mp_reserve/qc/` includes a manifest through `2026-04-01`, but the latest combined reserve parquet ends at the `20260331` run window.

In other words, the folder contents reflect multiple runs, not one perfectly synchronized snapshot.

### 2. MP is much less complete than the other daily datasets

The latest MP QC manifest has warnings on every single day in the combined window. If you are doing anything interval-sensitive, assume MP requires scrutiny.

### 3. `MP` and RTD regional `LMP_SMP` do not currently match well

The cross-check written by `download_market_data.py` currently reports:

- overlap groups: `36,090`
- exact match groups: `2,127`
- mismatch groups: `33,963`
- max absolute difference: `42,186.6828`
- MP-only groups: `12`
- RTD-only groups: `42,468`
- MP non-unique groups: `1,069`

Bottom line:

- Do not treat `MP` and RTD `LMP_SMP` as interchangeable.
- The repo already contains a verification artifact proving they diverge materially in the current snapshot.

### 4. A small set of interval gaps propagate through several datasets

The following anomalies appear in `RTD`, `RTDHS`, `RTDREG`, and then in both panel datasets:

- `2026-01-12 14:35:00` -> `2026-01-12 14:45:00`
- `2026-02-13 02:10:00` -> `2026-02-13 02:20:00`
- `2026-02-15 08:00:00` -> `2026-02-15 08:10:00`
- `2026-02-23 21:00:00` -> `2026-02-23 22:35:00`
- `2026-03-15 07:00:00` -> `2026-03-15 07:10:00`

Those are known holes in the current data snapshot and should not surprise you if counts are slightly short of the naive full-grid expectation.

### 5. Empty RTDCV files are not automatically a problem

Thirty RTDCV days are flagged `empty_file` in the current manifest. That usually means no congestion rows for that day, which downstream logic can handle.

### 6. Artifact discovery is filename-based

Many scripts do this:

```python
latest_matching_file(Path("data/..."), "PREFIX_*.parquet")
```

That is convenient, but it means:

- mixed vintages can silently combine
- the "latest" file may come from a partial rerun
- the file with the newest token is not necessarily the one you want analytically

If you need a clean reproducible batch, pass explicit paths instead of relying on auto-discovery.

## Practical Commands

### Refresh the retained source datasets

```bash
uv run python scripts/download_market_data.py --end-date YYYY-MM-DD --lookback-months 3
```

Useful flags:

- `--datasets ...` to limit the scope
- `--force` to re-download even if local raw files exist
- `--allow-partial` to write combined outputs even with QC failures
- `--include-errors-in-combined` to force error files into combined parquet creation

### Rebuild both panel datasets

```bash
uv run python scripts/build_rtd_panels.py
```

### Rebuild the main retained regression tables

```bash
uv run python scripts/run_panel_regressions.py
```

### Rebuild the main descriptive evening-spike report

```bash
uv run python scripts/run_evening_spike_visual_report.py
```

### Rebuild the main secondary outputs

```bash
uv run python scripts/run_large_effect_diagnostics.py
uv run python scripts/run_direct_pair_sign_flip_diagnostics.py
uv run python scripts/run_pair_price_difference_histograms.py
uv run python scripts/run_pair_gap_ols_clean_base_visual.py
uv run python scripts/run_pair_gap_ols_seasonality_fe_visual.py
uv run python scripts/run_luzon_visayas_ols_progressive.py
uv run python scripts/run_luzon_visayas_price_pooled_ols.py
uv run python scripts/run_luzon_visayas_price_split_ols.py
uv run python scripts/run_luzon_visayas_targeted_congestion_ols.py
uv run python scripts/run_evening_spike_congestion_visuals.py
uv run python scripts/run_mp_fast_raise_15min_quantiles_png.py
uv run python scripts/run_island_equipment_congestion_intraday_png.py
```

## Recommended Successor Workflow

If you are picking this project up fresh, this is the safest order to work in:

1. Read the latest QC manifests before trusting any combined parquet.
2. Decide whether you want to keep using the current rolling window convention or freeze explicit file paths.
3. Re-run `scripts/download_market_data.py` with an explicit `--end-date`.
4. Inspect:
   - `data/mp/qc/`
   - `data/mp_reserve/qc/`
   - `data/rtd/qc/`
   - `data/rtdcv/qc/`
   - `data/rtdhs/qc/`
   - `data/rtdreg/qc/`
5. Rebuild panels with `scripts/build_rtd_panels.py`.
6. Re-run core outputs:
   - `scripts/run_panel_regressions.py`
   - `scripts/run_evening_spike_visual_report.py`
7. Only then re-run the secondary diagnostics and PNG generators.

## Suggested Cleanup And Maintenance Priorities

If this project is going to continue, these are the highest-value improvements:

1. Add a small reproducibility manifest so a full run records the exact input files used by each downstream script.
2. Normalize how "latest" files are selected, or stop auto-selecting them for serious work.
3. Add a lightweight test or assertion suite around panel row counts and key uniqueness constraints.
4. Decide whether the MP-vs-RTD mismatch is expected by design or indicates a misunderstanding of the source products.
5. Decide whether the untracked `run_island_equipment_congestion_intraday_png.py` should become a tracked script.

## Final Notes

- The repo is functional, but it is best thought of as a research workspace with production-like data plumbing, not a fully productized package.
- The most trustworthy datasets in the current snapshot are the RTD-derived regional series and the two panel files built from them.
- The least trustworthy dataset in the current snapshot is `MP`, at least if you need dense interval coverage or one-to-one comparison with RTD regional prices.
- The most important hidden dependency is the RTDCV equipment-to-island mapping CSV.
- The most important hidden behavioral assumption is filename-based artifact auto-discovery.

If you are continuing the analysis, start from the panel layer and work backward only when QC tells you to.
