# RTD Panel QC Report

This report documents how the two RTD panels were built in [scripts/build_rtd_panels.py](../../scripts/build_rtd_panels.py) and records the QC checks run against the generated outputs.

## Source Files

| source           | path                                                                 | rows_used | intervals | time_min            | time_max            | notes                                                                                                             |
|------------------|----------------------------------------------------------------------|-----------|-----------|---------------------|---------------------|-------------------------------------------------------------------------------------------------------------------|
| RTD_ISLAND_PRICE | data/csv_exports_flat/RTD_ISLAND_PRICE_202512220000_202603230000.csv | 78,594    | 26,198    | 2025-12-21 23:05:00 | 2026-03-23 00:00:00 | Dense island-level price source with one row per interval-region and columns ISLAND_PRICE, WEIGHT_SUM.            |
| RTDREG (En only) | data/csv_exports_flat/RTDREG_20251218_20260320.csv                   | 80,286    | 26,762    | 2025-12-18 00:05:00 | 2026-03-21 00:00:00 | Filtered to COMMODITY_TYPE == En; MKT_REQT is used as island demand.                                              |
| RTDCV            | data/csv_exports_flat/RTDCV_20251218_20260318.csv                    | 4,082     | 3,572     | 2025-12-18 00:05:00 | 2026-03-18 13:15:00 | Sparse event-style congestion file; missing interval-island cells are interpreted as zero after aggregation.      |
| Congestion map   | data/rtd_congestion/rtd_congestion_resources_with_island_group.csv   | 19        |           |                     |                     | Maps RTDCV EQUIPMENT_NAME values into Luzon, Visayas, or Mindanao.                                                |
| RTDHS            | data/csv_exports_flat/RTDHS_20251218_20260318.csv                    | 52,372    | 26,186    | 2025-12-18 00:05:00 | 2026-03-19 00:00:00 | Dense direct-link schedule file with VISLUZ1 and MINVIS1 congestion flags. Missing rows are treated as true gaps. |

## Construction Logic

### Shared Definitions

- `time_interval` is the five-minute interval key shared across all sources.
- Region codes are kept as `CLUZ`, `CVIS`, and `CMIN`.
- Equipment congestion from `RTDCV` is mapped to islands using `rtd_congestion_resources_with_island_group.csv`.
- Fixed effects are derived directly from `time_interval`:
  - `fe_month = YYYY-MM`
  - `fe_week = ISO YYYY-Www`
  - `fe_day = YYYY-MM-DD`

### Island-System Panel Calculations

- Keep only intervals where all three island prices from `RTD_ISLAND_PRICE` and all three `RTDREG` `En` demand rows are present.
- For each interval:
  - `demand_total = demand_CLUZ + demand_CVIS + demand_CMIN`
  - `price_sys_dw = (price_CLUZ * demand_CLUZ + price_CVIS * demand_CVIS + price_CMIN * demand_CMIN) / demand_total`
- For each island-specific row:
  - `price_island = ISLAND_PRICE`
  - `dep_price_minus_sys = abs(price_island - price_sys_dw)`
  - `demand_island = MKT_REQT`
  - `weight_sum_island = WEIGHT_SUM`
  - `equip_cong_any = 1` if any mapped `RTDCV` row exists for that interval-island; otherwise `0`
  - `equip_excess_pct_sum = sum(PCT_MW - 100)` across mapped `RTDCV` rows for that interval-island; otherwise `0`

### Direct-Pair Panel Calculations

- Only direct inter-island pairs are included:
  - `CLUZ_CVIS` mapped to `VISLUZ1`
  - `CVIS_CMIN` mapped to `MINVIS1`
- Keep only intervals where island prices, `RTDREG` `En` demand rows, and the relevant `RTDHS` link row all exist.
- For each pair row:
  - `price_1`, `price_2` come directly from `RTD_ISLAND_PRICE`
  - `dep_abs_price_gap = abs(price_1 - price_2)`
  - `demand_1`, `demand_2` come from `RTDREG` `MKT_REQT`
  - `demand_total = demand_CLUZ + demand_CVIS + demand_CMIN`
  - `weight_sum_1`, `weight_sum_2` come from `WEIGHT_SUM`
  - `link_congested_any = 1` if the relevant `RTDHS` row has `CONGESTION_FLAG == 'Y'`; otherwise `0`
  - `equip_cong_any_1`, `equip_cong_any_2` and `equip_excess_pct_sum_1`, `equip_excess_pct_sum_2` come from the mapped `RTDCV` island aggregates for each side of the pair

## Coverage Notes

| window                         | intervals | time_min            | time_max            | notes                                                              |
|--------------------------------|-----------|---------------------|---------------------|--------------------------------------------------------------------|
| Island price source            | 26,198    | 2025-12-21 23:05:00 | 2026-03-23 00:00:00 | All three island prices present at every interval in file.         |
| Price + RTDREG (En) overlap    | 25,622    | 2025-12-21 23:05:00 | 2026-03-21 00:00:00 | Defines island-system panel coverage.                              |
| Price + RTDREG + RTDHS overlap | 25,046    | 2025-12-21 23:05:00 | 2026-03-19 00:00:00 | Defines direct-pair panel coverage because link rows are required. |
| direct_pair_panel output       | 25,046    | 2025-12-21 23:05:00 | 2026-03-19 00:00:00 | Two rows per usable interval: CLUZ_CVIS and CVIS_CMIN.             |
| island_system_panel output     | 25,622    | 2025-12-21 23:05:00 | 2026-03-21 00:00:00 | Three rows per usable interval: CLUZ, CVIS, CMIN.                  |

The price file is the dense driver here: it has all three islands at every interval in-file. The remaining coverage differences come from source boundaries:

- The pair panel stops when `RTDHS` stops providing dense link rows.
- The island-system panel continues longer because it does not require `RTDHS`.
- `RTDCV` is sparse by design, so absence is converted to `0` after interval-island aggregation rather than dropping rows.

## direct_pair_panel Summary

Output file: `data/panels/RTD_DIRECT_PAIR_PANEL_202512212305_202603190000.parquet`

| metric            | value                |
|-------------------|----------------------|
| rows              | 50,092               |
| unique intervals  | 25,046               |
| time min          | 2025-12-21 23:05:00  |
| time max          | 2026-03-19 00:00:00  |
| duplicate keys    | 0                    |
| null cells        | 0                    |
| key distribution  | CLUZ_CVIS, CVIS_CMIN |
| link distribution | MINVIS1, VISLUZ1     |

## direct_pair_panel QC Checks

| check                                                       | status | details                                                       |
|-------------------------------------------------------------|--------|---------------------------------------------------------------|
| Row count matches dense overlap and direct-link logic       | pass   | rows=50,092, expected=50,092                                  |
| Unique key (time_interval, pair_key)                        | pass   | duplicate_keys=0                                              |
| Pair set and link mapping                                   | pass   | mapping_errors=0, intervals=25,046, expected_intervals=25,046 |
| dep_abs_price_gap = abs(price_1 - price_2)                  | pass   | mismatches=0, max_abs_diff=0.000000                           |
| demand_total matches raw RTDREG En sum                      | pass   | mismatches=0, max_abs_diff=0.000000                           |
| WEIGHT_SUM audit fields match price source                  | pass   | weight_1_mismatches=0, weight_2_mismatches=0                  |
| Equipment congestion flags and sums match RTDCV aggregation | pass   | flag_mismatches=(0, 0), sum_mismatches=(0, 0)                 |
| HVDC congestion flag matches RTDHS aggregation              | pass   | mismatches=0                                                  |
| Fixed-effects columns match time_interval                   | pass   | fe_month=0, fe_week=0, fe_day=0                               |
| No null cells in output                                     | pass   | null_cells=0                                                  |

## island_system_panel Summary

Output file: `data/panels/RTD_ISLAND_SYSTEM_PANEL_202512212305_202603210000.parquet`

| metric           | value               |
|------------------|---------------------|
| rows             | 76,866              |
| unique intervals | 25,622              |
| time min         | 2025-12-21 23:05:00 |
| time max         | 2026-03-21 00:00:00 |
| duplicate keys   | 0                   |
| null cells       | 0                   |
| key distribution | CLUZ, CMIN, CVIS    |

## island_system_panel QC Checks

| check                                                       | status | details                                                                   |
|-------------------------------------------------------------|--------|---------------------------------------------------------------------------|
| Row count matches dense price-demand overlap                | pass   | rows=76,866, expected=76,866, intervals=25,622, expected_intervals=25,622 |
| Unique key (time_interval, island_code)                     | pass   | duplicate_keys=0                                                          |
| price_sys_dw matches raw demand-weighted system price       | pass   | mismatches=0, max_abs_diff=0.000000                                       |
| dep_price_minus_sys = abs(price_island - price_sys_dw)      | pass   | mismatches=0, max_abs_diff=0.000000                                       |
| Island price, demand, and weight fields match raw sources   | pass   | price_mismatches=0, demand_mismatches=0, weight_mismatches=0              |
| demand_total matches raw RTDREG En sum                      | pass   | mismatches=0, max_abs_diff=0.000000                                       |
| Equipment congestion flags and sums match RTDCV aggregation | pass   | flag_mismatches=0, sum_mismatches=0                                       |
| Fixed-effects columns match time_interval                   | pass   | fe_month=0, fe_week=0, fe_day=0                                           |
| No null cells in output                                     | pass   | null_cells=0                                                              |

## Notes On Source Behavior

- `RTD_ISLAND_PRICE` is already the correct island-level price input, so no price collapsing or marginal-resource aggregation is used here.
- `RTDREG` is filtered to `COMMODITY_TYPE == 'En'` because the panel treats `MKT_REQT` as island energy demand.
- `RTDCV` is event-style and sparse. The panel intentionally treats no event as no equipment congestion, not as missing data.
- `RTDHS` is used only for direct-link pair rows. There is no Luzon-Mindanao direct HVDC regressor, so that non-direct pair is excluded.
