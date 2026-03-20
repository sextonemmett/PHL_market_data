# RTD Dataset Column Profile

This report profiles the three newly combined RTD datasets requested for structure review: `RTDSL`, `RTDCV`, and `RTDHS`. It focuses on file-level sparsity, column-level missingness, value distributions, and numeric ranges for the `2025-12-18` through `2026-03-18` window.

## RTDSL: RTD Security Limits Used

Dense security-limit records keyed by resource and parameter type. No field-level nulls in the combined parquet for this window.

### Overview

| dataset                  | rows    | columns | date_min            | date_max            | warning_days | empty_days |
|--------------------------|---------|---------|---------------------|---------------------|--------------|------------|
| RTD Security Limits Used | 523,065 | 9       | 2025-12-18 00:05:00 | 2026-03-19 00:00:00 | 0            | 0          |

### File-Level QC

| status | days |
|--------|------|
| ok     | 91   |

Warning breakdown:

| warning_type | days | sample_dates |
|--------------|------|--------------|
| (none)       | 0    |              |

### Column Inventory

| column          | dtype          | non_null_rows | missing_rows | missing_pct | unique_non_null |
|-----------------|----------------|---------------|--------------|-------------|-----------------|
| RUN_TIME        | datetime64[ns] | 523,065       | 0            | 0.00%       | 26,186          |
| MKT_TYPE        | category       | 523,065       | 0            | 0.00%       | 1               |
| REGION_NAME     | category       | 523,065       | 0            | 0.00%       | 3               |
| RESOURCE_NAME   | category       | 523,065       | 0            | 0.00%       | 140             |
| RESOURCE_TYPE   | category       | 523,065       | 0            | 0.00%       | 2               |
| START_TIME      | datetime64[ns] | 523,065       | 0            | 0.00%       | 22,303          |
| END_TIME        | datetime64[ns] | 523,065       | 0            | 0.00%       | 21,975          |
| PARAMETER_TYPE  | category       | 523,065       | 0            | 0.00%       | 2               |
| PARAMETER_VALUE | float64        | 523,065       | 0            | 0.00%       | 8,680           |

### Datetime Columns

| column     | min                 | median              | max                 | unique_timestamps |
|------------|---------------------|---------------------|---------------------|-------------------|
| RUN_TIME   | 2025-12-18 00:05:00 | 2026-02-03 08:00:00 | 2026-03-19 00:00:00 | 26,186            |
| START_TIME | 2025-12-12 10:01:00 | 2026-02-03 07:01:00 | 2026-03-19 00:01:00 | 22,303            |
| END_TIME   | 2025-12-18 00:05:00 | 2026-02-03 09:00:00 | 2026-03-19 06:00:00 | 21,975            |

### Numeric Columns

| column          | non_null_rows | missing_pct | min     | p05  | median | mean  | p95    | max    | zeros  | negative |
|-----------------|---------------|-------------|---------|------|--------|-------|--------|--------|--------|----------|
| PARAMETER_VALUE | 523,065       | 0.00%       | -131.28 | 0.00 | 15.00  | 38.63 | 127.40 | 668.00 | 51,829 | 10,127   |

### Categorical / String Value Distributions

#### `MKT_TYPE`

| value | rows    | share   |
|-------|---------|---------|
| RTD   | 523,065 | 100.00% |

#### `REGION_NAME`

| value | rows    | share  |
|-------|---------|--------|
| CLUZ  | 379,984 | 72.65% |
| CVIS  | 102,096 | 19.52% |
| CMIN  | 40,985  | 7.84%  |

#### `RESOURCE_NAME`

| value           | rows   | share  |
|-----------------|--------|--------|
| 01CONSOL_G01    | 56,401 | 10.78% |
| 08PWIND_G02     | 50,707 | 9.69%  |
| 01BALWIND_G01   | 40,943 | 7.83%  |
| 03LUMBSOL_G01   | 37,571 | 7.18%  |
| 01ARAYSOL_G03   | 36,098 | 6.90%  |
| 01SNMANSOL_G01  | 33,098 | 6.33%  |
| 03LUNSOL_G01    | 32,850 | 6.28%  |
| 04ISIDROSOL_G01 | 28,175 | 5.39%  |
| 01OLONGSOL_G01  | 24,679 | 4.72%  |
| 01LIMSOL_G01    | 16,646 | 3.18%  |
| 01MAGAT_BAT     | 14,273 | 2.73%  |
| 11PULANAI_G01   | 13,042 | 2.49%  |

#### `RESOURCE_TYPE`

| value | rows    | share  |
|-------|---------|--------|
| G     | 521,338 | 99.67% |
| PS    | 1,727   | 0.33%  |

#### `PARAMETER_TYPE`

| value            | rows    | share  |
|------------------|---------|--------|
| MAX_OPERATING_MW | 261,593 | 50.01% |
| MIN_OPERATING_MW | 261,472 | 49.99% |

### Visuals

![RTDSL missingness](rtd_dataset_profile_assets/rtdsl_missingness.png)

![RTDSL daily rows](rtd_dataset_profile_assets/rtdsl_daily_rows.png)

![RTDSL numeric distributions](rtd_dataset_profile_assets/rtdsl_numeric_hist.png)

![RTDSL top categorical values](rtd_dataset_profile_assets/rtdsl_top_categories.png)

## RTDCV: RTD Congestions Manifesting

Sparse event-style congestion records. Most sparsity shows up as empty daily files rather than null columns inside the combined parquet.

### Overview

| dataset                     | rows  | columns | date_min            | date_max            | warning_days | empty_days |
|-----------------------------|-------|---------|---------------------|---------------------|--------------|------------|
| RTD Congestions Manifesting | 4,082 | 12      | 2025-12-18 00:00:00 | 2026-03-18 13:10:00 | 31           | 31         |

### File-Level QC

| status  | days |
|---------|------|
| ok      | 60   |
| warning | 31   |

Warning breakdown:

| warning_type | days | sample_dates                                               |
|--------------|------|------------------------------------------------------------|
| empty_file   | 31   | 2025-12-19, 2025-12-20, 2025-12-22, 2025-12-25, 2025-12-29 |

### Column Inventory

| column         | dtype          | non_null_rows | missing_rows | missing_pct | unique_non_null |
|----------------|----------------|---------------|--------------|-------------|-----------------|
| RUN_TIME       | datetime64[ns] | 4,082         | 0            | 0.00%       | 3,572           |
| MKT_TYPE       | category       | 4,082         | 0            | 0.00%       | 1               |
| TIME_INTERVAL  | datetime64[ns] | 4,082         | 0            | 0.00%       | 3,572           |
| CONGEST_TYPE   | category       | 4,082         | 0            | 0.00%       | 2               |
| RUN_TYPE       | category       | 4,082         | 0            | 0.00%       | 1               |
| EQUIPMENT_NAME | category       | 4,082         | 0            | 0.00%       | 19              |
| STATION_NAME   | category       | 4,082         | 0            | 0.00%       | 15              |
| VOLTAGE_LEVEL  | float64        | 4,082         | 0            | 0.00%       | 3               |
| BINDING_LIMIT  | float64        | 4,082         | 0            | 0.00%       | 1,894           |
| MW_FLOW        | float64        | 4,082         | 0            | 0.00%       | 2,290           |
| OVERLOAD_MW    | float64        | 4,082         | 0            | 0.00%       | 1,201           |
| PCT_MW         | float64        | 4,082         | 0            | 0.00%       | 823             |

### Datetime Columns

| column        | min                 | median              | max                 | unique_timestamps |
|---------------|---------------------|---------------------|---------------------|-------------------|
| RUN_TIME      | 2025-12-18 00:00:00 | 2026-01-21 16:25:00 | 2026-03-18 13:10:00 | 3,572             |
| TIME_INTERVAL | 2025-12-18 00:05:00 | 2026-01-21 16:30:00 | 2026-03-18 13:15:00 | 3,572             |

### Numeric Columns

| column        | non_null_rows | missing_pct | min    | p05    | median | mean   | p95    | max    | zeros | negative |
|---------------|---------------|-------------|--------|--------|--------|--------|--------|--------|-------|----------|
| VOLTAGE_LEVEL | 4,082         | 0.00%       | 115.00 | 230.00 | 230.00 | 229.83 | 230.00 | 230.00 | 0     | 0        |
| BINDING_LIMIT | 4,082         | 0.00%       | 36.22  | 191.24 | 223.06 | 373.34 | 991.62 | 995.08 | 0     | 0        |
| MW_FLOW       | 4,082         | 0.00%       | 36.47  | 191.24 | 239.13 | 380.73 | 991.62 | 995.08 | 0     | 0        |
| OVERLOAD_MW   | 4,082         | 0.00%       | 0.00   | 0.00   | 2.40   | 7.38   | 21.99  | 29.94  | 1,794 | 0        |
| PCT_MW        | 4,082         | 0.00%       | 100.00 | 100.00 | 101.08 | 103.35 | 110.12 | 111.33 | 0     | 0        |

### RTDCV Checks

The limit metrics behave like rounded arithmetic fields: `OVERLOAD_MW` is approximately `MW_FLOW - BINDING_LIMIT`, and `PCT_MW` is approximately `MW_FLOW / BINDING_LIMIT * 100`.

| check                                             | max_abs_diff | match_within_0.01 | match_within_0.05 |
|---------------------------------------------------|--------------|-------------------|-------------------|
| OVERLOAD_MW ~= MW_FLOW - BINDING_LIMIT            | 0.0100       | 92.14%            | 100.00%           |
| PCT_MW ~= MW_FLOW / BINDING_LIMIT * 100           | 0.0089       | 100.00%           | 100.00%           |
| PCT_MW - 100 ~= OVERLOAD_MW / BINDING_LIMIT * 100 | 0.0086       | 100.00%           | 100.00%           |

Equipment-to-station mapping:

| equipment_name | station_count | station_names      |
|----------------|---------------|--------------------|
| 1BAUA_1LAT1    | 2             | 01BAUANG, 01LATRIN |
| 1BAUA_1LAT2    | 2             | 01BAUANG, 01LATRIN |
| 5DAAN_4TAB1    | 2             | 04TABANG, 05DAANBN |
| 13MATA_13TOR1  | 1             | 13MATAN            |
| 1ANGAT_TR3     | 1             | 01ANGAT            |
| 1BALSIK_TR1    | 1             | 01BALSIK           |
| 1BING_1NGS2    | 1             | 01EHVNAG           |
| 1EHVNGS_TR1    | 1             | 01EHVNAG           |
| 1EHVNGS_TR2    | 1             | 01EHVNAG           |
| 1MEXI_1HER1    | 1             | 01MEXICO           |
| 1MEXI_1HER2    | 1             | 01MEXICO           |
| 1SRAF_1SJO1    | 1             | 01SNRAFA           |

![RTDCV congest type by station](rtd_dataset_profile_assets/rtdcv_congest_x_station.png)

![RTDCV congest type by equipment](rtd_dataset_profile_assets/rtdcv_congest_x_equipment.png)

### RTDCV Top 6 Equipment Deep Dive

These are the six equipment names with the most RTDCV rows in the current window. The table and visuals below focus on whether they are base or contingency, which stations they map to, and how their numeric fields are distributed.

| equipment_name | rows  | congest_type | stations           | binding_median | flow_median | overload_median | pct_median | row_share |
|----------------|-------|--------------|--------------------|----------------|-------------|-----------------|------------|-----------|
| 5DAAN_4TAB2    | 2,415 | BASE CASE    | 05DAANBN           | 221.93         | 235.28      | 13.03           | 105.88     | 59.16%    |
| 1BALSIK_TR1    | 760   | CONTINGENCY  | 01BALSIK           | 988.62         | 988.62      | 0.00            | 100.00     | 18.62%    |
| 5DAAN_4TAB1    | 442   | BASE CASE    | 04TABANG, 05DAANBN | 191.33         | 191.33      | 0.00            | 100.00     | 10.83%    |
| 1BAUA_1LAT1    | 160   | CONTINGENCY  | 01BAUANG, 01LATRIN | 288.30         | 288.30      | 0.00            | 100.00     | 3.92%     |
| 1BAUA_1LAT2    | 154   | CONTINGENCY  | 01BAUANG, 01LATRIN | 288.27         | 288.27      | 0.00            | 100.00     | 3.77%     |
| 1MEXI_1HER2    | 101   | CONTINGENCY  | 01MEXICO           | 461.93         | 462.04      | 0.00            | 100.00     | 2.47%     |

![RTDCV top 6 equipment overview](rtd_dataset_profile_assets/rtdcv_top6_equipment_overview.png)

![RTDCV top 6 equipment station heatmap](rtd_dataset_profile_assets/rtdcv_top6_equipment_station_heatmap.png)

![RTDCV top 6 equipment numeric boxplots](rtd_dataset_profile_assets/rtdcv_top6_equipment_numeric_boxplots.png)

### Categorical / String Value Distributions

#### `MKT_TYPE`

| value | rows  | share   |
|-------|-------|---------|
| RTD   | 4,082 | 100.00% |

#### `CONGEST_TYPE`

| value       | rows  | share  |
|-------------|-------|--------|
| BASE CASE   | 2,884 | 70.65% |
| CONTINGENCY | 1,198 | 29.35% |

#### `RUN_TYPE`

| value     | rows  | share   |
|-----------|-------|---------|
| SCHED_RUN | 4,082 | 100.00% |

#### `EQUIPMENT_NAME`

| value         | rows  | share  |
|---------------|-------|--------|
| 5DAAN_4TAB2   | 2,415 | 59.16% |
| 1BALSIK_TR1   | 760   | 18.62% |
| 5DAAN_4TAB1   | 442   | 10.83% |
| 1BAUA_1LAT1   | 160   | 3.92%  |
| 1BAUA_1LAT2   | 154   | 3.77%  |
| 1MEXI_1HER2   | 101   | 2.47%  |
| 1BING_1NGS2   | 13    | 0.32%  |
| 13MATA_13TOR1 | 10    | 0.24%  |
| 2QUEZON_TR1   | 5     | 0.12%  |
| 1MEXI_1HER1   | 5     | 0.12%  |
| 3AMAD_3CLA1   | 3     | 0.07%  |
| 1EHVNGS_TR2   | 3     | 0.07%  |

#### `STATION_NAME`

| value    | rows  | share  |
|----------|-------|--------|
| 05DAANBN | 2,850 | 69.82% |
| 01BALSIK | 760   | 18.62% |
| 01LATRIN | 267   | 6.54%  |
| 01MEXICO | 106   | 2.60%  |
| 01BAUANG | 47    | 1.15%  |
| 01EHVNAG | 17    | 0.42%  |
| 13MATAN  | 10    | 0.24%  |
| 04TABANG | 7     | 0.17%  |
| 02QUEZON | 5     | 0.12%  |
| 03CLACA  | 4     | 0.10%  |
| 09ZAMBO  | 3     | 0.07%  |
| 05MANDAU | 2     | 0.05%  |

### Visuals

![RTDCV missingness](rtd_dataset_profile_assets/rtdcv_missingness.png)

![RTDCV daily rows](rtd_dataset_profile_assets/rtdcv_daily_rows.png)

![RTDCV numeric distributions](rtd_dataset_profile_assets/rtdcv_numeric_hist.png)

![RTDCV top categorical values](rtd_dataset_profile_assets/rtdcv_top_categories.png)

## RTDHS: RTD HVDC Schedules

Regular interval schedules for the two HVDC links. `OVERLOAD_MW` is entirely null for the current three-month window.

### Overview

| dataset            | rows   | columns | date_min            | date_max            | warning_days | empty_days |
|--------------------|--------|---------|---------------------|---------------------|--------------|------------|
| RTD HVDC Schedules | 52,372 | 8       | 2025-12-18 00:00:00 | 2026-03-18 23:55:00 | 5            | 0          |

### File-Level QC

| status  | days |
|---------|------|
| ok      | 86   |
| warning | 5    |

Warning breakdown:

| warning_type                       | days | sample_dates                                   |
|------------------------------------|------|------------------------------------------------|
| unexpected_time_interval_count:287 | 4    | 2026-01-12, 2026-02-13, 2026-02-15, 2026-03-15 |
| unexpected_time_interval_count:270 | 1    | 2026-02-23                                     |

### Column Inventory

| column          | dtype          | non_null_rows | missing_rows | missing_pct | unique_non_null |
|-----------------|----------------|---------------|--------------|-------------|-----------------|
| RUN_TIME        | datetime64[ns] | 52,372        | 0            | 0.00%       | 26,186          |
| MKT_TYPE        | category       | 52,372        | 0            | 0.00%       | 1               |
| TIME_INTERVAL   | datetime64[ns] | 52,372        | 0            | 0.00%       | 26,186          |
| HVDC_NAME       | category       | 52,372        | 0            | 0.00%       | 2               |
| CONGESTION_FLAG | category       | 52,372        | 0            | 0.00%       | 2               |
| FLOW_FROM       | float64        | 52,372        | 0            | 0.00%       | 27,858          |
| FLOW_TO         | float64        | 52,372        | 0            | 0.00%       | 27,858          |
| OVERLOAD_MW     | float64        | 0             | 52,372       | 100.00%     | 0               |

### Datetime Columns

| column        | min                 | median              | max                 | unique_timestamps |
|---------------|---------------------|---------------------|---------------------|-------------------|
| RUN_TIME      | 2025-12-18 00:00:00 | 2026-02-01 11:10:00 | 2026-03-18 23:55:00 | 26,186            |
| TIME_INTERVAL | 2025-12-18 00:05:00 | 2026-02-01 11:15:00 | 2026-03-19 00:00:00 | 26,186            |

### Numeric Columns

| column      | non_null_rows | missing_pct | min     | p05     | median | mean   | p95    | max    | zeros | negative |
|-------------|---------------|-------------|---------|---------|--------|--------|--------|--------|-------|----------|
| FLOW_FROM   | 52,372        | 0.00%       | -366.66 | -180.00 | 71.69  | 78.23  | 388.05 | 450.00 | 3,366 | 17,076   |
| FLOW_TO     | 52,372        | 0.00%       | -450.00 | -388.05 | -71.69 | -78.23 | 180.00 | 366.66 | 3,366 | 31,930   |
| OVERLOAD_MW | 0             | 100.00%     |         |         |        |        |        |        | 0     | 0        |

### Categorical / String Value Distributions

#### `MKT_TYPE`

| value | rows   | share   |
|-------|--------|---------|
| RTD   | 52,372 | 100.00% |

#### `HVDC_NAME`

| value   | rows   | share  |
|---------|--------|--------|
| MINVIS1 | 26,186 | 50.00% |
| VISLUZ1 | 26,186 | 50.00% |

#### `CONGESTION_FLAG`

| value | rows   | share  |
|-------|--------|--------|
| N     | 37,541 | 71.68% |
| Y     | 14,831 | 28.32% |

### Visuals

![RTDHS missingness](rtd_dataset_profile_assets/rtdhs_missingness.png)

![RTDHS daily rows](rtd_dataset_profile_assets/rtdhs_daily_rows.png)

![RTDHS numeric distributions](rtd_dataset_profile_assets/rtdhs_numeric_hist.png)

![RTDHS top categorical values](rtd_dataset_profile_assets/rtdhs_top_categories.png)
