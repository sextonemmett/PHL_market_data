# RTD Island Price Visual Check

This report provides a quick visual QC pass for the concatenated `RTD_ISLAND_PRICE` file.

## Overview

| metric | value |
|--------|-------|
| rows | 78,594 |
| regions | CLUZ, CMIN, CVIS |
| min interval | 2025-12-21 23:05:00 |
| max interval | 2026-03-23 00:00:00 |
| median island price | 2925.43 |
| min island price | -216614.30 |
| max island price | 132547.87 |
| median weight sum | 4291.84 |
| min weight sum | 2532.51 |
| max weight sum | 24247.36 |
| ok hourly files | 2,184 |
| warning hourly files | 1 |
| partial but non-empty hourly files | 5 |

## Warning Hours

| file_token | status | interval_count | warnings |
|------------|--------|----------------|----------|
| 202602232200 | warning | 0 | empty_data_file |

## Partial Hours Included

| file_token | interval_count | min_interval | max_interval |
|------------|----------------|--------------|--------------|
| 202601121500 | 11 | 2026-01-12 14:05:00 | 2026-01-12 15:00:00 |
| 202602130300 | 11 | 2026-02-13 02:05:00 | 2026-02-13 03:00:00 |
| 202602150900 | 11 | 2026-02-15 08:10:00 | 2026-02-15 09:00:00 |
| 202602232300 | 6 | 2026-02-23 22:35:00 | 2026-02-23 23:00:00 |
| 202603150800 | 11 | 2026-03-15 07:10:00 | 2026-03-15 08:00:00 |

## Visuals

### Daily Price Trend by Region

![Daily island price trend](rtd_island_price_visual_assets/daily_region_prices.png)

### Median Intraday Price Profile

![Median intraday island price](rtd_island_price_visual_assets/intraday_profile.png)

### Daily Schedule-Weight Mass

![Daily schedule weight mass](rtd_island_price_visual_assets/daily_weight_sum.png)

### Hourly File Completeness

![Hourly file completeness heatmap](rtd_island_price_visual_assets/hourly_completeness_heatmap.png)
