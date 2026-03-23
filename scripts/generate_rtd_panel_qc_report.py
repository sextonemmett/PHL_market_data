#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_rtd_panels import (
    DIRECT_PAIRS,
    REGION_LABELS,
    load_congestion_wide,
    load_demand_wide,
    load_hvdc_wide,
    load_price_wide,
)

DEFAULT_PRICE_CSV = Path("data/csv_exports_flat/RTD_ISLAND_PRICE_202512220000_202603230000.csv")
DEFAULT_DEMAND_CSV = Path("data/csv_exports_flat/RTDREG_20251218_20260320.csv")
DEFAULT_CONGESTION_CSV = Path("data/csv_exports_flat/RTDCV_20251218_20260318.csv")
DEFAULT_CONGESTION_MAP_CSV = Path("data/rtd_congestion/rtd_congestion_resources_with_island_group.csv")
DEFAULT_HVDC_CSV = Path("data/csv_exports_flat/RTDHS_20251218_20260318.csv")
DEFAULT_DIRECT_PAIR_PANEL = Path("data/panels/RTD_DIRECT_PAIR_PANEL_202512212305_202603190000.parquet")
DEFAULT_ISLAND_SYSTEM_PANEL = Path("data/panels/RTD_ISLAND_SYSTEM_PANEL_202512212305_202603210000.parquet")
DEFAULT_REPORT_PATH = Path("reports/analysis/rtd_panel_qc_report.md")
FLOAT_TOL = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a markdown QC report for the RTD direct-pair and island-system panels."
    )
    parser.add_argument("--price-csv", default=str(DEFAULT_PRICE_CSV), help="Island-price CSV.")
    parser.add_argument("--demand-csv", default=str(DEFAULT_DEMAND_CSV), help="Demand CSV.")
    parser.add_argument("--congestion-csv", default=str(DEFAULT_CONGESTION_CSV), help="Congestion CSV.")
    parser.add_argument(
        "--congestion-map-csv",
        default=str(DEFAULT_CONGESTION_MAP_CSV),
        help="Equipment-to-island mapping CSV.",
    )
    parser.add_argument("--hvdc-csv", default=str(DEFAULT_HVDC_CSV), help="HVDC schedules CSV.")
    parser.add_argument(
        "--direct-pair-panel",
        default=str(DEFAULT_DIRECT_PAIR_PANEL),
        help="Direct-pair panel parquet.",
    )
    parser.add_argument(
        "--island-system-panel",
        default=str(DEFAULT_ISLAND_SYSTEM_PANEL),
        help="Island-system panel parquet.",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="Markdown report output path.",
    )
    return parser.parse_args()


def format_int(value: int) -> str:
    return f"{int(value):,}"


def format_float(value: float, digits: int = 6) -> str:
    return f"{float(value):,.{digits}f}"


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.astype(object).where(~frame.isna(), "")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy().tolist()]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def render_row(values: list[str]) -> str:
        return "|" + "|".join(f" {value.ljust(widths[idx])} " for idx, value in enumerate(values)) + "|"

    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join([render_row(headers), separator, *[render_row(row) for row in rows]])


def source_summary_table(
    price_csv: Path,
    demand_csv: Path,
    congestion_csv: Path,
    mapping_csv: Path,
    hvdc_csv: Path,
) -> pd.DataFrame:
    price = pd.read_csv(price_csv, parse_dates=["TIME_INTERVAL"])
    demand = pd.read_csv(demand_csv, parse_dates=["TIME_INTERVAL"])
    demand_en = demand.loc[demand["COMMODITY_TYPE"] == "En"].copy()
    congestion = pd.read_csv(congestion_csv, parse_dates=["TIME_INTERVAL"])
    mapping = pd.read_csv(mapping_csv)
    hvdc = pd.read_csv(hvdc_csv, parse_dates=["TIME_INTERVAL"])

    rows = [
        {
            "source": "RTD_ISLAND_PRICE",
            "path": str(price_csv),
            "rows_used": format_int(len(price)),
            "intervals": format_int(price["TIME_INTERVAL"].nunique()),
            "time_min": price["TIME_INTERVAL"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "time_max": price["TIME_INTERVAL"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "Dense island-level price source with one row per interval-region and columns ISLAND_PRICE, WEIGHT_SUM.",
        },
        {
            "source": "RTDREG (En only)",
            "path": str(demand_csv),
            "rows_used": format_int(len(demand_en)),
            "intervals": format_int(demand_en["TIME_INTERVAL"].nunique()),
            "time_min": demand_en["TIME_INTERVAL"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "time_max": demand_en["TIME_INTERVAL"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "Filtered to COMMODITY_TYPE == En; MKT_REQT is used as island demand.",
        },
        {
            "source": "RTDCV",
            "path": str(congestion_csv),
            "rows_used": format_int(len(congestion)),
            "intervals": format_int(congestion["TIME_INTERVAL"].nunique()),
            "time_min": congestion["TIME_INTERVAL"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "time_max": congestion["TIME_INTERVAL"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "Sparse event-style congestion file; missing interval-island cells are interpreted as zero after aggregation.",
        },
        {
            "source": "Congestion map",
            "path": str(mapping_csv),
            "rows_used": format_int(len(mapping)),
            "intervals": "",
            "time_min": "",
            "time_max": "",
            "notes": "Maps RTDCV EQUIPMENT_NAME values into Luzon, Visayas, or Mindanao.",
        },
        {
            "source": "RTDHS",
            "path": str(hvdc_csv),
            "rows_used": format_int(len(hvdc)),
            "intervals": format_int(hvdc["TIME_INTERVAL"].nunique()),
            "time_min": hvdc["TIME_INTERVAL"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "time_max": hvdc["TIME_INTERVAL"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "Dense direct-link schedule file with VISLUZ1 and MINVIS1 congestion flags. Missing rows are treated as true gaps.",
        },
    ]
    return pd.DataFrame(rows)


def expected_counts(
    price_wide: pd.DataFrame,
    demand_wide: pd.DataFrame,
    hvdc_wide: pd.DataFrame,
) -> tuple[int, int, int, int]:
    dense_base = price_wide.add_suffix("_price").join(demand_wide.add_suffix("_demand"), how="inner").dropna()
    expected_isys_rows = len(dense_base) * len(REGION_LABELS)
    hvdc_aligned = hvdc_wide.reindex(dense_base.index)
    expected_pair_rows = 0
    for pair in DIRECT_PAIRS:
        expected_pair_rows += int(hvdc_aligned[pair["link_name"]].notna().sum())
    pair_intervals = expected_pair_rows // len(DIRECT_PAIRS)
    return expected_pair_rows, expected_isys_rows, pair_intervals, len(dense_base)


def max_abs_diff(series_a: pd.Series, series_b: pd.Series) -> float:
    return float(np.max(np.abs(series_a.to_numpy(dtype=float) - series_b.to_numpy(dtype=float))))


def mismatch_count(series_a: pd.Series, series_b: pd.Series, tol: float = FLOAT_TOL) -> int:
    return int((~np.isclose(series_a.to_numpy(dtype=float), series_b.to_numpy(dtype=float), atol=tol, rtol=0)).sum())


def fixed_effect_mismatches(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "fe_month": int((frame["fe_month"] != frame["time_interval"].dt.strftime("%Y-%m")).sum()),
        "fe_week": int((frame["fe_week"] != frame["time_interval"].dt.strftime("%G-W%V")).sum()),
        "fe_day": int((frame["fe_day"] != frame["time_interval"].dt.strftime("%Y-%m-%d")).sum()),
    }


def build_direct_pair_qc(
    panel: pd.DataFrame,
    price_wide: pd.DataFrame,
    weight_wide: pd.DataFrame,
    demand_wide: pd.DataFrame,
    congestion_any_wide: pd.DataFrame,
    congestion_excess_wide: pd.DataFrame,
    hvdc_wide: pd.DataFrame,
    expected_rows: int,
    expected_intervals: int,
) -> pd.DataFrame:
    demand_total = demand_wide.sum(axis=1).rename("expected_demand_total")
    hvdc_aligned = hvdc_wide.reindex(panel["time_interval"])
    weight_1_expected = pd.Series(index=panel.index, dtype=float)
    weight_2_expected = pd.Series(index=panel.index, dtype=float)
    cong_1_expected = pd.Series(index=panel.index, dtype=float)
    cong_2_expected = pd.Series(index=panel.index, dtype=float)
    excess_1_expected = pd.Series(index=panel.index, dtype=float)
    excess_2_expected = pd.Series(index=panel.index, dtype=float)
    link_expected = pd.Series(index=panel.index, dtype=float)

    mapping_errors = 0
    for pair in DIRECT_PAIRS:
        mask = panel["pair_key"] == pair["pair_key"]
        mapping_errors += int((panel.loc[mask, "island_1"] != pair["island_1"]).sum())
        mapping_errors += int((panel.loc[mask, "island_2"] != pair["island_2"]).sum())
        mapping_errors += int((panel.loc[mask, "link_name"] != pair["link_name"]).sum())

        time_idx = panel.loc[mask, "time_interval"]
        weight_1_expected.loc[mask] = weight_wide.reindex(time_idx)[pair["island_1"]].to_numpy()
        weight_2_expected.loc[mask] = weight_wide.reindex(time_idx)[pair["island_2"]].to_numpy()
        cong_1_expected.loc[mask] = congestion_any_wide.reindex(time_idx, fill_value=0)[pair["island_1"]].to_numpy()
        cong_2_expected.loc[mask] = congestion_any_wide.reindex(time_idx, fill_value=0)[pair["island_2"]].to_numpy()
        excess_1_expected.loc[mask] = congestion_excess_wide.reindex(time_idx, fill_value=0.0)[pair["island_1"]].to_numpy()
        excess_2_expected.loc[mask] = congestion_excess_wide.reindex(time_idx, fill_value=0.0)[pair["island_2"]].to_numpy()
        link_expected.loc[mask] = hvdc_wide.reindex(time_idx)[pair["link_name"]].astype(int).to_numpy()

    fe_errors = fixed_effect_mismatches(panel)
    summary = [
        {
            "check": "Row count matches dense overlap and direct-link logic",
            "status": "pass" if len(panel) == expected_rows else "fail",
            "details": f"rows={len(panel):,}, expected={expected_rows:,}",
        },
        {
            "check": "Unique key (time_interval, pair_key)",
            "status": "pass" if int(panel.duplicated(['time_interval', 'pair_key']).sum()) == 0 else "fail",
            "details": f"duplicate_keys={int(panel.duplicated(['time_interval', 'pair_key']).sum())}",
        },
        {
            "check": "Pair set and link mapping",
            "status": "pass" if mapping_errors == 0 and sorted(panel['pair_key'].unique().tolist()) == ['CLUZ_CVIS', 'CVIS_CMIN'] else "fail",
            "details": f"mapping_errors={mapping_errors}, intervals={panel['time_interval'].nunique():,}, expected_intervals={expected_intervals:,}",
        },
        {
            "check": "dep_abs_price_gap = abs(price_1 - price_2)",
            "status": "pass" if mismatch_count(panel["dep_abs_price_gap"], (panel["price_1"] - panel["price_2"]).abs()) == 0 else "fail",
            "details": (
                f"mismatches={mismatch_count(panel['dep_abs_price_gap'], (panel['price_1'] - panel['price_2']).abs())}, "
                f"max_abs_diff={format_float(max_abs_diff(panel['dep_abs_price_gap'], (panel['price_1'] - panel['price_2']).abs()))}"
            ),
        },
        {
            "check": "demand_total matches raw RTDREG En sum",
            "status": "pass" if mismatch_count(panel["demand_total"], panel["time_interval"].map(demand_total)) == 0 else "fail",
            "details": (
                f"mismatches={mismatch_count(panel['demand_total'], panel['time_interval'].map(demand_total))}, "
                f"max_abs_diff={format_float(max_abs_diff(panel['demand_total'], panel['time_interval'].map(demand_total)))}"
            ),
        },
        {
            "check": "WEIGHT_SUM audit fields match price source",
            "status": "pass" if mismatch_count(panel["weight_sum_1"], weight_1_expected) == 0 and mismatch_count(panel["weight_sum_2"], weight_2_expected) == 0 else "fail",
            "details": (
                f"weight_1_mismatches={mismatch_count(panel['weight_sum_1'], weight_1_expected)}, "
                f"weight_2_mismatches={mismatch_count(panel['weight_sum_2'], weight_2_expected)}"
            ),
        },
        {
            "check": "Equipment congestion flags and sums match RTDCV aggregation",
            "status": "pass"
            if (
                mismatch_count(panel["equip_cong_any_1"], cong_1_expected) == 0
                and mismatch_count(panel["equip_cong_any_2"], cong_2_expected) == 0
                and mismatch_count(panel["equip_excess_pct_sum_1"], excess_1_expected) == 0
                and mismatch_count(panel["equip_excess_pct_sum_2"], excess_2_expected) == 0
            )
            else "fail",
            "details": (
                f"flag_mismatches=({mismatch_count(panel['equip_cong_any_1'], cong_1_expected)}, "
                f"{mismatch_count(panel['equip_cong_any_2'], cong_2_expected)}), "
                f"sum_mismatches=({mismatch_count(panel['equip_excess_pct_sum_1'], excess_1_expected)}, "
                f"{mismatch_count(panel['equip_excess_pct_sum_2'], excess_2_expected)})"
            ),
        },
        {
            "check": "HVDC congestion flag matches RTDHS aggregation",
            "status": "pass" if mismatch_count(panel["link_congested_any"], link_expected) == 0 else "fail",
            "details": f"mismatches={mismatch_count(panel['link_congested_any'], link_expected)}",
        },
        {
            "check": "Fixed-effects columns match time_interval",
            "status": "pass" if sum(fe_errors.values()) == 0 else "fail",
            "details": ", ".join(f"{key}={value}" for key, value in fe_errors.items()),
        },
        {
            "check": "No null cells in output",
            "status": "pass" if int(panel.isna().sum().sum()) == 0 else "fail",
            "details": f"null_cells={int(panel.isna().sum().sum())}",
        },
    ]
    return pd.DataFrame(summary)


def build_island_system_qc(
    panel: pd.DataFrame,
    price_wide: pd.DataFrame,
    weight_wide: pd.DataFrame,
    demand_wide: pd.DataFrame,
    congestion_any_wide: pd.DataFrame,
    congestion_excess_wide: pd.DataFrame,
    expected_rows: int,
    expected_intervals: int,
) -> pd.DataFrame:
    demand_total = demand_wide.sum(axis=1)
    price_sys = (price_wide * demand_wide).sum(axis=1) / demand_total
    price_expected = pd.Series(index=panel.index, dtype=float)
    weight_expected = pd.Series(index=panel.index, dtype=float)
    demand_expected = pd.Series(index=panel.index, dtype=float)
    cong_expected = pd.Series(index=panel.index, dtype=float)
    excess_expected = pd.Series(index=panel.index, dtype=float)
    psys_expected = panel["time_interval"].map(price_sys)

    for region in REGION_LABELS:
        mask = panel["island_code"] == region
        time_idx = panel.loc[mask, "time_interval"]
        price_expected.loc[mask] = price_wide.reindex(time_idx)[region].to_numpy()
        weight_expected.loc[mask] = weight_wide.reindex(time_idx)[region].to_numpy()
        demand_expected.loc[mask] = demand_wide.reindex(time_idx)[region].to_numpy()
        cong_expected.loc[mask] = congestion_any_wide.reindex(time_idx, fill_value=0)[region].to_numpy()
        excess_expected.loc[mask] = congestion_excess_wide.reindex(time_idx, fill_value=0.0)[region].to_numpy()

    fe_errors = fixed_effect_mismatches(panel)
    summary = [
        {
            "check": "Row count matches dense price-demand overlap",
            "status": "pass" if len(panel) == expected_rows else "fail",
            "details": f"rows={len(panel):,}, expected={expected_rows:,}, intervals={panel['time_interval'].nunique():,}, expected_intervals={expected_intervals:,}",
        },
        {
            "check": "Unique key (time_interval, island_code)",
            "status": "pass" if int(panel.duplicated(['time_interval', 'island_code']).sum()) == 0 else "fail",
            "details": f"duplicate_keys={int(panel.duplicated(['time_interval', 'island_code']).sum())}",
        },
        {
            "check": "price_sys_dw matches raw demand-weighted system price",
            "status": "pass" if mismatch_count(panel["price_sys_dw"], psys_expected) == 0 else "fail",
            "details": (
                f"mismatches={mismatch_count(panel['price_sys_dw'], psys_expected)}, "
                f"max_abs_diff={format_float(max_abs_diff(panel['price_sys_dw'], psys_expected))}"
            ),
        },
        {
            "check": "dep_price_minus_sys = abs(price_island - price_sys_dw)",
            "status": "pass" if mismatch_count(panel["dep_price_minus_sys"], (panel["price_island"] - panel["price_sys_dw"]).abs()) == 0 else "fail",
            "details": (
                f"mismatches={mismatch_count(panel['dep_price_minus_sys'], (panel['price_island'] - panel['price_sys_dw']).abs())}, "
                f"max_abs_diff={format_float(max_abs_diff(panel['dep_price_minus_sys'], (panel['price_island'] - panel['price_sys_dw']).abs()))}"
            ),
        },
        {
            "check": "Island price, demand, and weight fields match raw sources",
            "status": "pass"
            if (
                mismatch_count(panel["price_island"], price_expected) == 0
                and mismatch_count(panel["demand_island"], demand_expected) == 0
                and mismatch_count(panel["weight_sum_island"], weight_expected) == 0
            )
            else "fail",
            "details": (
                f"price_mismatches={mismatch_count(panel['price_island'], price_expected)}, "
                f"demand_mismatches={mismatch_count(panel['demand_island'], demand_expected)}, "
                f"weight_mismatches={mismatch_count(panel['weight_sum_island'], weight_expected)}"
            ),
        },
        {
            "check": "demand_total matches raw RTDREG En sum",
            "status": "pass" if mismatch_count(panel["demand_total"], panel["time_interval"].map(demand_total)) == 0 else "fail",
            "details": (
                f"mismatches={mismatch_count(panel['demand_total'], panel['time_interval'].map(demand_total))}, "
                f"max_abs_diff={format_float(max_abs_diff(panel['demand_total'], panel['time_interval'].map(demand_total)))}"
            ),
        },
        {
            "check": "Equipment congestion flags and sums match RTDCV aggregation",
            "status": "pass"
            if (
                mismatch_count(panel["equip_cong_any"], cong_expected) == 0
                and mismatch_count(panel["equip_excess_pct_sum"], excess_expected) == 0
            )
            else "fail",
            "details": (
                f"flag_mismatches={mismatch_count(panel['equip_cong_any'], cong_expected)}, "
                f"sum_mismatches={mismatch_count(panel['equip_excess_pct_sum'], excess_expected)}"
            ),
        },
        {
            "check": "Fixed-effects columns match time_interval",
            "status": "pass" if sum(fe_errors.values()) == 0 else "fail",
            "details": ", ".join(f"{key}={value}" for key, value in fe_errors.items()),
        },
        {
            "check": "No null cells in output",
            "status": "pass" if int(panel.isna().sum().sum()) == 0 else "fail",
            "details": f"null_cells={int(panel.isna().sum().sum())}",
        },
    ]
    return pd.DataFrame(summary)


def panel_summary_table(name: str, frame: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    rows = [
        {"metric": "rows", "value": format_int(len(frame))},
        {"metric": "unique intervals", "value": format_int(frame["time_interval"].nunique())},
        {"metric": "time min", "value": frame["time_interval"].min().strftime("%Y-%m-%d %H:%M:%S")},
        {"metric": "time max", "value": frame["time_interval"].max().strftime("%Y-%m-%d %H:%M:%S")},
        {"metric": "duplicate keys", "value": format_int(int(frame.duplicated(key_cols).sum()))},
        {"metric": "null cells", "value": format_int(int(frame.isna().sum().sum()))},
        {"metric": "key distribution", "value": ", ".join(sorted(map(str, frame[key_cols[-1]].unique().tolist())))},
    ]
    if name == "direct_pair_panel":
        rows.append({"metric": "link distribution", "value": ", ".join(sorted(frame["link_name"].unique().tolist()))})
    return pd.DataFrame(rows)


def coverage_table(
    price_wide: pd.DataFrame,
    demand_wide: pd.DataFrame,
    hvdc_wide: pd.DataFrame,
    direct_pair_panel: pd.DataFrame,
    island_system_panel: pd.DataFrame,
) -> pd.DataFrame:
    price_intervals = price_wide.index
    demand_intervals = demand_wide.index
    dense_base = (
        price_wide.add_suffix("_price").join(demand_wide.add_suffix("_demand"), how="inner").dropna().index
    )
    pair_base = hvdc_wide.reindex(dense_base).dropna().index
    rows = [
        {
            "window": "Island price source",
            "intervals": format_int(len(price_intervals)),
            "time_min": price_intervals.min().strftime("%Y-%m-%d %H:%M:%S"),
            "time_max": price_intervals.max().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "All three island prices present at every interval in file.",
        },
        {
            "window": "Price + RTDREG (En) overlap",
            "intervals": format_int(len(dense_base)),
            "time_min": dense_base.min().strftime("%Y-%m-%d %H:%M:%S"),
            "time_max": dense_base.max().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "Defines island-system panel coverage.",
        },
        {
            "window": "Price + RTDREG + RTDHS overlap",
            "intervals": format_int(len(pair_base)),
            "time_min": pair_base.min().strftime("%Y-%m-%d %H:%M:%S"),
            "time_max": pair_base.max().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "Defines direct-pair panel coverage because link rows are required.",
        },
        {
            "window": "direct_pair_panel output",
            "intervals": format_int(direct_pair_panel["time_interval"].nunique()),
            "time_min": direct_pair_panel["time_interval"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "time_max": direct_pair_panel["time_interval"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "Two rows per usable interval: CLUZ_CVIS and CVIS_CMIN.",
        },
        {
            "window": "island_system_panel output",
            "intervals": format_int(island_system_panel["time_interval"].nunique()),
            "time_min": island_system_panel["time_interval"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "time_max": island_system_panel["time_interval"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "Three rows per usable interval: CLUZ, CVIS, CMIN.",
        },
    ]
    return pd.DataFrame(rows)


def build_report(args: argparse.Namespace) -> str:
    price_csv = Path(args.price_csv)
    demand_csv = Path(args.demand_csv)
    congestion_csv = Path(args.congestion_csv)
    mapping_csv = Path(args.congestion_map_csv)
    hvdc_csv = Path(args.hvdc_csv)
    direct_pair_panel_path = Path(args.direct_pair_panel)
    island_system_panel_path = Path(args.island_system_panel)

    price_wide, weight_wide = load_price_wide(price_csv)
    demand_wide = load_demand_wide(demand_csv)
    congestion_any_wide, congestion_excess_wide = load_congestion_wide(congestion_csv, mapping_csv)
    hvdc_wide = load_hvdc_wide(hvdc_csv)
    direct_pair_panel = pd.read_parquet(direct_pair_panel_path)
    island_system_panel = pd.read_parquet(island_system_panel_path)

    direct_pair_panel["time_interval"] = pd.to_datetime(direct_pair_panel["time_interval"])
    island_system_panel["time_interval"] = pd.to_datetime(island_system_panel["time_interval"])

    expected_pair_rows, expected_isys_rows, expected_pair_intervals, expected_isys_intervals = expected_counts(
        price_wide,
        demand_wide,
        hvdc_wide,
    )
    source_table = source_summary_table(price_csv, demand_csv, congestion_csv, mapping_csv, hvdc_csv)
    coverage = coverage_table(price_wide, demand_wide, hvdc_wide, direct_pair_panel, island_system_panel)
    direct_summary = panel_summary_table("direct_pair_panel", direct_pair_panel, ["time_interval", "pair_key"])
    isys_summary = panel_summary_table("island_system_panel", island_system_panel, ["time_interval", "island_code"])
    direct_qc = build_direct_pair_qc(
        direct_pair_panel,
        price_wide,
        weight_wide,
        demand_wide,
        congestion_any_wide,
        congestion_excess_wide,
        hvdc_wide,
        expected_pair_rows,
        expected_pair_intervals,
    )
    isys_qc = build_island_system_qc(
        island_system_panel,
        price_wide,
        weight_wide,
        demand_wide,
        congestion_any_wide,
        congestion_excess_wide,
        expected_isys_rows,
        expected_isys_intervals,
    )

    return f"""# RTD Panel QC Report

This report documents how the two RTD panels were built in [scripts/build_rtd_panels.py](../../scripts/build_rtd_panels.py) and records the QC checks run against the generated outputs.

## Source Files

{markdown_table(source_table)}

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

{markdown_table(coverage)}

The price file is the dense driver here: it has all three islands at every interval in-file. The remaining coverage differences come from source boundaries:

- The pair panel stops when `RTDHS` stops providing dense link rows.
- The island-system panel continues longer because it does not require `RTDHS`.
- `RTDCV` is sparse by design, so absence is converted to `0` after interval-island aggregation rather than dropping rows.

## direct_pair_panel Summary

Output file: `{direct_pair_panel_path}`

{markdown_table(direct_summary)}

## direct_pair_panel QC Checks

{markdown_table(direct_qc)}

## island_system_panel Summary

Output file: `{island_system_panel_path}`

{markdown_table(isys_summary)}

## island_system_panel QC Checks

{markdown_table(isys_qc)}

## Notes On Source Behavior

- `RTD_ISLAND_PRICE` is already the correct island-level price input, so no price collapsing or marginal-resource aggregation is used here.
- `RTDREG` is filtered to `COMMODITY_TYPE == 'En'` because the panel treats `MKT_REQT` as island energy demand.
- `RTDCV` is event-style and sparse. The panel intentionally treats no event as no equipment congestion, not as missing data.
- `RTDHS` is used only for direct-link pair rows. There is no Luzon-Mindanao direct HVDC regressor, so that non-direct pair is excluded.
"""


def main() -> None:
    args = parse_args()
    report = build_report(args)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
