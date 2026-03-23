#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REGION_LABELS = {"CLUZ": "Luzon", "CVIS": "Visayas", "CMIN": "Mindanao"}
ISLAND_TO_REGION = {label: code for code, label in REGION_LABELS.items()}
DIRECT_PAIRS = (
    {"pair_key": "CLUZ_CVIS", "island_1": "CLUZ", "island_2": "CVIS", "link_name": "VISLUZ1"},
    {"pair_key": "CVIS_CMIN", "island_1": "CVIS", "island_2": "CMIN", "link_name": "MINVIS1"},
)


@dataclass(frozen=True)
class OutputPaths:
    parquet_path: Path
    csv_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build direct-pair and island-vs-system RTD panels using island prices, "
            "regional demand, congestion events, and HVDC congestion flags."
        )
    )
    parser.add_argument(
        "--price-csv",
        default="data/csv_exports_flat/RTD_ISLAND_PRICE_202512220000_202603230000.csv",
        help="Island-price CSV with TIME_INTERVAL, REGION_NAME, ISLAND_PRICE, and WEIGHT_SUM.",
    )
    parser.add_argument(
        "--demand-csv",
        default="data/csv_exports_flat/RTDREG_20251218_20260320.csv",
        help="RTD regional summaries CSV.",
    )
    parser.add_argument(
        "--congestion-csv",
        default="data/csv_exports_flat/RTDCV_20251218_20260318.csv",
        help="RTD congestion CSV.",
    )
    parser.add_argument(
        "--congestion-map-csv",
        default="data/rtd_congestion/rtd_congestion_resources_with_island_group.csv",
        help="Equipment-to-island-group mapping CSV.",
    )
    parser.add_argument(
        "--hvdc-csv",
        default="data/csv_exports_flat/RTDHS_20251218_20260318.csv",
        help="RTD HVDC schedules CSV.",
    )
    parser.add_argument(
        "--output-root",
        default="data/panels",
        help="Directory for panel parquet files.",
    )
    parser.add_argument(
        "--csv-export-dir",
        default="data/csv_exports_flat",
        help="Directory for flat CSV exports of the panel outputs.",
    )
    return parser.parse_args()


def ensure_unique(df: pd.DataFrame, keys: list[str], frame_name: str) -> None:
    duplicates = df.duplicated(subset=keys, keep=False)
    if duplicates.any():
        sample = df.loc[duplicates, keys].head(10).to_dict("records")
        raise ValueError(f"{frame_name} has duplicate keys on {keys}: {sample}")


def format_time_token(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y%m%d%H%M")


def output_paths(output_root: Path, csv_export_dir: Path, stem: str, frame: pd.DataFrame) -> OutputPaths:
    start = format_time_token(frame["time_interval"].min())
    end = format_time_token(frame["time_interval"].max())
    filename = f"{stem}_{start}_{end}"
    return OutputPaths(
        parquet_path=output_root / f"{filename}.parquet",
        csv_path=csv_export_dir / f"{filename}.csv",
    )


def load_price_wide(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    price = pd.read_csv(path, parse_dates=["TIME_INTERVAL"]).copy()
    price = price.loc[price["REGION_NAME"].isin(REGION_LABELS)].copy()
    ensure_unique(price, ["TIME_INTERVAL", "REGION_NAME"], "price")

    coverage = price.groupby("TIME_INTERVAL", observed=True)["REGION_NAME"].nunique()
    if coverage.empty or int(coverage.min()) != 3 or int(coverage.max()) != 3:
        raise ValueError("Island-price source does not have full 3-island coverage at every interval.")

    price_wide = (
        price.pivot(index="TIME_INTERVAL", columns="REGION_NAME", values="ISLAND_PRICE")
        .rename_axis(index="time_interval", columns=None)
        .sort_index()
    )
    weight_wide = (
        price.pivot(index="TIME_INTERVAL", columns="REGION_NAME", values="WEIGHT_SUM")
        .rename_axis(index="time_interval", columns=None)
        .sort_index()
    )
    return price_wide, weight_wide


def load_demand_wide(path: Path) -> pd.DataFrame:
    demand = pd.read_csv(path, parse_dates=["TIME_INTERVAL"]).copy()
    demand = demand.loc[
        (demand["COMMODITY_TYPE"] == "En") & demand["REGION_NAME"].isin(REGION_LABELS),
        ["TIME_INTERVAL", "REGION_NAME", "MKT_REQT"],
    ].copy()
    ensure_unique(demand, ["TIME_INTERVAL", "REGION_NAME"], "demand")

    demand_wide = (
        demand.pivot(index="TIME_INTERVAL", columns="REGION_NAME", values="MKT_REQT")
        .rename_axis(index="time_interval", columns=None)
        .sort_index()
    )
    if (demand_wide <= 0).any().any():
        bad = demand_wide[(demand_wide <= 0).any(axis=1)].head(10)
        raise ValueError(f"Demand contains non-positive values in sample rows: {bad}")
    return demand_wide


def load_congestion_wide(congestion_path: Path, mapping_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(mapping_path).copy()
    mapping["region_code"] = mapping["island_group"].map(ISLAND_TO_REGION)
    if mapping["region_code"].isna().any():
        missing = sorted(mapping.loc[mapping["region_code"].isna(), "island_group"].unique())
        raise ValueError(f"Unmapped island groups in equipment map: {missing}")

    congestion = pd.read_csv(congestion_path, parse_dates=["TIME_INTERVAL"]).copy()
    congestion = congestion.merge(
        mapping[["resource", "region_code"]],
        left_on="EQUIPMENT_NAME",
        right_on="resource",
        how="left",
    )
    if congestion["region_code"].isna().any():
        missing = sorted(congestion.loc[congestion["region_code"].isna(), "EQUIPMENT_NAME"].unique())
        raise ValueError(f"Unmapped congestion equipment names: {missing[:10]}")

    congestion["pct_excess"] = congestion["PCT_MW"] - 100.0
    grouped = (
        congestion.groupby(["TIME_INTERVAL", "region_code"], observed=True)
        .agg(
            equip_cong_any=("EQUIPMENT_NAME", "size"),
            equip_excess_pct_sum=("pct_excess", "sum"),
        )
        .reset_index()
    )
    grouped["equip_cong_any"] = 1

    any_wide = (
        grouped.pivot(index="TIME_INTERVAL", columns="region_code", values="equip_cong_any")
        .rename_axis(index="time_interval", columns=None)
        .sort_index()
        .fillna(0)
        .astype(int)
    )
    excess_wide = (
        grouped.pivot(index="TIME_INTERVAL", columns="region_code", values="equip_excess_pct_sum")
        .rename_axis(index="time_interval", columns=None)
        .sort_index()
        .fillna(0.0)
    )
    return any_wide, excess_wide


def load_hvdc_wide(path: Path) -> pd.DataFrame:
    hvdc = pd.read_csv(path, parse_dates=["TIME_INTERVAL"]).copy()
    hvdc = hvdc.loc[hvdc["HVDC_NAME"].isin({pair["link_name"] for pair in DIRECT_PAIRS})].copy()
    hvdc["link_congested_any"] = (hvdc["CONGESTION_FLAG"] == "Y").astype(int)

    grouped = (
        hvdc.groupby(["TIME_INTERVAL", "HVDC_NAME"], observed=True)["link_congested_any"]
        .max()
        .reset_index()
    )
    ensure_unique(grouped, ["TIME_INTERVAL", "HVDC_NAME"], "hvdc")

    return (
        grouped.pivot(index="TIME_INTERVAL", columns="HVDC_NAME", values="link_congested_any")
        .rename_axis(index="time_interval", columns=None)
        .sort_index()
    )


def build_island_system_panel(
    price_wide: pd.DataFrame,
    weight_wide: pd.DataFrame,
    demand_wide: pd.DataFrame,
    congestion_any_wide: pd.DataFrame,
    congestion_excess_wide: pd.DataFrame,
) -> pd.DataFrame:
    base = price_wide.join(demand_wide, how="inner", lsuffix="_price", rsuffix="_demand")
    base = base.dropna().copy()

    for region in REGION_LABELS:
        base[f"price_{region}"] = base[f"{region}_price"]
        base[f"demand_{region}"] = base[f"{region}_demand"]

    base["demand_total"] = sum(base[f"demand_{region}"] for region in REGION_LABELS)
    base["price_sys_dw"] = sum(
        base[f"price_{region}"] * base[f"demand_{region}"] for region in REGION_LABELS
    ) / base["demand_total"]

    congestion_any_aligned = congestion_any_wide.reindex(base.index, fill_value=0)
    congestion_excess_aligned = congestion_excess_wide.reindex(base.index, fill_value=0.0)
    weight_aligned = weight_wide.reindex(base.index)

    rows: list[pd.DataFrame] = []
    for region in REGION_LABELS:
        panel = pd.DataFrame(
            {
                "time_interval": base.index,
                "island_code": region,
                "price_island": base[f"price_{region}"].to_numpy(),
                "price_sys_dw": base["price_sys_dw"].to_numpy(),
                "dep_price_minus_sys": (base[f"price_{region}"] - base["price_sys_dw"]).abs().to_numpy(),
                "demand_island": base[f"demand_{region}"].to_numpy(),
                "demand_total": base["demand_total"].to_numpy(),
                "weight_sum_island": weight_aligned[region].to_numpy(),
                "equip_cong_any": congestion_any_aligned[region].to_numpy(),
                "equip_excess_pct_sum": congestion_excess_aligned[region].to_numpy(),
            }
        )
        rows.append(panel)

    result = pd.concat(rows, ignore_index=True)
    result = add_fixed_effects(result)
    ensure_unique(result, ["time_interval", "island_code"], "island_system_panel")
    return result.sort_values(["time_interval", "island_code"]).reset_index(drop=True)


def build_island_congestion_panel(
    price_wide: pd.DataFrame,
    demand_wide: pd.DataFrame,
    congestion_any_wide: pd.DataFrame,
    congestion_excess_wide: pd.DataFrame,
) -> pd.DataFrame:
    base = price_wide.add_suffix("_price").join(demand_wide.add_suffix("_demand"), how="inner").dropna().copy()
    congestion_any_aligned = congestion_any_wide.reindex(base.index, fill_value=0)
    congestion_excess_aligned = congestion_excess_wide.reindex(base.index, fill_value=0.0)

    rows: list[pd.DataFrame] = []
    for region in REGION_LABELS:
        panel = pd.DataFrame(
            {
                "time_interval": base.index,
                "island_1": region,
                "price_island_1": base[f"{region}_price"].to_numpy(),
                "demand_island_1": base[f"{region}_demand"].to_numpy(),
                "equipment_cong_bin_cluz": congestion_any_aligned["CLUZ"].to_numpy(),
                "equipment_cong_bin_cvis": congestion_any_aligned["CVIS"].to_numpy(),
                "equipment_cong_bin_cmin": congestion_any_aligned["CMIN"].to_numpy(),
                "equipment_cong_pct_cluz": congestion_excess_aligned["CLUZ"].to_numpy(),
                "equipment_cong_pct_cvis": congestion_excess_aligned["CVIS"].to_numpy(),
                "equipment_cong_pct_cmin": congestion_excess_aligned["CMIN"].to_numpy(),
            }
        )
        rows.append(panel)

    result = pd.concat(rows, ignore_index=True)
    result = add_fixed_effects(result)
    ensure_unique(result, ["time_interval", "island_1"], "island_congestion_panel")
    return result.sort_values(["time_interval", "island_1"]).reset_index(drop=True)


def build_direct_pair_panel(
    price_wide: pd.DataFrame,
    weight_wide: pd.DataFrame,
    demand_wide: pd.DataFrame,
    congestion_any_wide: pd.DataFrame,
    congestion_excess_wide: pd.DataFrame,
    hvdc_wide: pd.DataFrame,
) -> pd.DataFrame:
    price = price_wide.add_suffix("_price")
    demand = demand_wide.add_suffix("_demand")
    base = price.join(demand, how="inner").dropna().copy()

    for region in REGION_LABELS:
        base[f"price_{region}"] = base[f"{region}_price"]
        base[f"demand_{region}"] = base[f"{region}_demand"]
    base["demand_total"] = sum(base[f"demand_{region}"] for region in REGION_LABELS)

    congestion_any_aligned = congestion_any_wide.reindex(base.index, fill_value=0)
    congestion_excess_aligned = congestion_excess_wide.reindex(base.index, fill_value=0.0)
    weight_aligned = weight_wide.reindex(base.index)
    hvdc_aligned = hvdc_wide.reindex(base.index)

    rows: list[pd.DataFrame] = []
    for pair in DIRECT_PAIRS:
        link = pair["link_name"]
        pair_mask = hvdc_aligned[link].notna()
        pair_index = base.index[pair_mask]
        if len(pair_index) == 0:
            continue

        island_1 = pair["island_1"]
        island_2 = pair["island_2"]
        panel = pd.DataFrame(
            {
                "time_interval": pair_index,
                "pair_key": pair["pair_key"],
                "island_1": island_1,
                "island_2": island_2,
                "price_1": base.loc[pair_index, f"price_{island_1}"].to_numpy(),
                "price_2": base.loc[pair_index, f"price_{island_2}"].to_numpy(),
                "dep_abs_price_gap": (
                    base.loc[pair_index, f"price_{island_1}"] - base.loc[pair_index, f"price_{island_2}"]
                ).abs().to_numpy(),
                "demand_1": base.loc[pair_index, f"demand_{island_1}"].to_numpy(),
                "demand_2": base.loc[pair_index, f"demand_{island_2}"].to_numpy(),
                "demand_total": base.loc[pair_index, "demand_total"].to_numpy(),
                "weight_sum_1": weight_aligned.loc[pair_index, island_1].to_numpy(),
                "weight_sum_2": weight_aligned.loc[pair_index, island_2].to_numpy(),
                "equip_cong_any_1": congestion_any_aligned.loc[pair_index, island_1].to_numpy(),
                "equip_cong_any_2": congestion_any_aligned.loc[pair_index, island_2].to_numpy(),
                "equip_excess_pct_sum_1": congestion_excess_aligned.loc[pair_index, island_1].to_numpy(),
                "equip_excess_pct_sum_2": congestion_excess_aligned.loc[pair_index, island_2].to_numpy(),
                "link_name": link,
                "link_congested_any": hvdc_aligned.loc[pair_index, link].astype(int).to_numpy(),
            }
        )
        rows.append(panel)

    result = pd.concat(rows, ignore_index=True)
    result = add_fixed_effects(result)
    ensure_unique(result, ["time_interval", "pair_key"], "direct_pair_panel")
    return result.sort_values(["time_interval", "pair_key"]).reset_index(drop=True)


def add_fixed_effects(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["fe_month"] = result["time_interval"].dt.strftime("%Y-%m")
    result["fe_week"] = result["time_interval"].dt.strftime("%G-W%V")
    result["fe_day"] = result["time_interval"].dt.strftime("%Y-%m-%d")
    return result


def write_output(frame: pd.DataFrame, paths: OutputPaths) -> None:
    paths.parquet_path.parent.mkdir(parents=True, exist_ok=True)
    paths.csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(paths.parquet_path, index=False)
    frame.to_csv(paths.csv_path, index=False)


def summarize_frame(name: str, frame: pd.DataFrame) -> str:
    return (
        f"{name}: rows={len(frame):,}, "
        f"time_min={frame['time_interval'].min()}, "
        f"time_max={frame['time_interval'].max()}"
    )


def main() -> None:
    args = parse_args()
    price_wide, weight_wide = load_price_wide(Path(args.price_csv))
    demand_wide = load_demand_wide(Path(args.demand_csv))
    congestion_any_wide, congestion_excess_wide = load_congestion_wide(
        Path(args.congestion_csv),
        Path(args.congestion_map_csv),
    )
    hvdc_wide = load_hvdc_wide(Path(args.hvdc_csv))

    island_system_panel = build_island_system_panel(
        price_wide,
        weight_wide,
        demand_wide,
        congestion_any_wide,
        congestion_excess_wide,
    )
    island_congestion_panel = build_island_congestion_panel(
        price_wide,
        demand_wide,
        congestion_any_wide,
        congestion_excess_wide,
    )
    direct_pair_panel = build_direct_pair_panel(
        price_wide,
        weight_wide,
        demand_wide,
        congestion_any_wide,
        congestion_excess_wide,
        hvdc_wide,
    )

    output_root = Path(args.output_root)
    csv_export_dir = Path(args.csv_export_dir)
    island_paths = output_paths(output_root, csv_export_dir, "RTD_ISLAND_SYSTEM_PANEL", island_system_panel)
    island_congestion_paths = output_paths(
        output_root,
        csv_export_dir,
        "RTD_ISLAND_CONGESTION_PANEL",
        island_congestion_panel,
    )
    pair_paths = output_paths(output_root, csv_export_dir, "RTD_DIRECT_PAIR_PANEL", direct_pair_panel)

    write_output(island_system_panel, island_paths)
    write_output(island_congestion_panel, island_congestion_paths)
    write_output(direct_pair_panel, pair_paths)

    print(summarize_frame("island_system_panel", island_system_panel))
    print(f"  parquet: {island_paths.parquet_path}")
    print(f"  csv: {island_paths.csv_path}")
    print(summarize_frame("island_congestion_panel", island_congestion_panel))
    print(f"  parquet: {island_congestion_paths.parquet_path}")
    print(f"  csv: {island_congestion_paths.csv_path}")
    print(summarize_frame("direct_pair_panel", direct_pair_panel))
    print(f"  parquet: {pair_paths.parquet_path}")
    print(f"  csv: {pair_paths.csv_path}")


if __name__ == "__main__":
    main()
