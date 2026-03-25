#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

REGION_LABELS = {"CLUZ": "Luzon", "CVIS": "Visayas", "CMIN": "Mindanao"}
ISLAND_TO_REGION = {label: code for code, label in REGION_LABELS.items()}
CONTROL_COLUMNS = ("LOSSES", "GENERATION", "MKT_IMPORT", "MKT_EXPORT")
DIRECT_PAIRS = (
    {"pair_key": "CLUZ_CVIS", "island_1": "CLUZ", "island_2": "CVIS", "link_name": "VISLUZ1"},
    {"pair_key": "CVIS_CMIN", "island_1": "CVIS", "island_2": "CMIN", "link_name": "MINVIS1"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build cleaned direct-pair and island-vs-system RTD panels from RTD LMP_SMP prices, "
            "RTDREG controls, RTDCV congestion events, and RTDHS link congestion flags."
        )
    )
    parser.add_argument("--price-parquet", help="RTD LMP_SMP parquet path.")
    parser.add_argument("--regional-parquet", help="RTDREG parquet path.")
    parser.add_argument("--congestion-parquet", help="RTDCV parquet path.")
    parser.add_argument("--congestion-map-csv", help="Equipment-to-island mapping CSV.")
    parser.add_argument("--hvdc-parquet", help="RTDHS parquet path.")
    parser.add_argument(
        "--output-root",
        default="data/panels",
        help="Directory for panel parquet files.",
    )
    return parser.parse_args()


TIMESTAMP_TOKEN_RE = re.compile(r"(\d{8,12})")


def latest_matching_file(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} under {root}.")

    def sort_key(path: Path) -> tuple[str, str, str]:
        tokens = TIMESTAMP_TOKEN_RE.findall(path.stem)
        if not tokens:
            return ("", "", path.name)
        if len(tokens) == 1:
            return (tokens[0], tokens[0], path.name)
        return (tokens[-1], tokens[0], path.name)

    return max(matches, key=sort_key)


def resolve_congestion_map(path_arg: str | None) -> Path | None:
    if path_arg:
        return Path(path_arg)
    candidates = (
        Path("data/rtdcv/rtd_congestion_resources_with_island_group.csv"),
        Path("data/rtd_congestion/rtd_congestion_resources_with_island_group.csv"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ensure_unique(df: pd.DataFrame, keys: list[str], frame_name: str) -> None:
    duplicates = df.duplicated(subset=keys, keep=False)
    if duplicates.any():
        sample = df.loc[duplicates, keys].head(10).to_dict("records")
        raise ValueError(f"{frame_name} has duplicate keys on {keys}: {sample}")


def format_time_token(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y%m%d%H%M")


def output_path(output_root: Path, stem: str, frame: pd.DataFrame) -> Path:
    start = format_time_token(frame["time_interval"].min())
    end = format_time_token(frame["time_interval"].max())
    return output_root / f"{stem}_{start}_{end}.parquet"


def load_price_wide(path: Path) -> pd.DataFrame:
    price = pd.read_parquet(path).copy()
    price["TIME_INTERVAL"] = pd.to_datetime(price["TIME_INTERVAL"])
    price = price.loc[price["REGION_NAME"].isin(REGION_LABELS), ["TIME_INTERVAL", "REGION_NAME", "LMP_SMP"]].copy()
    ensure_unique(price, ["TIME_INTERVAL", "REGION_NAME"], "price")

    coverage = price.groupby("TIME_INTERVAL", observed=True)["REGION_NAME"].nunique()
    if coverage.empty or int(coverage.min()) != 3 or int(coverage.max()) != 3:
        raise ValueError("RTD LMP_SMP source does not have full 3-island coverage at every interval.")

    return (
        price.pivot(index="TIME_INTERVAL", columns="REGION_NAME", values="LMP_SMP")
        .rename_axis(index="time_interval", columns=None)
        .sort_index()
    )


def load_regional_inputs(path: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    regional = pd.read_parquet(path).copy()
    regional["TIME_INTERVAL"] = pd.to_datetime(regional["TIME_INTERVAL"])
    regional = regional.loc[
        (regional["COMMODITY_TYPE"] == "En") & regional["REGION_NAME"].isin(REGION_LABELS),
        ["TIME_INTERVAL", "REGION_NAME", "MKT_REQT", *CONTROL_COLUMNS],
    ].copy()
    ensure_unique(regional, ["TIME_INTERVAL", "REGION_NAME"], "regional")

    demand_wide = (
        regional.pivot(index="TIME_INTERVAL", columns="REGION_NAME", values="MKT_REQT")
        .rename_axis(index="time_interval", columns=None)
        .sort_index()
    )
    control_wides = {
        column: (
            regional.pivot(index="TIME_INTERVAL", columns="REGION_NAME", values=column)
            .rename_axis(index="time_interval", columns=None)
            .sort_index()
        )
        for column in CONTROL_COLUMNS
    }
    return demand_wide, control_wides


def load_congestion_wide(congestion_path: Path, mapping_path: Path | None) -> pd.DataFrame:
    congestion = pd.read_parquet(congestion_path).copy()
    congestion["TIME_INTERVAL"] = pd.to_datetime(congestion["TIME_INTERVAL"])
    if congestion.empty:
        return pd.DataFrame(columns=list(REGION_LABELS)).rename_axis(index="time_interval", columns=None)

    if mapping_path is None:
        raise FileNotFoundError(
            "RTDCV has rows but no rtd_congestion_resources_with_island_group.csv was found. "
            "Pass --congestion-map-csv to build nonzero congestion indicators."
        )

    mapping = pd.read_csv(mapping_path).copy()
    mapping["region_code"] = mapping["island_group"].map(ISLAND_TO_REGION)
    if mapping["region_code"].isna().any():
        missing = sorted(mapping.loc[mapping["region_code"].isna(), "island_group"].unique())
        raise ValueError(f"Unmapped island groups in equipment map: {missing}")

    congestion = congestion.merge(
        mapping[["resource", "region_code"]],
        left_on="EQUIPMENT_NAME",
        right_on="resource",
        how="left",
    )
    if congestion["region_code"].isna().any():
        missing = sorted(congestion.loc[congestion["region_code"].isna(), "EQUIPMENT_NAME"].unique())
        raise ValueError(f"Unmapped congestion equipment names: {missing[:10]}")

    grouped = (
        congestion.groupby(["TIME_INTERVAL", "region_code"], observed=True)["EQUIPMENT_NAME"]
        .size()
        .rename("equip_cong_any")
        .reset_index()
    )
    grouped["equip_cong_any"] = 1
    wide = (
        grouped.pivot(index="TIME_INTERVAL", columns="region_code", values="equip_cong_any")
        .rename_axis(index="time_interval", columns=None)
        .sort_index()
        .fillna(0)
        .astype(int)
    )
    for region in REGION_LABELS:
        if region not in wide.columns:
            wide[region] = 0
    return wide[list(REGION_LABELS)]


def load_hvdc_wide(path: Path) -> pd.DataFrame:
    hvdc = pd.read_parquet(path).copy()
    hvdc["TIME_INTERVAL"] = pd.to_datetime(hvdc["TIME_INTERVAL"])
    hvdc = hvdc.loc[hvdc["HVDC_NAME"].isin({pair["link_name"] for pair in DIRECT_PAIRS})].copy()
    hvdc["link_congested_any"] = (hvdc["CONGESTION_FLAG"] == "Y").astype(int)
    grouped = (
        hvdc.groupby(["TIME_INTERVAL", "HVDC_NAME"], observed=True)["link_congested_any"]
        .max()
        .reset_index()
    )
    ensure_unique(grouped, ["TIME_INTERVAL", "HVDC_NAME"], "hvdc")
    wide = (
        grouped.pivot(index="TIME_INTERVAL", columns="HVDC_NAME", values="link_congested_any")
        .rename_axis(index="time_interval", columns=None)
        .sort_index()
    )
    for pair in DIRECT_PAIRS:
        if pair["link_name"] not in wide.columns:
            wide[pair["link_name"]] = pd.NA
    return wide[[pair["link_name"] for pair in DIRECT_PAIRS]]


def add_fixed_effects(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["fe_day"] = result["time_interval"].dt.strftime("%Y-%m-%d")
    return result


def build_island_system_panel(
    price_wide: pd.DataFrame,
    demand_wide: pd.DataFrame,
    control_wides: dict[str, pd.DataFrame],
    congestion_any_wide: pd.DataFrame,
) -> pd.DataFrame:
    base = price_wide.add_suffix("_price").join(demand_wide.add_suffix("_demand"), how="inner").dropna().copy()
    for control_name, control_wide in control_wides.items():
        base = base.join(control_wide.add_suffix(f"_{control_name.lower()}"), how="inner")
    base = base.dropna().copy()

    for region in REGION_LABELS:
        base[f"price_{region}"] = base[f"{region}_price"]
        base[f"demand_{region}"] = base[f"{region}_demand"]

    base["demand_total"] = sum(base[f"demand_{region}"] for region in REGION_LABELS)
    base["price_sys_dw"] = sum(
        base[f"price_{region}"] * base[f"demand_{region}"] for region in REGION_LABELS
    ) / base["demand_total"]

    control_totals = {
        control_name: control_wides[control_name].reindex(base.index).sum(axis=1)
        for control_name in CONTROL_COLUMNS
    }
    congestion_any_aligned = congestion_any_wide.reindex(base.index, fill_value=0)

    rows: list[pd.DataFrame] = []
    for region in REGION_LABELS:
        panel = pd.DataFrame(
            {
                "time_interval": base.index,
                "island_code": region,
                "price_island": base[f"price_{region}"].to_numpy(),
                "price_sys_dw": base["price_sys_dw"].to_numpy(),
                "dep_price_minus_sys": (base[f"price_{region}"] - base["price_sys_dw"]).abs().to_numpy(),
                "equip_cong_any": congestion_any_aligned[region].to_numpy(),
            }
        )
        for control_name in CONTROL_COLUMNS:
            control_key = control_name.lower()
            control_aligned = control_wides[control_name].reindex(base.index)
            panel[f"{control_key}_island"] = control_aligned[region].to_numpy()
            panel[f"{control_key}_total"] = control_totals[control_name].to_numpy()
        rows.append(panel)

    result = pd.concat(rows, ignore_index=True)
    result = add_fixed_effects(result)
    ensure_unique(result, ["time_interval", "island_code"], "island_system_panel")
    return result.sort_values(["time_interval", "island_code"]).reset_index(drop=True)


def build_direct_pair_panel(
    price_wide: pd.DataFrame,
    control_wides: dict[str, pd.DataFrame],
    congestion_any_wide: pd.DataFrame,
    hvdc_wide: pd.DataFrame,
) -> pd.DataFrame:
    base = price_wide.add_suffix("_price").copy()
    for control_name, control_wide in control_wides.items():
        base = base.join(control_wide.add_suffix(f"_{control_name.lower()}"), how="inner")
    base = base.dropna().copy()

    for region in REGION_LABELS:
        base[f"price_{region}"] = base[f"{region}_price"]

    control_totals = {
        control_name: control_wides[control_name].reindex(base.index).sum(axis=1)
        for control_name in CONTROL_COLUMNS
    }
    congestion_any_aligned = congestion_any_wide.reindex(base.index, fill_value=0)
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
                "equip_cong_any_1": congestion_any_aligned.loc[pair_index, island_1].to_numpy(),
                "equip_cong_any_2": congestion_any_aligned.loc[pair_index, island_2].to_numpy(),
                "link_name": link,
                "link_congested_any": hvdc_aligned.loc[pair_index, link].astype(int).to_numpy(),
            }
        )
        for control_name in CONTROL_COLUMNS:
            control_key = control_name.lower()
            control_aligned = control_wides[control_name].reindex(base.index)
            panel[f"{control_key}_1"] = control_aligned.loc[pair_index, island_1].to_numpy()
            panel[f"{control_key}_2"] = control_aligned.loc[pair_index, island_2].to_numpy()
            panel[f"{control_key}_total"] = control_totals[control_name].loc[pair_index].to_numpy()
        rows.append(panel)

    result = pd.concat(rows, ignore_index=True)
    result = add_fixed_effects(result)
    ensure_unique(result, ["time_interval", "pair_key"], "direct_pair_panel")
    return result.sort_values(["time_interval", "pair_key"]).reset_index(drop=True)


def write_output(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def summarize_frame(name: str, frame: pd.DataFrame) -> str:
    return (
        f"{name}: rows={len(frame):,}, "
        f"time_min={frame['time_interval'].min()}, "
        f"time_max={frame['time_interval'].max()}"
    )


def main() -> None:
    args = parse_args()
    price_path = Path(args.price_parquet) if args.price_parquet else latest_matching_file(
        Path("data/rtd/combined"),
        "RTD_LMP_SMP_*.parquet",
    )
    regional_path = Path(args.regional_parquet) if args.regional_parquet else latest_matching_file(
        Path("data/rtdreg/combined"),
        "RTDREG_*.parquet",
    )
    congestion_path = Path(args.congestion_parquet) if args.congestion_parquet else latest_matching_file(
        Path("data/rtdcv/combined"),
        "RTDCV_*.parquet",
    )
    hvdc_path = Path(args.hvdc_parquet) if args.hvdc_parquet else latest_matching_file(
        Path("data/rtdhs/combined"),
        "RTDHS_*.parquet",
    )
    congestion_map_path = resolve_congestion_map(args.congestion_map_csv)

    price_wide = load_price_wide(price_path)
    demand_wide, control_wides = load_regional_inputs(regional_path)
    congestion_any_wide = load_congestion_wide(congestion_path, congestion_map_path)
    hvdc_wide = load_hvdc_wide(hvdc_path)

    island_system_panel = build_island_system_panel(
        price_wide,
        demand_wide,
        control_wides,
        congestion_any_wide,
    )
    direct_pair_panel = build_direct_pair_panel(
        price_wide,
        control_wides,
        congestion_any_wide,
        hvdc_wide,
    )

    output_root = Path(args.output_root)
    island_path = output_path(output_root, "RTD_ISLAND_SYSTEM_PANEL", island_system_panel)
    pair_path = output_path(output_root, "RTD_DIRECT_PAIR_PANEL", direct_pair_panel)
    write_output(island_system_panel, island_path)
    write_output(direct_pair_panel, pair_path)

    print(summarize_frame("island_system_panel", island_system_panel))
    print(f"  parquet: {island_path}")
    print(summarize_frame("direct_pair_panel", direct_pair_panel))
    print(f"  parquet: {pair_path}")


if __name__ == "__main__":
    main()
