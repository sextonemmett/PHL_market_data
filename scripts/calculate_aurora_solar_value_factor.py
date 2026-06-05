#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT_DIR = Path("data/aurora/prices")
DEFAULT_OUTPUT_CSV = Path("output/aurora_solar_value_factor/aurora_solar_value_factor_luzon.csv")
DEFAULT_REGION = "Luzon"
DEFAULT_TECHNOLOGY_GROUP = "Solar"
INTERVAL_HOURS = 0.5

TECH_PRICE_SOURCE = "aurora_technology_generation_weighted_price"
SYSTEM_PRICE_SOURCE = "luzon_system_wholesale_price"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Luzon solar capture price and value factor from Aurora 30-minute "
            "technology and system price CSV exports."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing phl_technology_30M_YYYY.csv and phl_system_30M_YYYY.csv files.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="CSV path for the value factor summary.",
    )
    parser.add_argument("--region", default=DEFAULT_REGION, help="Aurora region to calculate.")
    parser.add_argument(
        "--technology-group",
        default=DEFAULT_TECHNOLOGY_GROUP,
        help="Technology group to use as the solar generation profile.",
    )
    return parser.parse_args()


def year_from_path(path: Path) -> str:
    match = re.search(r"_(\d{4})\.csv$", path.name)
    if not match:
        raise ValueError(f"Could not infer year from {path.name}.")
    return match.group(1)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_interval_frame(input_dir: Path, region: str, technology_group: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    technology_paths = sorted(input_dir.glob("phl_technology_30M_*.csv"))
    if not technology_paths:
        raise FileNotFoundError(f"No technology CSVs found in {input_dir}.")

    for technology_path in technology_paths:
        file_year = year_from_path(technology_path)
        system_path = input_dir / f"phl_system_30M_{file_year}.csv"
        if not system_path.exists():
            raise FileNotFoundError(f"Expected system CSV for {file_year}: {system_path}")

        technology = pd.read_csv(technology_path)
        system = pd.read_csv(system_path)

        solar = technology.loc[
            (technology["region"] == region) & (technology["technology group"] == technology_group)
        ].copy()
        market = system.loc[system["region"] == region].copy()

        if solar.empty:
            raise ValueError(f"No {technology_group} rows found for {region} in {technology_path}.")
        if market.empty:
            raise ValueError(f"No system price rows found for {region} in {system_path}.")

        solar["generation_mw"] = numeric(solar["average generation, mw"])
        solar["technology_generation_weighted_price_php_per_mwh"] = numeric(
            solar["generation weighted average price, php/mwh"]
        )
        market["system_wholesale_price_php_per_mwh"] = numeric(market["wholesale price, php/mwh"])

        merged = solar.merge(
            market[
                [
                    "date_time_utc",
                    "region",
                    "system_wholesale_price_php_per_mwh",
                    "load, mw",
                ]
            ],
            on=["date_time_utc", "region"],
            how="left",
            validate="one_to_one",
        )
        if merged["system_wholesale_price_php_per_mwh"].isna().any():
            missing = int(merged["system_wholesale_price_php_per_mwh"].isna().sum())
            raise ValueError(f"{missing} {technology_group} intervals did not match a system price in {file_year}.")

        merged["file_year"] = file_year
        merged["date_time_utc"] = pd.to_datetime(merged["date_time_utc"], errors="raise")
        merged["local_date_time"] = pd.to_datetime(merged["local_date_time"], errors="raise")
        merged["load_mw"] = numeric(merged["load, mw"])
        frames.append(
            merged[
                [
                    "file_year",
                    "date_time_utc",
                    "local_date_time",
                    "region",
                    "technology group",
                    "technology subgroup",
                    "generation_mw",
                    "technology_generation_weighted_price_php_per_mwh",
                    "system_wholesale_price_php_per_mwh",
                    "load_mw",
                ]
            ]
        )

    return pd.concat(frames, ignore_index=True).sort_values(["date_time_utc", "region"]).reset_index(drop=True)


def weighted_capture_price(frame: pd.DataFrame, price_column: str) -> tuple[float, float, int]:
    clean = frame.loc[frame["generation_mw"].notna() & frame[price_column].notna()].copy()
    if clean.empty:
        return float("nan"), float("nan"), 0

    generation_mwh = clean["generation_mw"] * INTERVAL_HOURS
    denominator = generation_mwh.sum()
    if denominator <= 0:
        return float("nan"), float(denominator), len(clean)

    capture_price = float((generation_mwh * clean[price_column]).sum() / denominator)
    return capture_price, float(denominator), len(clean)


def summarize_period(frame: pd.DataFrame, period: str) -> list[dict[str, object]]:
    average_market_price = float(frame["system_wholesale_price_php_per_mwh"].mean())
    rows: list[dict[str, object]] = []

    for price_source, price_column in (
        (SYSTEM_PRICE_SOURCE, "system_wholesale_price_php_per_mwh"),
        (TECH_PRICE_SOURCE, "technology_generation_weighted_price_php_per_mwh"),
    ):
        capture_price, generation_mwh, intervals_with_price = weighted_capture_price(frame, price_column)
        value_factor = capture_price / average_market_price if average_market_price else float("nan")

        rows.append(
            {
                "period": period,
                "region": frame["region"].iat[0],
                "technology_group": frame["technology group"].iat[0],
                "technology_subgroup": frame["technology subgroup"].iat[0],
                "price_source": price_source,
                "start_local": frame["local_date_time"].min(),
                "end_local": frame["local_date_time"].max(),
                "interval_hours": INTERVAL_HOURS,
                "interval_count": len(frame),
                "intervals_with_price": intervals_with_price,
                "intervals_with_positive_generation": int((frame["generation_mw"] > 0).sum()),
                "generation_mwh_in_capture_calc": generation_mwh,
                "capture_price_php_per_mwh": capture_price,
                "average_market_price_php_per_mwh": average_market_price,
                "value_factor": value_factor,
            }
        )

    return rows


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for file_year, group in frame.groupby("file_year", sort=True):
        rows.extend(summarize_period(group, f"source_file_{file_year}"))
    rows.extend(summarize_period(frame, "all_available"))

    summary = pd.DataFrame(rows)
    numeric_columns = [
        "generation_mwh_in_capture_calc",
        "capture_price_php_per_mwh",
        "average_market_price_php_per_mwh",
        "value_factor",
    ]
    summary[numeric_columns] = summary[numeric_columns].replace([np.inf, -np.inf], np.nan).round(6)
    return summary


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    interval_frame = load_interval_frame(input_dir, args.region, args.technology_group)
    summary = build_summary(interval_frame)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    primary = summary.loc[
        (summary["period"] == "all_available") & (summary["price_source"] == SYSTEM_PRICE_SOURCE)
    ].iloc[0]
    print(f"Wrote {output_csv}")
    print(
        "Primary all-available Luzon solar value factor: "
        f"{primary['value_factor']:.6f} "
        f"(capture price {primary['capture_price_php_per_mwh']:.2f} PHP/MWh; "
        f"average market price {primary['average_market_price_php_per_mwh']:.2f} PHP/MWh)"
    )


if __name__ == "__main__":
    main()
