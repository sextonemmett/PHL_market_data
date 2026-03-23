#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

DATASET_CODE = "RTD_ISLAND_PRICE"
DEFAULT_PARQUET = Path("data/rtd_island_prices/combined/RTD_ISLAND_PRICE_202512220000_202603230000.parquet")
DEFAULT_QC = Path("data/rtd_island_prices/qc/rtd_island_price_qc_202512220000_202603230000.csv")
DEFAULT_REPORT = Path("reports/analysis/rtd_island_price_visual_check.md")
DEFAULT_ASSETS_DIR = Path("reports/analysis/rtd_island_price_visual_assets")
REGION_LABELS = {"CLUZ": "Luzon", "CVIS": "Visayas", "CMIN": "Mindanao"}
REGION_COLORS = {"CLUZ": "#7f2704", "CVIS": "#d95f0e", "CMIN": "#1f78b4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate verification visuals for the concatenated RTD island-price file."
    )
    parser.add_argument(
        "--parquet",
        default=str(DEFAULT_PARQUET),
        help="Combined RTD island-price parquet file.",
    )
    parser.add_argument(
        "--qc",
        default=str(DEFAULT_QC),
        help="Hourly QC manifest for the RTD island-price backfill.",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--assets-dir",
        default=str(DEFAULT_ASSETS_DIR),
        help="Directory for generated PNGs.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def apply_date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def slot_label(slot: int) -> str:
    hour = slot // 12
    minute = (slot % 12) * 5
    return f"{hour:02d}:{minute:02d}"


def load_frames(parquet_path: Path, qc_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(parquet_path).copy()
    df["TIME_INTERVAL"] = pd.to_datetime(df["TIME_INTERVAL"])
    df["RUN_DATE"] = df["TIME_INTERVAL"].dt.normalize()
    df["SLOT"] = df["TIME_INTERVAL"].dt.hour * 12 + df["TIME_INTERVAL"].dt.minute // 5

    qc = pd.read_csv(qc_path).copy()
    qc["file_time"] = pd.to_datetime(qc["file_token"], format="%Y%m%d%H%M")
    qc["RUN_DATE"] = qc["file_time"].dt.normalize()
    qc["hour"] = qc["file_time"].dt.hour
    return df, qc


def plot_daily_region_prices(df: pd.DataFrame, path: Path) -> None:
    daily = (
        df.groupby(["RUN_DATE", "REGION_NAME"], observed=True)
        .agg(
            median_price=("ISLAND_PRICE", "median"),
            p10_price=("ISLAND_PRICE", lambda s: s.quantile(0.10)),
            p90_price=("ISLAND_PRICE", lambda s: s.quantile(0.90)),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5.2))
    for region in sorted(daily["REGION_NAME"].unique()):
        subset = daily.loc[daily["REGION_NAME"] == region].sort_values("RUN_DATE")
        color = REGION_COLORS[region]
        ax.plot(
            subset["RUN_DATE"],
            subset["median_price"],
            color=color,
            linewidth=1.8,
            label=REGION_LABELS.get(region, region),
        )
        ax.fill_between(
            subset["RUN_DATE"],
            subset["p10_price"],
            subset["p90_price"],
            color=color,
            alpha=0.10,
        )
    ax.set_title("Daily Island Price by Region")
    ax.set_ylabel("Schedule-weighted price")
    apply_date_axis(ax)
    ax.legend(frameon=False, ncol=3)
    save_figure(fig, path)


def plot_intraday_profile(df: pd.DataFrame, path: Path) -> None:
    intraday = (
        df.groupby(["SLOT", "REGION_NAME"], observed=True)
        .agg(
            median_price=("ISLAND_PRICE", "median"),
            p10_price=("ISLAND_PRICE", lambda s: s.quantile(0.10)),
            p90_price=("ISLAND_PRICE", lambda s: s.quantile(0.90)),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5.2))
    for region in sorted(intraday["REGION_NAME"].unique()):
        subset = intraday.loc[intraday["REGION_NAME"] == region].sort_values("SLOT")
        color = REGION_COLORS[region]
        ax.plot(
            subset["SLOT"],
            subset["median_price"],
            color=color,
            linewidth=1.8,
            label=REGION_LABELS.get(region, region),
        )
        ax.fill_between(
            subset["SLOT"],
            subset["p10_price"],
            subset["p90_price"],
            color=color,
            alpha=0.10,
        )

    ticks = np.arange(0, 288, 24)
    ax.set_xticks(ticks)
    ax.set_xticklabels([slot_label(slot) for slot in ticks])
    ax.set_xlim(0, 287)
    ax.set_title("Median Intraday Island Price Profile")
    ax.set_xlabel("5-minute interval")
    ax.set_ylabel("Schedule-weighted price")
    ax.legend(frameon=False, ncol=3)
    save_figure(fig, path)


def plot_daily_weight_sum(df: pd.DataFrame, path: Path) -> None:
    daily = (
        df.groupby(["RUN_DATE", "REGION_NAME"], observed=True)
        .agg(
            median_weight=("WEIGHT_SUM", "median"),
            p10_weight=("WEIGHT_SUM", lambda s: s.quantile(0.10)),
            p90_weight=("WEIGHT_SUM", lambda s: s.quantile(0.90)),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5.2))
    for region in sorted(daily["REGION_NAME"].unique()):
        subset = daily.loc[daily["REGION_NAME"] == region].sort_values("RUN_DATE")
        color = REGION_COLORS[region]
        ax.plot(
            subset["RUN_DATE"],
            subset["median_weight"],
            color=color,
            linewidth=1.8,
            label=REGION_LABELS.get(region, region),
        )
        ax.fill_between(
            subset["RUN_DATE"],
            subset["p10_weight"],
            subset["p90_weight"],
            color=color,
            alpha=0.10,
        )
    ax.set_title("Daily Schedule-Weight Mass by Region")
    ax.set_ylabel("Sum of abs(SCHED_MW)")
    apply_date_axis(ax)
    ax.legend(frameon=False, ncol=3)
    save_figure(fig, path)


def plot_hourly_completeness(qc: pd.DataFrame, path: Path) -> None:
    heatmap = qc.pivot(index="RUN_DATE", columns="hour", values="interval_count").sort_index()
    dates = pd.to_datetime(heatmap.index)
    values = heatmap.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, max(4.8, 0.18 * len(heatmap.index))))
    cmap = LinearSegmentedColormap.from_list(
        "interval_count",
        ["#b2182b", "#fddbc7", "#f7f7f7", "#d9f0d3", "#1a9850"],
    )
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=12)

    ax.set_title("Hourly File Completeness Heatmap")
    ax.set_xlabel("File hour")
    ax.set_ylabel("Run date")
    ax.set_xticks(np.arange(24))
    ax.set_xticklabels([f"{hour:02d}" for hour in range(24)])

    tick_positions = np.linspace(0, len(dates) - 1, min(10, len(dates))).astype(int)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([dates[pos].strftime("%Y-%m-%d") for pos in tick_positions])

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("5-minute intervals present in hourly file")
    save_figure(fig, path)


def build_summary(df: pd.DataFrame, qc: pd.DataFrame) -> dict[str, object]:
    partial_hours = qc.loc[(qc["status"] == "ok") & (qc["output_row_count"] != 36)].copy()
    warning_hours = qc.loc[qc["status"] != "ok"].copy()
    return {
        "rows": len(df),
        "regions": ", ".join(sorted(df["REGION_NAME"].unique())),
        "time_min": df["TIME_INTERVAL"].min(),
        "time_max": df["TIME_INTERVAL"].max(),
        "min_price": float(df["ISLAND_PRICE"].min()),
        "median_price": float(df["ISLAND_PRICE"].median()),
        "max_price": float(df["ISLAND_PRICE"].max()),
        "min_weight": float(df["WEIGHT_SUM"].min()),
        "median_weight": float(df["WEIGHT_SUM"].median()),
        "max_weight": float(df["WEIGHT_SUM"].max()),
        "ok_hours": int((qc["status"] == "ok").sum()),
        "warning_hours": int((qc["status"] != "ok").sum()),
        "partial_hours": int(len(partial_hours)),
        "warning_details": warning_hours[
            ["file_token", "status", "interval_count", "warnings"]
        ].to_dict("records"),
        "partial_details": partial_hours[
            ["file_token", "interval_count", "min_interval", "max_interval"]
        ].to_dict("records"),
    }


def render_report(report_path: Path, assets_dir: Path, summary: dict[str, object]) -> None:
    assets_rel = assets_dir.relative_to(report_path.parent)

    warning_lines = []
    for row in summary["warning_details"]:
        warning_lines.append(
            f"| {row['file_token']} | {row['status']} | {row['interval_count']} | {row['warnings']} |"
        )
    if not warning_lines:
        warning_lines.append("| (none) | ok | 12 | |")

    partial_lines = []
    for row in summary["partial_details"]:
        partial_lines.append(
            f"| {row['file_token']} | {row['interval_count']} | {row['min_interval']} | {row['max_interval']} |"
        )
    if not partial_lines:
        partial_lines.append("| (none) | 12 | | |")

    report = f"""# RTD Island Price Visual Check

This report provides a quick visual QC pass for the concatenated `{DATASET_CODE}` file.

## Overview

| metric | value |
|--------|-------|
| rows | {summary['rows']:,} |
| regions | {summary['regions']} |
| min interval | {summary['time_min']} |
| max interval | {summary['time_max']} |
| median island price | {summary['median_price']:.2f} |
| min island price | {summary['min_price']:.2f} |
| max island price | {summary['max_price']:.2f} |
| median weight sum | {summary['median_weight']:.2f} |
| min weight sum | {summary['min_weight']:.2f} |
| max weight sum | {summary['max_weight']:.2f} |
| ok hourly files | {summary['ok_hours']:,} |
| warning hourly files | {summary['warning_hours']:,} |
| partial but non-empty hourly files | {summary['partial_hours']:,} |

## Warning Hours

| file_token | status | interval_count | warnings |
|------------|--------|----------------|----------|
{chr(10).join(warning_lines)}

## Partial Hours Included

| file_token | interval_count | min_interval | max_interval |
|------------|----------------|--------------|--------------|
{chr(10).join(partial_lines)}

## Visuals

### Daily Price Trend by Region

![Daily island price trend]({assets_rel}/daily_region_prices.png)

### Median Intraday Price Profile

![Median intraday island price]({assets_rel}/intraday_profile.png)

### Daily Schedule-Weight Mass

![Daily schedule weight mass]({assets_rel}/daily_weight_sum.png)

### Hourly File Completeness

![Hourly file completeness heatmap]({assets_rel}/hourly_completeness_heatmap.png)
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    parquet_path = Path(args.parquet)
    qc_path = Path(args.qc)
    report_path = Path(args.report_path)
    assets_dir = Path(args.assets_dir)

    configure_matplotlib()
    df, qc = load_frames(parquet_path, qc_path)

    plot_daily_region_prices(df, assets_dir / "daily_region_prices.png")
    plot_intraday_profile(df, assets_dir / "intraday_profile.png")
    plot_daily_weight_sum(df, assets_dir / "daily_weight_sum.png")
    plot_hourly_completeness(qc, assets_dir / "hourly_completeness_heatmap.png")

    summary = build_summary(df, qc)
    render_report(report_path, assets_dir, summary)

    print(f"Report written to {report_path}")
    print(f"PNG charts written to {assets_dir}")


if __name__ == "__main__":
    main()
