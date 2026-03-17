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

EXPECTED_INTERVALS_PER_DAY = 288
DATASETS = {
    "reserve": {
        "parquet": Path("data/mp_reserve/combined/MP_RESERVE_20251216_20260316.parquet"),
        "qc": Path("data/mp_reserve/qc/mp_reserve_qc_20251216_20260316.csv"),
        "title": "RTD Reserve Market Clearing Price",
    },
    "mp": {
        "parquet": Path("data/mp/combined/MP_20251216_20260316.parquet"),
        "qc": Path("data/mp/qc/mp_qc_20251216_20260316.csv"),
        "title": "RTD Market Clearing Price",
    },
}
STATUS_COLORS = {"ok": "#2f855a", "warning": "#dd6b20", "error": "#c53030"}
DATASET_COLORS = {"reserve": "#1f4e79", "mp": "#8b4513"}
RESERVE_PRODUCT_LABELS = {
    "Dr": "Dr\nDelayed Contingency Raise",
    "Fr": "Fr\nFast Contingency Raise",
    "Rd": "Rd\nRegulation Down",
    "Ru": "Ru\nRegulation Up",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate QC summary tables and PNG charts for the combined MP and reserve Parquet files."
    )
    parser.add_argument(
        "--output-root",
        default="reports/qc",
        help="Directory for generated summary CSVs and PNGs.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def load_qc_manifest(path: Path) -> pd.DataFrame:
    qc = pd.read_csv(path)
    qc["file_date"] = pd.to_datetime(qc["file_date"])
    return qc


def quantile_agg(q: float):
    return lambda series: series.quantile(q)


def build_dataset_summary(name: str, parquet_path: Path, qc_path: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(parquet_path).copy()
    df["RUN_DATE"] = df["RUN_TIME"].dt.normalize()
    qc = load_qc_manifest(qc_path)

    daily = (
        df.groupby("RUN_DATE", observed=True)
        .agg(
            rows=("RUN_TIME", "size"),
            intervals=("TIME_INTERVAL", "nunique"),
            resources=("RESOURCE_NAME", "nunique"),
            mean_price=("MARGINAL_PRICE", "mean"),
            median_price=("MARGINAL_PRICE", "median"),
            min_price=("MARGINAL_PRICE", "min"),
            p10_price=("MARGINAL_PRICE", quantile_agg(0.10)),
            p90_price=("MARGINAL_PRICE", quantile_agg(0.90)),
            max_price=("MARGINAL_PRICE", "max"),
        )
        .reset_index()
    )
    daily["pct_complete"] = daily["intervals"] / EXPECTED_INTERVALS_PER_DAY * 100.0
    daily = daily.merge(
        qc[
            [
                "file_date",
                "status",
                "missing_intervals",
                "warnings",
                "error",
                "http_status",
            ]
        ],
        left_on="RUN_DATE",
        right_on="file_date",
        how="left",
    ).drop(columns=["file_date"])
    daily["dataset"] = name

    region_daily = (
        df.groupby(["RUN_DATE", "REGION_NAME"], observed=True)
        .agg(
            rows=("RUN_TIME", "size"),
            intervals=("TIME_INTERVAL", "nunique"),
            resources=("RESOURCE_NAME", "nunique"),
            mean_price=("MARGINAL_PRICE", "mean"),
        )
        .reset_index()
    )
    region_daily["pct_complete"] = region_daily["intervals"] / EXPECTED_INTERVALS_PER_DAY * 100.0
    region_daily["dataset"] = name

    region_summary = (
        df.groupby("REGION_NAME", observed=True)
        .agg(
            rows=("REGION_NAME", "size"),
            resources=("RESOURCE_NAME", "nunique"),
            intervals=("TIME_INTERVAL", "nunique"),
            avg_price=("MARGINAL_PRICE", "mean"),
            p95_price=("MARGINAL_PRICE", quantile_agg(0.95)),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    region_summary["dataset"] = name

    commodity_summary = (
        df.groupby("COMMODITY_TYPE", observed=True)
        .agg(
            rows=("COMMODITY_TYPE", "size"),
            avg_price=("MARGINAL_PRICE", "mean"),
            median_price=("MARGINAL_PRICE", "median"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    commodity_summary["dataset"] = name

    hourly = (
        df.assign(hour=df["RUN_TIME"].dt.hour)
        .groupby("hour", observed=True)
        .agg(
            rows=("RUN_TIME", "size"),
            mean_price=("MARGINAL_PRICE", "mean"),
            median_price=("MARGINAL_PRICE", "median"),
        )
        .reset_index()
    )
    hourly["dataset"] = name

    resource_summary = (
        df.groupby("RESOURCE_NAME", observed=True)
        .agg(
            rows=("RESOURCE_NAME", "size"),
            active_days=("RUN_DATE", "nunique"),
            regions=("REGION_NAME", "nunique"),
            avg_price=("MARGINAL_PRICE", "mean"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    resource_summary["dataset"] = name

    return {
        "daily": daily,
        "region_daily": region_daily,
        "region_summary": region_summary,
        "commodity_summary": commodity_summary,
        "hourly": hourly,
        "resource_summary": resource_summary,
    }


def save_csvs(summaries: dict[str, dict[str, pd.DataFrame]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, dataset_summary in summaries.items():
        for summary_name, frame in dataset_summary.items():
            frame.to_csv(output_dir / f"{dataset_name}_{summary_name}.csv", index=False)

    reserve_resources = set(summaries["reserve"]["resource_summary"]["RESOURCE_NAME"].astype(str))
    mp_resources = set(summaries["mp"]["resource_summary"]["RESOURCE_NAME"].astype(str))
    overlap = pd.DataFrame(
        {
            "bucket": ["shared", "reserve_only", "mp_only"],
            "resource_count": [
                len(reserve_resources & mp_resources),
                len(reserve_resources - mp_resources),
                len(mp_resources - reserve_resources),
            ],
        }
    )
    overlap.to_csv(output_dir / "resource_overlap_summary.csv", index=False)


def apply_date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_daily_completeness(summaries: dict[str, dict[str, pd.DataFrame]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, dataset_name in zip(axes, ["reserve", "mp"]):
        daily = summaries[dataset_name]["daily"].sort_values("RUN_DATE")
        color = DATASET_COLORS[dataset_name]
        ax.plot(daily["RUN_DATE"], daily["pct_complete"], color=color, linewidth=2, label="Interval completeness")
        ax.scatter(
            daily["RUN_DATE"],
            daily["pct_complete"],
            s=18,
            c=daily["status"].map(STATUS_COLORS).fillna("#718096"),
            zorder=3,
            label="QC status",
        )
        ax.axhline(100, color="#718096", linestyle="--", linewidth=1)
        ax.set_ylim(75, 102)
        ax.set_ylabel("% of 288 Intervals")
        ax.set_title(f"{DATASETS[dataset_name]['title']}: Daily Completeness")
        rows_ax = ax.twinx()
        rows_ax.plot(daily["RUN_DATE"], daily["rows"], color="#4a5568", alpha=0.25, linewidth=1.5)
        rows_ax.set_ylabel("Rows / Day", color="#4a5568")
        rows_ax.tick_params(axis="y", colors="#4a5568")
        apply_date_axis(ax)

    handles = [
        plt.Line2D([], [], color=DATASET_COLORS["reserve"], linewidth=2, label="Completeness line"),
        plt.Line2D([], [], linestyle="", marker="o", color=STATUS_COLORS["ok"], label="ok"),
        plt.Line2D([], [], linestyle="", marker="o", color=STATUS_COLORS["warning"], label="warning"),
        plt.Line2D([], [], linestyle="", marker="o", color=STATUS_COLORS["error"], label="error"),
    ]
    axes[0].legend(handles=handles, loc="lower left", ncol=4, frameon=False)
    save_figure(fig, output_path)


def plot_region_heatmap(dataset_name: str, region_daily: pd.DataFrame, output_path: Path) -> None:
    pivot = (
        region_daily.pivot(index="REGION_NAME", columns="RUN_DATE", values="pct_complete")
        .sort_index()
        .astype(float)
    )
    dates = pd.to_datetime(pivot.columns)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    image = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    tick_positions = np.linspace(0, len(dates) - 1, min(10, len(dates)), dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([dates[pos].strftime("%Y-%m-%d") for pos in tick_positions], rotation=45, ha="right")
    ax.set_title(f"{DATASETS[dataset_name]['title']}: Regional Completeness Heatmap")
    ax.set_xlabel("Run Date")
    ax.set_ylabel("Region")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("% of 288 Intervals")
    save_figure(fig, output_path)


def plot_daily_price_trends(dataset_name: str, daily: pd.DataFrame, output_path: Path) -> None:
    daily = daily.sort_values("RUN_DATE")
    color = DATASET_COLORS[dataset_name]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.fill_between(
        daily["RUN_DATE"],
        daily["p10_price"],
        daily["p90_price"],
        color=color,
        alpha=0.18,
        label="10th-90th percentile",
    )
    ax.plot(daily["RUN_DATE"], daily["mean_price"], color=color, linewidth=2, label="Mean price")
    ax.plot(daily["RUN_DATE"], daily["median_price"], color="#2d3748", linewidth=1.5, label="Median price")
    ax.set_title(f"{DATASETS[dataset_name]['title']}: Daily Price Trend")
    ax.set_ylabel("Marginal Price (PHP/MWh)")
    apply_date_axis(ax)
    ax.legend(frameon=False, ncol=3)
    save_figure(fig, output_path)


def plot_hourly_profile(summaries: dict[str, dict[str, pd.DataFrame]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for ax, metric in zip(axes, ["mean_price", "rows"]):
        for dataset_name in ["reserve", "mp"]:
            hourly = summaries[dataset_name]["hourly"].sort_values("hour")
            ax.plot(hourly["hour"], hourly[metric], linewidth=2, label=dataset_name, color=DATASET_COLORS[dataset_name])
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("Hour of Day")
        ax.set_title("Hourly Mean Price" if metric == "mean_price" else "Hourly Row Volume")
        ax.set_ylabel("Price (PHP/MWh)" if metric == "mean_price" else "Rows")
    axes[0].legend(frameon=False)
    save_figure(fig, output_path)


def plot_resource_overlap(summaries: dict[str, dict[str, pd.DataFrame]], output_path: Path) -> None:
    reserve_resources = set(summaries["reserve"]["resource_summary"]["RESOURCE_NAME"].astype(str))
    mp_resources = set(summaries["mp"]["resource_summary"]["RESOURCE_NAME"].astype(str))
    overlap = {
        "shared": len(reserve_resources & mp_resources),
        "reserve_only": len(reserve_resources - mp_resources),
        "mp_only": len(mp_resources - reserve_resources),
    }

    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = list(overlap.keys())
    values = [overlap[label] for label in labels]
    colors = ["#4c78a8", "#72b7b2", "#f58518"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Resource Overlap Between Reserve and MP")
    ax.set_ylabel("Unique Resources")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, str(value), ha="center", va="bottom")
    save_figure(fig, output_path)


def plot_reserve_commodity_summary(commodity_summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = [RESERVE_PRODUCT_LABELS.get(code, code) for code in commodity_summary["COMMODITY_TYPE"].astype(str)]
    axes[0].bar(labels, commodity_summary["rows"], color="#4c78a8")
    axes[0].set_title("Reserve Commodity Row Counts")
    axes[0].set_ylabel("Rows")
    axes[1].bar(labels, commodity_summary["avg_price"], color="#f58518")
    axes[1].set_title("Reserve Commodity Average Price")
    axes[1].set_ylabel("Average Price (PHP/MWh)")
    save_figure(fig, output_path)


def main() -> int:
    args = parse_args()
    configure_matplotlib()

    output_root = Path(args.output_root)
    summary_dir = output_root / "summaries"
    png_dir = output_root / "png"

    summaries = {
        dataset_name: build_dataset_summary(dataset_name, config["parquet"], config["qc"])
        for dataset_name, config in DATASETS.items()
    }

    save_csvs(summaries, summary_dir)
    plot_daily_completeness(summaries, png_dir / "daily_completeness.png")
    plot_region_heatmap("reserve", summaries["reserve"]["region_daily"], png_dir / "reserve_region_heatmap.png")
    plot_region_heatmap("mp", summaries["mp"]["region_daily"], png_dir / "mp_region_heatmap.png")
    plot_daily_price_trends("reserve", summaries["reserve"]["daily"], png_dir / "reserve_daily_price_trends.png")
    plot_daily_price_trends("mp", summaries["mp"]["daily"], png_dir / "mp_daily_price_trends.png")
    plot_hourly_profile(summaries, png_dir / "hourly_profile_comparison.png")
    plot_resource_overlap(summaries, png_dir / "resource_overlap.png")
    plot_reserve_commodity_summary(summaries["reserve"]["commodity_summary"], png_dir / "reserve_commodity_summary.png")

    print(f"Summary CSVs written to {summary_dir}")
    print(f"PNG charts written to {png_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
