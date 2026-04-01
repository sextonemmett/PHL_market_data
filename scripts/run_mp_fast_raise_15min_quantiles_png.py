#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from run_evening_spike_visual_report import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    REGION_LABELS,
    apply_matplotlib_theme,
    build_merged_frame,
    date_bounds,
    latest_matching_file,
)

DEFAULT_OUTPUT_PNG = Path("regressions/mp_fast_raise_15min_quantiles.png")
DEFAULT_OUTPUT_CSV = Path("regressions/mp_fast_raise_15min_quantiles.csv")
DEFAULT_INPUT_CSV = Path("regressions/evening_spike_visual_15min.csv")
QUANTILES = (0.25, 0.50, 0.75)
METRIC_SPECS = (
    {"key": "mp", "label": "Market price", "column_template": "price_{region}", "color": "#c05621", "fill_alpha": 0.18},
    {"key": "fr", "label": "Fast contingency raise price", "column_template": "reserve_fr_{region}", "color": "#2b6cb0", "fill_alpha": 0.16},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a standalone PNG with 15-minute MP and fast contingency raise quantiles "
            "for Luzon, Visayas, and Mindanao."
        )
    )
    parser.add_argument("--merged-csv", default=str(DEFAULT_INPUT_CSV), help="Existing merged 15-minute analysis CSV.")
    parser.add_argument("--mp-parquet", help="Combined MP parquet path.")
    parser.add_argument("--reserve-parquet", help="Combined reserve parquet path.")
    parser.add_argument("--hvdc-parquet", help="Combined RTDHS parquet path.")
    parser.add_argument("--rtdreg-parquet", help="Combined RTDREG parquet path.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Analysis start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Analysis end date in YYYY-MM-DD.")
    parser.add_argument("--bin-minutes", type=int, default=15, help="Time-bin size in minutes if rebuilding the merged dataset.")
    parser.add_argument("--output-png", default=str(DEFAULT_OUTPUT_PNG), help="Output PNG path.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Output CSV path.")
    args = parser.parse_args()
    if args.bin_minutes != 15:
        raise SystemExit("This visual is fixed at 15-minute intervals.")
    return args


def load_or_build_merged(args: argparse.Namespace) -> pd.DataFrame:
    merged_csv = Path(args.merged_csv)
    start_ts, end_exclusive = date_bounds(args.start_date, args.end_date)
    if merged_csv.exists():
        frame = pd.read_csv(merged_csv, parse_dates=["bin_time", "date"]).copy()
        return frame.loc[(frame["bin_time"] >= start_ts) & (frame["bin_time"] < end_exclusive)].copy()

    mp_path = Path(args.mp_parquet) if args.mp_parquet else latest_matching_file(Path("data/mp/combined"), "MP_*.parquet")
    reserve_path = Path(args.reserve_parquet) if args.reserve_parquet else latest_matching_file(
        Path("data/mp_reserve/combined"),
        "MP_RESERVE_*.parquet",
    )
    hvdc_path = Path(args.hvdc_parquet) if args.hvdc_parquet else latest_matching_file(
        Path("data/rtdhs/combined"),
        "RTDHS_*.parquet",
    )
    rtdreg_path = Path(args.rtdreg_parquet) if args.rtdreg_parquet else latest_matching_file(
        Path("data/rtdreg/combined"),
        "RTDREG_*.parquet",
    )
    return build_merged_frame(
        mp_path=mp_path,
        reserve_path=reserve_path,
        hvdc_path=hvdc_path,
        rtdreg_path=rtdreg_path,
        start_ts=start_ts,
        end_exclusive=end_exclusive,
        bin_minutes=args.bin_minutes,
    )


def build_quantile_frame(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    time_order = sorted(merged["clock_label"].unique())
    time_index = {clock_label: position for position, clock_label in enumerate(time_order)}
    for region_code, region_label in REGION_LABELS.items():
        region_lower = region_code.lower()
        for metric in METRIC_SPECS:
            column = metric["column_template"].format(region=region_lower)
            grouped = merged.groupby("clock_label")[column]
            for quantile in QUANTILES:
                series = grouped.quantile(quantile).reindex(time_order)
                for clock_label, value in series.items():
                    rows.append(
                        {
                            "island_code": region_code,
                            "island_label": region_label,
                            "metric_key": metric["key"],
                            "metric_label": metric["label"],
                            "quantile": quantile,
                            "clock_label": clock_label,
                            "clock_order": time_index[clock_label],
                            "value": float(value) if pd.notna(value) else np.nan,
                        }
                    )
    return pd.DataFrame(rows).sort_values(["island_code", "metric_key", "quantile", "clock_order"]).reset_index(drop=True)


def compute_shared_limits(quantile_frame: pd.DataFrame) -> tuple[float, float]:
    values = quantile_frame["value"].dropna()
    if values.empty:
        return 0.0, 1.0
    lower = float(values.min())
    upper = float(values.max())
    padding = 0.04 * (upper - lower) if upper > lower else max(abs(upper) * 0.08, 1.0)
    return lower - padding, upper + padding


def build_png(quantile_frame: pd.DataFrame, output_png: Path) -> None:
    apply_matplotlib_theme()
    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True, sharey=True)

    clock_order = (
        quantile_frame[["clock_label", "clock_order"]]
        .drop_duplicates()
        .sort_values("clock_order")
        .reset_index(drop=True)
    )
    x_values = clock_order["clock_order"].to_numpy()
    x_labels = clock_order["clock_label"].tolist()
    y_min, y_max = compute_shared_limits(quantile_frame)

    for ax, (region_code, region_label) in zip(axes, REGION_LABELS.items()):
        subset = quantile_frame.loc[quantile_frame["island_code"] == region_code].copy()
        for metric in METRIC_SPECS:
            metric_subset = subset.loc[subset["metric_key"] == metric["key"]]
            p25 = metric_subset.loc[metric_subset["quantile"] == 0.25].sort_values("clock_order")
            p50 = metric_subset.loc[metric_subset["quantile"] == 0.50].sort_values("clock_order")
            p75 = metric_subset.loc[metric_subset["quantile"] == 0.75].sort_values("clock_order")

            ax.fill_between(
                p25["clock_order"].to_numpy(),
                p25["value"].to_numpy(),
                p75["value"].to_numpy(),
                color=metric["color"],
                alpha=metric["fill_alpha"],
                linewidth=0,
            )
            ax.plot(
                p50["clock_order"].to_numpy(),
                p50["value"].to_numpy(),
                color=metric["color"],
                linewidth=2.8,
                label=metric["label"],
            )

        ax.set_title(region_label, loc="left")
        ax.set_ylabel("PHP/MWh")
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", alpha=0.22)
        ax.grid(axis="x", alpha=0.08)

    tick_positions = [position for position, label in enumerate(x_labels) if label.endswith(":00")]
    tick_labels = [x_labels[position] for position in tick_positions]
    axes[-1].set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
    axes[-1].set_xlabel("15-minute interval")

    legend_handles = []
    for metric in METRIC_SPECS:
        legend_handles.append(Line2D([0], [0], color=metric["color"], linewidth=2.8, label=f"{metric['label']} median"))
        legend_handles.append(Patch(facecolor=metric["color"], alpha=metric["fill_alpha"], label=f"{metric['label']} p25-p75 band"))
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
    )
    fig.suptitle(
        "Intraday 15-minute market price and fast contingency raise quantiles",
        fontsize=17,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    merged = load_or_build_merged(args)
    quantile_frame = build_quantile_frame(merged)

    output_png = Path(args.output_png)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    quantile_frame.to_csv(output_csv, index=False)
    build_png(quantile_frame, output_png)

    print(f"Wrote {output_png}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
