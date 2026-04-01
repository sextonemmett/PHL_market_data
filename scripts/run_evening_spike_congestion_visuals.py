#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from run_evening_spike_visual_report import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    REGION_LABELS,
    apply_matplotlib_theme,
    build_merged_frame,
    date_bounds,
    full_clock_labels,
    latest_matching_file,
)

DEFAULT_INPUT_CSV = Path("regressions/evening_spike_visual_15min.csv")
DEFAULT_ECDF_PNG = Path("regressions/evening_spike_evening_ecdf_by_congestion.png")
DEFAULT_PROFILE_PNG = Path("regressions/evening_spike_intraday_profile_by_congestion.png")
DEFAULT_SUMMARY_CSV = Path("regressions/evening_spike_congestion_summary.csv")
WINDOW_START_LABEL = "17:00"
WINDOW_END_EXCLUSIVE_LABEL = "20:00"

REGIME_SPECS = (
    {"value": 0, "label": "Link uncongested", "color": "#4c78a8", "fill_alpha": 0.15},
    {"value": 1, "label": "Link congested", "color": "#c44e52", "fill_alpha": 0.18},
)
ECDF_METRICS = (
    {"key": "spike", "label": "MP", "linestyle": "-", "column_template": "spike_{region}"},
    {"key": "fr", "label": "Fast contingency raise", "linestyle": "--", "column_template": "reserve_fr_{region}"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create standalone visuals comparing evening price spikes under congested vs uncongested "
            "inter-island link regimes."
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
    parser.add_argument("--output-ecdf-png", default=str(DEFAULT_ECDF_PNG), help="Output PNG for evening ECDF visual.")
    parser.add_argument("--output-profile-png", default=str(DEFAULT_PROFILE_PNG), help="Output PNG for intraday regime profile visual.")
    parser.add_argument("--output-summary-csv", default=str(DEFAULT_SUMMARY_CSV), help="Output CSV for summary statistics.")
    args = parser.parse_args()
    if args.bin_minutes != 15:
        raise SystemExit("These visuals are fixed at 15-minute intervals.")
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


def in_target_window(frame: pd.DataFrame) -> pd.Series:
    return (frame["clock_label"] >= WINDOW_START_LABEL) & (frame["clock_label"] < WINDOW_END_EXCLUSIVE_LABEL)


def build_summary(merged: pd.DataFrame) -> pd.DataFrame:
    evening = merged.loc[in_target_window(merged)].copy()
    rows: list[dict[str, object]] = []
    for region_code, region_label in REGION_LABELS.items():
        region_lower = region_code.lower()
        regime_col = f"connected_link_congested_{region_lower}"
        spike_col = f"spike_{region_lower}"
        for regime_value, regime_frame in evening.groupby(regime_col, observed=True):
            series = regime_frame[spike_col].dropna()
            rows.append(
                {
                    "island_code": region_code,
                    "island_label": region_label,
                    "regime_value": int(regime_value),
                    "regime_label": "Link congested" if int(regime_value) == 1 else "Link uncongested",
                    "count": int(series.shape[0]),
                    "mean_spike": float(series.mean()) if not series.empty else np.nan,
                    "median_spike": float(series.median()) if not series.empty else np.nan,
                    "p95_spike": float(series.quantile(0.95)) if not series.empty else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["island_code", "regime_value"]).reset_index(drop=True)


def shared_spike_limits(merged: pd.DataFrame) -> tuple[float, float]:
    spike_columns = [f"spike_{region.lower()}" for region in REGION_LABELS]
    values = np.concatenate([merged[column].dropna().to_numpy() for column in spike_columns])
    if values.size == 0:
        return 0.0, 1.0
    lower = float(np.nanquantile(values, 0.01))
    upper = float(np.nanquantile(values, 0.99))
    if upper <= lower:
        upper = lower + 1.0
    padding = 0.05 * (upper - lower)
    return lower - padding, upper + padding


def shared_evening_ecdf_limits(merged: pd.DataFrame) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for region_code in REGION_LABELS:
        region_lower = region_code.lower()
        values.append(merged[f"spike_{region_lower}"].dropna().to_numpy())
        values.append(merged[f"reserve_fr_{region_lower}"].dropna().to_numpy())
    stacked = np.concatenate([array for array in values if array.size > 0]) if values else np.array([])
    if stacked.size == 0:
        return 0.0, 1.0
    lower = float(np.nanquantile(stacked, 0.01))
    upper = float(np.nanquantile(stacked, 0.99))
    if upper <= lower:
        upper = lower + 1.0
    padding = 0.05 * (upper - lower)
    return lower - padding, upper + padding


def ecdf_xy(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values.dropna().to_numpy())
    if ordered.size == 0:
        return np.array([]), np.array([])
    y = np.arange(1, ordered.size + 1) / ordered.size
    return ordered, y


def build_evening_ecdf_figure(merged: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    apply_matplotlib_theme()
    evening = merged.loc[in_target_window(merged)].copy()
    x_min, x_max = shared_evening_ecdf_limits(evening)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11.5), sharex=True, sharey=True)

    for ax, (region_code, region_label) in zip(axes, REGION_LABELS.items()):
        region_lower = region_code.lower()
        regime_col = f"connected_link_congested_{region_lower}"
        columns = [regime_col, f"spike_{region_lower}", f"reserve_fr_{region_lower}"]
        subset = evening[columns].copy()
        for regime in REGIME_SPECS:
            regime_frame = subset.loc[subset[regime_col] == regime["value"]].copy()
            for metric in ECDF_METRICS:
                metric_values = regime_frame[metric["column_template"].format(region=region_lower)]
                x, y = ecdf_xy(metric_values)
                ax.plot(
                    x,
                    y,
                    color=regime["color"],
                    linestyle=metric["linestyle"],
                    linewidth=2.4,
                )
        ax.set_title(region_label, loc="left")
        ax.set_ylabel("ECDF")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.18)

    axes[-1].set_xlabel("PHP/MWh")
    combined_handles = [
        Line2D([0], [0], color="#4c78a8", linestyle="-", linewidth=2.6, label="Uncongested, MP"),
        Line2D([0], [0], color="#4c78a8", linestyle="--", linewidth=2.6, label="Uncongested, fast contingency raise"),
        Line2D([0], [0], color="#c44e52", linestyle="-", linewidth=2.6, label="Congested, MP"),
        Line2D([0], [0], color="#c44e52", linestyle="--", linewidth=2.6, label="Congested, fast contingency raise"),
    ]
    fig.legend(
        handles=combined_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("17:00-20:00 ECDFs for MP and fast contingency raise by congestion regime", fontsize=17, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_intraday_profile_figure(merged: pd.DataFrame, output_path: Path) -> None:
    apply_matplotlib_theme()
    clocks = full_clock_labels(15)
    x_positions = np.arange(len(clocks))
    y_min, y_max = shared_spike_limits(merged)
    evening_start = clocks.index(WINDOW_START_LABEL)
    evening_end = clocks.index("19:45")

    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True, sharey=True)
    for ax, (region_code, region_label) in zip(axes, REGION_LABELS.items()):
        region_lower = region_code.lower()
        regime_col = f"connected_link_congested_{region_lower}"
        spike_col = f"spike_{region_lower}"
        subset = merged[["clock_label", regime_col, spike_col]].dropna().copy()
        for regime in REGIME_SPECS:
            regime_frame = subset.loc[subset[regime_col] == regime["value"]].copy()
            grouped = regime_frame.groupby("clock_label")[spike_col].agg(
                p25=lambda s: float(s.quantile(0.25)),
                p50="median",
                p75=lambda s: float(s.quantile(0.75)),
            ).reindex(clocks)
            ax.fill_between(
                x_positions,
                grouped["p25"].to_numpy(),
                grouped["p75"].to_numpy(),
                color=regime["color"],
                alpha=regime["fill_alpha"],
                linewidth=0,
            )
            ax.plot(
                x_positions,
                grouped["p50"].to_numpy(),
                color=regime["color"],
                linewidth=2.5,
                label=f"{regime['label']} median",
            )

        ax.add_patch(
            Rectangle(
                (evening_start - 0.5, y_min),
                evening_end - evening_start + 1,
                y_max - y_min,
                facecolor="#f0b429",
                alpha=0.08,
                linewidth=0,
            )
        )
        ax.set_title(region_label, loc="left")
        ax.set_ylabel("Spike (PHP/MWh)")
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", alpha=0.2)
        ax.grid(axis="x", alpha=0.08)

    tick_positions = [idx for idx, label in enumerate(clocks) if label.endswith(":00")]
    tick_labels = [clocks[idx] for idx in tick_positions]
    axes[-1].set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
    axes[-1].set_xlabel("15-minute interval")

    legend_handles = []
    for regime in REGIME_SPECS:
        legend_handles.append(Line2D([0], [0], color=regime["color"], linewidth=2.5, label=f"{regime['label']} median"))
        legend_handles.append(Patch(facecolor=regime["color"], alpha=regime["fill_alpha"], label=f"{regime['label']} p25-p75 band"))
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Intraday 15-minute spike profiles by congestion regime, with 17:00-20:00 highlighted", fontsize=17, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    merged = load_or_build_merged(args)
    summary = build_summary(merged)

    ecdf_path = Path(args.output_ecdf_png)
    profile_path = Path(args.output_profile_png)
    summary_path = Path(args.output_summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    build_evening_ecdf_figure(merged, summary, ecdf_path)
    build_intraday_profile_figure(merged, profile_path)

    print(f"Wrote {ecdf_path}")
    print(f"Wrote {profile_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
