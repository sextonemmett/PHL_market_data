#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from run_evening_spike_visual_report import REGION_LABELS, apply_matplotlib_theme, full_clock_labels, latest_matching_file

DEFAULT_INPUT_PARQUET = Path("data/panels")
DEFAULT_OUTPUT_PNG = Path("regressions/island_equipment_congestion_intraday_share.png")
DEFAULT_OVERLOAD_OUTPUT_PNG = Path("regressions/island_equipment_overload_intraday_share.png")
BIN_MINUTES = 15
BAR_COLOR = "#c05621"
METRIC_SPECS = {
    "equip_cong_any": {
        "title": "Share of 15-minute intervals with equipment congestion by island",
        "ylabel": "Share of intervals with\nequipment congestion",
    },
    "equip_overload_any": {
        "title": "Share of 15-minute intervals with equipment overload by island",
        "ylabel": "Share of intervals with\nequipment overload",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 3x1 PNG showing the share of intervals with island equipment congestion "
            "or overload by 15-minute time-of-day bin."
        )
    )
    parser.add_argument(
        "--panel-parquet",
        help="Island-vs-system panel parquet path. Defaults to the latest file under data/panels.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_SPECS),
        default="equip_cong_any",
        help="Island-level panel column to summarize.",
    )
    parser.add_argument("--output-png", help="Output PNG path.")
    args = parser.parse_args()
    if args.output_png is None:
        args.output_png = str(
            DEFAULT_OUTPUT_PNG if args.metric == "equip_cong_any" else DEFAULT_OVERLOAD_OUTPUT_PNG
        )
    return args


def load_panel(path: Path, metric: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["time_interval", "island_code", metric]).copy()
    frame["time_interval"] = pd.to_datetime(frame["time_interval"])
    frame["bin_time"] = frame["time_interval"].dt.floor(f"{BIN_MINUTES}min")
    frame["clock_label"] = frame["bin_time"].dt.strftime("%H:%M")
    frame["clock_hour"] = frame["bin_time"].dt.hour + (frame["bin_time"].dt.minute / 60.0)
    frame[metric] = frame[metric].astype(float)
    return frame


def build_share_frame(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    clocks = full_clock_labels(BIN_MINUTES)
    clock_hours = {
        label: (pd.Timestamp(label).hour + (pd.Timestamp(label).minute / 60.0))
        for label in clocks
    }

    share = (
        frame.groupby(["island_code", "clock_label"], observed=True)[metric]
        .mean()
        .reset_index(name="share_congested")
    )

    full_index = pd.MultiIndex.from_product([list(REGION_LABELS), clocks], names=["island_code", "clock_label"])
    share = share.set_index(["island_code", "clock_label"]).reindex(full_index).reset_index()
    share["clock_hour"] = share["clock_label"].map(clock_hours)
    share["island_label"] = share["island_code"].map(REGION_LABELS)
    return share


def plot_share_frame(share: pd.DataFrame, metric: str, output_png: Path) -> None:
    apply_matplotlib_theme()
    fig, axes = plt.subplots(3, 1, figsize=(16, 10.5), sharex=True, sharey=True)
    max_share = share["share_congested"].max(skipna=True)
    y_max = max(0.05, float(max_share) * 1.12 if pd.notna(max_share) else 0.05)
    metric_spec = METRIC_SPECS[metric]

    for ax, (island_code, island_label) in zip(axes, REGION_LABELS.items()):
        subset = (
            share.loc[share["island_code"] == island_code]
            .sort_values("clock_hour")
            .reset_index(drop=True)
        )
        ax.bar(
            subset["clock_hour"].to_numpy(),
            subset["share_congested"].to_numpy(),
            width=0.22,
            color=BAR_COLOR,
            edgecolor="#8c3d16",
            linewidth=0.35,
        )
        ax.set_title(island_label, loc="left")
        ax.set_ylabel(metric_spec["ylabel"])
        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.22)
        ax.grid(axis="x", alpha=0.08)

    tick_positions = np.arange(0, 25, 1)
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels([str(int(position)) for position in tick_positions])
    axes[-1].set_xlim(-0.125, 24.0)
    axes[-1].set_xlabel("Hour of day")

    fig.suptitle(metric_spec["title"], fontsize=17, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    panel_path = Path(args.panel_parquet) if args.panel_parquet else latest_matching_file(
        DEFAULT_INPUT_PARQUET,
        "RTD_ISLAND_SYSTEM_PANEL_*.parquet",
    )
    frame = load_panel(panel_path, args.metric)
    share = build_share_frame(frame, args.metric)
    output_png = Path(args.output_png)
    plot_share_frame(share, args.metric, output_png)
    print(f"Wrote {output_png}")


if __name__ == "__main__":
    main()
