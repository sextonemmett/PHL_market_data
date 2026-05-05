#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from run_evening_spike_visual_report import apply_matplotlib_theme

DEFAULT_INPUT_CSV = Path("data/source_prices/source_generation_prices_and_lcoe.csv")
DEFAULT_OUTPUT_PNG = Path("regressions/source_generation_price_lcoe_grouped.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a grouped bar chart comparing realized generation prices with LCOE-based comparators."
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV), help="Input CSV path.")
    parser.add_argument("--output-png", default=str(DEFAULT_OUTPUT_PNG), help="Output PNG path.")
    return parser.parse_args()


def load_frame(input_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_csv).copy()
    required_columns = {"source", "avg_price_php_per_kwh", "comparison_price_php_per_kwh"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    return frame


def compute_total_what_if(frame: pd.DataFrame) -> float:
    mix = frame.loc[frame["share_of_kwh_pct"].notna()].copy()
    weights = mix["share_of_kwh_pct"] / 100.0
    return float((weights * mix["comparison_price_php_per_kwh"]).sum())


def php_per_kwh(value: float, _position: int) -> str:
    return f"{value:.1f}"


def build_png(frame: pd.DataFrame, output_png: Path) -> None:
    apply_matplotlib_theme()
    fig, ax = plt.subplots(figsize=(12, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    source_order = ["Spot market", "Solar", "Hydro", "Gas", "Coal", "Average, all technologies"]
    frame["source"] = pd.Categorical(frame["source"], categories=source_order, ordered=True)
    frame = frame.sort_values("source").reset_index(drop=True)

    average_mask = frame["source"].eq("Average, all technologies")
    if average_mask.any():
        frame.loc[average_mask, "comparison_price_php_per_kwh"] = compute_total_what_if(frame)

    x = np.arange(len(frame))
    width = 0.34
    realized = frame["avg_price_php_per_kwh"].to_numpy(dtype=float)
    comparison = frame["comparison_price_php_per_kwh"].to_numpy(dtype=float)

    realized_color = "#2f6f73"
    comparison_color = "#c06c2b"
    average_color = "#8f5aa8"
    edge_color = "#102a43"

    realized_bars = ax.bar(
        x - width / 2,
        realized,
        width,
        color=realized_color,
        edgecolor="white",
        linewidth=0.8,
        label="Realized price",
    )

    comparison_positions = x + width / 2
    lcoe_mask = frame["source"].isin(["Gas", "Coal"])
    average_comparison_mask = frame["source"].eq("Average, all technologies")
    lcoe_bars = ax.bar(
        comparison_positions[lcoe_mask],
        comparison[lcoe_mask],
        width,
        color=comparison_color,
        edgecolor="white",
        linewidth=0.8,
        label="LCOE",
    )
    average_comparison_bars = ax.bar(
        comparison_positions[average_comparison_mask],
        comparison[average_comparison_mask],
        width,
        color=average_color,
        edgecolor="white",
        linewidth=0.8,
        label="LCOE-based average price",
    )

    for bar in realized_bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.12,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
        )

    for bar in lcoe_bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.12,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
        )

    for bar in average_comparison_bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.12,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
        )

    share_labels = []
    for source, share in zip(frame["source"], frame["share_of_kwh_pct"]):
        if pd.notna(share):
            share_labels.append(f"{source}\n({share:.1f}%)")
        else:
            share_labels.append(f"{source}\n(100.0%)")

    ax.set_ylabel("PhP/kWh", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(share_labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(php_per_kwh))
    ax.set_ylim(0, max(float(np.nanmax(realized)), float(np.nanmax(comparison))) + 1.3)
    ax.axvline(0.5, color="#6b7280", linestyle="--", linewidth=1.2, alpha=0.9)
    ax.axvline(4.5, color="#6b7280", linestyle="--", linewidth=1.2, alpha=0.9)
    ax.text(2.5, 10.15, "PSAs & IPPs", ha="center", va="bottom", fontsize=11, color=edge_color)
    average_realized = float(frame.loc[average_mask, "avg_price_php_per_kwh"].iloc[0])
    average_what_if = float(frame.loc[average_mask, "comparison_price_php_per_kwh"].iloc[0])
    ax.annotate(
        "What-if weighted average:\ncoal and gas replaced\nwith LCOE values",
        xy=(5 + width / 2, average_what_if + 0.45),
        xytext=(5 + width / 2, 9.6),
        ha="center",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#cbd5e1", "lw": 0.9},
        arrowprops={"arrowstyle": "->", "color": edge_color, "lw": 1.0},
    )
    ax.grid(axis="y", color="#d8d0c3", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False, fontsize=11)

    fig.text(
        0.5,
        0.01,
        "Coal and gas comparators use 2024 LCOE values converted at 57 PhP/USD. The total what-if average is weighted by each source's share of purchased kWh.",
        ha="center",
        fontsize=9,
        color=edge_color,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_png = Path(args.output_png)
    build_png(load_frame(input_csv), output_png)
    print(f"Wrote {output_png}")


if __name__ == "__main__":
    main()
