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

DEFAULT_INPUT_CSV = Path("data/subsidies/electricity_prices_subsidies_2024_selected_asian_countries.csv")
DEFAULT_OUTPUT_PNG = Path("regressions/residential_price_subsidy_stack.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a stacked residential price and subsidy PNG for selected Asian markets."
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV), help="Input CSV path.")
    parser.add_argument("--output-png", default=str(DEFAULT_OUTPUT_PNG), help="Output PNG path.")
    return parser.parse_args()


def load_frame(input_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_csv).copy()
    required_columns = {"country", "residential_usd_per_kwh", "subsidy_usd_per_kwh"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    return frame


def usd_per_kwh(value: float, _position: int) -> str:
    return f"${value:.2f}"


def build_png(frame: pd.DataFrame, output_png: Path) -> None:
    apply_matplotlib_theme()
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(frame))
    width = 0.62
    colors = {
        "residential": "#2f6f73",
        "subsidy": "#c06c2b",
        "edge": "#102a43",
    }

    residential = frame["residential_usd_per_kwh"].to_numpy()
    subsidy = frame["subsidy_usd_per_kwh"].to_numpy()
    total = residential + subsidy

    ax.bar(
        x,
        residential,
        width,
        color=colors["residential"],
        edgecolor="white",
        linewidth=0.8,
        label="Residential price",
    )
    ax.bar(
        x,
        subsidy,
        width,
        bottom=residential,
        color=colors["subsidy"],
        edgecolor="white",
        linewidth=0.8,
        label="Subsidy",
    )

    for position, x_value, z_value, total_value in zip(x, residential, subsidy, total):
        ax.text(position, total_value + 0.005, f"${total_value:.3f}", ha="center", va="bottom", fontsize=11)
        ax.text(position, x_value / 2, f"${x_value:.3f}", ha="center", va="center", fontsize=10.5, color="white")
        ax.text(position, x_value + (z_value / 2), f"${z_value:.3f}", ha="center", va="center", fontsize=10.5)

    ax.axhline(0, color=colors["edge"], linewidth=0.9)
    ax.set_ylabel("USD/kWh", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(frame["country"], fontsize=13)
    ax.yaxis.set_major_formatter(FuncFormatter(usd_per_kwh))
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0, float(total.max()) + 0.055)
    ax.set_xlim(-0.6, len(frame) - 0.4)
    ax.grid(axis="y", color="#d8d0c3", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)

    fig.text(
        0.5,
        0.01,
        "Base segment is residential price X. Top segment is subsidy Z = (explicit budget transfers + quasi-fiscal support) / residential use.",
        fontsize=8.5,
        color=colors["edge"],
        ha="center",
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
