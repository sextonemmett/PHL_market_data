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
from run_source_generation_price_lcoe_grouped_png import (
    DEFAULT_INPUT_CSV as DEFAULT_SOURCE_PRICE_INPUT_CSV,
    DEFAULT_LCOE_SOURCES,
    compute_total_what_if,
    format_source_list,
    load_frame as load_source_price_frame,
    SPOT_EXCESS_MARKUP_PHP_PER_KWH,
)

DEFAULT_INPUT_CSV = Path("data/subsidies/electricity_prices_subsidies_2024_selected_asian_countries.csv")
DEFAULT_OUTPUT_PNG = Path("regressions/residential_price_subsidy_stack.png")
DEFAULT_COUNTERFACTUAL_OUTPUT_PNG = Path("regressions/residential_price_subsidy_stack_meralco_counterfactual.png")
MERALCO_REALIZED_GENERATION_COST = 7.71
MERALCO_NON_GENERATION_COST = 3.97


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a stacked residential price and subsidy PNG for selected Asian markets."
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV), help="Input CSV path.")
    parser.add_argument(
        "--source-price-csv",
        default=str(DEFAULT_SOURCE_PRICE_INPUT_CSV),
        help="Input CSV path for the Meralco generation-cost counterfactual.",
    )
    parser.add_argument("--output-png", default=str(DEFAULT_OUTPUT_PNG), help="Output PNG path.")
    parser.add_argument(
        "--counterfactual-output-png",
        default=str(DEFAULT_COUNTERFACTUAL_OUTPUT_PNG),
        help="Output PNG path for the Meralco-counterfactual version.",
    )
    parser.add_argument(
        "--counterfactual-lcoe-sources",
        nargs="*",
        default=list(DEFAULT_LCOE_SOURCES),
        help="Sources whose generation costs should switch to LCOE in the counterfactual.",
    )
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


def meralco_counterfactual_generation_cost(
    source_price_frame: pd.DataFrame,
    lcoe_sources: tuple[str, ...] | list[str] | set[str],
) -> float:
    spot_realized_price = float(
        source_price_frame.loc[source_price_frame["source"] == "Spot market", "avg_price_php_per_kwh"].iloc[0]
    )
    spot_net_of_excess_price = spot_realized_price - SPOT_EXCESS_MARKUP_PHP_PER_KWH
    return compute_total_what_if(
        source_price_frame,
        lcoe_sources,
        spot_market_price_php_per_kwh=spot_net_of_excess_price,
    )


def meralco_counterfactual_factor(
    source_price_frame: pd.DataFrame,
    lcoe_sources: tuple[str, ...] | list[str] | set[str],
) -> float:
    counterfactual_generation_cost = meralco_counterfactual_generation_cost(source_price_frame, lcoe_sources)
    return (
        counterfactual_generation_cost + MERALCO_NON_GENERATION_COST
    ) / (
        MERALCO_REALIZED_GENERATION_COST + MERALCO_NON_GENERATION_COST
    )


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


def build_counterfactual_png(
    frame: pd.DataFrame,
    output_png: Path,
    source_price_frame: pd.DataFrame,
    lcoe_sources: tuple[str, ...] | list[str] | set[str],
) -> None:
    apply_matplotlib_theme()
    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = {
        "residential": "#2f6f73",
        "subsidy": "#c06c2b",
        "counterfactual": "#8f3f2b",
        "edge": "#102a43",
    }

    ph_row = frame.loc[frame["country"] == "Philippines"].iloc[0]
    ph_subsidy = float(ph_row["subsidy_usd_per_kwh"])
    ph_residential = float(ph_row["residential_usd_per_kwh"])
    ph_total = ph_residential + ph_subsidy
    factor = meralco_counterfactual_factor(source_price_frame, lcoe_sources)
    ph_counterfactual = ph_total * factor
    lcoe_source_set = {str(source) for source in lcoe_sources}

    display_rows: list[dict[str, float | str | bool]] = []
    for row in frame.itertuples(index=False):
        country = str(row.country)
        if country == "Philippines":
            display_rows.append(
                {
                    "country": country,
                    "residential": float(row.residential_usd_per_kwh),
                    "subsidy": 0.0,
                    "show_subsidy": False,
                    "kind": "actual",
                }
            )
            display_rows.append(
                {
                    "country": "Philippines\ncounterfactual",
                    "residential": ph_counterfactual,
                    "subsidy": 0.0,
                    "show_subsidy": False,
                    "kind": "counterfactual",
                }
            )
        else:
            display_rows.append(
                {
                    "country": country,
                    "residential": float(row.residential_usd_per_kwh),
                    "subsidy": float(row.subsidy_usd_per_kwh),
                    "show_subsidy": True,
                    "kind": "actual",
                }
            )

    display = pd.DataFrame(display_rows)

    x = np.arange(len(display))
    width = 0.62
    residential = display["residential"].to_numpy(dtype=float)
    subsidy = display["subsidy"].to_numpy(dtype=float)
    total = residential + subsidy
    kinds = display["kind"].tolist()
    bar_colors = [colors["counterfactual"] if kind == "counterfactual" else colors["residential"] for kind in kinds]

    ax.bar(
        x,
        residential,
        width,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8,
        label="Residential price",
    )

    subsidy_mask = display["show_subsidy"].to_numpy(dtype=bool)
    ax.bar(
        x[subsidy_mask],
        subsidy[subsidy_mask],
        width,
        bottom=residential[subsidy_mask],
        color=colors["subsidy"],
        edgecolor="white",
        linewidth=0.8,
        label="Subsidy",
    )

    for position, row in enumerate(display.itertuples(index=False)):
        total_value = float(row.residential + row.subsidy)
        label_color = "white" if str(row.kind) in {"actual", "counterfactual"} else colors["edge"]
        ax.text(position, row.residential / 2, f"${row.residential:.3f}", ha="center", va="center", fontsize=10.5, color=label_color)
        if str(row.country) not in {"Philippines", "Philippines\ncounterfactual"}:
            ax.text(position, total_value + 0.005, f"${total_value:.3f}", ha="center", va="bottom", fontsize=11)
        if bool(row.show_subsidy) and row.subsidy > 0:
            ax.text(position, row.residential + (row.subsidy / 2), f"${row.subsidy:.3f}", ha="center", va="center", fontsize=10.5)

    divider_x = 1.5
    ax.axvline(divider_x, color="#6b7280", linestyle="--", linewidth=1.2, alpha=0.95)

    bracket_x = 0.50
    cap_half = 0.10
    ax.vlines(bracket_x, ph_counterfactual, ph_residential, color=colors["edge"], linewidth=1.1)
    ax.hlines(ph_residential, bracket_x - cap_half, bracket_x + cap_half, color=colors["edge"], linewidth=1.1)
    ax.hlines(ph_counterfactual, bracket_x - cap_half, bracket_x + cap_half, color=colors["edge"], linewidth=1.1)
    annotation_text = (
        "Counterfactual if PSAs reflected LCOE\nand excess markups were limited\nin the spot market"
        if lcoe_source_set == set(DEFAULT_LCOE_SOURCES)
        else "Counterfactual if coal and gas PSAs\nreflected LCOE and excess markups\nwere limited in the spot market"
    )
    ax.annotate(
        annotation_text,
        xy=(bracket_x, (ph_residential + ph_counterfactual) / 2),
        xytext=(bracket_x, 0.238),
        ha="center",
        va="bottom",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#cbd5e1", "lw": 0.9},
    )

    ax.axhline(0, color=colors["edge"], linewidth=0.9)
    ax.set_ylabel("USD/kWh", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display["country"], fontsize=12.5)
    ax.yaxis.set_major_formatter(FuncFormatter(usd_per_kwh))
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0, float(total.max()) + 0.07)
    ax.set_xlim(-0.7, len(display) - 0.2)
    ax.grid(axis="y", color="#d8d0c3", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=colors["residential"]),
            plt.Rectangle((0, 0), 1, 1, color=colors["counterfactual"]),
            plt.Rectangle((0, 0), 1, 1, color=colors["subsidy"]),
        ],
        labels=["Residential price", "Philippines counterfactual", "Subsidy"],
        loc="upper right",
        frameon=False,
    )

    replacement_text = format_source_list(lcoe_sources)
    detail_line = (
        f"The generation-cost counterfactual replaces {replacement_text} with LCOE values and nets out excess spot markup."
        if lcoe_source_set == set(DEFAULT_LCOE_SOURCES)
        else "The generation-cost counterfactual replaces only coal and gas with LCOE values, keeps solar and hydro at realized prices, and nets out excess spot markup."
    )
    fig.text(
        0.5,
        0.03,
        "Counterfactual bar applies the Meralco scenario nationally by scaling the Philippines retail price by\n"
        "(counterfactual generation cost + non-generation cost) / (realized generation cost + non-generation cost).\n"
        f"{detail_line}\n"
        f"Philippines subsidy is omitted from the bar stack for readability and equals ${ph_subsidy:.4f}/kWh.",
        fontsize=8.7,
        color=colors["edge"],
        ha="center",
    )
    fig.tight_layout(rect=(0.02, 0.11, 0.98, 0.98))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    source_price_csv = Path(args.source_price_csv)
    output_png = Path(args.output_png)
    counterfactual_output_png = Path(args.counterfactual_output_png)
    frame = load_frame(input_csv)
    source_price_frame = load_source_price_frame(source_price_csv)
    build_png(frame, output_png)
    build_counterfactual_png(
        frame,
        counterfactual_output_png,
        source_price_frame,
        args.counterfactual_lcoe_sources,
    )
    print(f"Wrote {output_png}")
    print(f"Wrote {counterfactual_output_png}")


if __name__ == "__main__":
    main()
