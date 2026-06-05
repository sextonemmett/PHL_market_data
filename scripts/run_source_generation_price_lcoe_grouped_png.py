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
DEFAULT_LCOE_SOURCES = ("Solar", "Gas", "Coal")
LCOE_SOURCE_DISPLAY_ORDER = ("Solar", "Coal", "Gas")
SPOT_EXCESS_MARKUP_PHP_PER_MWH = 271.08
SPOT_EXCESS_MARKUP_PHP_PER_KWH = SPOT_EXCESS_MARKUP_PHP_PER_MWH / 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a grouped bar chart comparing realized generation prices with LCOE-based comparators."
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV), help="Input CSV path.")
    parser.add_argument("--output-png", default=str(DEFAULT_OUTPUT_PNG), help="Output PNG path.")
    parser.add_argument(
        "--lcoe-sources",
        nargs="*",
        default=list(DEFAULT_LCOE_SOURCES),
        help="Sources whose comparator bars should use LCOE values.",
    )
    return parser.parse_args()


def load_frame(input_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_csv).copy()
    required_columns = {"source", "avg_price_php_per_kwh", "comparison_price_php_per_kwh"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    return frame


def ordered_lcoe_sources(lcoe_sources: tuple[str, ...] | list[str] | set[str]) -> list[str]:
    selected = {str(source) for source in lcoe_sources}
    ordered = [source for source in LCOE_SOURCE_DISPLAY_ORDER if source in selected]
    extras = [source for source in lcoe_sources if str(source) not in set(ordered)]
    return ordered + [str(source) for source in extras]


def format_source_list(sources: tuple[str, ...] | list[str] | set[str]) -> str:
    ordered = [source.lower() for source in ordered_lcoe_sources(sources)]
    if not ordered:
        return "no sources"
    if len(ordered) == 1:
        return ordered[0]
    if len(ordered) == 2:
        return " and ".join(ordered)
    return f"{', '.join(ordered[:-1])}, and {ordered[-1]}"


def compute_total_what_if(
    frame: pd.DataFrame,
    lcoe_sources: tuple[str, ...] | list[str] | set[str],
    spot_market_price_php_per_kwh: float | None = None,
) -> float:
    lcoe_source_set = {str(source) for source in lcoe_sources}
    mix = frame.loc[frame["share_of_kwh_pct"].notna()].copy()
    mix["effective_price_php_per_kwh"] = mix["avg_price_php_per_kwh"]
    lcoe_mask = mix["source"].isin(lcoe_source_set)
    mix.loc[lcoe_mask, "effective_price_php_per_kwh"] = mix.loc[lcoe_mask, "comparison_price_php_per_kwh"]
    if spot_market_price_php_per_kwh is not None:
        mix.loc[mix["source"] == "Spot market", "effective_price_php_per_kwh"] = spot_market_price_php_per_kwh
    weights = mix["share_of_kwh_pct"] / 100.0
    return float((weights * mix["effective_price_php_per_kwh"]).sum())


def php_per_kwh(value: float, _position: int) -> str:
    return f"{value:.1f}"


def build_png(
    frame: pd.DataFrame,
    output_png: Path,
    lcoe_sources: tuple[str, ...] | list[str] | set[str],
) -> None:
    apply_matplotlib_theme()
    fig, ax = plt.subplots(figsize=(12, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    source_order = ["Spot market", "Solar", "Hydro", "Gas", "Coal", "Average, all technologies"]
    frame["source"] = pd.Categorical(frame["source"], categories=source_order, ordered=True)
    frame = frame.sort_values("source").reset_index(drop=True)
    lcoe_source_set = {str(source) for source in lcoe_sources}

    average_mask = frame["source"].eq("Average, all technologies")
    spot_mask = frame["source"].eq("Spot market")
    spot_realized_price = float(frame.loc[spot_mask, "avg_price_php_per_kwh"].iloc[0])
    spot_net_of_excess_price = spot_realized_price - SPOT_EXCESS_MARKUP_PHP_PER_KWH
    average_what_if = compute_total_what_if(frame, lcoe_sources)
    average_what_if_net_excess = compute_total_what_if(
        frame,
        lcoe_sources,
        spot_market_price_php_per_kwh=spot_net_of_excess_price,
    )
    if average_mask.any():
        frame.loc[average_mask, "comparison_price_php_per_kwh"] = average_what_if

    x = np.arange(len(frame))
    width = 0.24
    realized = frame["avg_price_php_per_kwh"].to_numpy(dtype=float)
    comparison = frame["comparison_price_php_per_kwh"].to_numpy(dtype=float)

    realized_color = "#2f6f73"
    comparison_color = "#c06c2b"
    average_color = "#8f5aa8"
    net_excess_color = "#c9485b"
    average_net_excess_color = "#2d5b9f"
    edge_color = "#102a43"
    average_label = (
        "LCOE-based average price"
        if lcoe_source_set == set(DEFAULT_LCOE_SOURCES)
        else "Coal/gas-LCOE average price"
    )
    average_net_label = (
        "LCOE-based average + spot net of excess markup"
        if lcoe_source_set == set(DEFAULT_LCOE_SOURCES)
        else "Coal/gas-LCOE average + spot net of excess markup"
    )

    legend_used: set[str] = set()
    bars_to_label: list[matplotlib.patches.Rectangle] = []

    def add_bar(
        xpos: float,
        height: float,
        color: str,
        label: str | None,
    ) -> matplotlib.patches.Rectangle:
        draw_label = label if label and label not in legend_used else None
        if draw_label is not None:
            legend_used.add(draw_label)
        container = ax.bar(
            [xpos],
            [height],
            width,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=draw_label,
        )
        bar = container.patches[0]
        bars_to_label.append(bar)
        return bar

    for xpos, row in zip(x, frame.itertuples(index=False)):
        source = str(row.source)
        realized_value = float(row.avg_price_php_per_kwh)
        comparison_value = float(row.comparison_price_php_per_kwh)

        if source in lcoe_source_set:
            add_bar(xpos - width / 2, realized_value, realized_color, "Realized price")
            add_bar(xpos + width / 2, comparison_value, comparison_color, "LCOE")
        elif source == "Spot market":
            add_bar(xpos - width / 2, realized_value, realized_color, "Realized price")
            add_bar(
                xpos + width / 2,
                spot_net_of_excess_price,
                net_excess_color,
                "Spot market, net of excess markup",
            )
        elif source == "Average, all technologies":
            add_bar(xpos - width, realized_value, realized_color, "Realized price")
            add_bar(xpos, average_what_if, average_color, average_label)
            add_bar(
                xpos + width,
                average_what_if_net_excess,
                average_net_excess_color,
                average_net_label,
            )
        else:
            add_bar(xpos, realized_value, realized_color, "Realized price")

    for bar in bars_to_label:
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
    y_max = max(
        float(np.nanmax(realized)),
        float(np.nanmax(comparison)),
        spot_net_of_excess_price,
        average_what_if_net_excess,
    )
    ax.set_ylim(0, y_max + 1.3)
    ax.set_xlim(-0.6, len(frame) - 0.15)
    ax.axvline(0.5, color="#6b7280", linestyle="--", linewidth=1.2, alpha=0.9)
    ax.axvline(4.5, color="#6b7280", linestyle="--", linewidth=1.2, alpha=0.9)
    ax.text(2.5, 10.15, "PSAs & IPPs", ha="center", va="bottom", fontsize=11, color=edge_color)
    average_realized = float(frame.loc[average_mask, "avg_price_php_per_kwh"].iloc[0])
    replacement_text = format_source_list(lcoe_sources)
    ax.annotate(
        f"What-if weighted average:\n{replacement_text}\nreplaced with LCOE values",
        xy=(5, average_what_if + 0.42),
        xytext=(5.02, 9.6),
        ha="center",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#cbd5e1", "lw": 0.9},
        arrowprops={"arrowstyle": "->", "color": edge_color, "lw": 1.0},
    )
    ax.annotate(
        "What-if weighted average:\nLCOE replacement plus\nspot net of excess markup",
        xy=(5 + width, average_what_if_net_excess + 0.38),
        xytext=(5.72, 8.15),
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

    footer_lines = [
        "Net-of-excess spot adjustment subtracts Luzon excess markup above 5,000 PhP/MWh = 271.08 PhP/MWh (0.2711 PhP/kWh) from the spot market bar."
    ]
    if lcoe_source_set == set(DEFAULT_LCOE_SOURCES):
        footer_lines.insert(
            0,
            "Solar value factor = generation-weighted Luzon solar capture price / average Luzon market price, using Aurora 30-minute prices and solar generation.",
        )
        footer_lines.insert(
            0,
            "Comparators: coal/gas use 2024 LCOE; solar uses 10.92 USD/MWh divided by the full 2025/26 Luzon solar value factor; all converted at 57 PhP/USD.",
        )
    else:
        footer_lines.insert(
            0,
            "Comparators: coal/gas use 2024 LCOE; solar, hydro, and spot stay at realized prices; LCOE values are converted at 57 PhP/USD.",
        )
    fig.text(
        0.5,
        0.03,
        "\n".join(footer_lines),
        ha="center",
        fontsize=9,
        color=edge_color,
    )
    fig.tight_layout(rect=(0.03, 0.11, 0.98, 0.98))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_png = Path(args.output_png)
    build_png(load_frame(input_csv), output_png, args.lcoe_sources)
    print(f"Wrote {output_png}")


if __name__ == "__main__":
    main()
