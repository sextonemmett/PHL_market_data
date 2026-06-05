#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

DEFAULT_INPUT_XLSX = Path("data/aurora/World Bank_May26_Granular SRMC.xlsx")
DEFAULT_OUTPUT_DIR = Path("output/aurora_economic_markup_ridgelines")
SHEET_PATH = "xl/worksheets/sheet1.xml"
SHARED_STRINGS_PATH = "xl/sharedStrings.xml"
XML_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
TARGET_REGIONS = ("Luzon", "Visayas", "Mindanao")
REGION_COLORS = {
    "Luzon": "#8f3b1b",
    "Visayas": "#0f766e",
    "Mindanao": "#556b2f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one ridgeline PNG per island using 30-minute economic markups "
            "from the Aurora World Bank SRMC workbook."
        )
    )
    parser.add_argument("--input-xlsx", default=str(DEFAULT_INPUT_XLSX), help="Aurora workbook path.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the three ridgeline PNGs.",
    )
    return parser.parse_args()


def load_shared_strings(zf: ZipFile) -> list[str]:
    root = ET.fromstring(zf.read(SHARED_STRINGS_PATH))
    strings: list[str] = []
    for item in root.findall("a:si", XML_NS):
        text = "".join(node.text or "" for node in item.iterfind(".//a:t", XML_NS))
        strings.append(text)
    return strings


def cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iterfind(".//a:t", XML_NS))

    value_node = cell.find("a:v", XML_NS)
    if value_node is None or value_node.text is None:
        return ""

    value = value_node.text
    if cell_type == "s":
        return shared_strings[int(value)]
    return value


def iter_sheet_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with ZipFile(path) as zf:
        shared_strings = load_shared_strings(zf)
        with zf.open(SHEET_PATH) as sheet_file:
            for _, elem in ET.iterparse(sheet_file, events=("end",)):
                if elem.tag != f"{{{XML_NS['a']}}}row":
                    continue

                row_values: dict[str, str] = {}
                for cell in elem.findall("a:c", XML_NS):
                    reference = cell.attrib.get("r", "")
                    column = "".join(ch for ch in reference if ch.isalpha())
                    if column in {"A", "C", "G", "J"}:
                        row_values[column] = cell_value(cell, shared_strings)

                local_date_time = row_values.get("A")
                region = row_values.get("C")
                generation_mw = row_values.get("G")
                profit = row_values.get("J")

                if local_date_time == "local_date_time":
                    elem.clear()
                    continue

                if region in TARGET_REGIONS and local_date_time and generation_mw and profit:
                    records.append(
                        {
                            "excel_serial": float(local_date_time),
                            "region": region,
                            "generation_mw": float(generation_mw),
                            "economic_markup": float(profit),
                        }
                    )

                elem.clear()
    return records


def excel_serial_to_datetime(serials: pd.Series) -> pd.Series:
    values = serials.to_numpy(dtype=float)
    whole_days = np.floor(values).astype("int64")
    seconds = np.rint((values - whole_days) * 86400.0).astype("int64")
    origin = pd.Timestamp("1899-12-30")
    timestamps = origin + pd.to_timedelta(whole_days, unit="D") + pd.to_timedelta(seconds, unit="s")
    return pd.Series(timestamps, index=serials.index)


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    clean = pd.DataFrame({"values": values, "weights": weights}).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("nan")
    if clean["weights"].sum() <= 0:
        return float(clean["values"].mean())
    return float(np.average(clean["values"], weights=clean["weights"]))


def load_markup_frame(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(iter_sheet_records(path))
    if frame.empty:
        raise ValueError(f"No Aurora markup records found in {path}.")

    frame["timestamp"] = excel_serial_to_datetime(frame["excel_serial"]).dt.round("30min")

    aggregated_rows: list[dict[str, object]] = []
    for (timestamp, region), group in frame.groupby(["timestamp", "region"], sort=True):
        aggregated_rows.append(
            {
                "timestamp": timestamp,
                "region": region,
                "economic_markup": weighted_average(group["economic_markup"], group["generation_mw"]),
            }
        )

    aggregated = pd.DataFrame(aggregated_rows).sort_values(["region", "timestamp"]).reset_index(drop=True)
    aggregated["clock_label"] = aggregated["timestamp"].dt.strftime("%H:%M")
    aggregated["clock_order"] = aggregated["timestamp"].dt.hour * 2 + (aggregated["timestamp"].dt.minute // 30)
    return aggregated


def month_range_suffix(frame: pd.DataFrame) -> str:
    start_month = frame["timestamp"].min().strftime("%Y%m")
    end_month = frame["timestamp"].max().strftime("%Y%m")
    return f"{start_month}_{end_month}"


def output_paths(frame: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    suffix = month_range_suffix(frame)
    return {
        region: output_dir / f"aurora_economic_markup_ridgeline_{region.lower()}_{suffix}.png"
        for region in TARGET_REGIONS
    }


def pooled_output_path(frame: pd.DataFrame, output_dir: Path) -> Path:
    suffix = month_range_suffix(frame)
    return output_dir / f"aurora_economic_markup_density_pooled_3x1_{suffix}.png"


def format_thousands(value: float, _: int) -> str:
    return f"{int(value):,}"


def density_curve(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return np.zeros_like(x_grid)
    unique_values = np.unique(values)
    if unique_values.size < 2:
        return np.zeros_like(x_grid)
    density = gaussian_kde(values)(x_grid)
    peak = float(density.max())
    if peak <= 0:
        return np.zeros_like(x_grid)
    return density / peak


def pooled_density_curve(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return np.zeros_like(x_grid)
    unique_values = np.unique(values)
    if unique_values.size < 2:
        return np.zeros_like(x_grid)
    return gaussian_kde(values)(x_grid)


def plot_region_ridgeline(frame: pd.DataFrame, region: str, output_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#222222",
        }
    )

    subset = frame.loc[frame["region"] == region].copy()
    if subset.empty:
        raise ValueError(f"No records found for region {region}.")

    by_clock = {
        clock_order: group["economic_markup"].dropna().to_numpy(dtype=float)
        for clock_order, group in subset.groupby("clock_order", sort=True)
    }
    all_values = subset["economic_markup"].dropna().to_numpy(dtype=float)
    x_min = float(all_values.min())
    x_max = float(all_values.max())
    padding = max((x_max - x_min) * 0.05, 250.0)
    x_grid = np.linspace(x_min - padding, x_max + padding, 500)

    base_color = REGION_COLORS[region]
    fill_color = to_rgba(base_color, alpha=0.78)
    line_color = to_rgba(base_color, alpha=0.98)
    median_color = "#1f1f1f"
    ridge_scale = 0.86

    fig, ax = plt.subplots(figsize=(13.5, 17.0))

    ax.axvline(0.0, color="#c53030", linewidth=1.1, linestyle="--", alpha=0.95, zorder=1.5)

    for clock_order in range(48):
        baseline = float(clock_order)
        values = by_clock.get(clock_order, np.array([], dtype=float))
        density = density_curve(values, x_grid) * ridge_scale

        ax.hlines(baseline, x_grid[0], x_grid[-1], color="#d8d8d8", linewidth=0.6, zorder=1)
        if np.any(density > 0):
            ax.fill_between(x_grid, baseline, baseline + density, color=fill_color, linewidth=0.0, zorder=2)
            ax.plot(x_grid, baseline + density, color=line_color, linewidth=1.0, zorder=3)
            median = float(np.median(values))
            ax.vlines(median, baseline, baseline + 0.72, color=median_color, linewidth=0.8, alpha=0.9, zorder=4)

    tick_positions = list(range(0, 48, 4))
    tick_labels = [f"{hour:02d}:00" for hour in range(0, 24, 2)]
    ax.set_yticks(tick_positions, tick_labels)
    ax.set_ylim(-0.5, 47.9)
    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.xaxis.set_major_formatter(FuncFormatter(format_thousands))
    ax.grid(axis="x", color="#d0d0d0", alpha=0.4, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.set_xlabel("Economic markup (PHP/MWh)")
    ax.set_ylabel("Half-hour of day")
    ax.set_title(f"{region}: Distribution of 30-minute economic markups", loc="left", fontsize=16, pad=12)
    fig.text(
        0.125,
        0.985,
        "Generation-weighted island markups by time-of-day bin",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.06, 0.03, 0.99, 0.97))
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pooled_density_figure(frame: pd.DataFrame, output_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#222222",
        }
    )

    all_values = frame["economic_markup"].dropna().to_numpy(dtype=float)
    x_min = float(all_values.min())
    x_max = float(all_values.max())
    padding = max((x_max - x_min) * 0.05, 250.0)
    x_grid = np.linspace(x_min - padding, x_max + padding, 600)

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 9.5), sharex=True)

    for ax, region in zip(axes, TARGET_REGIONS):
        subset = frame.loc[frame["region"] == region, "economic_markup"].dropna().to_numpy(dtype=float)
        density = pooled_density_curve(subset, x_grid)
        base_color = REGION_COLORS[region]
        fill_color = to_rgba(base_color, alpha=0.72)
        line_color = to_rgba(base_color, alpha=0.98)

        ax.fill_between(x_grid, 0.0, density, color=fill_color, linewidth=0.0)
        ax.plot(x_grid, density, color=line_color, linewidth=1.5)
        ax.axvline(float(np.median(subset)), color="#1f1f1f", linewidth=0.9, alpha=0.9)
        ax.set_title(region, loc="left", fontsize=14, pad=6)
        ax.set_ylabel("Density")
        ax.grid(axis="x", color="#d0d0d0", alpha=0.4, linewidth=0.8)
        ax.grid(axis="y", color="#d0d0d0", alpha=0.22, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)

    axes[-1].set_xlabel("Economic markup (PHP/MWh)")
    axes[-1].xaxis.set_major_formatter(FuncFormatter(format_thousands))
    fig.suptitle("Pooled distribution of generation-weighted economic markups by island", fontsize=17, y=0.985)
    fig.tight_layout(rect=(0.06, 0.04, 0.99, 0.965))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_xlsx)
    output_dir = Path(args.output_dir)

    frame = load_markup_frame(input_path)
    outputs = output_paths(frame, output_dir)
    for region, output_png in outputs.items():
        plot_region_ridgeline(frame, region, output_png)
        print(f"Wrote {output_png}")

    pooled_png = pooled_output_path(frame, output_dir)
    plot_pooled_density_figure(frame, pooled_png)
    print(f"Wrote {pooled_png}")


if __name__ == "__main__":
    main()
