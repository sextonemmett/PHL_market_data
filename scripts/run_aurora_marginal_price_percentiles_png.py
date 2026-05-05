#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

DEFAULT_INPUT_XLSX = Path("data/aurora/World Bank_May26_Granular SRMC.xlsx")
DEFAULT_OUTPUT_DIR = Path("regressions")
SHEET_PATH = "xl/worksheets/sheet1.xml"
SHARED_STRINGS_PATH = "xl/sharedStrings.xml"
XML_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
TARGET_REGIONS = ("Luzon", "Visayas", "Mindanao")
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
P10_P90_COLOR = "#f6d7b0"
P25_P75_COLOR = "#f6a77d"
P50_COLOR = "#8f201d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate the Aurora marginal price percentile figure using the "
            "30-minute granular workbook in data/aurora."
        )
    )
    parser.add_argument("--input-xlsx", default=str(DEFAULT_INPUT_XLSX), help="Aurora workbook path.")
    parser.add_argument("--output-png", help="Output PNG path. Defaults to a month-ranged filename under regressions/.")
    parser.add_argument("--output-csv", help="Output CSV path. Defaults to a month-ranged filename under regressions/.")
    parser.add_argument("--y-min", type=float, default=-10000.0, help="Lower y-axis bound.")
    parser.add_argument("--y-max", type=float, default=33000.0, help="Upper y-axis bound.")
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
                    if column in {"A", "C", "H"}:
                        row_values[column] = cell_value(cell, shared_strings)

                local_date_time = row_values.get("A")
                region = row_values.get("C")
                wholesale_price = row_values.get("H")
                if local_date_time == "local_date_time":
                    elem.clear()
                    continue

                if region in TARGET_REGIONS and local_date_time and wholesale_price:
                    records.append(
                        {
                            "excel_serial": float(local_date_time),
                            "region": region,
                            "wholesale_price": float(wholesale_price),
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


def load_price_frame(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(iter_sheet_records(path))
    if frame.empty:
        raise ValueError(f"No Aurora price records found in {path}.")

    frame["timestamp"] = excel_serial_to_datetime(frame["excel_serial"]).dt.round("30min")
    # The workbook repeats the same regional wholesale price across technology rows.
    frame = (
        frame.groupby(["timestamp", "region"], as_index=False)["wholesale_price"]
        .median()
        .sort_values(["region", "timestamp"])
        .reset_index(drop=True)
    )
    frame["clock_label"] = frame["timestamp"].dt.strftime("%H:%M")
    frame["clock_order"] = frame["timestamp"].dt.hour * 2 + (frame["timestamp"].dt.minute // 30)
    return frame


def month_range_suffix(frame: pd.DataFrame) -> str:
    start_month = frame["timestamp"].min().strftime("%Y%m")
    end_month = frame["timestamp"].max().strftime("%Y%m")
    return f"{start_month}_{end_month}"


def default_output_paths(frame: pd.DataFrame) -> tuple[Path, Path]:
    suffix = month_range_suffix(frame)
    stem = f"aurora_marginal_price_percentiles_{suffix}"
    return DEFAULT_OUTPUT_DIR / f"{stem}.png", DEFAULT_OUTPUT_DIR / f"{stem}.csv"


def full_clock_labels() -> list[str]:
    start = pd.Timestamp("2026-01-01 00:00:00")
    return [(start + pd.Timedelta(minutes=30 * i)).strftime("%H:%M") for i in range(48)]


def build_quantile_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clock_labels = full_clock_labels()
    clock_order = {label: index for index, label in enumerate(clock_labels)}
    quantiles = (
        frame.groupby(["region", "clock_label"], observed=True)["wholesale_price"]
        .quantile(QUANTILES)
        .unstack()
        .reindex(pd.MultiIndex.from_product([TARGET_REGIONS, clock_labels], names=["region", "clock_label"]))
        .reset_index()
    )
    quantiles["clock_order"] = quantiles["clock_label"].map(clock_order)
    quantiles = quantiles.rename(
        columns={
            0.10: "p10",
            0.25: "p25",
            0.50: "p50",
            0.75: "p75",
            0.90: "p90",
        }
    )
    return quantiles.sort_values(["region", "clock_order"]).reset_index(drop=True)


def format_thousands(value: float, _: int) -> str:
    return f"{int(value):,}"


def plot_quantiles(quantiles: pd.DataFrame, output_png: Path, y_min: float, y_max: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4a4a4a",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#222222",
            "axes.titlesize": 16,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 9.0), sharex=True, sharey=True)

    for ax, region in zip(axes, TARGET_REGIONS):
        subset = (
            quantiles.loc[quantiles["region"] == region]
            .sort_values("clock_order")
            .reset_index(drop=True)
        )
        x_values = subset["clock_order"].to_numpy()

        ax.fill_between(x_values, subset["p10"].to_numpy(), subset["p90"].to_numpy(), color=P10_P90_COLOR, alpha=0.55)
        ax.fill_between(x_values, subset["p25"].to_numpy(), subset["p75"].to_numpy(), color=P25_P75_COLOR, alpha=0.7)
        ax.plot(x_values, subset["p50"].to_numpy(), color=P50_COLOR, linewidth=1.7)

        ax.set_ylabel(region, rotation=0, labelpad=42, va="center", ha="right", fontsize=12)
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
        ax.grid(True, axis="both", color="#d9d9d9", alpha=0.45, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

    hour_ticks = list(range(0, 48, 4))
    hour_labels = [f"{hour:02d}:00" for hour in range(0, 24, 2)]
    axes[-1].set_xticks(hour_ticks, hour_labels)
    axes[-1].set_xlim(0, 47)

    fig.suptitle("Marginal Price Percentiles by Region (PHP/MWh)", y=0.98)
    legend_handles = [
        Patch(facecolor=P10_P90_COLOR, edgecolor="none", alpha=0.55, label="10-90 percentile"),
        Patch(facecolor=P25_P75_COLOR, edgecolor="none", alpha=0.7, label="25-75 percentile"),
        Line2D([0], [0], color=P50_COLOR, linewidth=1.7, label="50 percentile"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False)
    fig.tight_layout(rect=(0.06, 0.08, 0.98, 0.965))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_xlsx)

    frame = load_price_frame(input_path)
    quantiles = build_quantile_frame(frame)
    default_png, default_csv = default_output_paths(frame)
    output_png = Path(args.output_png) if args.output_png else default_png
    output_csv = Path(args.output_csv) if args.output_csv else default_csv

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    quantiles.to_csv(output_csv, index=False)
    plot_quantiles(quantiles, output_png, y_min=args.y_min, y_max=args.y_max)

    print(f"Wrote {output_png}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
