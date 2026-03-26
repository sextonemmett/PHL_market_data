#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TIMESTAMP_TOKEN_RE = re.compile(r"(\d{8,12})")
PAIR_SPECS = (
    {"pair_key": "CLUZ_CVIS", "title": "Luzon-Visayas", "island_1": "Luzon", "island_2": "Visayas"},
    {"pair_key": "CVIS_CMIN", "title": "Visayas-Mindanao", "island_1": "Visayas", "island_2": "Mindanao"},
)
DEFAULT_OUTPUT_FIGURE = Path("regressions/direct_pair_price_difference_histograms.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot signed direct-pair price-difference histograms with density scaling, "
            "split by pair-specific link congestion."
        )
    )
    parser.add_argument("--direct-pair-panel", help="Direct-pair panel parquet path.")
    parser.add_argument(
        "--output-figure",
        default=str(DEFAULT_OUTPUT_FIGURE),
        help="Matplotlib figure output path, for example .png, .pdf, or .svg.",
    )
    return parser.parse_args()


def latest_matching_file(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} under {root}.")

    def sort_key(path: Path) -> tuple[str, str, str]:
        tokens = TIMESTAMP_TOKEN_RE.findall(path.stem)
        if not tokens:
            return ("", "", path.name)
        if len(tokens) == 1:
            return (tokens[0], tokens[0], path.name)
        return (tokens[-1], tokens[0], path.name)

    return max(matches, key=sort_key)


def load_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["time_interval"] = pd.to_datetime(frame["time_interval"])
    frame["signed_price_difference"] = frame["price_1"] - frame["price_2"]
    return frame


def choose_bins(series: pd.Series, count: int = 70) -> list[float]:
    lower = float(series.quantile(0.005))
    upper = float(series.quantile(0.995))
    if lower == upper:
        lower = float(series.min())
        upper = float(series.max())
    if lower == upper:
        lower -= 1.0
        upper += 1.0
    return list(pd.interval_range(start=lower, end=upper, periods=count).left) + [upper]


def render_subplot(ax: plt.Axes, pair_frame: pd.DataFrame, spec: dict[str, str]) -> None:
    signed_diff = pair_frame["signed_price_difference"]
    bins = choose_bins(signed_diff)
    uncongested = pair_frame.loc[pair_frame["link_congested_any"] == 0, "signed_price_difference"]
    congested = pair_frame.loc[pair_frame["link_congested_any"] == 1, "signed_price_difference"]

    ax.hist(
        uncongested,
        bins=bins,
        density=True,
        alpha=0.45,
        color="#4c78a8",
        label=f"Link congestion = 0 (n = {len(uncongested):,})",
    )
    ax.hist(
        congested,
        bins=bins,
        density=True,
        alpha=0.45,
        color="#c44e52",
        label=f"Link congestion = 1 (n = {len(congested):,})",
    )
    ax.axvline(0, color="#000000", linewidth=1.2, linestyle="--")
    ax.set_title(spec["title"], fontsize=16, color="#000000", fontweight="bold")
    ax.set_xlabel(
        f"Signed price difference: {spec['island_1']} price - {spec['island_2']} price (PHP)",
        fontsize=13,
        color="#000000",
    )
    ax.set_ylabel("Density", fontsize=13, color="#000000")
    ax.tick_params(axis="both", labelsize=12, colors="#000000")
    ax.grid(axis="y", color="#000000", linewidth=1.0, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")
    ax.legend(frameon=False, fontsize=11, loc="upper right")


def build_figure(frame: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), sharey=True)
    fig.patch.set_facecolor("#ffffff")

    for ax, spec in zip(axes, PAIR_SPECS):
        pair_frame = frame.loc[frame["pair_key"] == spec["pair_key"]].copy()
        render_subplot(ax, pair_frame, spec)

    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    panel_path = Path(args.direct_pair_panel) if args.direct_pair_panel else latest_matching_file(
        Path("data/panels"),
        "RTD_DIRECT_PAIR_PANEL_*.parquet",
    )
    output_figure = Path(args.output_figure)

    frame = load_panel(panel_path)
    fig = build_figure(frame)

    output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output_figure}")


if __name__ == "__main__":
    main()
