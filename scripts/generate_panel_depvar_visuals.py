#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_DIRECT_PANEL = Path("data/panels/RTD_DIRECT_PAIR_PANEL_202512212305_202603190000.parquet")
DEFAULT_ISLAND_SYSTEM_PANEL = Path("data/panels/RTD_ISLAND_SYSTEM_PANEL_202512212305_202603210000.parquet")
DEFAULT_ISLAND_CONGESTION_PANEL = Path("data/panels/RTD_ISLAND_CONGESTION_PANEL_202512212305_202603210000.parquet")
DEFAULT_OUTPUT_ROOT = Path("regressions")
DEFAULT_ASSETS_DIR = DEFAULT_OUTPUT_ROOT / "panel_depvar_visual_assets"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_ROOT / "panel_depvar_visuals.html"

REGION_LABELS = {"CLUZ": "Luzon", "CVIS": "Visayas", "CMIN": "Mindanao"}
REGION_COLORS = {"CLUZ": "#7f2704", "CVIS": "#d95f0e", "CMIN": "#1f78b4"}
PAIR_LABELS = {"CLUZ_CVIS": "Luzon-Visayas", "CVIS_CMIN": "Visayas-Mindanao"}
PAIR_COLORS = {"CLUZ_CVIS": "#0b4f6c", "CVIS_CMIN": "#a63d40"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visuals for panel dependent variables.")
    parser.add_argument("--direct-pair-panel", default=str(DEFAULT_DIRECT_PANEL), help="Direct-pair panel parquet path.")
    parser.add_argument(
        "--island-system-panel",
        default=str(DEFAULT_ISLAND_SYSTEM_PANEL),
        help="Island-system panel parquet path.",
    )
    parser.add_argument(
        "--island-congestion-panel",
        default=str(DEFAULT_ISLAND_CONGESTION_PANEL),
        help="Focal-island congestion panel parquet path.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output directory for report assets.")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH), help="HTML report output path.")
    parser.add_argument("--assets-dir", default=str(DEFAULT_ASSETS_DIR), help="Directory for generated PNGs.")
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def apply_date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def load_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["time_interval"] = pd.to_datetime(frame["time_interval"])
    frame["run_date"] = frame["time_interval"].dt.normalize()
    return frame


def build_summary_stats(frame: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    summary = (
        frame.groupby(group_col, observed=True)[value_col]
        .agg(
            observations="size",
            mean="mean",
            median="median",
            p90=lambda s: s.quantile(0.90),
            p99=lambda s: s.quantile(0.99),
            maximum="max",
        )
        .reset_index()
    )
    return summary


def plot_daily_profile(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    labels: dict[str, str],
    colors: dict[str, str],
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    daily = (
        frame.groupby(["run_date", group_col], observed=True)[value_col]
        .agg(
            median_value="median",
            p10_value=lambda s: s.quantile(0.10),
            p90_value=lambda s: s.quantile(0.90),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5.2))
    for key in daily[group_col].drop_duplicates():
        subset = daily.loc[daily[group_col] == key].sort_values("run_date")
        color = colors[key]
        ax.plot(
            subset["run_date"],
            subset["median_value"],
            color=color,
            linewidth=1.9,
            label=labels.get(key, key),
        )
        ax.fill_between(
            subset["run_date"],
            subset["p10_value"],
            subset["p90_value"],
            color=color,
            alpha=0.12,
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    apply_date_axis(ax)
    ax.legend(frameon=False, ncol=min(3, len(labels)))
    save_figure(fig, path)


def plot_box_distribution(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    labels: dict[str, str],
    colors: dict[str, str],
    title: str,
    xlabel: str,
    path: Path,
) -> None:
    order = [key for key in labels if key in set(frame[group_col].unique())]
    values = [frame.loc[frame[group_col] == key, value_col].to_numpy() for key in order]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    box = ax.boxplot(
        values,
        patch_artist=True,
        tick_labels=[labels.get(key, key) for key in order],
        showfliers=False,
        widths=0.55,
        medianprops={"color": "#0b1f33", "linewidth": 1.5},
        whiskerprops={"color": "#486581"},
        capprops={"color": "#486581"},
    )
    for patch, key in zip(box["boxes"], order):
        patch.set_facecolor(colors[key])
        patch.set_alpha(0.55)
        patch.set_edgecolor(colors[key])

    ax.set_title(title)
    ax.set_ylabel(xlabel)
    ax.grid(axis="y", alpha=0.18)
    save_figure(fig, path)


def summary_to_html(summary: pd.DataFrame, key_col: str, labels: dict[str, str], digits: int = 1) -> str:
    table = summary.copy()
    table[key_col] = table[key_col].map(lambda value: labels.get(value, value))
    table = table.rename(columns={key_col: "Group"})
    for col in ["mean", "median", "p90", "p99", "maximum"]:
        table[col] = table[col].map(lambda value: f"{value:,.{digits}f}")
    table["observations"] = table["observations"].map(lambda value: f"{int(value):,}")
    return table.to_html(index=False, escape=False, classes=["summary-table"])


def render_report(
    report_path: Path,
    direct_panel_path: Path,
    island_panel_path: Path,
    focal_panel_path: Path,
    figures: dict[str, Path],
    tables: dict[str, str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    figure_paths = {key: str(path.relative_to(report_path.parent)) for key, path in figures.items()}

    css = """
body { font-family: Georgia, "Times New Roman", serif; margin: 32px auto; max-width: 1280px; color: #102a43; line-height: 1.55; background: #f4f1ea; }
h1, h2, h3 { color: #0b1f33; }
h1 { margin-bottom: 10px; }
p { margin: 0 0 14px; }
code { background: #dde7f0; color: #0b1f33; padding: 2px 5px; border-radius: 4px; }
.lead { font-size: 17px; color: #243b53; margin-bottom: 22px; }
.panel-card { background: #faf7f1; border: 1px solid #d9e2ec; border-radius: 14px; padding: 22px 24px; margin: 24px 0 30px; box-shadow: 0 10px 28px rgba(16, 42, 67, 0.08); }
.meta { color: #486581; font-size: 14px; margin-bottom: 12px; }
.formula { background: #102a43; color: #fdfdfd; border-radius: 12px; padding: 12px 14px; font-size: 18px; margin: 0 0 18px; }
.figure-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin: 18px 0; }
.figure-card { background: #eef3f7; border-radius: 12px; padding: 12px; }
.figure-card img { width: 100%; border-radius: 8px; display: block; }
.figure-card .caption { font-size: 14px; color: #243b53; margin-top: 8px; }
.summary-table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 14px; }
.summary-table th { background: #0b1f33; color: #fdfdfd; padding: 10px 12px; text-align: center; border: 1px solid #102a43; }
.summary-table td { border: 1px solid #bcccdc; padding: 8px 12px; background: #fffdf8; text-align: right; }
.summary-table td:first-child { text-align: left; font-weight: 600; background: #e6ecf2; }
.notes { color: #243b53; font-size: 14px; background: #e6ecf2; border-radius: 12px; padding: 16px 18px; margin-top: 26px; }
"""

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Panel Dependent Variable Visuals</title>
  <style>{css}</style>
</head>
<body>
  <h1>Panel Dependent Variable Visuals</h1>
  <p class="lead">This page summarizes the regression dependent variables from the three panel constructions: direct-pair price gaps, island-vs-system price gaps, and focal-island price levels. Each panel includes a daily profile chart and a distribution chart, plus summary statistics for quick comparison.</p>

  <section class="panel-card">
    <h2>Direct-Pair Panel</h2>
    <div class="meta">Source: <code>{direct_panel_path}</code></div>
    <div class="formula">Dependent variable: <em>|P<sub>i,t</sub> - P<sub>j,t</sub>|</em> (PHP/MWh)</div>
    <div class="figure-grid">
      <div class="figure-card">
        <img src="{figure_paths["direct_daily"]}" alt="Daily direct-pair price gap profile">
        <div class="caption">Daily median direct-pair price gap with 10th to 90th percentile band.</div>
      </div>
      <div class="figure-card">
        <img src="{figure_paths["direct_box"]}" alt="Direct-pair price gap distribution">
        <div class="caption">Distribution of direct-pair price gaps by island pair. Outlier points are hidden to keep the central mass readable.</div>
      </div>
    </div>
    {tables["direct"]}
  </section>

  <section class="panel-card">
    <h2>Island-System Panel</h2>
    <div class="meta">Source: <code>{island_panel_path}</code></div>
    <div class="formula">Dependent variable: <em>|P<sub>i,t</sub> - P<sub>sys,t</sub>|</em> (PHP/MWh)</div>
    <div class="figure-grid">
      <div class="figure-card">
        <img src="{figure_paths["island_daily"]}" alt="Daily island-system price gap profile">
        <div class="caption">Daily median island-versus-system price gap with 10th to 90th percentile band.</div>
      </div>
      <div class="figure-card">
        <img src="{figure_paths["island_box"]}" alt="Island-system price gap distribution">
        <div class="caption">Distribution of island-versus-system price gaps by island. Outlier points are hidden to keep the central mass readable.</div>
      </div>
    </div>
    {tables["island"]}
  </section>

  <section class="panel-card">
    <h2>Focal-Island Price Panel</h2>
    <div class="meta">Source: <code>{focal_panel_path}</code></div>
    <div class="formula">Dependent variable: <em>P<sub>i,t</sub></em> (PHP/MWh)</div>
    <div class="figure-grid">
      <div class="figure-card">
        <img src="{figure_paths["focal_daily"]}" alt="Daily focal-island price profile">
        <div class="caption">Daily median focal-island price with 10th to 90th percentile band.</div>
      </div>
      <div class="figure-card">
        <img src="{figure_paths["focal_box"]}" alt="Focal-island price distribution">
        <div class="caption">Distribution of focal-island price levels by island. Outlier points are hidden to keep the central mass readable.</div>
      </div>
    </div>
    {tables["focal"]}
  </section>

  <div class="notes">
    <p>All visuals are based on the current regression panel outputs on disk.</p>
    <p>The direct-pair and island-system dependent variables are absolute price-gap measures. The focal-island panel uses the raw island price level as the dependent variable for the focal-island regressions.</p>
  </div>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    direct_panel_path = Path(args.direct_pair_panel)
    island_panel_path = Path(args.island_system_panel)
    focal_panel_path = Path(args.island_congestion_panel)
    report_path = Path(args.report_path)
    assets_dir = Path(args.assets_dir)

    direct = load_panel(direct_panel_path)
    island = load_panel(island_panel_path)
    focal = load_panel(focal_panel_path)

    figures = {
        "direct_daily": assets_dir / "direct_pair_daily_gap.png",
        "direct_box": assets_dir / "direct_pair_gap_boxplot.png",
        "island_daily": assets_dir / "island_system_daily_gap.png",
        "island_box": assets_dir / "island_system_gap_boxplot.png",
        "focal_daily": assets_dir / "focal_island_daily_price.png",
        "focal_box": assets_dir / "focal_island_price_boxplot.png",
    }

    plot_daily_profile(
        direct,
        "pair_key",
        "dep_abs_price_gap",
        PAIR_LABELS,
        PAIR_COLORS,
        "Daily Direct-Pair Price Gap",
        "Absolute price gap (PHP/MWh)",
        figures["direct_daily"],
    )
    plot_box_distribution(
        direct,
        "pair_key",
        "dep_abs_price_gap",
        PAIR_LABELS,
        PAIR_COLORS,
        "Distribution of Direct-Pair Price Gaps",
        "Absolute price gap (PHP/MWh)",
        figures["direct_box"],
    )

    plot_daily_profile(
        island,
        "island_code",
        "dep_price_minus_sys",
        REGION_LABELS,
        REGION_COLORS,
        "Daily Island-Versus-System Price Gap",
        "Absolute price gap (PHP/MWh)",
        figures["island_daily"],
    )
    plot_box_distribution(
        island,
        "island_code",
        "dep_price_minus_sys",
        REGION_LABELS,
        REGION_COLORS,
        "Distribution of Island-Versus-System Price Gaps",
        "Absolute price gap (PHP/MWh)",
        figures["island_box"],
    )

    plot_daily_profile(
        focal,
        "island_1",
        "price_island_1",
        REGION_LABELS,
        REGION_COLORS,
        "Daily Focal-Island Price Level",
        "Price (PHP/MWh)",
        figures["focal_daily"],
    )
    plot_box_distribution(
        focal,
        "island_1",
        "price_island_1",
        REGION_LABELS,
        REGION_COLORS,
        "Distribution of Focal-Island Price Levels",
        "Price (PHP/MWh)",
        figures["focal_box"],
    )

    tables = {
        "direct": summary_to_html(build_summary_stats(direct, "pair_key", "dep_abs_price_gap"), "pair_key", PAIR_LABELS),
        "island": summary_to_html(
            build_summary_stats(island, "island_code", "dep_price_minus_sys"),
            "island_code",
            REGION_LABELS,
        ),
        "focal": summary_to_html(
            build_summary_stats(focal, "island_1", "price_island_1"),
            "island_1",
            REGION_LABELS,
        ),
    }

    render_report(report_path, direct_panel_path, island_panel_path, focal_panel_path, figures, tables)
    print(f"Wrote {report_path}")
    for path in figures.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
