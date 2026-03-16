#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

DATASETS = {
    "reserve": {
        "parquet": Path("data/mp_reserve/combined/MP_RESERVE_20251216_20260316.parquet"),
        "title": "Reserve",
        "color": "#1f4e79",
    },
    "mp": {
        "parquet": Path("data/mp/combined/MP_20251216_20260316.parquet"),
        "title": "Market Price",
        "color": "#8b4513",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate analysis figures for the combined reserve and MP Parquet datasets."
    )
    parser.add_argument(
        "--output-root",
        default="reports/analysis",
        help="Directory for analysis PNGs and summary CSVs.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
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


def extract_prefix(resource_name: str) -> str:
    return resource_name.split("_")[0] if "_" in resource_name else resource_name


def load_data() -> dict[str, pd.DataFrame]:
    data = {}
    for name, config in DATASETS.items():
        frame = pd.read_parquet(config["parquet"]).copy()
        frame["RUN_DATE"] = frame["RUN_TIME"].dt.normalize()
        frame["SLOT"] = frame["RUN_TIME"].dt.hour * 12 + frame["RUN_TIME"].dt.minute // 5
        data[name] = frame
    return data


def build_daily_region_mean(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(["RUN_DATE", "REGION_NAME"], observed=True)
        .agg(
            mean_price=("MARGINAL_PRICE", "mean"),
            median_price=("MARGINAL_PRICE", "median"),
            rows=("RUN_TIME", "size"),
        )
        .reset_index()
    )
    return daily


def build_effective_setter_summary(
    df: pd.DataFrame,
    extra_group_cols: list[str] | None = None,
    entity_mode: str = "resource",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    extra_group_cols = extra_group_cols or []
    unique_cols = ["RUN_TIME", "REGION_NAME", "MARGINAL_PRICE", "RESOURCE_NAME", *extra_group_cols]
    interval_group_cols = ["RUN_TIME", "REGION_NAME", *extra_group_cols]

    unique_rows = df[unique_cols].drop_duplicates()
    unique_rows["ENTITY_PREFIX"] = unique_rows["RESOURCE_NAME"].astype(str).map(extract_prefix)
    if entity_mode == "prefix":
        unique_rows["ENTITY_NAME"] = unique_rows["ENTITY_PREFIX"]
    else:
        unique_rows["ENTITY_NAME"] = unique_rows["RESOURCE_NAME"].astype(str)

    unique_entity_rows = unique_rows[
        ["RUN_TIME", "REGION_NAME", "MARGINAL_PRICE", "ENTITY_NAME", "ENTITY_PREFIX", *extra_group_cols]
    ].drop_duplicates()

    price_counts = (
        unique_entity_rows.groupby([*interval_group_cols, "MARGINAL_PRICE"], observed=True)
        .size()
        .rename("price_resource_count")
        .reset_index()
    )
    price_counts["max_price_resource_count"] = price_counts.groupby(
        interval_group_cols, observed=True
    )["price_resource_count"].transform("max")
    dominant_prices = price_counts[
        price_counts["price_resource_count"] == price_counts["max_price_resource_count"]
    ].copy()

    setters = unique_rows.merge(
        dominant_prices[
            [*interval_group_cols, "MARGINAL_PRICE", "price_resource_count"]
        ],
        on=[*interval_group_cols, "MARGINAL_PRICE"],
        how="inner",
    )
    setters["setter_credit"] = 1.0 / setters["price_resource_count"]

    total_interval_regions = unique_entity_rows[interval_group_cols].drop_duplicates().shape[0]
    summary = (
        setters.groupby([*extra_group_cols, "ENTITY_NAME", "ENTITY_PREFIX"], observed=True)
        .agg(
            setter_credit=("setter_credit", "sum"),
            active_regions=("REGION_NAME", "nunique"),
            active_intervals=("RUN_TIME", "nunique"),
            mean_setter_price=("MARGINAL_PRICE", "mean"),
        )
        .reset_index()
        .sort_values("setter_credit", ascending=False)
    )
    summary["setter_share_pct"] = summary["setter_credit"] / total_interval_regions * 100.0

    region_summary = (
        setters.groupby([*extra_group_cols, "ENTITY_NAME", "ENTITY_PREFIX", "REGION_NAME"], observed=True)
        .agg(
            setter_credit=("setter_credit", "sum"),
            active_intervals=("RUN_TIME", "nunique"),
            mean_setter_price=("MARGINAL_PRICE", "mean"),
        )
        .reset_index()
        .sort_values(
            [*extra_group_cols, "REGION_NAME", "setter_credit"],
            ascending=[True] * len(extra_group_cols) + [True, False],
        )
    )
    return summary, region_summary


def save_summaries(
    daily_region_means: dict[str, pd.DataFrame],
    setter_summaries: dict[str, pd.DataFrame],
    setter_region_summaries: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in daily_region_means.items():
        frame.to_csv(output_dir / f"{name}_daily_region_mean.csv", index=False)
    for name, frame in setter_summaries.items():
        frame.to_csv(output_dir / f"{name}_effective_setters.csv", index=False)
    for name, frame in setter_region_summaries.items():
        frame.to_csv(output_dir / f"{name}_effective_setters_by_region.csv", index=False)

    reserve_overlap = (
        setter_summaries["reserve"]
        .groupby(["COMMODITY_TYPE", "ENTITY_PREFIX"], observed=True)["setter_share_pct"]
        .sum()
        .reset_index()
        .rename(columns={"setter_share_pct": "reserve_setter_share_pct"})
    )
    overlap = (
        reserve_overlap
        .merge(
            setter_summaries["mp"][["ENTITY_PREFIX", "setter_share_pct"]].rename(
                columns={"setter_share_pct": "mp_setter_share_pct"}
            ),
            on="ENTITY_PREFIX",
            how="outer",
        )
    )
    overlap["reserve_setter_share_pct"] = overlap["reserve_setter_share_pct"].fillna(0.0)
    overlap["mp_setter_share_pct"] = overlap["mp_setter_share_pct"].fillna(0.0)
    overlap["COMMODITY_TYPE"] = overlap["COMMODITY_TYPE"].astype("string").fillna("none")
    overlap = overlap.sort_values(
        ["COMMODITY_TYPE", "reserve_setter_share_pct", "mp_setter_share_pct"],
        ascending=[True, False, False],
    )
    overlap.to_csv(output_dir / "setter_overlap.csv", index=False)


def plot_regional_daily_heatmaps(
    reserve_daily: pd.DataFrame,
    mp_daily: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.6), sharex=False)
    global_min = min(reserve_daily["mean_price"].min(), mp_daily["mean_price"].min())
    global_max = max(reserve_daily["mean_price"].max(), mp_daily["mean_price"].max())
    cmap = LinearSegmentedColormap.from_list("white_red", ["#ffffff", "#b30000"])

    image = None
    for ax, (dataset_name, frame) in zip(axes, [("reserve", reserve_daily), ("mp", mp_daily)]):
        pivot = (
            frame.pivot(index="REGION_NAME", columns="RUN_DATE", values="mean_price")
            .sort_index()
            .astype(float)
        )
        dates = pd.to_datetime(pivot.columns)
        image = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=global_min, vmax=global_max)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        tick_positions = np.linspace(0, len(dates) - 1, min(10, len(dates)), dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([dates[pos].strftime("%Y-%m-%d") for pos in tick_positions], rotation=45, ha="right")
        ax.set_title(f"{DATASETS[dataset_name]['title']}: Average Daily Regional Price")

    colorbar = fig.colorbar(image, ax=axes, shrink=0.9)
    colorbar.set_label("Average Price")
    save_figure(fig, output_path)


def plot_top_effective_setters(
    reserve_setters: pd.DataFrame,
    mp_setters: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=False)
    product_axes = axes.flatten()
    reserve_products = ["Dr", "Fr", "Rd", "Ru"]

    for ax, product in zip(product_axes[:4], reserve_products):
        frame = reserve_setters[reserve_setters["COMMODITY_TYPE"].astype(str) == product]
        top = frame.head(12).sort_values("setter_share_pct")
        ax.barh(top["ENTITY_NAME"].astype(str), top["setter_share_pct"], color=DATASETS["reserve"]["color"])
        ax.set_title(f"Reserve {product}: Top Effective Setters")
        ax.set_xlabel("Effective Setter Share (%)")
        ax.set_ylabel("Resource")

    mp_ax = product_axes[4]
    mp_top = mp_setters.head(12).sort_values("setter_share_pct")
    mp_ax.barh(mp_top["ENTITY_NAME"].astype(str), mp_top["setter_share_pct"], color=DATASETS["mp"]["color"])
    mp_ax.set_title("Market Price: Top Effective Setters (Collapsed Prefix)")
    mp_ax.set_xlabel("Effective Setter Share (%)")
    mp_ax.set_ylabel("Prefix")

    note_ax = product_axes[5]
    note_ax.axis("off")
    note_ax.text(
        0.0,
        0.95,
        "Reserve setters stay at the\nresource level inside each reserve\nproduct bucket.\n\n"
        "MP setters are collapsed to the\nresource prefix before shares are\ncomputed.",
        va="top",
        ha="left",
        fontsize=11,
    )

    save_figure(fig, output_path)


def plot_setter_interaction_scatter(
    reserve_setters: pd.DataFrame,
    mp_setters: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    product_colors = {"Dr": "#2b6cb0", "Fr": "#d69e2e", "Rd": "#c53030", "Ru": "#805ad5"}

    for ax, product in zip(axes.flatten(), ["Dr", "Fr", "Rd", "Ru"]):
        reserve_prefix = (
            reserve_setters[reserve_setters["COMMODITY_TYPE"].astype(str) == product]
            .groupby("ENTITY_PREFIX", observed=True)["setter_share_pct"]
            .sum()
            .reset_index()
            .rename(columns={"setter_share_pct": "reserve_share"})
        )
        overlap = reserve_prefix.merge(
            mp_setters[["ENTITY_PREFIX", "setter_share_pct"]].rename(
                columns={"setter_share_pct": "mp_share"}
            ),
            on="ENTITY_PREFIX",
            how="inner",
        )
        overlap["combined_share"] = overlap["reserve_share"] + overlap["mp_share"]
        ax.scatter(
            overlap["reserve_share"],
            overlap["mp_share"],
            s=28 + overlap["combined_share"] * 28,
            alpha=0.72,
            color=product_colors[product],
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_title(f"Reserve {product} vs MP Setter Share")
        ax.set_xlabel("Reserve Setter Share (%)")
        ax.set_ylabel("MP Setter Share (%)")

        for _, row in overlap.nlargest(5, "combined_share").iterrows():
            ax.annotate(
                row["ENTITY_PREFIX"],
                (row["reserve_share"], row["mp_share"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
            )

    save_figure(fig, output_path)


def plot_mp_intraday_spaghetti(mp: pd.DataFrame, output_path: Path) -> None:
    daily_slot = (
        mp.groupby(["RUN_DATE", "SLOT"], observed=True)
        .agg(mean_price=("MARGINAL_PRICE", "mean"))
        .reset_index()
    )
    dates = sorted(daily_slot["RUN_DATE"].unique())
    color_map = plt.get_cmap("viridis")
    colors = color_map(np.linspace(0.08, 0.95, len(dates)))

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for color, run_date in zip(colors, dates):
        frame = daily_slot[daily_slot["RUN_DATE"] == run_date]
        ax.plot(frame["SLOT"], frame["mean_price"], color=color, alpha=0.23, linewidth=1.0)

    median_line = (
        daily_slot.groupby("SLOT", observed=True)["mean_price"]
        .median()
        .reset_index()
    )
    ax.plot(median_line["SLOT"], median_line["mean_price"], color="#111827", linewidth=2.5, label="Median day")
    ax.set_title("MP Intraday Shape: One Mean Price Line per Day")
    ax.set_xlabel("5-Minute Slot in Day")
    ax.set_ylabel("Average MP Price")
    ax.set_xticks(np.arange(0, 289, 24))
    ax.set_xticklabels([f"{int(slot // 12):02d}:00" for slot in np.arange(0, 289, 24)])
    ax.legend(frameon=False)
    save_figure(fig, output_path)


def plot_mp_intraday_spaghetti_by_region(mp: pd.DataFrame, output_path: Path) -> None:
    regional = (
        mp.groupby(["REGION_NAME", "RUN_DATE", "SLOT"], observed=True)
        .agg(mean_price=("MARGINAL_PRICE", "mean"))
        .reset_index()
    )
    dates = sorted(regional["RUN_DATE"].unique())
    color_map = plt.get_cmap("plasma")
    colors = color_map(np.linspace(0.08, 0.95, len(dates)))

    regions = sorted(regional["REGION_NAME"].astype(str).unique())
    fig, axes = plt.subplots(len(regions), 1, figsize=(13, 9), sharex=True)
    if len(regions) == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        region_frame = regional[regional["REGION_NAME"].astype(str) == region]
        for color, run_date in zip(colors, dates):
            frame = region_frame[region_frame["RUN_DATE"] == run_date]
            ax.plot(frame["SLOT"], frame["mean_price"], color=color, alpha=0.18, linewidth=0.9)
        median_line = (
            region_frame.groupby("SLOT", observed=True)["mean_price"]
            .median()
            .reset_index()
        )
        ax.plot(median_line["SLOT"], median_line["mean_price"], color="#111827", linewidth=2.0)
        ax.set_title(f"MP Intraday Shape by Region: {region}")
        ax.set_ylabel("Mean Price")

    axes[-1].set_xlabel("5-Minute Slot in Day")
    axes[-1].set_xticks(np.arange(0, 289, 24))
    axes[-1].set_xticklabels([f"{int(slot // 12):02d}:00" for slot in np.arange(0, 289, 24)])
    save_figure(fig, output_path)


def main() -> int:
    args = parse_args()
    configure_matplotlib()

    data = load_data()
    output_root = Path(args.output_root)
    summary_dir = output_root / "summaries"
    png_dir = output_root / "png"

    daily_region_means = {name: build_daily_region_mean(frame) for name, frame in data.items()}
    setter_summaries = {}
    setter_region_summaries = {}
    for name, frame in data.items():
        extra_group_cols = ["COMMODITY_TYPE"] if name == "reserve" else []
        entity_mode = "resource" if name == "reserve" else "prefix"
        setter_summaries[name], setter_region_summaries[name] = build_effective_setter_summary(
            frame,
            extra_group_cols=extra_group_cols,
            entity_mode=entity_mode,
        )

    save_summaries(daily_region_means, setter_summaries, setter_region_summaries, summary_dir)

    plot_regional_daily_heatmaps(
        daily_region_means["reserve"],
        daily_region_means["mp"],
        png_dir / "regional_daily_price_heatmaps.png",
    )
    plot_top_effective_setters(
        setter_summaries["reserve"],
        setter_summaries["mp"],
        png_dir / "top_effective_setters.png",
    )
    plot_setter_interaction_scatter(
        setter_summaries["reserve"],
        setter_summaries["mp"],
        png_dir / "setter_interaction_scatter.png",
    )
    plot_mp_intraday_spaghetti(data["mp"], png_dir / "mp_intraday_spaghetti.png")
    plot_mp_intraday_spaghetti_by_region(data["mp"], png_dir / "mp_intraday_spaghetti_by_region.png")

    print(f"Analysis summaries written to {summary_dir}")
    print(f"Analysis PNGs written to {png_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
