#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

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
REGION_LABELS = {"CLUZ": "Luzon", "CVIS": "Visayas", "CMIN": "Mindanao"}
REGION_COLORS = {"CLUZ": "#7f2704", "CVIS": "#d95f0e", "CMIN": "#f16913"}
RESERVE_PRODUCT_LABELS = {
    "Dr": "Delayed Contingency Raise",
    "Fr": "Fast Contingency Raise",
    "Rd": "Regulation Down",
    "Ru": "Regulation Up",
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
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def reserve_product_label(code: str) -> str:
    return RESERVE_PRODUCT_LABELS.get(code, code)


def extract_prefix(resource_name: str) -> str:
    return resource_name.split("_")[0] if "_" in resource_name else resource_name


def load_data() -> dict[str, pd.DataFrame]:
    data = {}
    for name, config in DATASETS.items():
        frame = pd.read_parquet(config["parquet"]).copy()
        frame["RUN_DATE"] = frame["RUN_TIME"].dt.normalize()
        frame["SLOT"] = frame["RUN_TIME"].dt.hour * 12 + frame["RUN_TIME"].dt.minute // 5
        frame["SLOT_15M"] = frame["RUN_TIME"].dt.hour * 4 + frame["RUN_TIME"].dt.minute // 15
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


def build_daily_region_product_mean(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(["RUN_DATE", "REGION_NAME", "COMMODITY_TYPE"], observed=True)
        .agg(
            mean_price=("MARGINAL_PRICE", "mean"),
            median_price=("MARGINAL_PRICE", "median"),
            rows=("RUN_TIME", "size"),
        )
        .reset_index()
    )
    return daily


def build_daily_mp_reserve_comparison(
    mp_daily: pd.DataFrame,
    reserve_daily: pd.DataFrame,
) -> pd.DataFrame:
    comparison = reserve_daily.merge(
        mp_daily[["RUN_DATE", "REGION_NAME", "mean_price"]].rename(columns={"mean_price": "mp_mean_price"}),
        on=["RUN_DATE", "REGION_NAME"],
        how="inner",
    ).rename(columns={"mean_price": "reserve_mean_price"})
    return comparison.sort_values(["COMMODITY_TYPE", "REGION_NAME", "RUN_DATE"]).reset_index(drop=True)


def add_within_group_percentiles(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str = "mean_price",
    output_col: str = "percentile_value",
) -> pd.DataFrame:
    ranked = df.copy()
    ranked[output_col] = (
        ranked.groupby(group_cols, observed=True)[value_col]
        .rank(method="average", pct=True)
        .mul(100.0)
    )
    return ranked


def build_diverging_norm(values: pd.Series) -> Normalize:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmin < 0.0 and vmax > 0.0:
        bound = max(abs(vmin), abs(vmax))
        return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    return Normalize(vmin=vmin, vmax=vmax)


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
        ax.set_title(f"{reserve_product_label(product)}: Top Effective Setters")
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
        ax.set_title(f"{reserve_product_label(product)} vs MP Setter Share")
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


def plot_mp_daily_price_heatmap(mp_daily: pd.DataFrame, output_path: Path) -> None:
    region_order = ["CLUZ", "CVIS", "CMIN"]
    region_labels = {"CLUZ": "Luzon", "CVIS": "Visayas", "CMIN": "Mindanao"}
    cmap = LinearSegmentedColormap.from_list(
        "green_yellow_red",
        [(0.0, "#0b5d1e"), (0.5, "#fff2a8"), (1.0, "#8b0000")],
    )
    norm = build_diverging_norm(mp_daily["mean_price"])
    regions = [region for region in region_order if region in set(mp_daily["REGION_NAME"].astype(str))]

    fig, axes = plt.subplots(len(regions), 1, figsize=(13, 3.9), sharex=True, gridspec_kw={"hspace": 0.02})
    if len(regions) == 1:
        axes = [axes]

    image = None
    for ax, region in zip(axes, regions):
        region_frame = mp_daily[mp_daily["REGION_NAME"].astype(str) == region]
        pivot = (
            region_frame.pivot(index="REGION_NAME", columns="RUN_DATE", values="mean_price")
            .sort_index()
            .astype(float)
        )
        dates = pd.to_datetime(pivot.columns)
        image = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)
        ax.set_yticks([0])
        ax.set_yticklabels([region_labels.get(region, region)], fontsize=13)
        ax.tick_params(axis="y", length=0)
        if ax is not axes[-1]:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
        else:
            tick_positions = np.linspace(0, len(dates) - 1, min(10, len(dates)), dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([dates[pos].strftime("%Y-%m-%d") for pos in tick_positions], rotation=45, ha="right", fontsize=11)

    fig.suptitle("Average Daily Market Price by Region", fontsize=16, y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.1, right=0.88, hspace=0.02)
    colorbar_ax = fig.add_axes([0.9, 0.17, 0.015, 0.66])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Average Price (PHP/MWh)", fontsize=12)
    colorbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    save_figure(fig, output_path)


def plot_stacked_daily_heatmap(
    df: pd.DataFrame,
    row_specs: list[tuple[tuple[str, ...], str]],
    output_path: Path,
    title: str,
    value_col: str,
    colorbar_label: str,
    cmap,
    norm: Normalize | None = None,
    colorbar_formatter=None,
    y_fontsize: int = 11,
    spacer_after_indices: list[int] | None = None,
    y_fontfamily: str | None = None,
    left_margin: float = 0.2,
) -> None:
    spacer_after_indices = spacer_after_indices or []
    total_rows = len(row_specs) + len(spacer_after_indices)
    height_ratios = []
    for idx in range(len(row_specs)):
        height_ratios.append(1.0)
        if idx in spacer_after_indices:
            height_ratios.append(0.22)

    fig = plt.figure(figsize=(13, 0.62 * total_rows + 1.9))
    grid = fig.add_gridspec(total_rows, 1, hspace=0.02, height_ratios=height_ratios)
    axes = []
    grid_row = 0
    for idx in range(len(row_specs)):
        ax = fig.add_subplot(grid[grid_row, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
        grid_row += 1
        if idx in spacer_after_indices:
            spacer_ax = fig.add_subplot(grid[grid_row, 0])
            spacer_ax.axis("off")
            grid_row += 1

    image = None
    for ax, (row_key, row_label) in zip(axes, row_specs):
        if len(row_key) == 1:
            region = row_key[0]
            row_frame = df[df["REGION_NAME"].astype(str) == region]
        else:
            region, product = row_key
            row_frame = df[
                (df["REGION_NAME"].astype(str) == region)
                & (df["COMMODITY_TYPE"].astype(str) == product)
            ]
        pivot = (
            row_frame.assign(_ROW_LABEL=row_label)
            .pivot(index="_ROW_LABEL", columns="RUN_DATE", values=value_col)
            .astype(float)
        )
        dates = pd.to_datetime(pivot.columns)
        image = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap=cmap,
            norm=norm if norm is not None else Normalize(vmin=float(df[value_col].min()), vmax=float(df[value_col].max())),
        )
        ax.set_yticks([0])
        ax.set_yticklabels([row_label], fontsize=y_fontsize, fontfamily=y_fontfamily)
        ax.tick_params(axis="y", length=0)
        if ax is not axes[-1]:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
        else:
            tick_positions = np.linspace(0, len(dates) - 1, min(10, len(dates)), dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                [dates[pos].strftime("%Y-%m-%d") for pos in tick_positions],
                rotation=45,
                ha="right",
                fontsize=11,
            )

    fig.suptitle(title, fontsize=16, y=0.98)
    fig.subplots_adjust(top=0.9, bottom=0.12, left=left_margin, right=0.88, hspace=0.02)
    colorbar_ax = fig.add_axes([0.9, 0.16, 0.015, 0.68])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(colorbar_label, fontsize=12)
    if colorbar_formatter is not None:
        colorbar.ax.yaxis.set_major_formatter(colorbar_formatter)
    save_figure(fig, output_path)


def plot_mp_daily_price_heatmap_percentile(mp_daily: pd.DataFrame, output_path: Path) -> None:
    region_order = ["CLUZ", "CVIS", "CMIN"]
    region_labels = {"CLUZ": "Luzon", "CVIS": "Visayas", "CMIN": "Mindanao"}
    row_specs = [((region,), region_labels.get(region, region)) for region in region_order]
    percentile_daily = add_within_group_percentiles(mp_daily, ["REGION_NAME"])
    plot_stacked_daily_heatmap(
        percentile_daily,
        row_specs=row_specs,
        output_path=output_path,
        title="Average Daily Market Price by Region (Within-Region Percentiles)",
        value_col="percentile_value",
        colorbar_label="Within-Region Percentile",
        cmap=plt.get_cmap("YlOrRd"),
        norm=Normalize(vmin=0.0, vmax=100.0),
        colorbar_formatter=plt.FuncFormatter(lambda x, _: f"{x:.0f}"),
        y_fontsize=13,
    )


def plot_reserve_daily_price_heatmap_by_region_product(
    reserve_daily: pd.DataFrame,
    output_path: Path,
) -> None:
    region_order = ["CLUZ", "CVIS", "CMIN"]
    region_labels = {"CLUZ": "Luzon", "CVIS": "Visayas", "CMIN": "Mindanao"}
    product_order = ["Dr", "Fr", "Rd", "Ru"]
    row_specs = [
        ((region, product), f"{region_labels.get(region, region)} - {reserve_product_label(product)}")
        for region in region_order
        for product in product_order
    ]
    plot_stacked_daily_heatmap(
        reserve_daily,
        row_specs=row_specs,
        output_path=output_path,
        title="Reserve Daily Price by Region and Product (PHP/MWh)",
        value_col="mean_price",
        colorbar_label="Average Price (PHP/MWh)",
        cmap=plt.get_cmap("YlOrRd"),
        norm=Normalize(vmin=float(reserve_daily["mean_price"].min()), vmax=float(reserve_daily["mean_price"].max())),
        colorbar_formatter=plt.FuncFormatter(lambda x, _: f"{x:,.0f}"),
        y_fontsize=10,
        spacer_after_indices=[3, 7],
    )


def plot_daily_mp_vs_reserve_scatter(comparison: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=False)
    product_order = ["Dr", "Fr", "Rd", "Ru"]

    for ax, product in zip(axes.flatten(), product_order):
        frame = comparison[comparison["COMMODITY_TYPE"].astype(str) == product].copy()
        for region in ["CLUZ", "CVIS", "CMIN"]:
            region_frame = frame[frame["REGION_NAME"].astype(str) == region]
            if region_frame.empty:
                continue
            ax.scatter(
                region_frame["mp_mean_price"],
                region_frame["reserve_mean_price"],
                s=24,
                alpha=0.75,
                color=REGION_COLORS[region],
                label=REGION_LABELS[region],
                edgecolor="white",
                linewidth=0.4,
            )
        if len(frame) >= 2:
            x = frame["mp_mean_price"].to_numpy(dtype=float)
            y = frame["reserve_mean_price"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color="#1f2937", linewidth=1.8, linestyle="--")
        ax.set_title(reserve_product_label(product), fontsize=13)
        ax.set_xlabel("Daily Average MP (PHP/MWh)")
        ax.set_ylabel("Daily Average Reserve (PHP/MWh)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Daily Average MP vs Reserve Price by Product", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save_figure(fig, output_path)


def plot_daily_mp_reserve_rolling_correlation(
    comparison: pd.DataFrame,
    output_path: Path,
    window_days: int = 14,
    min_periods: int = 7,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)
    region_order = ["CLUZ", "CVIS", "CMIN"]
    product_order = ["Dr", "Fr", "Rd", "Ru"]
    product_colors = {"Dr": "#2b6cb0", "Fr": "#d69e2e", "Rd": "#c53030", "Ru": "#805ad5"}

    for ax, region in zip(axes, region_order):
        frame = comparison[comparison["REGION_NAME"].astype(str) == region].copy()
        for product in product_order:
            product_frame = frame[frame["COMMODITY_TYPE"].astype(str) == product].sort_values("RUN_DATE").copy()
            if product_frame.empty:
                continue
            product_frame["rolling_corr"] = (
                product_frame["mp_mean_price"]
                .rolling(window=window_days, min_periods=min_periods)
                .corr(product_frame["reserve_mean_price"])
            )
            ax.plot(
                product_frame["RUN_DATE"],
                product_frame["rolling_corr"],
                color=product_colors[product],
                linewidth=2.0,
                label=reserve_product_label(product),
            )
        ax.axhline(0.0, color="#4a5568", linewidth=1.0, linestyle=":")
        ax.set_title(REGION_LABELS[region], fontsize=13)
        ax.set_ylim(-1.0, 1.0)
        ax.set_ylabel("Correlation")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

    axes[-1].tick_params(axis="x", rotation=45)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"{window_days}-Day Rolling Correlation: Daily Average MP vs Reserve", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    save_figure(fig, output_path)


def plot_mp_intraday_percentiles(mp: pd.DataFrame, output_path: Path) -> None:
    slot_stats = (
        mp.groupby(["RUN_DATE", "SLOT_15M"], observed=True)
        .agg(mean_price=("MARGINAL_PRICE", "mean"))
        .reset_index()
        .groupby("SLOT_15M", observed=True)["mean_price"]
        .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        .unstack()
        .reset_index()
        .rename(columns={0.10: "p10", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.90: "p90"})
    )
    fig, ax = plt.subplots(figsize=(13, 5.8))
    outer = ax.fill_between(
        slot_stats["SLOT_15M"],
        slot_stats["p10"],
        slot_stats["p90"],
        color="#fdd0a2",
        alpha=0.45,
        label="10-90 percentile",
    )
    inner = ax.fill_between(
        slot_stats["SLOT_15M"],
        slot_stats["p25"],
        slot_stats["p75"],
        color="#fc8d59",
        alpha=0.6,
        label="25-75 percentile",
    )
    median_line, = ax.plot(
        slot_stats["SLOT_15M"],
        slot_stats["p50"],
        color="#7f0000",
        linewidth=2.5,
        label="50 percentile",
    )
    ax.plot(slot_stats["SLOT_15M"], slot_stats["p90"], color="#b30000", linewidth=1.0, alpha=0.8, linestyle="--")
    ax.plot(slot_stats["SLOT_15M"], slot_stats["p10"], color="#b30000", linewidth=1.0, alpha=0.8, linestyle="--")
    ax.set_title("Marginal Price Percentiles", fontsize=16)
    ax.set_ylabel("Marginal Price (PHP/MWh)", fontsize=13)
    ax.set_xticks(np.arange(0, 97, 8))
    ax.set_xticklabels([f"{int(slot // 4):02d}:00" for slot in np.arange(0, 97, 8)], fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(
        [outer, inner, median_line],
        ["10-90 percentile", "25-75 percentile", "50 percentile"],
        frameon=False,
        ncol=3,
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    save_figure(fig, output_path)


def plot_mp_intraday_region_medians(mp: pd.DataFrame, output_path: Path) -> None:
    region_stats = (
        mp.groupby(["REGION_NAME", "RUN_DATE", "SLOT_15M"], observed=True)
        .agg(mean_price=("MARGINAL_PRICE", "mean"))
        .reset_index()
        .groupby(["REGION_NAME", "SLOT_15M"], observed=True)["mean_price"]
        .median()
        .reset_index(name="p50_region")
    )
    region_labels = {"CLUZ": "Luzon", "CVIS": "Visayas", "CMIN": "Mindanao"}
    region_styles = {
        "CLUZ": {"color": "#1f77b4", "linestyle": "-", "linewidth": 2.4},
        "CVIS": {"color": "#2ca02c", "linestyle": "-", "linewidth": 2.4},
        "CMIN": {"color": "#9467bd", "linestyle": "-", "linewidth": 2.4},
    }

    fig, ax = plt.subplots(figsize=(13, 5.8))
    region_handles = []
    for region in sorted(region_stats["REGION_NAME"].astype(str).unique()):
        region_frame = region_stats[region_stats["REGION_NAME"].astype(str) == region]
        style = region_styles.get(region, {"color": "#444444", "linestyle": "-", "linewidth": 2.2})
        region_line, = ax.plot(
            region_frame["SLOT_15M"],
            region_frame["p50_region"],
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            label=region_labels.get(region, region),
        )
        region_handles.append(region_line)

    ax.set_title("Regional Median Marginal Prices", fontsize=16)
    ax.set_ylabel("Marginal Price (PHP/MWh)", fontsize=13)
    ax.set_xticks(np.arange(0, 97, 8))
    ax.set_xticklabels([f"{int(slot // 4):02d}:00" for slot in np.arange(0, 97, 8)], fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(
        region_handles,
        [handle.get_label() for handle in region_handles],
        frameon=False,
        ncol=3,
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    save_figure(fig, output_path)


def plot_mp_intraday_percentiles_by_region(
    mp: pd.DataFrame,
    output_path: Path,
    slot_min: int | None = None,
    slot_max: int | None = None,
    title: str = "Marginal Price Percentiles by Region (PHP/MWh)",
) -> None:
    regional = (
        mp.groupby(["REGION_NAME", "RUN_DATE", "SLOT_15M"], observed=True)
        .agg(mean_price=("MARGINAL_PRICE", "mean"))
        .reset_index()
        .groupby(["REGION_NAME", "SLOT_15M"], observed=True)["mean_price"]
        .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        .unstack()
        .reset_index()
        .rename(columns={0.10: "p10", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.90: "p90"})
    )
    if slot_min is not None:
        regional = regional[regional["SLOT_15M"] >= slot_min]
    if slot_max is not None:
        regional = regional[regional["SLOT_15M"] <= slot_max]

    regions = sorted(regional["REGION_NAME"].astype(str).unique())
    fig, axes = plt.subplots(len(regions), 1, figsize=(13, 9), sharex=True)
    if len(regions) == 1:
        axes = [axes]

    fill_outer = "#fdd0a2"
    fill_inner = "#fc8d59"
    line_color = "#7f0000"
    region_labels = {"CLUZ": "Luzon", "CVIS": "Visayas", "CMIN": "Mindanao"}
    y_min = float(regional["p10"].min())
    y_max = float(regional["p90"].max())
    handles = None

    for ax, region in zip(axes, regions):
        region_frame = regional[regional["REGION_NAME"].astype(str) == region]
        outer = ax.fill_between(
            region_frame["SLOT_15M"],
            region_frame["p10"],
            region_frame["p90"],
            color=fill_outer,
            alpha=0.45,
            label="10-90 percentile",
        )
        inner = ax.fill_between(
            region_frame["SLOT_15M"],
            region_frame["p25"],
            region_frame["p75"],
            color=fill_inner,
            alpha=0.55,
            label="25-75 percentile",
        )
        median_line, = ax.plot(
            region_frame["SLOT_15M"],
            region_frame["p50"],
            color=line_color,
            linewidth=2.0,
            label="50 percentile",
        )
        ax.set_ylabel(
            region_labels.get(region, region),
            rotation=0,
            ha="right",
            va="center",
            labelpad=36,
            fontsize=13,
        )
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.tick_params(axis="y", labelsize=12)
        if handles is None:
            handles = [outer, inner, median_line]

    tick_start = int(regional["SLOT_15M"].min())
    tick_end = int(regional["SLOT_15M"].max())
    tick_step = 4 if (tick_end - tick_start) <= 56 else 8
    ticks = np.arange(tick_start, tick_end + 1, tick_step)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([f"{int(slot // 4):02d}:{int((slot % 4) * 15):02d}" for slot in ticks], fontsize=12)
    axes[-1].tick_params(axis="x", labelsize=12)
    fig.suptitle(title, fontsize=16, y=0.98)
    fig.legend(
        handles,
        ["10-90 percentile", "25-75 percentile", "50 percentile"],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save_figure(fig, output_path)


def plot_reserve_intraday_percentiles_by_product(
    reserve: pd.DataFrame,
    output_path: Path,
    slot_min: int | None = None,
    slot_max: int | None = None,
    title: str = "Reserve Price Percentiles by Region and Product (PHP/MWh)",
) -> None:
    region_order = ["CLUZ", "CVIS", "CMIN"]
    region_labels = {"CLUZ": "Luzon", "CVIS": "Visayas", "CMIN": "Mindanao"}
    product_order = ["Dr", "Fr", "Rd", "Ru"]
    product_stats = (
        reserve.groupby(["REGION_NAME", "COMMODITY_TYPE", "RUN_DATE", "SLOT_15M"], observed=True)
        .agg(mean_price=("MARGINAL_PRICE", "mean"))
        .reset_index()
        .groupby(["REGION_NAME", "COMMODITY_TYPE", "SLOT_15M"], observed=True)["mean_price"]
        .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        .unstack()
        .reset_index()
        .rename(columns={0.10: "p10", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.90: "p90"})
    )
    if slot_min is not None:
        product_stats = product_stats[product_stats["SLOT_15M"] >= slot_min]
    if slot_max is not None:
        product_stats = product_stats[product_stats["SLOT_15M"] <= slot_max]

    regions = [region for region in region_order if region in set(product_stats["REGION_NAME"].astype(str))]
    products = [product for product in product_order if product in set(product_stats["COMMODITY_TYPE"].astype(str))]
    fig, axes = plt.subplots(len(regions), len(products), figsize=(18, 11), sharex=True, sharey=True)
    if len(regions) == 1 and len(products) == 1:
        axes = np.array([[axes]])
    elif len(regions) == 1:
        axes = np.array([axes])
    elif len(products) == 1:
        axes = np.array([[ax] for ax in axes])

    fill_outer = "#fdd0a2"
    fill_inner = "#fc8d59"
    line_color = "#7f0000"
    y_min = float(product_stats["p10"].min())
    y_max = float(product_stats["p90"].max())
    handles = None

    for row_idx, region in enumerate(regions):
        for col_idx, product in enumerate(products):
            ax = axes[row_idx, col_idx]
            panel = product_stats[
                (product_stats["REGION_NAME"].astype(str) == region)
                & (product_stats["COMMODITY_TYPE"].astype(str) == product)
            ]
            outer = ax.fill_between(
                panel["SLOT_15M"],
                panel["p10"],
                panel["p90"],
                color=fill_outer,
                alpha=0.45,
                label="10-90 percentile",
            )
            inner = ax.fill_between(
                panel["SLOT_15M"],
                panel["p25"],
                panel["p75"],
                color=fill_inner,
                alpha=0.55,
                label="25-75 percentile",
            )
            median_line, = ax.plot(
                panel["SLOT_15M"],
                panel["p50"],
                color=line_color,
                linewidth=2.0,
                label="50 percentile",
            )
            ax.set_ylim(y_min, y_max)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax.tick_params(axis="y", labelsize=12)
            ax.tick_params(axis="x", labelsize=12)
            if row_idx == 0:
                ax.set_title(reserve_product_label(product), fontsize=13)
            if col_idx == 0:
                ax.set_ylabel(region_labels.get(region, region), fontsize=13)
            if handles is None:
                handles = [outer, inner, median_line]

    tick_start = int(product_stats["SLOT_15M"].min())
    tick_end = int(product_stats["SLOT_15M"].max())
    tick_step = 8 if (tick_end - tick_start) <= 56 else 16
    ticks = np.arange(tick_start, tick_end + 1, tick_step)
    for ax in axes[-1, :]:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(slot // 4):02d}:{int((slot % 4) * 15):02d}" for slot in ticks], fontsize=12)
        ax.tick_params(axis="x", labelsize=12)
    fig.suptitle(title, fontsize=16, y=0.98)
    fig.legend(
        handles,
        ["10-90 percentile", "25-75 percentile", "50 percentile"],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save_figure(fig, output_path)


def main() -> int:
    args = parse_args()
    configure_matplotlib()

    data = load_data()
    output_root = Path(args.output_root)
    summary_dir = output_root / "summaries"
    png_dir = output_root / "png"

    daily_region_means = {name: build_daily_region_mean(frame) for name, frame in data.items()}
    reserve_daily_region_product = build_daily_region_product_mean(data["reserve"])
    daily_mp_reserve_comparison = build_daily_mp_reserve_comparison(
        daily_region_means["mp"],
        reserve_daily_region_product,
    )
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

    plot_mp_daily_price_heatmap(
        daily_region_means["mp"],
        png_dir / "mp_daily_price_heatmap.png",
    )
    plot_mp_daily_price_heatmap_percentile(
        daily_region_means["mp"],
        png_dir / "mp_daily_price_heatmap_percentile.png",
    )
    plot_reserve_daily_price_heatmap_by_region_product(
        reserve_daily_region_product,
        png_dir / "reserve_daily_price_heatmap_by_region_product.png",
    )
    plot_daily_mp_vs_reserve_scatter(
        daily_mp_reserve_comparison,
        png_dir / "daily_mp_vs_reserve_scatter.png",
    )
    plot_daily_mp_reserve_rolling_correlation(
        daily_mp_reserve_comparison,
        png_dir / "daily_mp_reserve_rolling_correlation.png",
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
    plot_mp_intraday_percentiles(data["mp"], png_dir / "mp_intraday_percentiles.png")
    plot_mp_intraday_region_medians(data["mp"], png_dir / "mp_intraday_percentiles_with_region_medians.png")
    plot_mp_intraday_percentiles_by_region(data["mp"], png_dir / "mp_intraday_percentiles_by_region.png")
    plot_mp_intraday_percentiles_by_region(
        data["mp"],
        png_dir / "mp_intraday_percentiles_by_region_0800_2200.png",
        slot_min=32,
        slot_max=88,
        title="Marginal Price Percentiles by Region (PHP/MWh, 08:00-22:00)",
    )
    plot_reserve_intraday_percentiles_by_product(
        data["reserve"],
        png_dir / "reserve_intraday_percentiles_by_product.png",
    )
    plot_reserve_intraday_percentiles_by_product(
        data["reserve"],
        png_dir / "reserve_intraday_percentiles_by_product_0800_2200.png",
        slot_min=32,
        slot_max=88,
        title="Reserve Price Percentiles by Region and Product (PHP/MWh, 08:00-22:00)",
    )

    print(f"Analysis summaries written to {summary_dir}")
    print(f"Analysis PNGs written to {png_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
