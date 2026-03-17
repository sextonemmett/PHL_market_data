#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

DATASETS = {
    "mp": {
        "parquet": Path("data/mp/combined/MP_20251216_20260316.parquet"),
        "title": "Market Price",
        "color": "#8b4513",
    },
    "reserve": {
        "parquet": Path("data/mp_reserve/combined/MP_RESERVE_20251216_20260316.parquet"),
        "title": "Reserve",
        "color": "#1f4e79",
    },
}
REGION_LABELS = {"CLUZ": "Luzon", "CVIS": "Visayas", "CMIN": "Mindanao"}
RESERVE_PRODUCT_LABELS = {
    "Dr": "Delayed Contingency Raise",
    "Fr": "Fast Contingency Raise",
    "Rd": "Regulation Down",
    "Ru": "Regulation Up",
}
FAMILY_MEMBERSHIP_LABELS = {
    1: "MP only",
    2: "Reserve only",
    3: "Both",
}
FAMILY_MEMBERSHIP_COLORS = {
    0: "#f7f7f7",
    1: "#2b6cb0",
    2: "#d95f0e",
    3: "#6a3d9a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate resource-name focused summaries and figures for MP and reserve data."
    )
    parser.add_argument(
        "--output-root",
        default="reports/analysis/resources",
        help="Directory for generated resource analysis summaries and PNGs.",
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


def split_resource_name(resource_name: str) -> tuple[str, str, bool]:
    if "_" in resource_name:
        prefix, suffix = resource_name.rsplit("_", 1)
        return prefix, suffix, True
    return resource_name, "", False


def commodity_label(code: str) -> str:
    return RESERVE_PRODUCT_LABELS.get(code, code)


def suffix_label(suffix: str) -> str:
    return suffix if suffix else "<none>"


def natural_suffix_key(value: str) -> tuple[int, str]:
    return (0 if value == "" else 1, value)


def load_dataset(name: str, config: dict[str, object]) -> pd.DataFrame:
    columns = ["RUN_TIME", "RESOURCE_NAME", "REGION_NAME", "MARGINAL_PRICE"]
    if name == "reserve":
        columns.append("COMMODITY_TYPE")

    frame = pd.read_parquet(config["parquet"], columns=columns).copy()
    frame["dataset"] = name
    frame["RESOURCE_NAME"] = frame["RESOURCE_NAME"].astype(str)
    frame["REGION_NAME"] = frame["REGION_NAME"].astype(str)
    frame["RUN_DATE"] = frame["RUN_TIME"].dt.normalize()
    if name == "reserve":
        frame["COMMODITY_TYPE"] = frame["COMMODITY_TYPE"].astype(str)
    else:
        frame["COMMODITY_TYPE"] = ""

    parts = frame["RESOURCE_NAME"].map(split_resource_name)
    frame["resource_full"] = frame["RESOURCE_NAME"]
    frame["resource_prefix"] = parts.str[0]
    frame["resource_suffix"] = parts.str[1]
    frame["has_suffix"] = parts.str[2]
    return frame


def join_sorted(values: pd.Series) -> str:
    unique = sorted({str(value) for value in values if str(value) != ""})
    return "|".join(unique)


def build_resource_full_summary(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset, frame in data.items():
        total_rows = len(frame)
        summary = (
            frame.groupby(["dataset", "resource_full", "resource_prefix", "resource_suffix", "has_suffix"], observed=True)
            .agg(
                region_set=("REGION_NAME", join_sorted),
                commodity_set=("COMMODITY_TYPE", join_sorted),
                active_days=("RUN_DATE", "nunique"),
                active_regions=("REGION_NAME", "nunique"),
                row_count=("resource_full", "size"),
                mean_price=("MARGINAL_PRICE", "mean"),
            )
            .reset_index()
        )
        summary["row_share_pct"] = summary["row_count"] / total_rows * 100.0
        summary["dataset_title"] = DATASETS[dataset]["title"]
        frames.append(summary)
    return pd.concat(frames, ignore_index=True).sort_values(["dataset", "row_count", "resource_full"], ascending=[True, False, True])


def build_resource_prefix_summary(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset, frame in data.items():
        total_rows = len(frame)
        summary = (
            frame.groupby(["dataset", "resource_prefix"], observed=True)
            .agg(
                suffix_set=("resource_suffix", join_sorted),
                suffix_variant_count=("resource_suffix", "nunique"),
                unique_full_resource_count=("resource_full", "nunique"),
                region_set=("REGION_NAME", join_sorted),
                commodity_set=("COMMODITY_TYPE", join_sorted),
                active_days=("RUN_DATE", "nunique"),
                active_regions=("REGION_NAME", "nunique"),
                row_count=("resource_full", "size"),
                mean_price=("MARGINAL_PRICE", "mean"),
            )
            .reset_index()
        )
        summary["row_share_pct"] = summary["row_count"] / total_rows * 100.0
        summary["dataset_title"] = DATASETS[dataset]["title"]
        frames.append(summary)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["dataset", "suffix_variant_count", "row_count", "resource_prefix"],
        ascending=[True, False, False, True],
    )


def build_overlap_summary(full_summary: pd.DataFrame, family_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    full_sets = {
        dataset: set(frame["resource_full"].astype(str))
        for dataset, frame in full_summary.groupby("dataset", observed=True)
    }
    family_sets = {
        dataset: set(frame["resource_prefix"].astype(str))
        for dataset, frame in family_summary.groupby("dataset", observed=True)
    }
    for level, value_sets in [("full", full_sets), ("family", family_sets)]:
        mp_values = value_sets["mp"]
        reserve_values = value_sets["reserve"]
        shared = mp_values & reserve_values
        rows.append(
            {
                "level": level,
                "mp_unique": len(mp_values),
                "reserve_unique": len(reserve_values),
                "shared": len(shared),
                "mp_only": len(mp_values - reserve_values),
                "reserve_only": len(reserve_values - mp_values),
            }
        )
    return pd.DataFrame(rows)


def build_family_membership_summary(full_summary: pd.DataFrame) -> pd.DataFrame:
    resource_rows = full_summary[["dataset", "resource_full", "resource_prefix", "resource_suffix"]].drop_duplicates()
    prefix_suffix_presence = (
        resource_rows.assign(value=1)
        .pivot_table(
            index=["resource_prefix", "resource_suffix"],
            columns="dataset",
            values="value",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )
    prefix_suffix_presence["present_in_mp"] = prefix_suffix_presence.get("mp", 0).astype(int)
    prefix_suffix_presence["present_in_reserve"] = prefix_suffix_presence.get("reserve", 0).astype(int)
    prefix_suffix_presence["presence_code"] = (
        prefix_suffix_presence["present_in_mp"] + 2 * prefix_suffix_presence["present_in_reserve"]
    )
    prefix_suffix_presence["resource_suffix_label"] = prefix_suffix_presence["resource_suffix"].map(suffix_label)
    prefix_suffix_presence["_suffix_sort"] = prefix_suffix_presence["resource_suffix"].map(
        lambda value: natural_suffix_key(str(value))
    )
    return (
        prefix_suffix_presence[
            [
                "resource_prefix",
                "resource_suffix",
                "resource_suffix_label",
                "present_in_mp",
                "present_in_reserve",
                "presence_code",
                "_suffix_sort",
            ]
        ]
        .sort_values(["resource_prefix", "_suffix_sort"], ascending=[True, True])
        .drop(columns="_suffix_sort")
    )


def build_marginal_supplier_interval_share_summary(
    df: pd.DataFrame,
    entity_mode: str,
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    extra_group_cols = extra_group_cols or []
    interval_group_cols = ["RUN_TIME", "REGION_NAME", *extra_group_cols]
    unique_rows = df[
        ["RUN_TIME", "REGION_NAME", "MARGINAL_PRICE", "resource_full", "resource_prefix", *extra_group_cols]
    ].drop_duplicates()

    if entity_mode == "full":
        unique_rows["entity_name"] = unique_rows["resource_full"]
    else:
        unique_rows["entity_name"] = unique_rows["resource_prefix"]

    unique_entities = unique_rows[
        ["RUN_TIME", "REGION_NAME", "MARGINAL_PRICE", "entity_name", *extra_group_cols]
    ].drop_duplicates()
    total_groups = unique_entities[interval_group_cols].drop_duplicates().shape[0]
    summary = (
        unique_entities.groupby([*extra_group_cols, "entity_name"], observed=True)
        .agg(
            interval_count=("entity_name", "size"),
            active_regions=("REGION_NAME", "nunique"),
            active_intervals=("RUN_TIME", "nunique"),
            mean_marginal_price=("MARGINAL_PRICE", "mean"),
        )
        .reset_index()
    )
    summary["interval_share_pct"] = summary["interval_count"] / total_groups * 100.0
    return summary.sort_values(["interval_share_pct", "entity_name"], ascending=[False, True])


def write_summary_tables(
    output_dir: Path,
    full_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    family_membership: pd.DataFrame,
    marginal_supplier_full_summary: pd.DataFrame,
    marginal_supplier_family_summary: pd.DataFrame,
    reserve_family_marginal_supplier_by_product: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_summary.to_csv(output_dir / "resource_full_summary.csv", index=False)
    family_summary.to_csv(output_dir / "resource_family_summary.csv", index=False)
    overlap_summary.to_csv(output_dir / "resource_overlap_full_vs_family.csv", index=False)
    family_membership.to_csv(output_dir / "resource_family_membership.csv", index=False)
    marginal_supplier_full_summary.to_csv(output_dir / "marginal_supplier_full_summary.csv", index=False)
    marginal_supplier_family_summary.to_csv(output_dir / "marginal_supplier_family_summary.csv", index=False)
    reserve_family_marginal_supplier_by_product.to_csv(output_dir / "reserve_family_marginal_supplier_by_product.csv", index=False)


def build_entity_column_share_matrix(
    marginal_supplier_summary: pd.DataFrame,
    reserve_marginal_supplier_by_product: pd.DataFrame,
) -> pd.DataFrame:
    mp = (
        marginal_supplier_summary[marginal_supplier_summary["dataset"] == "mp"][
            ["entity_name", "interval_count"]
        ]
        .rename(columns={"interval_count": "MP"})
    )
    reserve = (
        marginal_supplier_summary[marginal_supplier_summary["dataset"] == "reserve"][
            ["entity_name", "interval_count"]
        ]
        .rename(columns={"interval_count": "Reserve"})
    )
    reserve_by_product = (
        reserve_marginal_supplier_by_product.pivot(
            index="entity_name",
            columns="COMMODITY_TYPE",
            values="interval_count",
        )
        .reset_index()
    )
    for product in ["Dr", "Fr", "Rd", "Ru"]:
        if product not in reserve_by_product.columns:
            reserve_by_product[product] = 0

    matrix = mp.merge(reserve, on="entity_name", how="outer").merge(
        reserve_by_product[["entity_name", "Dr", "Fr", "Rd", "Ru"]],
        on="entity_name",
        how="outer",
    )
    matrix = matrix.fillna(0.0)
    value_cols = ["MP", "Reserve", "Dr", "Fr", "Rd", "Ru"]
    for col in value_cols:
        col_sum = float(matrix[col].sum())
        matrix[col] = 0.0 if col_sum == 0.0 else matrix[col] / col_sum * 100.0
    matrix["combined_share"] = matrix[value_cols].sum(axis=1)
    return matrix.sort_values(["combined_share", "entity_name"], ascending=[False, True]).drop(columns="combined_share")


def build_entity_column_share_matrix_from_data(
    data: dict[str, pd.DataFrame],
    entity_mode: str,
    region_name: str | None = None,
) -> pd.DataFrame:
    supplier_frames = []
    for dataset, frame in data.items():
        scoped = frame if region_name is None else frame[frame["REGION_NAME"] == region_name].copy()
        supplier = build_marginal_supplier_interval_share_summary(scoped, entity_mode=entity_mode)
        supplier["dataset"] = dataset
        supplier_frames.append(supplier)

    marginal_supplier_summary = pd.concat(supplier_frames, ignore_index=True).sort_values(
        ["dataset", "interval_share_pct", "entity_name"], ascending=[True, False, True]
    )
    reserve_scoped = data["reserve"] if region_name is None else data["reserve"][data["reserve"]["REGION_NAME"] == region_name].copy()
    reserve_marginal_supplier_by_product = build_marginal_supplier_interval_share_summary(
        reserve_scoped, entity_mode=entity_mode, extra_group_cols=["COMMODITY_TYPE"]
    ).sort_values(["COMMODITY_TYPE", "interval_share_pct", "entity_name"], ascending=[True, False, True])
    return build_entity_column_share_matrix(
        marginal_supplier_summary,
        reserve_marginal_supplier_by_product,
    )


def filter_top_families_by_any_column(
    family_column_share: pd.DataFrame,
    value_cols: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    selected: set[str] = set()
    for col in value_cols:
        top_entities = (
            family_column_share.sort_values([col, "entity_name"], ascending=[False, True])
            .head(top_n)["entity_name"]
            .astype(str)
            .tolist()
        )
        selected.update(top_entities)

    filtered = family_column_share[family_column_share["entity_name"].isin(selected)].copy()
    return filtered.sort_values(["MP", "entity_name"], ascending=[False, True])


def build_top_entities_by_column(
    column_share: pd.DataFrame,
    value_cols: list[str],
    top_n: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in value_cols:
        ranked = column_share.sort_values([col, "entity_name"], ascending=[False, True]).head(top_n)
        for rank, (_, record) in enumerate(ranked.iterrows(), start=1):
            rows.append(
                {
                    "column_name": col,
                    "rank": rank,
                    "entity_name": str(record["entity_name"]),
                    "share_pct": float(record[col]),
                }
            )
    return pd.DataFrame(rows)


def plot_marginal_supplier_family_column_share_heatmap(
    top_entities_by_column: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    columns = ["MP", "Dr", "Fr", "Rd", "Ru"]
    column_labels = {
        "MP": "MP",
        "Dr": "Delayed CR",
        "Fr": "Fast CR",
        "Rd": "Reg Down",
        "Ru": "Reg Up",
    }
    max_rows = int(top_entities_by_column["rank"].max()) if not top_entities_by_column.empty else 0
    fig_height = max(7.0, 0.68 * max_rows + 1.8)
    fig, axes = plt.subplots(
        1,
        len(columns),
        figsize=(15.5, fig_height),
        gridspec_kw={"wspace": 1.0},
    )
    vmax = float(top_entities_by_column["share_pct"].max()) if not top_entities_by_column.empty else 1.0
    image = None
    for ax, col in zip(axes, columns):
        subset = top_entities_by_column[top_entities_by_column["column_name"] == col].sort_values("rank")
        values = subset["share_pct"].to_numpy(dtype=float).reshape(-1, 1)
        image = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax)
        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels(subset["entity_name"].astype(str), fontsize=10)
        ax.set_xticks([0])
        ax.set_xticklabels([column_labels[col]], fontsize=12)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, pad=8, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row_idx, value in enumerate(subset["share_pct"].to_numpy(dtype=float)):
            ax.text(0, row_idx, f"{value:.1f}%", ha="center", va="center", color="black", fontsize=9)
        ax.set_ylim(len(subset) - 0.5, -0.5)

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.subplots_adjust(left=0.2, right=0.9, top=0.91, bottom=0.04, wspace=1.0)
    colorbar_ax = fig.add_axes([0.92, 0.12, 0.018, 0.72])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Column Share (%)", fontsize=12)
    colorbar.ax.tick_params(labelsize=11)
    colorbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    save_figure(fig, output_path)


def plot_family_membership_matrix_all(
    family_membership: pd.DataFrame,
    family_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    family_order = (
        family_summary.groupby("resource_prefix", observed=True)
        .agg(
            total_full_resources=("unique_full_resource_count", "sum"),
            total_rows=("row_count", "sum"),
        )
        .reset_index()
        .sort_values(["total_full_resources", "total_rows", "resource_prefix"], ascending=[False, False, True])
    )["resource_prefix"].tolist()
    suffixes = sorted(
        family_membership["resource_suffix"].astype(str).unique(),
        key=natural_suffix_key,
    )
    matrix = (
        family_membership.pivot_table(
            index="resource_prefix",
            columns="resource_suffix",
            values="presence_code",
            aggfunc="max",
            fill_value=0,
        )
        .reindex(index=family_order, fill_value=0)
        .reindex(columns=suffixes, fill_value=0)
    )

    cmap = ListedColormap([FAMILY_MEMBERSHIP_COLORS[i] for i in range(4)])
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), cmap.N)
    fig_height = max(12.0, 0.18 * len(matrix) + 1.6)
    fig_width = max(11.0, 0.42 * len(matrix.columns) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(matrix.values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([suffix_label(col) for col in matrix.columns], rotation=45, ha="right", fontsize=8)
    ax.set_title("Family by Suffix Membership Matrix", fontsize=15, pad=14)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend_handles = [
        Patch(facecolor=FAMILY_MEMBERSHIP_COLORS[key], label=label)
        for key, label in FAMILY_MEMBERSHIP_LABELS.items()
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.subplots_adjust(left=0.18, right=0.84, top=0.94, bottom=0.08)
    save_figure(fig, output_path)


def run_validation(
    data: dict[str, pd.DataFrame],
    full_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    overlap_summary: pd.DataFrame,
) -> list[str]:
    messages = []
    mapped = full_summary[["resource_full", "resource_prefix", "resource_suffix"]].drop_duplicates()
    duplicate_mappings = mapped.duplicated(subset=["resource_full"]).sum()
    if duplicate_mappings != 0:
        raise ValueError("Some full resources mapped to multiple prefixes or suffixes.")
    messages.append("Resource full-to-prefix mapping is one-to-one.")

    for dataset, frame in data.items():
        full_rows = int(full_summary[full_summary["dataset"] == dataset]["row_count"].sum())
        prefix_rows = int(family_summary[family_summary["dataset"] == dataset]["row_count"].sum())
        if full_rows != len(frame):
            raise ValueError(f"Full summary row count mismatch for {dataset}.")
        if prefix_rows != len(frame):
            raise ValueError(f"Family summary row count mismatch for {dataset}.")
        messages.append(f"{dataset}: full and family row counts match source parquet.")

    full_shared = overlap_summary.loc[overlap_summary["level"] == "full", "shared"].iloc[0]
    prefix_shared = overlap_summary.loc[overlap_summary["level"] == "family", "shared"].iloc[0]
    messages.append(f"Overlap reproducible: shared full={int(full_shared):,}, shared family={int(prefix_shared):,}.")

    for family in ["01MAGAT", "04IASMOD", "10GNPK", "13MACO"]:
        mp_present = family in set(family_summary[family_summary["dataset"] == "mp"]["resource_prefix"].astype(str))
        reserve_present = family in set(family_summary[family_summary["dataset"] == "reserve"]["resource_prefix"].astype(str))
        messages.append(f"Family check {family}: mp={mp_present} reserve={reserve_present}")
    return messages


def main() -> int:
    args = parse_args()
    configure_matplotlib()

    data = {name: load_dataset(name, config) for name, config in DATASETS.items()}
    output_root = Path(args.output_root)
    summary_dir = output_root / "summaries"
    png_dir = output_root / "png"

    full_summary = build_resource_full_summary(data)
    family_summary = build_resource_prefix_summary(data)
    overlap_summary = build_overlap_summary(full_summary, family_summary)
    family_membership = build_family_membership_summary(full_summary)

    full_supplier_frames = []
    family_supplier_frames = []
    for dataset, frame in data.items():
        full_supplier = build_marginal_supplier_interval_share_summary(frame, entity_mode="full")
        full_supplier["dataset"] = dataset
        full_supplier_frames.append(full_supplier)

        family_supplier = build_marginal_supplier_interval_share_summary(frame, entity_mode="prefix")
        family_supplier["dataset"] = dataset
        family_supplier_frames.append(family_supplier)

    marginal_supplier_full_summary = pd.concat(full_supplier_frames, ignore_index=True).sort_values(
        ["dataset", "interval_share_pct", "entity_name"], ascending=[True, False, True]
    )
    marginal_supplier_family_summary = pd.concat(family_supplier_frames, ignore_index=True).sort_values(
        ["dataset", "interval_share_pct", "entity_name"], ascending=[True, False, True]
    )
    reserve_full_marginal_supplier_by_product = build_marginal_supplier_interval_share_summary(
        data["reserve"], entity_mode="full", extra_group_cols=["COMMODITY_TYPE"]
    ).sort_values(["COMMODITY_TYPE", "interval_share_pct", "entity_name"], ascending=[True, False, True])
    reserve_family_marginal_supplier_by_product = build_marginal_supplier_interval_share_summary(
        data["reserve"], entity_mode="prefix", extra_group_cols=["COMMODITY_TYPE"]
    ).sort_values(["COMMODITY_TYPE", "interval_share_pct", "entity_name"], ascending=[True, False, True])
    full_column_share = build_entity_column_share_matrix(
        marginal_supplier_full_summary,
        reserve_full_marginal_supplier_by_product,
    )
    family_column_share = build_entity_column_share_matrix(
        marginal_supplier_family_summary,
        reserve_family_marginal_supplier_by_product,
    )
    column_share_value_cols = ["MP", "Dr", "Fr", "Rd", "Ru"]
    full_column_share_top = build_top_entities_by_column(
        full_column_share,
        column_share_value_cols,
        top_n=10,
    )
    family_column_share_top = build_top_entities_by_column(
        family_column_share,
        column_share_value_cols,
        top_n=5,
    )
    regional_full_column_shares = {}
    regional_full_column_share_top = {}
    regional_family_column_shares = {}
    regional_family_column_share_top = {}
    for region_code in ["CLUZ", "CVIS", "CMIN"]:
        region_full_matrix = build_entity_column_share_matrix_from_data(data, entity_mode="full", region_name=region_code)
        regional_full_column_shares[region_code] = region_full_matrix
        regional_full_column_share_top[region_code] = build_top_entities_by_column(
            region_full_matrix,
            column_share_value_cols,
            top_n=10,
        )
        region_family_matrix = build_entity_column_share_matrix_from_data(data, entity_mode="prefix", region_name=region_code)
        regional_family_column_shares[region_code] = region_family_matrix
        regional_family_column_share_top[region_code] = build_top_entities_by_column(
            region_family_matrix,
            column_share_value_cols,
            top_n=5,
        )

    write_summary_tables(
        summary_dir,
        full_summary,
        family_summary,
        overlap_summary,
        family_membership,
        marginal_supplier_full_summary,
        marginal_supplier_family_summary,
        reserve_family_marginal_supplier_by_product,
    )
    reserve_full_marginal_supplier_by_product.to_csv(
        summary_dir / "reserve_full_marginal_supplier_by_product.csv",
        index=False,
    )
    full_column_share.to_csv(summary_dir / "marginal_supplier_full_column_share.csv", index=False)
    full_column_share_top.to_csv(summary_dir / "marginal_supplier_full_column_share_top10_by_column.csv", index=False)
    family_column_share.to_csv(summary_dir / "marginal_supplier_family_column_share.csv", index=False)
    family_column_share_top.to_csv(summary_dir / "marginal_supplier_family_column_share_top5_by_column.csv", index=False)
    for region_code, region_matrix in regional_full_column_shares.items():
        region_label = REGION_LABELS.get(region_code, region_code).lower()
        region_matrix.to_csv(
            summary_dir / f"marginal_supplier_full_column_share_{region_label}.csv",
            index=False,
        )
        regional_full_column_share_top[region_code].to_csv(
            summary_dir / f"marginal_supplier_full_column_share_{region_label}_top10_by_column.csv",
            index=False,
        )
    for region_code, region_matrix in regional_family_column_shares.items():
        region_label = REGION_LABELS.get(region_code, region_code).lower()
        region_matrix.to_csv(
            summary_dir / f"marginal_supplier_family_column_share_{region_label}.csv",
            index=False,
        )
        regional_family_column_share_top[region_code].to_csv(
            summary_dir / f"marginal_supplier_family_column_share_{region_label}_top5_by_column.csv",
            index=False,
        )

    plot_marginal_supplier_family_column_share_heatmap(
        full_column_share_top,
        png_dir / "marginal_supplier_full_column_share_heatmap.png",
        "Marginal Supplier Resource Column Share",
    )
    for region_code, region_matrix in regional_full_column_share_top.items():
        region_label = REGION_LABELS.get(region_code, region_code)
        plot_marginal_supplier_family_column_share_heatmap(
            region_matrix,
            png_dir / f"marginal_supplier_full_column_share_heatmap_{region_label.lower()}.png",
            f"Marginal Supplier Resource Column Share: {region_label}",
        )

    plot_marginal_supplier_family_column_share_heatmap(
        family_column_share_top,
        png_dir / "marginal_supplier_family_column_share_heatmap.png",
        "Marginal Supplier Family Column Share",
    )
    plot_family_membership_matrix_all(
        family_membership,
        family_summary,
        png_dir / "family_membership_matrix.png",
    )
    for region_code, region_matrix in regional_family_column_share_top.items():
        region_label = REGION_LABELS.get(region_code, region_code)
        plot_marginal_supplier_family_column_share_heatmap(
            region_matrix,
            png_dir / f"marginal_supplier_family_column_share_heatmap_{region_label.lower()}.png",
            f"Marginal Supplier Family Column Share: {region_label}",
        )

    checks = run_validation(data, full_summary, family_summary, overlap_summary)
    for message in checks:
        print(message)
    print(f"Resource analysis summaries written to {summary_dir}")
    print(f"Resource analysis PNGs written to {png_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
