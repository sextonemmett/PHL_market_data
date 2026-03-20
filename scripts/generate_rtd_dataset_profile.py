#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

DATASETS = {
    "RTDSL": {
        "title": "RTD Security Limits Used",
        "parquet": Path("data/rtd_security_limits/combined/RTDSL_20251218_20260318.parquet"),
        "qc": Path("data/rtd_security_limits/qc/rtdsl_qc_20251218_20260318.csv"),
        "notes": (
            "Dense security-limit records keyed by resource and parameter type. "
            "No field-level nulls in the combined parquet for this window."
        ),
    },
    "RTDCV": {
        "title": "RTD Congestions Manifesting",
        "parquet": Path("data/rtd_congestion/combined/RTDCV_20251218_20260318.parquet"),
        "qc": Path("data/rtd_congestion/qc/rtdcv_qc_20251218_20260318.csv"),
        "notes": (
            "Sparse event-style congestion records. Most sparsity shows up as empty daily files "
            "rather than null columns inside the combined parquet."
        ),
    },
    "RTDHS": {
        "title": "RTD HVDC Schedules",
        "parquet": Path("data/rtd_hvdc_schedules/combined/RTDHS_20251218_20260318.parquet"),
        "qc": Path("data/rtd_hvdc_schedules/qc/rtdhs_qc_20251218_20260318.csv"),
        "notes": (
            "Regular interval schedules for the two HVDC links. `OVERLOAD_MW` is entirely null "
            "for the current three-month window."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a markdown profile report for the RTDSL, RTDCV, and RTDHS datasets."
    )
    parser.add_argument(
        "--output-markdown",
        default="reports/analysis/rtd_dataset_profile.md",
        help="Path for the generated markdown report.",
    )
    parser.add_argument(
        "--assets-dir",
        default="reports/analysis/rtd_dataset_profile_assets",
        help="Directory for generated plot assets.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def format_int(value: int) -> str:
    return f"{int(value):,}"


def format_float(value: float, digits: int = 2) -> str:
    return f"{float(value):,.{digits}f}"


def format_pct(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}%"


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.astype(object).where(~frame.isna(), "")
    headers = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy().tolist()]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt_row(values: list[str]) -> str:
        cells = [f" {value.ljust(widths[idx])} " for idx, value in enumerate(values)]
        return "|" + "|".join(cells) + "|"

    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    parts = [fmt_row(headers), separator]
    parts.extend(fmt_row(row) for row in rows)
    return "\n".join(parts)


def summarize_qc(qc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    status_counts = (
        qc["status"]
        .value_counts(dropna=False)
        .rename_axis("status")
        .reset_index(name="days")
        .sort_values("status")
    )
    status_counts["days"] = status_counts["days"].map(format_int)

    warning_rows = qc[qc["status"] == "warning"].copy()
    warning_rows["warnings"] = warning_rows["warnings"].fillna("(none)")
    if warning_rows.empty:
        warning_summary = pd.DataFrame(
            [{"warning_type": "(none)", "days": "0", "sample_dates": ""}]
        )
    else:
        warning_summary = (
            warning_rows.groupby("warnings", dropna=False)["file_date"]
            .agg(["count", lambda values: ", ".join(sorted(values)[:5])])
            .reset_index()
            .rename(
                columns={
                    "warnings": "warning_type",
                    "count": "days",
                    "<lambda_0>": "sample_dates",
                }
            )
            .sort_values(["days", "warning_type"], ascending=[False, True])
        )
        warning_summary["days"] = warning_summary["days"].map(format_int)
    return status_counts, warning_summary


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_rows = len(df)
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null_rows": format_int(total_rows - missing),
                "missing_rows": format_int(missing),
                "missing_pct": format_pct(series.isna().mean() * 100.0),
                "unique_non_null": format_int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def datetime_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.select_dtypes(include=["datetime64[ns]"]).columns:
        series = df[column]
        rows.append(
            {
                "column": column,
                "min": series.min().strftime("%Y-%m-%d %H:%M:%S"),
                "median": series.sort_values().iloc[len(series) // 2].strftime("%Y-%m-%d %H:%M:%S"),
                "max": series.max().strftime("%Y-%m-%d %H:%M:%S"),
                "unique_timestamps": format_int(series.nunique()),
            }
        )
    return pd.DataFrame(rows)


def categorical_distribution_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        counts = series.astype("string").fillna("<NA>").value_counts(dropna=False).head(12)
        table = counts.rename_axis("value").reset_index(name="rows")
        table["share"] = table["rows"] / len(df) * 100.0
        table["rows"] = table["rows"].map(format_int)
        table["share"] = table["share"].map(format_pct)
        tables[column] = table
    return tables


def numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.select_dtypes(include=["number"]).columns:
        series = df[column]
        non_null = series.dropna()
        if non_null.empty:
            rows.append(
                {
                    "column": column,
                    "non_null_rows": "0",
                    "missing_pct": format_pct(100.0),
                    "min": "",
                    "p05": "",
                    "median": "",
                    "mean": "",
                    "p95": "",
                    "max": "",
                    "zeros": "0",
                    "negative": "0",
                }
            )
            continue

        rows.append(
            {
                "column": column,
                "non_null_rows": format_int(non_null.size),
                "missing_pct": format_pct(series.isna().mean() * 100.0),
                "min": format_float(non_null.min()),
                "p05": format_float(non_null.quantile(0.05)),
                "median": format_float(non_null.median()),
                "mean": format_float(non_null.mean()),
                "p95": format_float(non_null.quantile(0.95)),
                "max": format_float(non_null.max()),
                "zeros": format_int((non_null == 0).sum()),
                "negative": format_int((non_null < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_missingness(df: pd.DataFrame, title: str, path: Path) -> None:
    missing_pct = df.isna().mean().sort_values(ascending=False) * 100.0
    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = ["#c44e52" if value > 0 else "#4c72b0" for value in missing_pct.values]
    ax.bar(missing_pct.index, missing_pct.values, color=colors)
    ax.set_title(f"{title}: Column Missingness")
    ax.set_ylabel("Missing %")
    ax.set_ylim(0, max(5, missing_pct.max() * 1.15 if len(missing_pct) else 5))
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")
    save_figure(fig, path)


def plot_daily_rows(df: pd.DataFrame, title: str, path: Path) -> None:
    datetime_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    primary_time = "RUN_TIME" if "RUN_TIME" in datetime_columns else datetime_columns[0]
    daily = df.assign(run_date=df[primary_time].dt.normalize()).groupby("run_date").size()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(daily.index, daily.values, linewidth=1.8, color="#1f4e79")
    ax.fill_between(daily.index, daily.values, alpha=0.15, color="#1f4e79")
    ax.set_title(f"{title}: Rows per Day")
    ax.set_ylabel("Rows")
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")
    save_figure(fig, path)


def plot_numeric_histograms(df: pd.DataFrame, title: str, path: Path) -> None:
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    if not numeric_cols:
        return

    ncols = 2
    nrows = (len(numeric_cols) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.5 * nrows))
    axes_list = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
    for ax, column in zip(axes_list, numeric_cols):
        series = df[column].dropna()
        if series.empty:
            ax.text(0.5, 0.5, "all null", ha="center", va="center")
            ax.set_title(column)
            continue
        ax.hist(series, bins=40, color="#55a868", alpha=0.85)
        ax.set_title(column)
        ax.set_ylabel("Rows")
    for ax in axes_list[len(numeric_cols):]:
        ax.axis("off")
    fig.suptitle(f"{title}: Numeric Distributions", fontsize=12)
    save_figure(fig, path)


def plot_top_categories(df: pd.DataFrame, title: str, path: Path) -> None:
    categorical_cols = [
        column
        for column in df.columns
        if not pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_datetime64_any_dtype(df[column])
    ]
    if not categorical_cols:
        return

    chosen = categorical_cols[:4]
    fig, axes = plt.subplots(len(chosen), 1, figsize=(9, 3 * len(chosen)))
    axes_list = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
    for ax, column in zip(axes_list, chosen):
        counts = df[column].astype("string").fillna("<NA>").value_counts().head(8).sort_values()
        ax.barh(counts.index, counts.values, color="#8172b3")
        ax.set_title(column)
        ax.set_xlabel("Rows")
    fig.suptitle(f"{title}: Top Categorical Values", fontsize=12)
    save_figure(fig, path)


def build_overview_table(df: pd.DataFrame, qc: pd.DataFrame, title: str) -> pd.DataFrame:
    primary_time = "RUN_TIME" if "RUN_TIME" in df.columns else df.select_dtypes(include=["datetime64[ns]"]).columns[0]
    warning_days = int((qc["status"] == "warning").sum())
    empty_days = int(qc["empty_file"].fillna(False).sum())
    return pd.DataFrame(
        [
            {
                "dataset": title,
                "rows": format_int(len(df)),
                "columns": format_int(len(df.columns)),
                "date_min": df[primary_time].min().strftime("%Y-%m-%d %H:%M:%S"),
                "date_max": df[primary_time].max().strftime("%Y-%m-%d %H:%M:%S"),
                "warning_days": format_int(warning_days),
                "empty_days": format_int(empty_days),
            }
        ]
    )


def build_rtdcv_relationship_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    calc_overload = df["MW_FLOW"] - df["BINDING_LIMIT"]
    calc_pct = df["MW_FLOW"] / df["BINDING_LIMIT"] * 100.0
    calc_pct_overload = df["OVERLOAD_MW"] / df["BINDING_LIMIT"] * 100.0

    relationship = pd.DataFrame(
        [
            {
                "check": "OVERLOAD_MW ~= MW_FLOW - BINDING_LIMIT",
                "max_abs_diff": format_float((df["OVERLOAD_MW"] - calc_overload).abs().max(), 4),
                "match_within_0.01": format_pct(
                    ((df["OVERLOAD_MW"] - calc_overload).abs() <= 0.01).mean() * 100.0
                ),
                "match_within_0.05": format_pct(
                    ((df["OVERLOAD_MW"] - calc_overload).abs() <= 0.05).mean() * 100.0
                ),
            },
            {
                "check": "PCT_MW ~= MW_FLOW / BINDING_LIMIT * 100",
                "max_abs_diff": format_float((df["PCT_MW"] - calc_pct).abs().max(), 4),
                "match_within_0.01": format_pct(
                    ((df["PCT_MW"] - calc_pct).abs() <= 0.01).mean() * 100.0
                ),
                "match_within_0.05": format_pct(
                    ((df["PCT_MW"] - calc_pct).abs() <= 0.05).mean() * 100.0
                ),
            },
            {
                "check": "PCT_MW - 100 ~= OVERLOAD_MW / BINDING_LIMIT * 100",
                "max_abs_diff": format_float(
                    (((df["PCT_MW"] - 100.0) - calc_pct_overload * 100.0 / 100.0).abs().max()),
                    4,
                ),
                "match_within_0.01": format_pct(
                    ((((df["PCT_MW"] - 100.0) - calc_pct_overload).abs()) <= 0.01).mean() * 100.0
                ),
                "match_within_0.05": format_pct(
                    ((((df["PCT_MW"] - 100.0) - calc_pct_overload).abs()) <= 0.05).mean() * 100.0
                ),
            },
        ]
    )

    equipment_station = (
        df.groupby("EQUIPMENT_NAME", observed=True)["STATION_NAME"]
        .agg(["nunique", lambda values: ", ".join(sorted(set(values.astype(str))))])
        .reset_index()
        .rename(
            columns={
                "EQUIPMENT_NAME": "equipment_name",
                "nunique": "station_count",
                "<lambda_0>": "station_names",
            }
        )
        .sort_values(["station_count", "equipment_name"], ascending=[False, True])
    )
    equipment_station["station_count"] = equipment_station["station_count"].map(format_int)
    return relationship, equipment_station


def build_rtdcv_top_equipment_table(df: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    top = (
        df.groupby("EQUIPMENT_NAME", observed=True)
        .agg(
            rows=("EQUIPMENT_NAME", "size"),
            congest_type=("CONGEST_TYPE", lambda s: ", ".join(sorted(set(s.astype(str))))),
            stations=("STATION_NAME", lambda s: ", ".join(sorted(set(s.astype(str))))),
            binding_median=("BINDING_LIMIT", "median"),
            flow_median=("MW_FLOW", "median"),
            overload_median=("OVERLOAD_MW", "median"),
            pct_median=("PCT_MW", "median"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
        .head(top_n)
    )
    total_rows = len(df)
    top["row_share"] = top["rows"] / total_rows * 100.0
    top["rows"] = top["rows"].map(format_int)
    top["row_share"] = top["row_share"].map(format_pct)
    for column in ["binding_median", "flow_median", "overload_median", "pct_median"]:
        top[column] = top[column].map(format_float)
    return top.rename(columns={"EQUIPMENT_NAME": "equipment_name"})


def plot_rtdcv_cross_bars(df: pd.DataFrame, axis_column: str, title: str, path: Path) -> None:
    counts = (
        df.groupby([axis_column, "CONGEST_TYPE"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_values(list(df["CONGEST_TYPE"].astype(str).unique()), ascending=False)
    )
    top_counts = counts.sum(axis=1).sort_values(ascending=False).head(12)
    counts = counts.loc[top_counts.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    left = pd.Series(0, index=counts.index, dtype="float64")
    colors = {"BASE CASE": "#4c72b0", "CONTINGENCY": "#dd8452"}
    for congest_type in counts.columns:
        ax.barh(
            counts.index.astype(str),
            counts[congest_type].values,
            left=left.values,
            label=str(congest_type),
            color=colors.get(str(congest_type), None),
        )
        left += counts[congest_type]
    ax.set_title(title)
    ax.set_xlabel("Rows")
    ax.legend(frameon=False)
    save_figure(fig, path)


def plot_rtdcv_top_equipment_overview(df: pd.DataFrame, path: Path, top_n: int = 6) -> None:
    top = (
        df.groupby("EQUIPMENT_NAME", observed=True)
        .agg(
            rows=("EQUIPMENT_NAME", "size"),
            congest_type=("CONGEST_TYPE", lambda s: ", ".join(sorted(set(s.astype(str))))),
            stations=("STATION_NAME", lambda s: ", ".join(sorted(set(s.astype(str))))),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
        .head(top_n)
        .sort_values("rows")
    )
    colors = {"BASE CASE": "#4c72b0", "CONTINGENCY": "#dd8452"}

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = [colors.get(value, "#888888") for value in top["congest_type"]]
    ax.barh(top["EQUIPMENT_NAME"], top["rows"], color=bar_colors, alpha=0.9)
    ax.set_title("RTDCV: Top 6 Equipment by Rows")
    ax.set_xlabel("Rows")
    for idx, (_, row) in enumerate(top.iterrows()):
        ax.text(
            row["rows"] * 1.01,
            idx,
            f"{row['congest_type']} | {row['stations']}",
            va="center",
            fontsize=9,
        )
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=colors["BASE CASE"], label="BASE CASE"),
            plt.Rectangle((0, 0), 1, 1, color=colors["CONTINGENCY"], label="CONTINGENCY"),
        ],
        frameon=False,
        loc="lower right",
    )
    save_figure(fig, path)


def plot_rtdcv_top_equipment_station_heatmap(df: pd.DataFrame, path: Path, top_n: int = 6) -> None:
    top_equipment = (
        df["EQUIPMENT_NAME"].value_counts().head(top_n).index.tolist()
    )
    subset = df[df["EQUIPMENT_NAME"].isin(top_equipment)].copy()
    pivot = (
        subset.groupby(["EQUIPMENT_NAME", "STATION_NAME"], observed=True)
        .size()
        .unstack(fill_value=0)
        .loc[top_equipment]
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
    ax.set_title("RTDCV: Top 6 Equipment x Station Row Counts")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(str), rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            value = int(pivot.iat[row_idx, col_idx])
            if value:
                ax.text(col_idx, row_idx, str(value), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Rows")
    save_figure(fig, path)


def plot_rtdcv_top_equipment_numeric_boxplots(df: pd.DataFrame, path: Path, top_n: int = 6) -> None:
    top_equipment = (
        df["EQUIPMENT_NAME"].value_counts().head(top_n).index.tolist()
    )
    subset = df[df["EQUIPMENT_NAME"].isin(top_equipment)].copy()
    metrics = ["BINDING_LIMIT", "MW_FLOW", "OVERLOAD_MW", "PCT_MW"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes_list = list(axes.flatten())
    for ax, metric in zip(axes_list, metrics):
        data = [
            subset.loc[subset["EQUIPMENT_NAME"] == equipment, metric].dropna().values
            for equipment in top_equipment
        ]
        ax.boxplot(data, vert=False, tick_labels=top_equipment, patch_artist=True)
        ax.set_title(metric)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2)
    fig.suptitle("RTDCV: Numeric Distributions for Top 6 Equipment", fontsize=12)
    save_figure(fig, path)


def build_markdown(output_markdown: Path, assets_dir: Path) -> str:
    parts: list[str] = []
    parts.append("# RTD Dataset Column Profile")
    parts.append("")
    parts.append(
        "This report profiles the three newly combined RTD datasets requested for structure review: "
        "`RTDSL`, `RTDCV`, and `RTDHS`. It focuses on file-level sparsity, column-level missingness, "
        "value distributions, and numeric ranges for the `2025-12-18` through `2026-03-18` window."
    )
    parts.append("")

    for dataset_code, meta in DATASETS.items():
        df = pd.read_parquet(meta["parquet"])
        qc = pd.read_csv(meta["qc"])

        overview = build_overview_table(df, qc, meta["title"])
        qc_status, qc_warnings = summarize_qc(qc)
        col_profile = column_profile(df)
        dt_profile = datetime_profile(df)
        num_profile = numeric_profile(df)
        cat_tables = categorical_distribution_tables(df)

        missingness_path = assets_dir / f"{dataset_code.lower()}_missingness.png"
        daily_rows_path = assets_dir / f"{dataset_code.lower()}_daily_rows.png"
        numeric_hist_path = assets_dir / f"{dataset_code.lower()}_numeric_hist.png"
        categorical_path = assets_dir / f"{dataset_code.lower()}_top_categories.png"
        cross_station_path = assets_dir / f"{dataset_code.lower()}_congest_x_station.png"
        cross_equipment_path = assets_dir / f"{dataset_code.lower()}_congest_x_equipment.png"
        top6_overview_path = assets_dir / f"{dataset_code.lower()}_top6_equipment_overview.png"
        top6_station_heatmap_path = assets_dir / f"{dataset_code.lower()}_top6_equipment_station_heatmap.png"
        top6_numeric_boxplots_path = assets_dir / f"{dataset_code.lower()}_top6_equipment_numeric_boxplots.png"

        plot_missingness(df, meta["title"], missingness_path)
        plot_daily_rows(df, meta["title"], daily_rows_path)
        plot_numeric_histograms(df, meta["title"], numeric_hist_path)
        plot_top_categories(df, meta["title"], categorical_path)
        if dataset_code == "RTDCV":
            plot_rtdcv_cross_bars(
                df,
                axis_column="STATION_NAME",
                title="RTDCV: Congest Type by Station",
                path=cross_station_path,
            )
            plot_rtdcv_cross_bars(
                df,
                axis_column="EQUIPMENT_NAME",
                title="RTDCV: Congest Type by Equipment",
                path=cross_equipment_path,
            )
            plot_rtdcv_top_equipment_overview(df, top6_overview_path)
            plot_rtdcv_top_equipment_station_heatmap(df, top6_station_heatmap_path)
            plot_rtdcv_top_equipment_numeric_boxplots(df, top6_numeric_boxplots_path)

        rel_missingness = missingness_path.relative_to(output_markdown.parent)
        rel_daily = daily_rows_path.relative_to(output_markdown.parent)
        rel_numeric = numeric_hist_path.relative_to(output_markdown.parent)
        rel_categorical = categorical_path.relative_to(output_markdown.parent)

        parts.append(f"## {dataset_code}: {meta['title']}")
        parts.append("")
        parts.append(meta["notes"])
        parts.append("")
        parts.append("### Overview")
        parts.append("")
        parts.append(markdown_table(overview))
        parts.append("")
        parts.append("### File-Level QC")
        parts.append("")
        parts.append(markdown_table(qc_status))
        parts.append("")
        parts.append("Warning breakdown:")
        parts.append("")
        parts.append(markdown_table(qc_warnings))
        parts.append("")
        parts.append("### Column Inventory")
        parts.append("")
        parts.append(markdown_table(col_profile))
        parts.append("")
        if not dt_profile.empty:
            parts.append("### Datetime Columns")
            parts.append("")
            parts.append(markdown_table(dt_profile))
            parts.append("")
        if not num_profile.empty:
            parts.append("### Numeric Columns")
            parts.append("")
            parts.append(markdown_table(num_profile))
            parts.append("")
        if dataset_code == "RTDCV":
            relationship, equipment_station = build_rtdcv_relationship_tables(df)
            top_equipment = build_rtdcv_top_equipment_table(df)
            rel_cross_station = cross_station_path.relative_to(output_markdown.parent)
            rel_cross_equipment = cross_equipment_path.relative_to(output_markdown.parent)
            rel_top6_overview = top6_overview_path.relative_to(output_markdown.parent)
            rel_top6_station_heatmap = top6_station_heatmap_path.relative_to(output_markdown.parent)
            rel_top6_numeric_boxplots = top6_numeric_boxplots_path.relative_to(output_markdown.parent)
            parts.append("### RTDCV Checks")
            parts.append("")
            parts.append(
                "The limit metrics behave like rounded arithmetic fields: `OVERLOAD_MW` is approximately "
                "`MW_FLOW - BINDING_LIMIT`, and `PCT_MW` is approximately `MW_FLOW / BINDING_LIMIT * 100`."
            )
            parts.append("")
            parts.append(markdown_table(relationship))
            parts.append("")
            parts.append("Equipment-to-station mapping:")
            parts.append("")
            parts.append(markdown_table(equipment_station.head(12)))
            parts.append("")
            parts.append(f"![RTDCV congest type by station]({rel_cross_station})")
            parts.append("")
            parts.append(f"![RTDCV congest type by equipment]({rel_cross_equipment})")
            parts.append("")
            parts.append("### RTDCV Top 6 Equipment Deep Dive")
            parts.append("")
            parts.append(
                "These are the six equipment names with the most RTDCV rows in the current window. "
                "The table and visuals below focus on whether they are base or contingency, "
                "which stations they map to, and how their numeric fields are distributed."
            )
            parts.append("")
            parts.append(markdown_table(top_equipment))
            parts.append("")
            parts.append(f"![RTDCV top 6 equipment overview]({rel_top6_overview})")
            parts.append("")
            parts.append(f"![RTDCV top 6 equipment station heatmap]({rel_top6_station_heatmap})")
            parts.append("")
            parts.append(f"![RTDCV top 6 equipment numeric boxplots]({rel_top6_numeric_boxplots})")
            parts.append("")
        parts.append("### Categorical / String Value Distributions")
        parts.append("")
        for column, table in cat_tables.items():
            parts.append(f"#### `{column}`")
            parts.append("")
            parts.append(markdown_table(table))
            parts.append("")
        parts.append("### Visuals")
        parts.append("")
        parts.append(f"![{dataset_code} missingness]({rel_missingness})")
        parts.append("")
        parts.append(f"![{dataset_code} daily rows]({rel_daily})")
        parts.append("")
        parts.append(f"![{dataset_code} numeric distributions]({rel_numeric})")
        parts.append("")
        parts.append(f"![{dataset_code} top categorical values]({rel_categorical})")
        parts.append("")

    return "\n".join(parts)


def main() -> int:
    args = parse_args()
    output_markdown = Path(args.output_markdown)
    assets_dir = Path(args.assets_dir)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    markdown = build_markdown(output_markdown, assets_dir)
    output_markdown.write_text(markdown, encoding="utf-8")
    print(f"Wrote markdown report to {output_markdown}")
    print(f"Wrote assets to {assets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
