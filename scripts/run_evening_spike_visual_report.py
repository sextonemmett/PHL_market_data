#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import html
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

DEFAULT_START_DATE = "2026-01-01"
DEFAULT_END_DATE = "2026-03-24"
DEFAULT_OUTPUT_HTML = Path("regressions/evening_spike_visual_report.html")
DEFAULT_OUTPUT_CSV = Path("regressions/evening_spike_visual_15min.csv")
TIMESTAMP_TOKEN_RE = re.compile(r"(\d{8,12})")
REGION_LABELS = {"CLUZ": "Luzon", "CVIS": "Visayas", "CMIN": "Mindanao"}
PAIR_SPECS = (
    {"key": "cluz_cvis", "title": "Luzon-Visayas", "left": "CLUZ", "right": "CVIS", "link": "VISLUZ1"},
    {"key": "cvis_cmin", "title": "Visayas-Mindanao", "left": "CVIS", "right": "CMIN", "link": "MINVIS1"},
    {"key": "cluz_cmin", "title": "Luzon-Mindanao", "left": "CLUZ", "right": "CMIN", "link": None},
)
DIRECT_PAIR_KEYS = ("cluz_cvis", "cvis_cmin")
RESERVE_COMMODITIES = ("Dr", "Fr", "Rd", "Ru")
RTDREG_NUMERIC_COLUMNS = (
    "MKT_REQT",
    "LOAD_BID",
    "LOAD_CURTAILED",
    "LOSSES",
    "GENERATION",
    "MKT_IMPORT",
    "MKT_EXPORT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a polished HTML visual report on evening price spikes, inter-island congestion, "
            "reserve prices, and time-of-day relationships using 15-minute bins."
        )
    )
    parser.add_argument("--mp-parquet", help="Combined MP parquet path.")
    parser.add_argument("--reserve-parquet", help="Combined reserve parquet path.")
    parser.add_argument("--hvdc-parquet", help="Combined RTDHS parquet path.")
    parser.add_argument("--rtdreg-parquet", help="Combined RTDREG parquet path.")
    parser.add_argument("--reserve-qc-manifest", help="Reserve QC manifest CSV path.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Analysis start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Analysis end date in YYYY-MM-DD.")
    parser.add_argument("--rolling-days", type=int, default=7, help="Rolling window length in days.")
    parser.add_argument("--bin-minutes", type=int, default=15, help="Time-bin size in minutes.")
    parser.add_argument("--output-html", default=str(DEFAULT_OUTPUT_HTML), help="Output HTML path.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Output CSV path.")
    args = parser.parse_args()
    if args.rolling_days < 2:
        raise SystemExit("--rolling-days must be at least 2.")
    if args.bin_minutes < 1 or 60 % args.bin_minutes != 0:
        raise SystemExit("--bin-minutes must be a positive divisor of 60.")
    return args


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


def date_bounds(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if start_ts > end_ts:
        raise ValueError("start_date must be on or before end_date.")
    return start_ts, end_ts + pd.Timedelta(days=1)


def filter_time_range(frame: pd.DataFrame, timestamp_column: str, start_ts: pd.Timestamp, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    timestamps = pd.to_datetime(frame[timestamp_column])
    return frame.loc[(timestamps >= start_ts) & (timestamps < end_exclusive)].copy()


def bin_frame(frame: pd.DataFrame, timestamp_column: str, bin_minutes: int) -> pd.DataFrame:
    result = frame.copy()
    result["bin_time"] = pd.to_datetime(result[timestamp_column]).dt.floor(f"{bin_minutes}min")
    return result


def load_mp_wide(path: Path, start_ts: pd.Timestamp, end_exclusive: pd.Timestamp, bin_minutes: int) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["TIME_INTERVAL", "REGION_NAME", "MARGINAL_PRICE"]).copy()
    frame = filter_time_range(frame, "TIME_INTERVAL", start_ts, end_exclusive)
    frame = bin_frame(frame, "TIME_INTERVAL", bin_minutes)
    grouped = (
        frame.groupby(["bin_time", "REGION_NAME"], observed=True)["MARGINAL_PRICE"]
        .median()
        .unstack("REGION_NAME")
        .sort_index()
    )
    return grouped.rename(columns={code: f"price_{code.lower()}" for code in REGION_LABELS})


def load_reserve_wide(path: Path, start_ts: pd.Timestamp, end_exclusive: pd.Timestamp, bin_minutes: int) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=["TIME_INTERVAL", "REGION_NAME", "COMMODITY_TYPE", "MARGINAL_PRICE"],
    ).copy()
    frame = filter_time_range(frame, "TIME_INTERVAL", start_ts, end_exclusive)
    frame = frame.loc[frame["COMMODITY_TYPE"].isin(RESERVE_COMMODITIES)].copy()
    frame = bin_frame(frame, "TIME_INTERVAL", bin_minutes)
    grouped = (
        frame.groupby(["bin_time", "REGION_NAME", "COMMODITY_TYPE"], observed=True)["MARGINAL_PRICE"]
        .median()
        .unstack(["REGION_NAME", "COMMODITY_TYPE"])
        .sort_index()
    )
    grouped.columns = [
        f"reserve_{commodity.lower()}_{region.lower()}"
        for region, commodity in grouped.columns.to_flat_index()
    ]
    return grouped


def load_hvdc_wide(path: Path, start_ts: pd.Timestamp, end_exclusive: pd.Timestamp, bin_minutes: int) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=["TIME_INTERVAL", "HVDC_NAME", "CONGESTION_FLAG", "FLOW_FROM", "FLOW_TO", "OVERLOAD_MW"],
    ).copy()
    frame = filter_time_range(frame, "TIME_INTERVAL", start_ts, end_exclusive)
    frame = bin_frame(frame, "TIME_INTERVAL", bin_minutes)
    frame["link_congested"] = (frame["CONGESTION_FLAG"] == "Y").astype(int)
    frame["OVERLOAD_MW"] = frame["OVERLOAD_MW"].fillna(0.0)
    grouped = (
        frame.groupby(["bin_time", "HVDC_NAME"], observed=True)
        .agg(
            link_congested=("link_congested", "max"),
            flow_from=("FLOW_FROM", "mean"),
            flow_to=("FLOW_TO", "mean"),
            overload_mw=("OVERLOAD_MW", "max"),
        )
        .unstack("HVDC_NAME")
        .sort_index()
    )
    grouped.columns = [
        f"{metric}_{link.lower()}"
        for metric, link in grouped.columns.to_flat_index()
    ]
    grouped = grouped.fillna(
        {
            "link_congested_visluz1": 0.0,
            "link_congested_minvis1": 0.0,
            "overload_mw_visluz1": 0.0,
            "overload_mw_minvis1": 0.0,
        }
    )
    grouped["connected_link_congested_cluz"] = grouped.get("link_congested_visluz1", 0.0)
    grouped["connected_link_congested_cvis"] = grouped.get("link_congested_visluz1", 0.0).combine(
        grouped.get("link_congested_minvis1", 0.0),
        max,
    )
    grouped["connected_link_congested_cmin"] = grouped.get("link_congested_minvis1", 0.0)
    grouped["pair_link_congested_cluz_cvis"] = grouped.get("link_congested_visluz1", 0.0)
    grouped["pair_link_congested_cvis_cmin"] = grouped.get("link_congested_minvis1", 0.0)
    return grouped


def load_rtdreg_wide(path: Path, start_ts: pd.Timestamp, end_exclusive: pd.Timestamp, bin_minutes: int) -> pd.DataFrame:
    columns = ["TIME_INTERVAL", "REGION_NAME", "COMMODITY_TYPE", *RTDREG_NUMERIC_COLUMNS]
    frame = pd.read_parquet(path, columns=columns).copy()
    frame = filter_time_range(frame, "TIME_INTERVAL", start_ts, end_exclusive)
    frame = frame.loc[frame["COMMODITY_TYPE"] == "En"].copy()
    frame = bin_frame(frame, "TIME_INTERVAL", bin_minutes)
    grouped = (
        frame.groupby(["bin_time", "REGION_NAME"], observed=True)[list(RTDREG_NUMERIC_COLUMNS)]
        .mean()
        .unstack("REGION_NAME")
        .sort_index()
    )
    grouped.columns = [
        f"{metric.lower()}_{region.lower()}"
        for metric, region in grouped.columns.to_flat_index()
    ]
    return grouped


def build_merged_frame(
    mp_path: Path,
    reserve_path: Path,
    hvdc_path: Path,
    rtdreg_path: Path,
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    bin_minutes: int,
) -> pd.DataFrame:
    mp_wide = load_mp_wide(mp_path, start_ts, end_exclusive, bin_minutes)
    reserve_wide = load_reserve_wide(reserve_path, start_ts, end_exclusive, bin_minutes)
    hvdc_wide = load_hvdc_wide(hvdc_path, start_ts, end_exclusive, bin_minutes)
    rtdreg_wide = load_rtdreg_wide(rtdreg_path, start_ts, end_exclusive, bin_minutes)

    merged = mp_wide.join(hvdc_wide, how="inner").join(rtdreg_wide, how="inner").join(reserve_wide, how="left")
    merged = merged.reset_index()
    merged["date"] = merged["bin_time"].dt.normalize()
    merged["clock_label"] = merged["bin_time"].dt.strftime("%H:%M")
    merged["clock_minutes"] = merged["bin_time"].dt.hour * 60 + merged["bin_time"].dt.minute
    merged["hour"] = merged["bin_time"].dt.hour
    merged["day_of_week"] = merged["bin_time"].dt.day_name()
    merged["evening_window_flag"] = (merged["clock_minutes"] >= 16 * 60) & (merged["clock_minutes"] < 19 * 60)

    for pair in PAIR_SPECS:
        left = pair["left"].lower()
        right = pair["right"].lower()
        signed_col = f"signed_gap_{pair['key']}"
        abs_col = f"abs_gap_{pair['key']}"
        merged[signed_col] = merged[f"price_{left}"] - merged[f"price_{right}"]
        merged[abs_col] = merged[signed_col].abs()

    for region_code in REGION_LABELS:
        region_lower = region_code.lower()
        outside_evening = merged.loc[~merged["evening_window_flag"]].groupby("date")[f"price_{region_lower}"].median()
        merged[f"baseline_{region_lower}"] = merged["date"].map(outside_evening)
        merged[f"spike_{region_lower}"] = merged[f"price_{region_lower}"] - merged[f"baseline_{region_lower}"]

    return merged.sort_values("bin_time").reset_index(drop=True)


def build_island_daily_summary(merged: pd.DataFrame) -> pd.DataFrame:
    evening = merged.loc[merged["evening_window_flag"]].copy()
    rows: list[dict[str, object]] = []
    for region_code, region_label in REGION_LABELS.items():
        region_lower = region_code.lower()
        reserve_columns = {commodity: f"reserve_{commodity.lower()}_{region_lower}" for commodity in RESERVE_COMMODITIES}
        grouped = evening.groupby("date").agg(
            mean_spike=(f"spike_{region_lower}", "mean"),
            p95_spike=(f"spike_{region_lower}", lambda s: float(s.quantile(0.95))),
            mean_price=(f"price_{region_lower}", "mean"),
            connected_share=(f"connected_link_congested_{region_lower}", "mean"),
            mean_demand=(f"mkt_reqt_{region_lower}", "mean"),
        )
        for commodity, column in reserve_columns.items():
            grouped[f"reserve_mean_{commodity.lower()}"] = evening.groupby("date")[column].mean()
        grouped = grouped.reset_index()
        grouped["island_code"] = region_code
        grouped["island_label"] = region_label
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def build_pair_daily_summary(merged: pd.DataFrame) -> pd.DataFrame:
    evening = merged.loc[merged["evening_window_flag"]].copy()
    rows: list[dict[str, object]] = []
    for pair in PAIR_SPECS:
        if pair["key"] not in DIRECT_PAIR_KEYS:
            continue
        grouped = evening.groupby("date").agg(
            mean_abs_gap=(f"abs_gap_{pair['key']}", "mean"),
            p95_abs_gap=(f"abs_gap_{pair['key']}", lambda s: float(s.quantile(0.95))),
            link_share=(f"pair_link_congested_{pair['key']}", "mean"),
        )
        grouped = grouped.reset_index()
        grouped["pair_key"] = pair["key"]
        grouped["pair_title"] = pair["title"]
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def build_evening_reserve_long(merged: pd.DataFrame) -> pd.DataFrame:
    evening = merged.loc[merged["evening_window_flag"]].copy()
    rows: list[pd.DataFrame] = []
    for region_code, region_label in REGION_LABELS.items():
        region_lower = region_code.lower()
        subset = pd.DataFrame(
            {
                "bin_time": evening["bin_time"],
                "date": evening["date"],
                "island_code": region_code,
                "island_label": region_label,
                "spike": evening[f"spike_{region_lower}"],
                "connected_congestion": evening[f"connected_link_congested_{region_lower}"],
            }
        )
        for commodity in RESERVE_COMMODITIES:
            subset[f"reserve_{commodity.lower()}"] = evening[f"reserve_{commodity.lower()}_{region_lower}"]
        rows.append(subset)
    long = pd.concat(rows, ignore_index=True)
    return long.melt(
        id_vars=["bin_time", "date", "island_code", "island_label", "spike", "connected_congestion"],
        value_vars=[f"reserve_{commodity.lower()}" for commodity in RESERVE_COMMODITIES],
        var_name="reserve_series",
        value_name="reserve_price",
    ).assign(
        commodity=lambda df: df["reserve_series"].str.replace("reserve_", "", regex=False).str.upper()
    )


def build_rolling_reserve_summary(island_daily: pd.DataFrame, rolling_days: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for commodity in RESERVE_COMMODITIES:
        reserve_col = f"reserve_mean_{commodity.lower()}"
        subset = island_daily[["date", "island_code", "island_label", "mean_spike", reserve_col]].rename(
            columns={reserve_col: "reserve_mean"}
        )
        subset["commodity"] = commodity
        subset["rolling_corr"] = np.nan
        for island_code, island_group in subset.groupby("island_code", observed=True):
            rolling = compute_rolling_spearman(
                island_group.sort_values("date"),
                x_col="reserve_mean",
                y_col="mean_spike",
                window=rolling_days,
            )
            subset.loc[rolling.index, "rolling_corr"] = rolling["rolling_corr"].to_numpy()
        rows.append(subset)
    return pd.concat(rows, ignore_index=True)


def compute_rolling_spearman(frame: pd.DataFrame, x_col: str, y_col: str, window: int) -> pd.DataFrame:
    ordered = frame.sort_values("date").copy()
    results: list[float] = []
    for idx in range(len(ordered)):
        if idx + 1 < window:
            results.append(np.nan)
            continue
        window_frame = ordered.iloc[idx + 1 - window : idx + 1][[x_col, y_col]].dropna()
        if len(window_frame) < window or window_frame[x_col].nunique() < 2 or window_frame[y_col].nunique() < 2:
            results.append(np.nan)
            continue
        results.append(float(window_frame[x_col].rank().corr(window_frame[y_col].rank())))
    ordered["rolling_corr"] = results
    return ordered


def build_rolling_island_congestion(island_daily: pd.DataFrame, rolling_days: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for island_code, island_group in island_daily.groupby("island_code", observed=True):
        rolling = compute_rolling_spearman(
            island_group.sort_values("date"),
            x_col="connected_share",
            y_col="mean_spike",
            window=rolling_days,
        )
        rows.append(rolling.assign(metric="Spike vs connected-link congestion share"))
    return pd.concat(rows, ignore_index=True)


def build_rolling_pair_congestion(pair_daily: pd.DataFrame, rolling_days: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for pair_key, pair_group in pair_daily.groupby("pair_key", observed=True):
        rolling = compute_rolling_spearman(
            pair_group.sort_values("date"),
            x_col="link_share",
            y_col="mean_abs_gap",
            window=rolling_days,
        )
        rows.append(rolling.assign(metric="Absolute gap vs pair-link congestion share"))
    return pd.concat(rows, ignore_index=True)


def load_reserve_qc(path: Path | None, start_ts: pd.Timestamp, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path).copy()
    if "file_date" in frame.columns:
        frame["file_date"] = pd.to_datetime(frame["file_date"], errors="coerce")
        frame = frame.loc[(frame["file_date"] >= start_ts) & (frame["file_date"] < end_exclusive)].copy()
    return frame


def build_coverage_summary(merged: pd.DataFrame, reserve_qc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage_rows: list[dict[str, object]] = []
    for commodity in RESERVE_COMMODITIES:
        for region_code, region_label in REGION_LABELS.items():
            column = f"reserve_{commodity.lower()}_{region_code.lower()}"
            coverage_rows.append(
                {
                    "commodity": commodity,
                    "island_code": region_code,
                    "island_label": region_label,
                    "coverage_pct": 100.0 * float(merged[column].notna().mean()),
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    warnings = reserve_qc.loc[reserve_qc.get("status", pd.Series(dtype=str)).eq("warning")].copy() if not reserve_qc.empty else pd.DataFrame()
    if not warnings.empty:
        warnings["coverage_pct"] = 100.0 * warnings["data_row_count"] / (288.0 * 3.0 * 4.0)
    return coverage, warnings


def figure_to_data_uri(fig: plt.Figure) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def full_clock_labels(bin_minutes: int) -> list[str]:
    start = pd.Timestamp("2026-01-01 00:00:00")
    periods = int((24 * 60) / bin_minutes)
    return [(start + pd.Timedelta(minutes=bin_minutes * i)).strftime("%H:%M") for i in range(periods)]


def evening_clock_labels(bin_minutes: int) -> list[str]:
    labels = full_clock_labels(bin_minutes)
    return [label for label in labels if "16:00" <= label <= "18:45"]


def apply_matplotlib_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.facecolor": "#faf7f1",
            "figure.facecolor": "#f4f1ea",
            "axes.edgecolor": "#102a43",
            "axes.labelcolor": "#102a43",
            "xtick.color": "#102a43",
            "ytick.color": "#102a43",
            "text.color": "#102a43",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
        }
    )


def date_ticks(date_labels: list[str], step: int = 7) -> tuple[list[int], list[str]]:
    positions = list(range(0, len(date_labels), step))
    return positions, [date_labels[position] for position in positions]


def clock_ticks(clock_labels: list[str], every_n: int) -> tuple[list[int], list[str]]:
    positions = list(range(0, len(clock_labels), every_n))
    return positions, [clock_labels[position] for position in positions]


def build_spike_heatmap_figure(merged: pd.DataFrame, bin_minutes: int) -> str:
    apply_matplotlib_theme()
    full_clocks = full_clock_labels(bin_minutes)
    heatmap_source = merged.copy()
    heatmap_source["date_label"] = heatmap_source["date"].dt.strftime("%Y-%m-%d")
    date_labels = sorted(heatmap_source["date_label"].unique())
    evening_start = full_clocks.index("16:00")
    evening_end = full_clocks.index("18:45")
    value_columns = [f"spike_{code.lower()}" for code in REGION_LABELS]
    all_values = np.concatenate([heatmap_source[column].dropna().to_numpy() for column in value_columns])
    vmax = float(np.nanquantile(np.abs(all_values), 0.98)) if len(all_values) else 1.0
    vmax = max(vmax, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 12), sharey=True)
    for ax, (region_code, region_label) in zip(axes, REGION_LABELS.items()):
        pivot = (
            heatmap_source.pivot(index="clock_label", columns="date_label", values=f"spike_{region_code.lower()}")
            .reindex(index=full_clocks, columns=date_labels)
        )
        masked = np.ma.masked_invalid(pivot.to_numpy())
        im = ax.imshow(masked, aspect="auto", origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.add_patch(
            Rectangle(
                (-0.5, evening_start - 0.5),
                len(date_labels),
                evening_end - evening_start + 1,
                fill=False,
                edgecolor="#f0b429",
                linewidth=2.0,
            )
        )
        xticks, xticklabels = date_ticks(date_labels, step=7)
        yticks, yticklabels = clock_ticks(full_clocks, every_n=max(1, int(60 / bin_minutes) * 2))
        ax.set_xticks(xticks, xticklabels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(yticks, yticklabels, fontsize=9)
        ax.set_title(f"{region_label} spike heatmap")
        ax.set_xlabel("Date")
        if ax is axes[0]:
            ax.set_ylabel("15-minute clock bin")
    colorbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    colorbar.set_label("Spike vs same-day outside-window baseline (PHP/MWh)")
    fig.suptitle("Section 1: Calendar-by-clock heatmaps of island price spikes", fontsize=16, y=1.02)
    return figure_to_data_uri(fig)


def build_intraday_profile_figure(merged: pd.DataFrame, bin_minutes: int) -> str:
    apply_matplotlib_theme()
    evening = merged.loc[merged["evening_window_flag"]].copy()
    clocks = evening_clock_labels(bin_minutes)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for ax, (region_code, region_label) in zip(axes, REGION_LABELS.items()):
        region_lower = region_code.lower()
        profile = evening.groupby("clock_label").agg(
            median_spike=(f"spike_{region_lower}", "median"),
            p75_spike=(f"spike_{region_lower}", lambda s: float(s.quantile(0.75))),
            p90_spike=(f"spike_{region_lower}", lambda s: float(s.quantile(0.90))),
            congestion_share=(f"connected_link_congested_{region_lower}", "mean"),
        ).reindex(clocks)
        x = np.arange(len(clocks))
        ax.plot(x, profile["median_spike"], color="#c05621", linewidth=2.6, label="Median spike")
        ax.plot(x, profile["p75_spike"], color="#dd6b20", linewidth=1.8, linestyle="--", label="p75 spike")
        ax.plot(x, profile["p90_spike"], color="#7b341e", linewidth=1.8, linestyle=":", label="p90 spike")
        ax2 = ax.twinx()
        ax2.plot(x, 100.0 * profile["congestion_share"], color="#2f855a", linewidth=2.0, alpha=0.75, label="Congestion share")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Congested bins (%)", color="#2f855a")
        tick_positions = list(range(0, len(clocks), 2))
        ax.set_xticks(tick_positions, [clocks[i] for i in tick_positions], rotation=45, ha="right")
        ax.axhline(0, color="#718096", linewidth=1.0, linestyle="--")
        ax.set_title(region_label)
        ax.set_xlabel("Clock time")
        if ax is axes[0]:
            ax.set_ylabel("Spike (PHP/MWh)")
        ax.grid(axis="y", alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Section 2: Evening intraday spike profiles with congestion overlay", fontsize=16, y=1.03)
    fig.tight_layout()
    return figure_to_data_uri(fig)


def build_rolling_island_congestion_figure(rolling: pd.DataFrame) -> str:
    apply_matplotlib_theme()
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=True)
    for ax, (region_code, region_label) in zip(axes, REGION_LABELS.items()):
        subset = rolling.loc[rolling["island_code"] == region_code].sort_values("date")
        ax.plot(subset["date"], subset["rolling_corr"], color="#2b6cb0", linewidth=2.2)
        ax.axhline(0, color="#718096", linestyle="--", linewidth=1.0)
        ax.set_title(region_label)
        ax.set_xlabel("Date")
        if ax is axes[0]:
            ax.set_ylabel("7-day rolling Spearman corr.")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Spike vs connected-link congestion share", fontsize=15, y=1.02)
    fig.tight_layout()
    return figure_to_data_uri(fig)


def build_rolling_pair_congestion_figure(rolling: pd.DataFrame) -> str:
    apply_matplotlib_theme()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    pair_titles = {pair["key"]: pair["title"] for pair in PAIR_SPECS if pair["key"] in DIRECT_PAIR_KEYS}
    for ax, pair_key in zip(axes, DIRECT_PAIR_KEYS):
        subset = rolling.loc[rolling["pair_key"] == pair_key].sort_values("date")
        ax.plot(subset["date"], subset["rolling_corr"], color="#805ad5", linewidth=2.2)
        ax.axhline(0, color="#718096", linestyle="--", linewidth=1.0)
        ax.set_title(pair_titles[pair_key])
        ax.set_xlabel("Date")
        if ax is axes[0]:
            ax.set_ylabel("7-day rolling Spearman corr.")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Pair absolute gap vs direct-link congestion share", fontsize=15, y=1.02)
    fig.tight_layout()
    return figure_to_data_uri(fig)


def build_rolling_reserve_figure(rolling: pd.DataFrame) -> str:
    apply_matplotlib_theme()
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True, sharey=True)
    for row_index, (region_code, region_label) in enumerate(REGION_LABELS.items()):
        for col_index, commodity in enumerate(RESERVE_COMMODITIES):
            ax = axes[row_index, col_index]
            subset = rolling.loc[
                (rolling["island_code"] == region_code) & (rolling["commodity"] == commodity)
            ].sort_values("date")
            ax.plot(subset["date"], subset["rolling_corr"], color="#c05621", linewidth=2.0)
            ax.axhline(0, color="#718096", linestyle="--", linewidth=1.0)
            if row_index == 0:
                ax.set_title(commodity)
            if col_index == 0:
                ax.set_ylabel(f"{region_label}\n7-day corr.")
            if row_index == len(REGION_LABELS) - 1:
                ax.tick_params(axis="x", rotation=45)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Island spike vs own-island reserve price by commodity", fontsize=15, y=1.01)
    fig.tight_layout()
    return figure_to_data_uri(fig)


def build_reserve_scatter_figure(evening_reserve_long: pd.DataFrame) -> str:
    apply_matplotlib_theme()
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharey=False)
    colors = {0: "#4c78a8", 1: "#c44e52"}
    for row_index, (region_code, region_label) in enumerate(REGION_LABELS.items()):
        for col_index, commodity in enumerate(RESERVE_COMMODITIES):
            ax = axes[row_index, col_index]
            subset = evening_reserve_long.loc[
                (evening_reserve_long["island_code"] == region_code)
                & (evening_reserve_long["commodity"] == commodity)
                & evening_reserve_long["reserve_price"].notna()
                & evening_reserve_long["spike"].notna()
            ].copy()
            for regime, regime_frame in subset.groupby("connected_congestion", observed=True):
                ax.scatter(
                    regime_frame["reserve_price"],
                    regime_frame["spike"],
                    s=13,
                    alpha=0.35,
                    color=colors[int(regime)],
                    label="Congested link" if int(regime) == 1 else "Uncongested link",
                )
            if len(subset) >= 8 and subset["reserve_price"].nunique() > 1 and subset["spike"].nunique() > 1:
                corr = float(subset["reserve_price"].rank().corr(subset["spike"].rank()))
                ax.text(0.03, 0.95, f"Spearman {corr:0.2f}", transform=ax.transAxes, va="top", fontsize=8)
            if row_index == 0:
                ax.set_title(commodity)
            if col_index == 0:
                ax.set_ylabel(f"{region_label}\nSpike (PHP/MWh)")
            if row_index == len(REGION_LABELS) - 1:
                ax.set_xlabel("Reserve price (PHP/MWh)")
            ax.grid(alpha=0.15)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[0], label="Uncongested link", markersize=6),
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[1], label="Congested link", markersize=6),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Section 4: Evening reserve price vs spike relationships", fontsize=15, y=1.01)
    fig.tight_layout()
    return figure_to_data_uri(fig)


def build_time_of_day_surface_figure(merged: pd.DataFrame, bin_minutes: int) -> str:
    apply_matplotlib_theme()
    full_clocks = full_clock_labels(bin_minutes)
    evening_start = full_clocks.index("16:00")
    evening_end = full_clocks.index("18:45")
    fig, axes = plt.subplots(3, 2, figsize=(18, 11), gridspec_kw={"width_ratios": [1.1, 1.7]})

    spike_values = np.concatenate([merged[f"spike_{code.lower()}"].dropna().to_numpy() for code in REGION_LABELS])
    spike_limit = max(float(np.nanquantile(np.abs(spike_values), 0.98)), 1.0)
    reserve_values = np.concatenate(
        [
            merged[f"reserve_{commodity.lower()}_{code.lower()}"].dropna().to_numpy()
            for commodity in RESERVE_COMMODITIES
            for code in REGION_LABELS
        ]
    )
    reserve_limit = max(float(np.nanquantile(reserve_values, 0.98)), 1.0)

    spike_im = None
    reserve_im = None
    for row_index, (region_code, region_label) in enumerate(REGION_LABELS.items()):
        region_lower = region_code.lower()
        spike_profile = (
            merged.groupby("clock_label")[f"spike_{region_lower}"].mean().reindex(full_clocks)
        )
        spike_matrix = np.ma.masked_invalid(spike_profile.to_numpy().reshape(1, -1))
        ax_spike = axes[row_index, 0]
        spike_im = ax_spike.imshow(
            spike_matrix,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=-spike_limit,
            vmax=spike_limit,
        )
        ax_spike.add_patch(
            Rectangle(
                (evening_start - 0.5, -0.5),
                evening_end - evening_start + 1,
                1.0,
                fill=False,
                edgecolor="#f0b429",
                linewidth=2.0,
            )
        )
        xticks, xticklabels = clock_ticks(full_clocks, every_n=max(1, int(60 / bin_minutes) * 2))
        ax_spike.set_xticks(xticks, xticklabels, rotation=45, ha="right", fontsize=8)
        ax_spike.set_yticks([0], [region_label], fontsize=10)
        ax_spike.set_title(f"{region_label}: average spike by clock")

        reserve_rows = []
        for commodity in RESERVE_COMMODITIES:
            reserve_profile = (
                merged.groupby("clock_label")[f"reserve_{commodity.lower()}_{region_lower}"].mean().reindex(full_clocks)
            )
            reserve_rows.append(reserve_profile.to_numpy())
        reserve_matrix = np.ma.masked_invalid(np.vstack(reserve_rows))
        ax_reserve = axes[row_index, 1]
        reserve_im = ax_reserve.imshow(
            reserve_matrix,
            aspect="auto",
            origin="lower",
            cmap="YlOrRd",
            vmin=0,
            vmax=reserve_limit,
        )
        ax_reserve.add_patch(
            Rectangle(
                (evening_start - 0.5, -0.5),
                evening_end - evening_start + 1,
                len(RESERVE_COMMODITIES),
                fill=False,
                edgecolor="#f0b429",
                linewidth=2.0,
            )
        )
        ax_reserve.set_xticks(xticks, xticklabels, rotation=45, ha="right", fontsize=8)
        ax_reserve.set_yticks(range(len(RESERVE_COMMODITIES)), RESERVE_COMMODITIES, fontsize=9)
        ax_reserve.set_title(f"{region_label}: average reserve price by clock")

    spike_colorbar = fig.colorbar(spike_im, ax=axes[:, 0], shrink=0.8, pad=0.02)
    spike_colorbar.set_label("Average spike (PHP/MWh)")
    reserve_colorbar = fig.colorbar(reserve_im, ax=axes[:, 1], shrink=0.8, pad=0.02)
    reserve_colorbar.set_label("Average reserve price (PHP/MWh)")
    fig.suptitle("Section 5: Time-of-day relationship surfaces", fontsize=15, y=1.01)
    fig.tight_layout()
    return figure_to_data_uri(fig)


def build_coverage_qc_figure(coverage: pd.DataFrame, warnings: pd.DataFrame) -> str:
    apply_matplotlib_theme()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    coverage_pivot = (
        coverage.pivot(index="commodity", columns="island_label", values="coverage_pct")
        .reindex(index=RESERVE_COMMODITIES, columns=list(REGION_LABELS.values()))
    )
    ax = axes[0]
    im = ax.imshow(coverage_pivot.to_numpy(), aspect="auto", origin="lower", cmap="YlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(coverage_pivot.columns)), coverage_pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(coverage_pivot.index)), coverage_pivot.index)
    ax.set_title("Reserve coverage by island and product")
    for row_index, commodity in enumerate(coverage_pivot.index):
        for col_index, island_label in enumerate(coverage_pivot.columns):
            value = coverage_pivot.loc[commodity, island_label]
            ax.text(col_index, row_index, f"{value:0.1f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03).set_label("Coverage (%)")

    ax_warn = axes[1]
    if warnings.empty:
        ax_warn.text(0.5, 0.5, "No reserve warning days in analysis window", ha="center", va="center", fontsize=12)
        ax_warn.set_axis_off()
    else:
        warning_dates = warnings["file_date"].dt.strftime("%Y-%m-%d")
        ax_warn.bar(warning_dates, warnings["coverage_pct"], color="#dd6b20", alpha=0.85)
        ax_warn.set_title("Reserve warning-day completeness")
        ax_warn.set_ylabel("Observed reserve rows (% of expected)")
        ax_warn.tick_params(axis="x", rotation=45)
        ax_warn.set_ylim(0, 105)
        ax_warn.grid(axis="y", alpha=0.2)

    fig.suptitle("Section 6: QC and data coverage", fontsize=15, y=1.02)
    fig.tight_layout()
    return figure_to_data_uri(fig)


def dataframe_to_html(frame: pd.DataFrame, float_digits: int = 2) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:,.{float_digits}f}" if pd.notna(value) else "")
    return display.to_html(index=False, escape=False, classes=["summary-table"])


def render_figure_card(title: str, image_uri: str, caption: str) -> str:
    return f"""
  <section class="figure-card">
    <h2>{html.escape(title)}</h2>
    <img src="{image_uri}" alt="{html.escape(title)}">
    <p class="caption">{html.escape(caption)}</p>
  </section>
"""


def build_html(
    args: argparse.Namespace,
    merged: pd.DataFrame,
    island_daily: pd.DataFrame,
    pair_daily: pd.DataFrame,
    coverage: pd.DataFrame,
    warnings: pd.DataFrame,
    figure_cards: list[str],
    output_csv: Path,
    reserve_qc_path: Path | None,
) -> str:
    css = """
body { font-family: Georgia, "Times New Roman", serif; margin: 32px auto; max-width: 1500px; color: #102a43; line-height: 1.55; background: #f4f1ea; }
h1, h2 { color: #0b1f33; }
h1 { margin-bottom: 8px; }
h2 { margin: 0 0 12px; }
p { margin: 0 0 14px; }
code { background: #dde7f0; color: #0b1f33; padding: 2px 5px; border-radius: 4px; }
.lead { font-size: 17px; color: #243b53; margin-bottom: 22px; }
.meta-card, .figure-card, .notes { background: #faf7f1; border: 1px solid #d9e2ec; border-radius: 14px; padding: 22px 24px; margin: 24px 0; box-shadow: 0 10px 28px rgba(16, 42, 67, 0.08); }
.meta-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px 16px; margin-top: 14px; }
.meta-item { background: #eef3f7; border-radius: 10px; padding: 10px 12px; }
.meta-label { display: block; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #486581; margin-bottom: 4px; }
.figure-card img { width: 100%; border-radius: 12px; border: 1px solid #d9e2ec; background: #ffffff; }
.caption { color: #486581; font-size: 14px; margin-top: 12px; }
.summary-table { border-collapse: collapse; width: 100%; margin-top: 14px; font-size: 14px; }
.summary-table th { background: #0b1f33; color: #ffffff; padding: 10px 12px; text-align: left; }
.summary-table td { border: 1px solid #bcccdc; padding: 8px 12px; background: #fffdf8; }
.summary-table tr:nth-child(even) td { background: #eef3f7; }
"""

    overlap_start = merged["bin_time"].min().strftime("%Y-%m-%d %H:%M")
    overlap_end = merged["bin_time"].max().strftime("%Y-%m-%d %H:%M")
    summary_frame = pd.DataFrame(
        [
            {"Metric": "15-minute bins in merged frame", "Value": len(merged)},
            {"Metric": "Analysis days", "Value": merged["date"].nunique()},
            {"Metric": "Evening bins (16:00-18:45)", "Value": int(merged["evening_window_flag"].sum())},
            {"Metric": "Island daily summary rows", "Value": len(island_daily)},
            {"Metric": "Direct-pair daily summary rows", "Value": len(pair_daily)},
            {"Metric": "Reserve warning days in window", "Value": len(warnings)},
        ]
    )

    warning_table_html = ""
    if not warnings.empty:
        warning_frame = warnings[["file_date", "data_row_count", "coverage_pct", "warnings"]].copy()
        warning_frame["file_date"] = warning_frame["file_date"].dt.strftime("%Y-%m-%d")
        warning_table_html = """
    <h3>Reserve warning days</h3>
""" + dataframe_to_html(warning_frame.rename(columns={"file_date": "Date", "data_row_count": "Reserve rows", "coverage_pct": "Coverage (%)", "warnings": "Warning"}), float_digits=1)

    reserve_qc_note = f"Reserve QC manifest used: {reserve_qc_path}" if reserve_qc_path is not None else "Reserve QC manifest was not available."

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Evening Spike Visual Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>Evening Spike Visual Report</h1>
  <p class="lead">This report studies price spikes across Luzon, Visayas, and Mindanao using <code>{args.bin_minutes}-minute</code> bins, with a focus on the evening window from <code>16:00</code> through <code>18:45</code>. It links island MP spikes to direct-link congestion, reserve market clearing prices, and time-of-day structure.</p>

  <section class="meta-card">
    <h2>Scope and Setup</h2>
    <p>The analysis uses the shared data overlap from <code>{html.escape(overlap_start)}</code> through <code>{html.escape(overlap_end)}</code>. MP and reserve prices are collapsed by median within each <code>{args.bin_minutes}-minute</code> bin. HVDC congestion flags use bin-level maxima, and RTDREG demand controls use bin-level means.</p>
    <div class="meta-grid">
      <div class="meta-item"><span class="meta-label">Output CSV</span><span><code>{html.escape(str(output_csv))}</code></span></div>
      <div class="meta-item"><span class="meta-label">Rolling Window</span><span>{args.rolling_days} days, Spearman correlation</span></div>
      <div class="meta-item"><span class="meta-label">Reserve Products</span><span>{", ".join(RESERVE_COMMODITIES)}</span></div>
      <div class="meta-item"><span class="meta-label">Direct Links</span><span><code>VISLUZ1</code>, <code>MINVIS1</code></span></div>
      <div class="meta-item"><span class="meta-label">Spike Baseline</span><span>Same-day median MP price outside the evening window</span></div>
      <div class="meta-item"><span class="meta-label">QC Source</span><span>{html.escape(reserve_qc_note)}</span></div>
    </div>
    {dataframe_to_html(summary_frame, float_digits=0)}
  </section>

  {''.join(figure_cards)}

  <section class="notes">
    <h2>QC and Caveats</h2>
    <p>The reserve download attempt for <code>2026-04-01</code> returned <code>404</code>, so reserve analysis is effectively capped before that date and the report uses the earlier shared overlap ending on <code>2026-03-24</code>.</p>
    <p>Direct pair congestion relationships are only available for Luzon-Visayas and Visayas-Mindanao because those are the two observed inter-island links in <code>RTDHS</code>. Luzon-Mindanao price gaps are still included in the merged analysis dataset for descriptive comparisons.</p>
    {warning_table_html}
  </section>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    mp_path = Path(args.mp_parquet) if args.mp_parquet else latest_matching_file(Path("data/mp/combined"), "MP_*.parquet")
    reserve_path = Path(args.reserve_parquet) if args.reserve_parquet else latest_matching_file(
        Path("data/mp_reserve/combined"),
        "MP_RESERVE_*.parquet",
    )
    hvdc_path = Path(args.hvdc_parquet) if args.hvdc_parquet else latest_matching_file(
        Path("data/rtdhs/combined"),
        "RTDHS_*.parquet",
    )
    rtdreg_path = Path(args.rtdreg_parquet) if args.rtdreg_parquet else latest_matching_file(
        Path("data/rtdreg/combined"),
        "RTDREG_*.parquet",
    )
    reserve_qc_path = Path(args.reserve_qc_manifest) if args.reserve_qc_manifest else (
        latest_matching_file(Path("data/mp_reserve/qc"), "mp_reserve_qc_*.csv")
        if Path("data/mp_reserve/qc").exists()
        else None
    )
    output_html = Path(args.output_html)
    output_csv = Path(args.output_csv)

    start_ts, end_exclusive = date_bounds(args.start_date, args.end_date)
    merged = build_merged_frame(
        mp_path=mp_path,
        reserve_path=reserve_path,
        hvdc_path=hvdc_path,
        rtdreg_path=rtdreg_path,
        start_ts=start_ts,
        end_exclusive=end_exclusive,
        bin_minutes=args.bin_minutes,
    )
    island_daily = build_island_daily_summary(merged)
    pair_daily = build_pair_daily_summary(merged)
    evening_reserve_long = build_evening_reserve_long(merged)
    rolling_island = build_rolling_island_congestion(island_daily, rolling_days=args.rolling_days)
    rolling_pair = build_rolling_pair_congestion(pair_daily, rolling_days=args.rolling_days)
    rolling_reserve = build_rolling_reserve_summary(island_daily, rolling_days=args.rolling_days)
    reserve_qc = load_reserve_qc(reserve_qc_path, start_ts, end_exclusive)
    coverage, warnings = build_coverage_summary(merged, reserve_qc)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    figure_cards = [
        render_figure_card(
            "Section 1: Evening Spike Heatmaps",
            build_spike_heatmap_figure(merged, bin_minutes=args.bin_minutes),
            "Calendar-by-clock heatmaps show when island prices rise above the same-day outside-window baseline, with the evening window outlined in gold.",
        ),
        render_figure_card(
            "Section 2: Intraday Spike Profiles",
            build_intraday_profile_figure(merged, bin_minutes=args.bin_minutes),
            "Median and upper-tail spike paths are shown alongside the share of evening bins with connected-link congestion.",
        ),
        render_figure_card(
            "Section 3A: Rolling Island Congestion Relationships",
            build_rolling_island_congestion_figure(rolling_island),
            "Seven-day rolling Spearman correlations track how each island's evening spike comoves with connected inter-island congestion.",
        ),
        render_figure_card(
            "Section 3B: Rolling Direct-Pair Relationships",
            build_rolling_pair_congestion_figure(rolling_pair),
            "Direct-pair price gaps are compared against the share of evening bins with direct-link congestion on the same pair.",
        ),
        render_figure_card(
            "Section 3C: Rolling Reserve Relationships",
            build_rolling_reserve_figure(rolling_reserve),
            "Reserve products remain separate so it is easy to see which reserve price moves most strongly with evening island spikes.",
        ),
        render_figure_card(
            "Section 4: Reserve vs Spike Visuals",
            build_reserve_scatter_figure(evening_reserve_long),
            "Each panel shows evening-bin reserve prices against island spike values, colored by whether the connected inter-island link is congested.",
        ),
        render_figure_card(
            "Section 5: Time-of-Day Relationship Surfaces",
            build_time_of_day_surface_figure(merged, bin_minutes=args.bin_minutes),
            "Average spike and reserve-price surfaces across the full day make it easy to see whether the evening concentration stands out from the rest of the clock.",
        ),
        render_figure_card(
            "Section 6: QC and Data Coverage",
            build_coverage_qc_figure(coverage, warnings),
            "Coverage by island and reserve product is shown alongside reserve warning-day completeness within the analysis window.",
        ),
    ]
    output_html.write_text(
        build_html(
            args=args,
            merged=merged,
            island_daily=island_daily,
            pair_daily=pair_daily,
            coverage=coverage,
            warnings=warnings,
            figure_cards=figure_cards,
            output_csv=output_csv,
            reserve_qc_path=reserve_qc_path,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {output_html}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
