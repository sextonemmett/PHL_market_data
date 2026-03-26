#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

CONTROL_COLUMNS = ("losses", "generation", "mkt_import", "mkt_export")
TIMESTAMP_TOKEN_RE = re.compile(r"(\d{8,12})")
PAIR_SPECS = (
    {"pair_key": "CLUZ_CVIS", "title": "Luzon-Visayas", "island_1": "Luzon", "island_2": "Visayas"},
    {"pair_key": "CVIS_CMIN", "title": "Visayas-Mindanao", "island_1": "Visayas", "island_2": "Mindanao"},
)
CHART_TERMS = (
    {"term": "link_congested_any", "label_template": "Link congestion", "family": "link"},
    {"term": "equip_cong_any_1", "label_template": "{island_1} equipment congestion", "family": "equipment"},
    {"term": "equip_overload_any_1", "label_template": "{island_1} overload", "family": "overload"},
    {"term": "equip_cong_any_2", "label_template": "{island_2} equipment congestion", "family": "equipment"},
    {"term": "equip_overload_any_2", "label_template": "{island_2} overload", "family": "overload"},
)
SEASONALITY_SPECS = (
    {
        "spec_key": "month_week_day_of_week_fe",
        "label": "Month + ISO Week + Day-of-Week FE",
        "fe_term": "C(fe_month) + C(fe_iso_week) + C(fe_day_of_week)",
    },
)
DEFAULT_OUTPUT_FIGURE = Path("regressions/direct_pair_ols_seasonality_fe_visual.png")
DEFAULT_OUTPUT_CSV = Path("regressions/direct_pair_ols_seasonality_fe_coefficients.csv")
EXCLUDED_PLOT_TERMS = {
    ("CLUZ_CVIS", "equip_overload_any_1"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit full-sample pair-specific levels OLS for direct-pair gaps using month, ISO week, "
            "and day-of-week fixed effects together, and write a stacked-coefficient visual scaled "
            "as a percent of the clean pair-price base."
        )
    )
    parser.add_argument("--direct-pair-panel", help="Direct-pair panel parquet path.")
    parser.add_argument(
        "--output-figure",
        default=str(DEFAULT_OUTPUT_FIGURE),
        help="Matplotlib figure output path, for example .png, .pdf, or .svg.",
    )
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Output CSV path.")
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
    frame["fe_month"] = frame["time_interval"].dt.month.astype(int)
    frame["fe_iso_week"] = frame["time_interval"].dt.isocalendar().week.astype(int)
    frame["fe_day_of_week"] = frame["time_interval"].dt.day_name()
    return frame


def select_estimable_terms(frame: pd.DataFrame, rhs_terms: list[str]) -> list[str]:
    kept_terms: list[str] = []
    for term in rhs_terms:
        if frame[term].nunique(dropna=False) <= 1:
            continue
        if any(frame[term].equals(frame[kept_term]) for kept_term in kept_terms):
            continue
        kept_terms.append(term)
    return kept_terms


def format_number(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def fit_pair_levels_ols(frame: pd.DataFrame, fe_term: str) -> object:
    rhs_terms = [
        chart_term["term"] for chart_term in CHART_TERMS
    ] + [f"{control}_{side}" for control in CONTROL_COLUMNS for side in ("1", "2")]
    estimable_rhs = select_estimable_terms(frame, rhs_terms)
    formula = f"dep_abs_price_gap ~ {' + '.join([*estimable_rhs, fe_term])}"
    return smf.ols(formula=formula, data=frame).fit(cov_type="HC1")


def fully_clean_base(frame: pd.DataFrame) -> tuple[float, float, float, int]:
    clean = frame.loc[
        (frame["link_congested_any"] == 0)
        & (frame["equip_cong_any_1"] == 0)
        & (frame["equip_overload_any_1"] == 0)
        & (frame["equip_cong_any_2"] == 0)
        & (frame["equip_overload_any_2"] == 0)
    ].copy()
    mean_price_1 = float(clean["price_1"].mean())
    mean_price_2 = float(clean["price_2"].mean())
    base = 0.5 * (mean_price_1 + mean_price_2)
    return mean_price_1, mean_price_2, base, int(len(clean))


def build_pair_rows(frame: pd.DataFrame, pair_spec: dict[str, str], seasonality_spec: dict[str, str]) -> list[dict[str, object]]:
    result = fit_pair_levels_ols(frame, fe_term=seasonality_spec["fe_term"])
    mean_price_1, mean_price_2, base_price, clean_rows = fully_clean_base(frame)
    rows: list[dict[str, object]] = []
    for chart_term in CHART_TERMS:
        term = chart_term["term"]
        term_label = chart_term["label_template"].format(
            island_1=pair_spec["island_1"],
            island_2=pair_spec["island_2"],
        )
        if term not in result.params.index:
            coef = np.nan
            std_err = np.nan
            pvalue = np.nan
        else:
            coef = float(result.params[term])
            std_err = float(result.bse[term])
            pvalue = float(result.pvalues[term])
        scaled_pct = 100.0 * coef / base_price if not pd.isna(coef) else np.nan
        scaled_se_pct = 100.0 * std_err / base_price if not pd.isna(std_err) else np.nan
        rows.append(
            {
                "pair_key": pair_spec["pair_key"],
                "pair_title": pair_spec["title"],
                "island_1": pair_spec["island_1"],
                "island_2": pair_spec["island_2"],
                "seasonality_key": seasonality_spec["spec_key"],
                "seasonality_label": seasonality_spec["label"],
                "fe_term": seasonality_spec["fe_term"],
                "term": term,
                "term_label": term_label,
                "coef": coef,
                "std_err": std_err,
                "pvalue": pvalue,
                "scaled_pct_of_clean_base": scaled_pct,
                "scaled_se_pct_of_clean_base": scaled_se_pct,
                "mean_price_1_clean": mean_price_1,
                "mean_price_2_clean": mean_price_2,
                "clean_pair_base": base_price,
                "clean_rows": clean_rows,
                "total_rows": int(len(frame)),
                "nobs": int(result.nobs),
            }
        )
    return rows


def render_subplot(ax: plt.Axes, pair_rows: pd.DataFrame) -> None:
    plotted_rows = pair_rows.loc[
        pair_rows["scaled_pct_of_clean_base"].notna()
        & ~pair_rows.apply(lambda row: (row["pair_key"], row["term"]) in EXCLUDED_PLOT_TERMS, axis=1)
    ].copy()
    positive_bottom = 0.0
    negative_bottom = 0.0

    for _, row in plotted_rows.iterrows():
        value = float(row["scaled_pct_of_clean_base"])
        family = next(chart_term["family"] for chart_term in CHART_TERMS if chart_term["term"] == row["term"])
        if value >= 0:
            color = {
                "link": "#c0392b",
                "equipment": "#e67e22",
                "overload": "#8e2d1f",
            }[family]
        else:
            color = {
                "link": "#5a8f29",
                "equipment": "#74a33a",
                "overload": "#1f6b3a",
            }[family]
        bottom = positive_bottom if value >= 0 else negative_bottom
        ax.bar([0], [value], width=0.58, bottom=bottom, color=color, edgecolor="none", linewidth=0.0)
        label_y = bottom + (value / 2.0)
        label_text = f"{row['term_label']}\n{format_number(value, 0)}%"
        if abs(value) >= 6:
            ax.text(
                0,
                label_y,
                label_text,
                ha="center",
                va="center",
                color="white",
                fontsize=8.5,
                fontweight="bold",
            )
        else:
            ax.text(
                0.38,
                bottom + value,
                label_text,
                ha="left",
                va="center",
                color="#102a43",
                fontsize=8,
            )
        if value >= 0:
            positive_bottom += value
        else:
            negative_bottom += value

    total = float(plotted_rows["scaled_pct_of_clean_base"].sum())
    top = max(positive_bottom, 0.0)
    ax.text(
        0,
        top + max(abs(total) * 0.03, 4.0),
        f"Net: {format_number(total, 0)}%",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#102a43",
    )

    first = pair_rows.iloc[0]
    ax.text(
        0.5,
        1.15,
        first["pair_title"],
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=15,
        color="#0b1f33",
        fontweight="bold",
    )
    ax.text(
        0.5,
        1.09,
        first["seasonality_label"],
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="#000000",
    )
    ax.text(
        0.5,
        1.03,
        (
            "Average pair price without equipment or link congestion = PHP "
            f"{format_number(float(first['clean_pair_base']), 0)}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12.5,
        color="#000000",
    )
    ax.set_xticks([0], ["Stacked congestion effects"], fontsize=12)
    ax.tick_params(axis="x", length=4, colors="#000000")
    ax.set_xlim(-0.75, 0.75)
    ax.axhline(0, color="#000000", linewidth=1.2)
    ax.grid(axis="y", color="#000000", linewidth=1.1, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=12, colors="#000000")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")


def build_figure(rows: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.6), sharey=True)
    fig.patch.set_facecolor("#f4f1ea")
    for ax in axes:
        ax.set_facecolor("#faf7f1")

    combined_spec = SEASONALITY_SPECS[0]
    for ax, pair_spec in zip(axes, PAIR_SPECS):
        pair_rows = rows.loc[
            (rows["pair_key"] == pair_spec["pair_key"])
            & (rows["seasonality_key"] == combined_spec["spec_key"])
        ].copy()
        render_subplot(ax, pair_rows)

    axes[0].set_ylabel("% of average pair price without equipment or link congestion", color="#000000", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 1])
    return fig


def main() -> None:
    args = parse_args()
    panel_path = Path(args.direct_pair_panel) if args.direct_pair_panel else latest_matching_file(
        Path("data/panels"),
        "RTD_DIRECT_PAIR_PANEL_*.parquet",
    )
    output_figure = Path(args.output_figure)
    output_csv = Path(args.output_csv)

    frame = load_panel(panel_path)
    rows: list[dict[str, object]] = []
    for pair_spec in PAIR_SPECS:
        pair_frame = frame.loc[frame["pair_key"] == pair_spec["pair_key"]].copy()
        for seasonality_spec in SEASONALITY_SPECS:
            rows.extend(build_pair_rows(pair_frame, pair_spec, seasonality_spec))

    result_frame = pd.DataFrame(rows)
    fig = build_figure(result_frame)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output_csv, index=False)
    fig.savefig(output_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output_figure}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
