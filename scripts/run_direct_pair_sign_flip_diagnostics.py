#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

CONTROL_COLUMNS = ("losses", "generation", "mkt_import", "mkt_export")
TIMESTAMP_TOKEN_RE = re.compile(r"(\d{8,12})")
DEFAULT_OUTPUT_HTML = Path("regressions/direct_pair_sign_flip_diagnostics.html")
SCARCITY_PRICE_CAP = 30_000.0
SCARCITY_PRICE_FLOOR = -9_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run direct-pair diagnostics to understand the equipment-congestion sign flip "
            "between the non-elasticity and elasticity-style regressions."
        )
    )
    parser.add_argument("--direct-pair-panel", help="Direct-pair panel parquet path.")
    parser.add_argument(
        "--output-html",
        default=str(DEFAULT_OUTPUT_HTML),
        help="Output HTML report path.",
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


def format_number(value: float, digits: int = 4) -> str:
    return f"{value:,.{digits}f}"


def significance_stars(pvalue: float) -> str:
    if pd.isna(pvalue):
        return ""
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.10:
        return "*"
    return ""


def add_log1p_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if (result[column] < 0).any():
            raise ValueError(f"Column {column} contains negative values; cannot apply log1p.")
        result[f"log1p_{column}"] = np.log1p(result[column].astype(float))
    return result


def fit_direct_pair_model(frame: pd.DataFrame, dep_var: str, use_log_controls: bool) -> object:
    rhs_terms = ["link_congested_any", "equip_cong_any_1", "equip_cong_any_2"]
    rhs_terms += [
        f"log1p_{control}_{side}" if use_log_controls else f"{control}_{side}"
        for control in CONTROL_COLUMNS
        for side in ("1", "2", "total")
    ]
    formula = f"{dep_var} ~ {' + '.join(rhs_terms + ['C(pair_key)', 'C(fe_day)'])}"
    return smf.ols(formula=formula, data=frame).fit(cov_type="HC1")


def describe_gap_by_island1_congestion(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    quantiles = [0.75, 0.9, 0.95, 0.99]
    for indicator_value, group in frame.groupby("equip_cong_any_1", observed=True):
        row = {
            "island_1_congested": int(indicator_value),
            "rows": int(len(group)),
            "share_of_sample": len(group) / len(frame),
            "mean_gap": float(group["dep_abs_price_gap"].mean()),
            "median_gap": float(group["dep_abs_price_gap"].median()),
            "std_gap": float(group["dep_abs_price_gap"].std()),
            "max_gap": float(group["dep_abs_price_gap"].max()),
        }
        for q in quantiles:
            row[f"p{int(q * 100)}_gap"] = float(group["dep_abs_price_gap"].quantile(q))
        rows.append(row)
    return pd.DataFrame(rows)


def build_indicator_summary(frame: pd.DataFrame) -> pd.DataFrame:
    indicators = ["link_congested_any", "equip_cong_any_1", "equip_cong_any_2"]
    rows: list[dict[str, object]] = []
    for indicator in indicators:
        for indicator_value, group in frame.groupby(indicator, observed=True):
            rows.append(
                {
                    "indicator": indicator,
                    "state": int(indicator_value),
                    "rows": int(len(group)),
                    "mean_gap": float(group["dep_abs_price_gap"].mean()),
                    "median_gap": float(group["dep_abs_price_gap"].median()),
                }
            )
    return pd.DataFrame(rows)


def build_model_comparison(frame: pd.DataFrame) -> tuple[pd.DataFrame, float, int]:
    p99_gap = float(frame["dep_abs_price_gap"].quantile(0.99))
    comparison_frame = frame.copy()
    comparison_frame["dep_abs_price_gap_winsor_99"] = comparison_frame["dep_abs_price_gap"].clip(upper=p99_gap)
    comparison_frame["log1p_dep_abs_price_gap_winsor_99"] = np.log1p(comparison_frame["dep_abs_price_gap_winsor_99"])

    scarcity_mask = (
        (comparison_frame["price_1"] >= SCARCITY_PRICE_CAP)
        | (comparison_frame["price_2"] >= SCARCITY_PRICE_CAP)
        | (comparison_frame["price_1"] <= SCARCITY_PRICE_FLOOR)
        | (comparison_frame["price_2"] <= SCARCITY_PRICE_FLOOR)
    )

    models = [
        (
            "Baseline level",
            fit_direct_pair_model(comparison_frame, "dep_abs_price_gap", use_log_controls=False),
            "dep_abs_price_gap",
        ),
        (
            "Baseline elasticity-style",
            fit_direct_pair_model(comparison_frame, "log1p_dep_abs_price_gap", use_log_controls=True),
            "log1p_dep_abs_price_gap",
        ),
        (
            "Winsorized level",
            fit_direct_pair_model(comparison_frame, "dep_abs_price_gap_winsor_99", use_log_controls=False),
            "dep_abs_price_gap_winsor_99",
        ),
        (
            "Winsorized elasticity-style",
            fit_direct_pair_model(comparison_frame, "log1p_dep_abs_price_gap_winsor_99", use_log_controls=True),
            "log1p_dep_abs_price_gap_winsor_99",
        ),
        (
            "Restricted-sample level",
            fit_direct_pair_model(comparison_frame.loc[~scarcity_mask].copy(), "dep_abs_price_gap", use_log_controls=False),
            "dep_abs_price_gap",
        ),
    ]

    rows: list[dict[str, object]] = []
    for model_name, result, dependent_variable in models:
        for term in ("link_congested_any", "equip_cong_any_1", "equip_cong_any_2"):
            rows.append(
                {
                    "model": model_name,
                    "dependent_variable": dependent_variable,
                    "term": term,
                    "coef": float(result.params[term]),
                    "std_err": float(result.bse[term]),
                    "pvalue": float(result.pvalues[term]),
                    "coef_with_se": f"{format_number(float(result.params[term]))}{significance_stars(float(result.pvalues[term]))}<br>({format_number(float(result.bse[term]))})",
                    "nobs": int(result.nobs),
                    "rsquared": float(result.rsquared),
                }
            )
    return pd.DataFrame(rows), p99_gap, int(scarcity_mask.sum())


def build_correlation_and_vif(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vif_columns = [
        "link_congested_any",
        "equip_cong_any_1",
        "equip_cong_any_2",
        "losses_total",
        "generation_total",
        "mkt_import_total",
        "mkt_export_total",
    ]
    correlation = frame[vif_columns].corr().round(4)

    def compute_vif_table(columns: list[str]) -> pd.DataFrame:
        X = frame[columns].astype(float)
        rows = []
        for index, column in enumerate(columns):
            rows.append({"variable": column, "vif": float(variance_inflation_factor(X.values, index))})
        return pd.DataFrame(rows)

    full_vif = compute_vif_table(vif_columns)
    reduced_vif = compute_vif_table([column for column in vif_columns if column != "mkt_export_total"])
    return correlation, full_vif, reduced_vif


def html_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda x: format_number(float(x), float_digits))
    return display.to_html(index=False, escape=False, classes=["report-table"])


def make_indicator_plot_svg(indicator_summary: pd.DataFrame, indicator: str, title: str) -> str:
    subset = indicator_summary.loc[indicator_summary["indicator"] == indicator].sort_values("state")
    if subset.empty:
        return ""

    width = 420
    height = 220
    margin_left = 70
    margin_bottom = 40
    chart_width = width - margin_left - 20
    chart_height = height - 30 - margin_bottom
    max_value = max(float(subset["mean_gap"].max()), float(subset["median_gap"].max()), 1.0)
    scale = chart_height / max_value
    bar_width = 38
    group_gap = 70
    colors = {"mean_gap": "#b85c38", "median_gap": "#355c7d"}
    labels = {"mean_gap": "Mean", "median_gap": "Median"}

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" class="mini-chart" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{width / 2}" y="18" text-anchor="middle" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 10}" y2="{height - margin_bottom}" class="axis-line" />',
        f'<line x1="{margin_left}" y1="30" x2="{margin_left}" y2="{height - margin_bottom}" class="axis-line" />',
    ]

    for tick_fraction in (0.0, 0.5, 1.0):
        tick_value = max_value * tick_fraction
        y = height - margin_bottom - (tick_value * scale)
        svg_parts.append(f'<line x1="{margin_left - 4}" y1="{y}" x2="{margin_left}" y2="{y}" class="axis-line" />')
        svg_parts.append(
            f'<text x="{margin_left - 8}" y="{y + 4}" text-anchor="end" class="tick-label">{html.escape(format_number(tick_value, 0))}</text>'
        )

    for group_index, (_, row) in enumerate(subset.iterrows()):
        group_x = margin_left + 30 + group_index * (2 * bar_width + group_gap)
        for bar_index, metric in enumerate(("mean_gap", "median_gap")):
            value = float(row[metric])
            bar_height = value * scale
            x = group_x + bar_index * bar_width
            y = height - margin_bottom - bar_height
            svg_parts.append(
                f'<rect x="{x}" y="{y}" width="{bar_width - 6}" height="{bar_height}" fill="{colors[metric]}" rx="4" ry="4" />'
            )
            svg_parts.append(
                f'<text x="{x + (bar_width - 6) / 2}" y="{max(y - 6, 24)}" text-anchor="middle" class="bar-label">{html.escape(format_number(value, 0))}</text>'
            )
        svg_parts.append(
            f'<text x="{group_x + bar_width - 3}" y="{height - 12}" text-anchor="middle" class="tick-label">{int(row["state"])}</text>'
        )

    legend_x = width - 155
    legend_y = 36
    for legend_index, metric in enumerate(("mean_gap", "median_gap")):
        y = legend_y + legend_index * 18
        svg_parts.append(f'<rect x="{legend_x}" y="{y - 10}" width="10" height="10" fill="{colors[metric]}" rx="2" ry="2" />')
        svg_parts.append(f'<text x="{legend_x + 16}" y="{y - 1}" class="legend-label">{labels[metric]}</text>')

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def write_report(
    output_path: Path,
    panel_path: Path,
    model_comparison: pd.DataFrame,
    dep_distribution: pd.DataFrame,
    indicator_summary: pd.DataFrame,
    correlation: pd.DataFrame,
    full_vif: pd.DataFrame,
    reduced_vif: pd.DataFrame,
    winsor_cutoff: float,
    scarcity_removed: int,
    total_rows: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sign_flip_focus = model_comparison.loc[model_comparison["term"] == "equip_cong_any_1", [
        "model",
        "coef_with_se",
        "pvalue",
        "nobs",
        "rsquared",
    ]].copy()
    sign_flip_focus.rename(
        columns={
            "model": "Model",
            "coef_with_se": "Equipment congestion island 1",
            "pvalue": "p-value",
            "nobs": "Observations",
            "rsquared": "R-squared",
        },
        inplace=True,
    )
    fe_summary = pd.DataFrame(
        {
            "Model": sign_flip_focus["Model"],
            "Unit FE": ["Pair"] * len(sign_flip_focus),
            "Calendar FE": ["Day"] * len(sign_flip_focus),
        }
    )

    congestion_table = model_comparison.pivot(index="term", columns="model", values="coef_with_se").reset_index()
    congestion_table.rename(columns={"term": "Congestion variable"}, inplace=True)

    dep_distribution_display = dep_distribution.copy()
    dep_distribution_display["share_of_sample"] = dep_distribution_display["share_of_sample"].map(lambda x: f"{100 * x:.2f}%")

    indicator_plot_html = "".join(
        [
            make_indicator_plot_svg(indicator_summary, "link_congested_any", "Gap by Link Congestion"),
            make_indicator_plot_svg(indicator_summary, "equip_cong_any_1", "Gap by Island 1 Equipment Congestion"),
            make_indicator_plot_svg(indicator_summary, "equip_cong_any_2", "Gap by Island 2 Equipment Congestion"),
        ]
    )

    correlation_display = correlation.reset_index().rename(columns={"index": "variable"})

    css = """
body { font-family: Georgia, "Times New Roman", serif; margin: 32px auto; max-width: 1420px; color: #102a43; line-height: 1.55; background: #f4f1ea; }
h1, h2, h3 { color: #0b1f33; }
h1 { margin-bottom: 10px; }
h2 { margin: 28px 0 12px; }
h3 { margin: 20px 0 10px; }
p { margin: 0 0 14px; }
code { background: #dde7f0; color: #0b1f33; padding: 2px 5px; border-radius: 4px; }
.lead { font-size: 17px; color: #243b53; margin-bottom: 24px; }
.card { background: #faf7f1; border: 1px solid #d9e2ec; border-radius: 14px; padding: 22px 24px; margin: 22px 0 28px; box-shadow: 0 10px 28px rgba(16, 42, 67, 0.08); }
.meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 20px; margin: 0 0 16px; }
.meta-grid div { background: #eef3f7; border-radius: 10px; padding: 10px 12px; }
.meta-label { display: block; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #486581; margin-bottom: 4px; }
.report-table { border-collapse: collapse; width: 100%; margin: 16px 0 10px; font-size: 14px; box-shadow: 0 8px 24px rgba(16, 42, 67, 0.10); }
.report-table th { background: #0b1f33; color: #fdfdfd; padding: 11px 12px; text-align: center; border: 1px solid #102a43; }
.report-table td { border: 1px solid #bcccdc; padding: 9px 12px; vertical-align: top; background: #fffdf8; color: #102a43; }
.report-table td:first-child { font-weight: 600; background: #e6ecf2; color: #0b1f33; }
.report-table tr:nth-child(even) td:not(:first-child) { background: #eef3f7; }
.plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; margin-top: 14px; }
.mini-chart { width: 100%; height: auto; background: #fffdf8; border: 1px solid #d9e2ec; border-radius: 12px; }
.chart-title { font-size: 14px; font-weight: 700; fill: #102a43; }
.axis-line { stroke: #7b8794; stroke-width: 1; }
.tick-label { font-size: 11px; fill: #486581; }
.bar-label { font-size: 11px; fill: #102a43; }
.legend-label { font-size: 11px; fill: #102a43; }
.notes { color: #243b53; font-size: 14px; background: #e6ecf2; border-radius: 12px; padding: 16px 18px; margin-top: 28px; }
"""

    html_body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Direct Pair Sign-Flip Diagnostics</title>
  <style>{css}</style>
</head>
<body>
  <h1>Direct Pair Sign-Flip Diagnostics</h1>
  <p class="lead">This report checks why the coefficient on island-1 equipment congestion can flip sign between the non-elasticity and elasticity-style direct-pair regressions.</p>

  <section class="card">
    <div class="meta-grid">
      <div><span class="meta-label">Source panel</span><code>{html.escape(str(panel_path))}</code></div>
      <div><span class="meta-label">Baseline sample size</span><span>{html.escape(format_number(total_rows, 0))} rows</span></div>
      <div><span class="meta-label">Winsor cutoff</span><span>Top 1% gap cutoff at <code>{html.escape(format_number(winsor_cutoff, 4))}</code></span></div>
      <div><span class="meta-label">Scarcity restriction</span><span>Remove rows where either island price is at least <code>{html.escape(format_number(SCARCITY_PRICE_CAP, 0))}</code> or at most <code>{html.escape(format_number(SCARCITY_PRICE_FLOOR, 0))}</code>; removed <code>{html.escape(format_number(scarcity_removed, 0))}</code> rows</span></div>
    </div>
    <p>The baseline level and elasticity-style models match the direct-pair regression structure in the main report: pair fixed effects, day fixed effects, binary link and equipment congestion indicators, and RTDREG controls for each island and the system total.</p>
  </section>

  <section class="card">
    <h2>Coefficient Path</h2>
    <p>This table focuses on the coefficient for <code>equip_cong_any_1</code>, the term whose sign flips between the level and elasticity-style versions.</p>
    {html_table(sign_flip_focus, float_digits=4)}
    <h3>Fixed Effects Used</h3>
    {html_table(fe_summary, float_digits=4)}
    <h3>All Congestion Coefficients</h3>
    {html_table(congestion_table, float_digits=4)}
  </section>

  <section class="card">
    <h2>Dependent Variable During Island-1 Congestion Episodes</h2>
    <p>The summary below compares the direct-pair price gap distribution when <code>equip_cong_any_1 = 1</code> versus <code>0</code>.</p>
    {html_table(dep_distribution_display, float_digits=4)}
  </section>

  <section class="card">
    <h2>Mean And Median Gap By Congestion Indicator</h2>
    <p>Each plot compares the mean and median direct-pair gap when the indicator is off versus on.</p>
    <div class="plot-grid">{indicator_plot_html}</div>
    <h3>Underlying Values</h3>
    {html_table(indicator_summary, float_digits=4)}
  </section>

  <section class="card">
    <h2>Correlation And Variance Inflation</h2>
    <p>The correlation matrix and VIF tables check whether the sign flip is plausibly tied to multicollinearity among the congestion indicators and RTDREG totals.</p>
    <h3>Pairwise Correlation</h3>
    {html_table(correlation_display, float_digits=4)}
    <h3>Full VIF</h3>
    {html_table(full_vif, float_digits=4)}
    <h3>Reduced VIF Without <code>mkt_export_total</code></h3>
    <p><code>mkt_import_total</code> and <code>mkt_export_total</code> are identical in this panel, so the reduced VIF table drops one of them to show the remaining collinearity more clearly.</p>
    {html_table(reduced_vif, float_digits=4)}
  </section>

  <div class="notes">
    <p>The direct-pair panel contains a heavy upper tail in price gaps. The diagnostics above separate three possibilities: tail sensitivity, scarcity episodes, and multicollinearity among congestion and RTDREG totals.</p>
    <p>Significance stars follow the main report convention: <code>* p&lt;0.10</code>, <code>** p&lt;0.05</code>, <code>*** p&lt;0.01</code>.</p>
  </div>
</body>
</html>
"""
    output_path.write_text(html_body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    panel_path = Path(args.direct_pair_panel) if args.direct_pair_panel else latest_matching_file(
        Path("data/panels"),
        "RTD_DIRECT_PAIR_PANEL_*.parquet",
    )
    output_html = Path(args.output_html)

    frame = pd.read_parquet(panel_path).copy()
    frame["time_interval"] = pd.to_datetime(frame["time_interval"])
    frame = add_log1p_columns(
        frame,
        ["dep_abs_price_gap", *[f"{control}_{side}" for control in CONTROL_COLUMNS for side in ("1", "2", "total")]],
    )

    model_comparison, winsor_cutoff, scarcity_removed = build_model_comparison(frame)
    dep_distribution = describe_gap_by_island1_congestion(frame)
    indicator_summary = build_indicator_summary(frame)
    correlation, full_vif, reduced_vif = build_correlation_and_vif(frame)

    write_report(
        output_html,
        panel_path,
        model_comparison,
        dep_distribution,
        indicator_summary,
        correlation,
        full_vif,
        reduced_vif,
        winsor_cutoff,
        scarcity_removed,
        len(frame),
    )

    print(f"Wrote {output_html}")


if __name__ == "__main__":
    main()
