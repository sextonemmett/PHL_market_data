#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

CONTROL_COLUMNS = ("losses", "generation", "mkt_import", "mkt_export")
DEFAULT_OUTPUT_ROOT = Path("regressions")

TERM_LABELS = {
    "Intercept": "Intercept",
    "link_congested_any": "Inter-island link congestion indicator",
    "equip_cong_any_1": "Equipment congestion indicator, island 1",
    "equip_cong_any_2": "Equipment congestion indicator, island 2",
    "equip_cong_any": "Equipment congestion indicator",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cleaned day-FE binary log1p regressions on the retained RTD panels."
    )
    parser.add_argument("--direct-pair-panel", help="Direct-pair panel parquet path.")
    parser.add_argument("--island-system-panel", help="Island-system panel parquet path.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory for regression tables and coefficient exports.",
    )
    return parser.parse_args()


TIMESTAMP_TOKEN_RE = re.compile(r"(\d{8,12})")


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
    return frame


def add_log1p_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if (result[column] < 0).any():
            raise ValueError(f"Column {column} contains negative values; cannot apply log1p.")
        result[f"log1p_{column}"] = np.log1p(result[column].astype(float))
    return result


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


def format_estimate_cell(result: object, term: str) -> str:
    coef = float(result.params[term])
    se = float(result.bse[term])
    pvalue = float(result.pvalues[term]) if term in result.pvalues.index else float("nan")
    return f"{format_number(coef)}{significance_stars(pvalue)}<br>({format_number(se)})"


def tidy_results(spec_name: str, dep_var: str, rhs_terms: list[str], result: object) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for term in ["Intercept", *rhs_terms]:
        rows.append(
            {
                "specification": spec_name,
                "dependent_variable": dep_var,
                "term": term,
                "coef": float(result.params[term]),
                "std_err": float(result.bse[term]),
                "pvalue": float(result.pvalues[term]),
                "nobs": int(result.nobs),
                "rsquared": float(result.rsquared),
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_html_table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, escape=False, classes=["reg-table"])


def build_display_table(rhs_terms: list[str], result: object, dependent_row_html: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for term in rhs_terms:
        rows.append(
            {
                "Variable": prettify_log_term(term),
                "Day FE": format_estimate_cell(result, term),
            }
        )

    stats_rows = [
        ("Observations", f"{int(result.nobs):,}"),
        ("R-squared", format_number(float(result.rsquared), digits=3)),
        ("Dependent variable", dependent_row_html),
        ("Robust SE", "HC1"),
    ]
    for label, value in stats_rows:
        rows.append({"Variable": label, "Day FE": value})
    return pd.DataFrame(rows)


def render_spec_section(spec: dict[str, str], panel_path: Path, table: pd.DataFrame) -> str:
    return f"""
  <section class="spec-card">
    <h2>{html.escape(spec["title"])}</h2>
    <div class="meta-grid">
      <div><span class="meta-label">Source panel</span><code>{html.escape(str(panel_path))}</code></div>
      <div><span class="meta-label">Unit of observation</span><span>{html.escape(spec["unit_of_observation"])}</span></div>
      <div><span class="meta-label">Dependent variable</span><span class="formula-inline"><em>{spec["dependent_label_html"]}</em></span></div>
      <div><span class="meta-label">Interpretation</span><span>{html.escape(spec["dependent_description"])}</span></div>
    </div>
    <p class="spec-description">{html.escape(spec["spec_description"])}</p>
    <div class="formula-box">
      <div class="formula-label">Regression specification</div>
      <div class="formula">{spec["formula_html"]}</div>
    </div>
    {dataframe_to_html_table(table)}
  </section>
"""


def write_report(
    output_root: Path,
    direct_panel_path: Path,
    island_panel_path: Path,
    spec_tables: dict[str, pd.DataFrame],
    tidy_results_by_spec: dict[str, pd.DataFrame],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    coeff_path = output_root / "panel_regression_coefficients.csv"
    pd.concat(list(tidy_results_by_spec.values()), ignore_index=True).to_csv(coeff_path, index=False)

    html_path = output_root / "panel_regression_tables.html"
    css = """
body { font-family: Georgia, "Times New Roman", serif; margin: 32px auto; max-width: 1280px; color: #102a43; line-height: 1.55; background: #f4f1ea; }
h1, h2 { color: #0b1f33; }
h1 { margin-bottom: 10px; }
h2 { margin: 0 0 14px; }
p { margin: 0 0 14px; }
code { background: #dde7f0; color: #0b1f33; padding: 2px 5px; border-radius: 4px; }
.lead { font-size: 17px; color: #243b53; margin-bottom: 24px; }
.spec-card { background: #faf7f1; border: 1px solid #d9e2ec; border-radius: 14px; padding: 22px 24px; margin: 26px 0 32px; box-shadow: 0 10px 28px rgba(16, 42, 67, 0.08); }
.meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 20px; margin: 0 0 16px; }
.meta-grid div { background: #eef3f7; border-radius: 10px; padding: 10px 12px; }
.meta-label { display: block; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #486581; margin-bottom: 4px; }
.formula-inline { color: #0b1f33; font-size: 18px; }
.spec-description { color: #243b53; margin-bottom: 14px; }
.formula-box { background: #102a43; color: #fdfdfd; border-radius: 12px; padding: 14px 16px; margin: 0 0 18px; }
.formula-label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.85; margin-bottom: 8px; }
.formula { font-size: 18px; line-height: 1.5; }
.reg-table { border-collapse: collapse; width: 100%; margin: 18px 0 10px; font-size: 14px; box-shadow: 0 8px 24px rgba(16, 42, 67, 0.10); }
.reg-table th { background: #0b1f33; color: #fdfdfd; padding: 11px 12px; text-align: center; border: 1px solid #102a43; }
.reg-table td { border: 1px solid #bcccdc; padding: 9px 12px; vertical-align: top; background: #fffdf8; color: #102a43; }
.reg-table td:first-child { font-weight: 600; width: 360px; background: #e6ecf2; color: #0b1f33; }
.reg-table tr:nth-child(even) td:not(:first-child) { background: #eef3f7; }
.notes { color: #243b53; font-size: 14px; background: #e6ecf2; border-radius: 12px; padding: 16px 18px; margin-top: 28px; }
"""
    spec_sections = [
        render_spec_section(SPECIFICATIONS["direct_pair_binary_log"], direct_panel_path, spec_tables["direct_pair_binary_log"]),
        render_spec_section(SPECIFICATIONS["island_system_binary_log"], island_panel_path, spec_tables["island_system_binary_log"]),
    ]
    html_body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Panel Regression Tables</title>
  <style>{css}</style>
</head>
<body>
  <h1>Panel Regression Tables</h1>
  <p class="lead">These regressions keep only the cleaned day-fixed-effect binary congestion specifications. Dependent variables and continuous controls use <code>log(1 + x)</code>, so continuous-control coefficients are elasticity-style estimates and congestion-indicator coefficients are semi-elasticities with day and entity fixed effects.</p>

  {''.join(spec_sections)}

  <div class="notes">
    <p>All specifications use <code>Day FE</code> plus entity fixed effects and <code>HC1</code> robust standard errors.</p>
    <p>Significance stars: <code>* p&lt;0.10</code>, <code>** p&lt;0.05</code>, <code>*** p&lt;0.01</code>.</p>
    <p>Full tidy coefficient export: <code>{html.escape(str(coeff_path))}</code></p>
  </div>
</body>
</html>
"""
    html_path.write_text(html_body, encoding="utf-8")


def prettify_log_term(term: str) -> str:
    if not term.startswith("log1p_"):
        return TERM_LABELS.get(term, term)
    base = term.removeprefix("log1p_")
    if base.endswith("_total"):
        return f"log(1 + {base.removesuffix('_total').upper()} total)"
    if base.endswith("_island"):
        return f"log(1 + {base.removesuffix('_island').upper()} island)"
    if base.endswith("_1"):
        return f"log(1 + {base.removesuffix('_1').upper()} island 1)"
    if base.endswith("_2"):
        return f"log(1 + {base.removesuffix('_2').upper()} island 2)"
    return f"log(1 + {base.upper()})"


SPECIFICATIONS = {
    "direct_pair_binary_log": {
        "title": "Specification 1: Direct Pair Panel, Day FE Binary Congestion",
        "panel_path_key": "direct",
        "dep_var": "log1p_dep_abs_price_gap",
        "dependent_label_html": "log(1 + |P<sub>i,t</sub> - P<sub>j,t</sub>|)",
        "dependent_row_html": "log(1 + |P<sub>i,t</sub> - P<sub>j,t</sub>|)",
        "dependent_description": "Log1p absolute price gap between the two directly connected islands at 5-minute interval t.",
        "unit_of_observation": "Island-pair by 5-minute interval",
        "rhs_terms": [
            "link_congested_any",
            "equip_cong_any_1",
            "equip_cong_any_2",
            *[f"log1p_{control}_{side}" for control in CONTROL_COLUMNS for side in ("1", "2", "total")],
        ],
        "always_terms": ["C(pair_key)", "C(fe_day)"],
        "spec_description": "Uses binary congestion indicators for the inter-island link and each side of the pair, plus log1p RTDREG controls for LOSSES, GENERATION, MKT_IMPORT, and MKT_EXPORT at island-1, island-2, and system-total levels.",
        "formula_html": (
            "<em>log(1 + |P<sub>i,t</sub> - P<sub>j,t</sub>|)</em> = "
            "&beta;<sub>1</sub> Link congestion + &beta;<sub>2</sub> Equipment congestion (island 1)"
            " + &beta;<sub>3</sub> Equipment congestion (island 2)"
            " + log(1 + controls)"
            " + pair fixed effects + day fixed effects + &epsilon;<sub>i,j,t</sub>"
        ),
    },
    "island_system_binary_log": {
        "title": "Specification 2: Island-System Panel, Day FE Binary Congestion",
        "panel_path_key": "island",
        "dep_var": "log1p_dep_price_minus_sys",
        "dependent_label_html": "log(1 + |P<sub>i,t</sub> - P<sub>sys,t</sub>|)",
        "dependent_row_html": "log(1 + |P<sub>i,t</sub> - P<sub>sys,t</sub>|)",
        "dependent_description": "Log1p absolute deviation between an island price and the demand-weighted system price at 5-minute interval t.",
        "unit_of_observation": "Island by 5-minute interval",
        "rhs_terms": [
            "equip_cong_any",
            *[f"log1p_{control}_{side}" for control in CONTROL_COLUMNS for side in ("island", "total")],
        ],
        "always_terms": ["C(island_code)", "C(fe_day)"],
        "spec_description": "Uses the binary island equipment-congestion indicator plus log1p RTDREG controls for LOSSES, GENERATION, MKT_IMPORT, and MKT_EXPORT at island and system-total levels.",
        "formula_html": (
            "<em>log(1 + |P<sub>i,t</sub> - P<sub>sys,t</sub>|)</em> = "
            "&beta;<sub>1</sub> Equipment congestion + log(1 + controls)"
            " + island fixed effects + day fixed effects + &epsilon;<sub>i,t</sub>"
        ),
    },
}


def fit_model(frame: pd.DataFrame, dep_var: str, rhs_terms: list[str], always_terms: list[str]) -> object:
    rhs = " + ".join([*rhs_terms, *always_terms])
    formula = f"{dep_var} ~ {rhs}"
    return smf.ols(formula=formula, data=frame).fit(cov_type="HC1")


def main() -> None:
    args = parse_args()
    direct_panel_path = Path(args.direct_pair_panel) if args.direct_pair_panel else latest_matching_file(
        Path("data/panels"),
        "RTD_DIRECT_PAIR_PANEL_*.parquet",
    )
    island_panel_path = Path(args.island_system_panel) if args.island_system_panel else latest_matching_file(
        Path("data/panels"),
        "RTD_ISLAND_SYSTEM_PANEL_*.parquet",
    )
    output_root = Path(args.output_root)

    direct_frame = load_panel(direct_panel_path)
    island_frame = load_panel(island_panel_path)
    direct_frame = add_log1p_columns(
        direct_frame,
        ["dep_abs_price_gap", *[f"{control}_{side}" for control in CONTROL_COLUMNS for side in ("1", "2", "total")]],
    )
    island_frame = add_log1p_columns(
        island_frame,
        ["dep_price_minus_sys", *[f"{control}_{side}" for control in CONTROL_COLUMNS for side in ("island", "total")]],
    )

    frames = {"direct": direct_frame, "island": island_frame}
    spec_tables: dict[str, pd.DataFrame] = {}
    tidy_results_by_spec: dict[str, pd.DataFrame] = {}

    for spec_key, spec in SPECIFICATIONS.items():
        frame = frames[spec["panel_path_key"]]
        result = fit_model(frame, spec["dep_var"], spec["rhs_terms"], spec["always_terms"])
        tidy_results_by_spec[spec_key] = tidy_results(spec_key, spec["dep_var"], spec["rhs_terms"], result)
        spec_tables[spec_key] = build_display_table(spec["rhs_terms"], result, spec["dependent_row_html"])

    write_report(
        output_root,
        direct_panel_path,
        island_panel_path,
        spec_tables,
        tidy_results_by_spec,
    )

    print(f"Wrote {output_root / 'panel_regression_tables.html'}")
    print(f"Wrote {output_root / 'panel_regression_coefficients.csv'}")


if __name__ == "__main__":
    main()
