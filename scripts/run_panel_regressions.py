#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

CONTROL_COLUMNS = ("losses", "generation", "mkt_import", "mkt_export")
DEFAULT_OUTPUT_ROOT = Path("regressions")
TIMESTAMP_TOKEN_RE = re.compile(r"(\d{8,12})")
ISLAND_LABELS = {"CLUZ": "Luzon", "CVIS": "Visayas", "CMIN": "Mindanao"}
ISLAND_COLUMN_TITLES = {
    "CLUZ": "Luzon vs System",
    "CVIS": "Visayas vs System",
    "CMIN": "Mindanao vs System",
}
PAIR_LABELS = {"CLUZ_CVIS": "Luzon-Visayas", "CVIS_CMIN": "Visayas-Mindanao"}
PAIR_COLUMN_TITLES = {"CLUZ_CVIS": "Luzon-Visayas", "CVIS_CMIN": "Visayas-Mindanao"}
PPML_CONTINUOUS_DELTA = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run retained RTD PPML regressions for the direct-pair and island-vs-system "
            "panels and write the formatted HTML tables plus a tidy coefficient export."
        )
    )
    parser.add_argument("--direct-pair-panel", help="Direct-pair panel parquet path.")
    parser.add_argument("--island-system-panel", help="Island-system panel parquet path.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory for regression tables and coefficient exports.",
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
    return frame


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


def is_binary_term(term: str) -> bool:
    return term in {
        "link_congested_any",
        "interlink_congested_any",
        "equip_cong_any_1",
        "equip_cong_any_2",
        "equip_cong_any",
        "equip_overload_any_1",
        "equip_overload_any_2",
        "equip_overload_any",
    }


def is_continuous_mw_term(term: str) -> bool:
    return term.startswith(("losses_", "generation_", "mkt_import_", "mkt_export_"))


def reporting_delta(term: str) -> float | None:
    if is_binary_term(term):
        return 1.0
    if is_continuous_mw_term(term):
        return PPML_CONTINUOUS_DELTA
    return None


def reporting_unit_label(term: str) -> str:
    if is_binary_term(term):
        return "0 to 1"
    if is_continuous_mw_term(term):
        return "+100 MW"
    return ""


def transformed_effect(result: object, term: str, delta: float) -> tuple[float, float]:
    coef = float(result.params[term])
    se = float(result.bse[term])
    transformed_coef = 100.0 * (np.exp(coef * delta) - 1.0)
    transformed_se = 100.0 * np.exp(coef * delta) * delta * se
    return transformed_coef, transformed_se


def format_percent_effect_cell(result: object, term: str) -> str:
    delta = reporting_delta(term)
    if delta is None:
        coef = float(result.params[term])
        se = float(result.bse[term])
        pvalue = float(result.pvalues[term]) if term in result.pvalues.index else float("nan")
        return f"{format_number(coef)}{significance_stars(pvalue)}<br>({format_number(se)})"

    pvalue = float(result.pvalues[term]) if term in result.pvalues.index else float("nan")
    transformed_coef, transformed_se = transformed_effect(result, term, delta)
    return (
        f"{format_number(transformed_coef, digits=2)}%{significance_stars(pvalue)}"
        f"<br>({format_number(transformed_se, digits=2)}%)"
    )


def fit_ppml(frame: pd.DataFrame, dep_var: str, rhs_terms: list[str], fixed_effect_terms: list[str]) -> object:
    rhs = " + ".join([*rhs_terms, *fixed_effect_terms])
    formula = f"{dep_var} ~ {rhs}"
    return smf.glm(
        formula=formula,
        data=frame,
        family=sm.families.Poisson(),
    ).fit(cov_type="HC1", maxiter=200)


def select_estimable_terms(frame: pd.DataFrame, rhs_terms: list[str]) -> list[str]:
    kept_terms: list[str] = []
    for term in rhs_terms:
        if term not in frame.columns:
            continue
        if frame[term].nunique(dropna=False) <= 1:
            continue
        if any(frame[term].equals(frame[kept_term]) for kept_term in kept_terms):
            continue
        kept_terms.append(term)
    return kept_terms


def fit_metric_cell(column: dict[str, object]) -> str:
    result = column["result"]
    llnull = getattr(result, "llnull", None)
    llf = getattr(result, "llf", None)
    if llnull is not None and llf is not None and llnull != 0:
        pseudo_r2 = 1.0 - (llf / llnull)
        return f"Pseudo R² {format_number(float(pseudo_r2), digits=3)}"
    return "Pseudo R² n/a"


def tidy_results(
    section_key: str,
    column_key: str,
    dependent_variable: str,
    rhs_terms: list[str],
    result: object,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for term in ["Intercept", *rhs_terms]:
        delta = reporting_delta(term)
        reported_effect_pct = np.nan
        reported_effect_se_pct = np.nan
        if delta is not None and term in result.params.index:
            reported_effect_pct, reported_effect_se_pct = transformed_effect(result, term, delta)

        rows.append(
            {
                "section": section_key,
                "column_key": column_key,
                "dependent_variable": dependent_variable,
                "term": term,
                "coef": float(result.params[term]),
                "std_err": float(result.bse[term]),
                "pvalue": float(result.pvalues[term]),
                "nobs": int(result.nobs),
                "pseudo_r2": (
                    float(1.0 - (result.llf / result.llnull))
                    if getattr(result, "llnull", None) not in {None, 0}
                    else np.nan
                ),
                "reported_change": reporting_unit_label(term),
                "reported_effect_pct": reported_effect_pct,
                "reported_effect_se_pct": reported_effect_se_pct,
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_html_table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, escape=False, classes=["reg-table"])


def prettify_variable_label(label: str) -> str:
    labels = {
        "link_congested_any": "Pair-Specific Link Congestion",
        "interlink_congested_any": "Any Connected Inter-Link Congestion",
        "equip_cong_any_1": "Equipment Congestion: Island 1",
        "equip_cong_any_2": "Equipment Congestion: Island 2",
        "equip_cong_any": "Equipment Congestion: Focal Island",
        "equip_overload_any_1": "Overload > 0: Island 1",
        "equip_overload_any_2": "Overload > 0: Island 2",
        "equip_overload_any": "Overload > 0: Focal Island",
        "losses_1": "Losses: Island 1",
        "losses_2": "Losses: Island 2",
        "generation_1": "Generation: Island 1",
        "generation_2": "Generation: Island 2",
        "mkt_import_1": "Market Imports: Island 1",
        "mkt_import_2": "Market Imports: Island 2",
        "mkt_export_1": "Market Exports: Island 1",
        "mkt_export_2": "Market Exports: Island 2",
        "losses_island": "Losses: Focal Island",
        "generation_island": "Generation: Focal Island",
        "mkt_import_island": "Market Imports: Focal Island",
        "mkt_export_island": "Market Exports: Focal Island",
    }
    pretty = labels.get(label, label)
    if is_continuous_mw_term(label):
        return f"{pretty} (+100 MW)"
    return pretty


def format_display_cell(column: dict[str, object], concept: dict[str, str]) -> str:
    term = concept["term"]
    result = column["result"]
    if term not in result.params.index:
        return ""
    return format_percent_effect_cell(result, term)


def build_multi_column_table(
    concepts: list[dict[str, str]],
    columns: list[dict[str, object]],
    extra_stats: list[tuple[str, object]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    headers = ["Variable", *[str(column["title"]) for column in columns]]

    for concept in concepts:
        row = {"Variable": prettify_variable_label(concept["label"])}
        for column in columns:
            row[str(column["title"])] = format_display_cell(column, concept)
        rows.append(row)

    stats_rows = [
        ("Estimator", lambda column: "PPML"),
        ("Subsample", lambda column: str(column["sample_label"])),
        ("Observations", lambda column: f"{int(column['result'].nobs):,}"),
        ("Fit metric", fit_metric_cell),
        ("Dependent variable", lambda column: str(column["dependent_label"])),
        *([] if extra_stats is None else extra_stats),
        ("Robust SE", lambda column: "HC1"),
    ]
    for label, value_fn in stats_rows:
        row = {"Variable": label}
        for column in columns:
            row[str(column["title"])] = value_fn(column)
        rows.append(row)

    return pd.DataFrame(rows, columns=headers)


def render_section(
    title: str,
    panel_path: Path,
    sample_label: str,
    unit_of_observation: str,
    description: str,
    transform_note_html: str,
    formula_html: str,
    table: pd.DataFrame,
) -> str:
    return f"""
  <section class="spec-card">
    <h2>{html.escape(title)}</h2>
    <div class="meta-grid">
      <div><span class="meta-label">Source panel</span><code>{html.escape(str(panel_path))}</code></div>
      <div><span class="meta-label">Sample</span><span>{html.escape(sample_label)}</span></div>
      <div><span class="meta-label">Unit of observation</span><span>{html.escape(unit_of_observation)}</span></div>
      <div><span class="meta-label">Column meaning</span><span>{transform_note_html}</span></div>
    </div>
    <p class="spec-description">{html.escape(description)}</p>
    <div class="formula-box">
      <div class="formula-label">Regression Formula</div>
      <div class="formula">{formula_html}</div>
    </div>
    {dataframe_to_html_table(table)}
  </section>
"""


def write_report(
    output_root: Path,
    direct_panel_path: Path,
    island_panel_path: Path,
    section_tables: list[str],
    tidy_frames: list[pd.DataFrame],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    coeff_path = output_root / "panel_regression_coefficients.csv"
    pd.concat(tidy_frames, ignore_index=True).to_csv(coeff_path, index=False)

    html_path = output_root / "panel_regression_tables.html"
    css = """
body { font-family: Georgia, "Times New Roman", serif; margin: 32px auto; max-width: 1440px; color: #102a43; line-height: 1.55; background: #f4f1ea; }
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
.spec-description { color: #243b53; margin-bottom: 14px; }
.formula-box { background: #102a43; color: #fdfdfd; border-radius: 12px; padding: 14px 16px; margin: 0 0 18px; }
.formula-label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.85; margin-bottom: 8px; }
.formula { font-size: 18px; line-height: 1.5; }
.reg-table { border-collapse: collapse; width: 100%; margin: 18px 0 10px; font-size: 14px; box-shadow: 0 8px 24px rgba(16, 42, 67, 0.10); }
.reg-table th { background: #0b1f33; color: #fdfdfd; padding: 11px 12px; text-align: center; border: 1px solid #102a43; }
.reg-table td { border: 1px solid #bcccdc; padding: 9px 12px; vertical-align: top; background: #fffdf8; color: #102a43; }
.reg-table td:first-child { font-weight: 600; width: 320px; background: #e6ecf2; color: #0b1f33; }
.reg-table tr:nth-child(even) td:not(:first-child) { background: #eef3f7; }
.notes { color: #243b53; font-size: 14px; background: #e6ecf2; border-radius: 12px; padding: 16px 18px; margin-top: 28px; }
"""
    html_body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Panel Regression Tables</title>
  <style>{css}</style>
</head>
<body>
  <h1>RTD PPML Regression Tables</h1>
  <p class="lead">All reported estimates come from Poisson pseudo-maximum-likelihood models. Binary rows are exact percent effects for a dummy moving from 0 to 1. RTDREG continuous rows are reported as exact percent effects for a <code>+100 MW</code> change, which is usually easier to interpret here than a per-1-MW coefficient.</p>

  {''.join(section_tables)}

  <div class="notes">
    <p>All specifications use day fixed effects and <code>HC1</code> robust standard errors.</p>
    <p>The pooled direct-pair column includes pair fixed effects; the pair-specific columns do not because each sample contains one pair.</p>
    <p>The pooled island-vs-system column includes island fixed effects; the island-specific columns do not because each sample contains one island.</p>
    <p>The system price is the demand-weighted average of island prices using <code>MKT_REQT</code> weights at each 5-minute interval.</p>
    <p>System-total RTDREG controls are excluded from all reported regressions.</p>
    <p>Significance stars: <code>* p&lt;0.10</code>, <code>** p&lt;0.05</code>, <code>*** p&lt;0.01</code>.</p>
    <p>Full tidy coefficient export: <code>{html.escape(str(coeff_path))}</code></p>
  </div>
</body>
</html>
"""
    html_path.write_text(html_body, encoding="utf-8")


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

    tidy_frames: list[pd.DataFrame] = []
    section_tables: list[str] = []

    direct_concepts = [
        {"label": "link_congested_any", "term": "link_congested_any"},
        {"label": "equip_cong_any_1", "term": "equip_cong_any_1"},
        {"label": "equip_overload_any_1", "term": "equip_overload_any_1"},
        {"label": "equip_cong_any_2", "term": "equip_cong_any_2"},
        {"label": "equip_overload_any_2", "term": "equip_overload_any_2"},
        *[
            {"label": f"{control}_{side}", "term": f"{control}_{side}"}
            for control in CONTROL_COLUMNS
            for side in ("1", "2")
        ],
    ]
    direct_rhs = [concept["term"] for concept in direct_concepts]

    direct_columns: list[dict[str, object]] = []
    direct_pooled_rhs = select_estimable_terms(direct_frame, direct_rhs)
    direct_pooled_result = fit_ppml(
        direct_frame,
        "dep_abs_price_gap",
        direct_pooled_rhs,
        ["C(pair_key)", "C(fe_day)"],
    )
    tidy_frames.append(
        tidy_results(
            "direct_pair",
            "pooled",
            "dep_abs_price_gap",
            direct_pooled_rhs,
            direct_pooled_result,
        )
    )
    direct_columns.append(
        {
            "title": "Pooled (Both Pairs)",
            "sample_label": "Both direct pairs pooled",
            "dependent_label": "|P_i,t - P_j,t|",
            "result": direct_pooled_result,
            "unit_fe": "Pair",
            "calendar_fe": "Day",
        }
    )
    for pair_key in ("CLUZ_CVIS", "CVIS_CMIN"):
        pair_frame = direct_frame.loc[direct_frame["pair_key"] == pair_key].copy()
        pair_rhs = select_estimable_terms(pair_frame, direct_rhs)
        pair_result = fit_ppml(
            pair_frame,
            "dep_abs_price_gap",
            pair_rhs,
            ["C(fe_day)"],
        )
        tidy_frames.append(
            tidy_results(
                "direct_pair",
                pair_key.lower(),
                "dep_abs_price_gap",
                pair_rhs,
                pair_result,
            )
        )
        direct_columns.append(
            {
                "title": PAIR_COLUMN_TITLES[pair_key],
                "sample_label": PAIR_LABELS[pair_key],
                "dependent_label": "|P_i,t - P_j,t|",
                "result": pair_result,
                "unit_fe": "No",
                "calendar_fe": "Day",
            }
        )

    direct_table = build_multi_column_table(
        direct_concepts,
        direct_columns,
        extra_stats=[
            ("Unit FE", lambda column: str(column["unit_fe"])),
            ("Calendar FE", lambda column: str(column["calendar_fe"])),
        ],
    )
    section_tables.append(
        render_section(
            "Specification 1: Direct Pair Price Gap",
            direct_panel_path,
            "Pooled and pair-specific direct-link samples",
            "Island-pair by 5-minute interval",
            (
                "Explains the direct-pair absolute RTD price gap using the pair-specific inter-island "
                "link congestion flag, equipment congestion and "
                "overload indicators on each side of the pair, and island-level RTDREG controls. "
                "System-total RTDREG controls are excluded."
            ),
            (
                "All columns are PPML exact percent effects. Dummy rows are 0-to-1 effects. "
                "Continuous RTDREG rows are exact percent effects for a <em>+100 MW</em> change."
            ),
            (
                "<em>E[g<sub>ij,t</sub> | X]</em> = exp("
                "&beta;<sub>1</sub> LinkCong<sub>ij,t</sub> + "
                "&beta;<sub>2</sub> EquipCong<sub>i,t</sub> + "
                "&beta;<sub>3</sub> Overload<sub>i,t</sub> + "
                "&beta;<sub>4</sub> EquipCong<sub>j,t</sub> + "
                "&beta;<sub>5</sub> Overload<sub>j,t</sub> + "
                "&Gamma;X<sub>ij,t</sub> + &alpha;<sub>ij</sub> + &delta;<sub>d</sub>), "
                "where <em>g<sub>ij,t</sub></em> = |P<sub>i,t</sub> - P<sub>j,t</sub>|."
            ),
            direct_table,
        )
    )

    island_concepts = [
        {"label": "interlink_congested_any", "term": "interlink_congested_any"},
        {"label": "equip_cong_any", "term": "equip_cong_any"},
        {"label": "equip_overload_any", "term": "equip_overload_any"},
        *[
            {"label": f"{control}_island", "term": f"{control}_island"}
            for control in CONTROL_COLUMNS
        ],
    ]
    island_rhs = [concept["term"] for concept in island_concepts]

    island_columns: list[dict[str, object]] = []
    island_pooled_rhs = select_estimable_terms(island_frame, island_rhs)
    island_pooled_result = fit_ppml(
        island_frame,
        "dep_price_minus_sys",
        island_pooled_rhs,
        ["C(island_code)", "C(fe_day)"],
    )
    tidy_frames.append(
        tidy_results(
            "island_vs_system",
            "pooled",
            "dep_price_minus_sys",
            island_pooled_rhs,
            island_pooled_result,
        )
    )
    island_columns.append(
        {
            "title": "Pooled (All Islands)",
            "sample_label": "All islands pooled",
            "dependent_label": "|P_i,t - P_sys,t|",
            "result": island_pooled_result,
            "unit_fe": "Island",
            "calendar_fe": "Day",
        }
    )
    for island_code in ("CLUZ", "CVIS", "CMIN"):
        focal_frame = island_frame.loc[island_frame["island_code"] == island_code].copy()
        focal_rhs = select_estimable_terms(focal_frame, island_rhs)
        focal_result = fit_ppml(
            focal_frame,
            "dep_price_minus_sys",
            focal_rhs,
            ["C(fe_day)"],
        )
        tidy_frames.append(
            tidy_results(
                "island_vs_system",
                island_code.lower(),
                "dep_price_minus_sys",
                focal_rhs,
                focal_result,
            )
        )
        island_columns.append(
            {
                "title": ISLAND_COLUMN_TITLES[island_code],
                "sample_label": ISLAND_LABELS[island_code],
                "dependent_label": "|P_i,t - P_sys,t|",
                "result": focal_result,
                "unit_fe": "No",
                "calendar_fe": "Day",
            }
        )

    island_table = build_multi_column_table(
        island_concepts,
        island_columns,
        extra_stats=[
            ("Unit FE", lambda column: str(column["unit_fe"])),
            ("Calendar FE", lambda column: str(column["calendar_fe"])),
        ],
    )
    section_tables.append(
        render_section(
            "Specification 2: Island Against System",
            island_panel_path,
            "Pooled and island-specific samples",
            "Island by 5-minute interval",
            (
                "Explains the absolute deviation between the focal island RTD price and the demand-weighted "
                "system RTD price using an island-specific any-connected-inter-link congestion flag, focal-island equipment congestion "
                "and overload indicators, and focal-island RTDREG controls. System-total RTDREG controls are excluded."
            ),
            (
                "All columns are PPML exact percent effects. Dummy rows are 0-to-1 effects. "
                "Continuous RTDREG rows are exact percent effects for a <em>+100 MW</em> change."
            ),
            (
                "<em>E[g<sub>i,t</sub> | X]</em> = exp("
                "&beta;<sub>1</sub> AnyConnectedInterLinkCong<sub>i,t</sub> + "
                "&beta;<sub>2</sub> EquipCong<sub>i,t</sub> + "
                "&beta;<sub>3</sub> Overload<sub>i,t</sub> + "
                "&Gamma;X<sub>i,t</sub> + &alpha;<sub>i</sub> + &delta;<sub>d</sub>), "
                "where <em>g<sub>i,t</sub></em> = |P<sub>i,t</sub> - P<sub>sys,t</sub>| and "
                "<em>P<sub>sys,t</sub></em> is the demand-weighted system price."
            ),
            island_table,
        )
    )

    write_report(
        output_root,
        direct_panel_path,
        island_panel_path,
        section_tables,
        tidy_frames,
    )

    print(f"Wrote {output_root / 'panel_regression_tables.html'}")
    print(f"Wrote {output_root / 'panel_regression_coefficients.csv'}")


if __name__ == "__main__":
    main()
