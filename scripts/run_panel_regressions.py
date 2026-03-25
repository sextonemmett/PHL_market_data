#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

DEFAULT_DIRECT_PAIR_PANEL = Path("data/panels/RTD_DIRECT_PAIR_PANEL_202512212305_202603190000.parquet")
DEFAULT_ISLAND_SYSTEM_PANEL = Path("data/panels/RTD_ISLAND_SYSTEM_PANEL_202512212305_202603210000.parquet")
DEFAULT_ISLAND_CONGESTION_PANEL = Path("data/panels/RTD_ISLAND_CONGESTION_PANEL_202512212305_202603210000.parquet")
DEFAULT_OUTPUT_ROOT = Path("regressions")

FE_OPTIONS = {
    "No FE": "",
    "Month FE": " + C(fe_month)",
    "Week FE": " + C(fe_week)",
    "Day FE": " + C(fe_day)",
}

DIRECT_PAIR_DEP = "dep_abs_price_gap"
ISLAND_SYSTEM_DEP = "dep_price_minus_sys"
ISLAND_PRICE_DEP = "price_island_1"

TERM_LABELS = {
    "Intercept": "Intercept",
    "link_congested_any": "Inter-island link congestion indicator (1 = congested)",
    "equip_cong_any_1": "Equipment congestion indicator, island 1 (1 = any congested resource)",
    "equip_cong_any_2": "Equipment congestion indicator, island 2 (1 = any congested resource)",
    "equip_excess_pct_sum_1": "Sum of equipment overload margins, island 1 (sum of PCT_MW - 100, pct points)",
    "equip_excess_pct_sum_2": "Sum of equipment overload margins, island 2 (sum of PCT_MW - 100, pct points)",
    "demand_1": "Island 1 demand (MWh requirement)",
    "demand_2": "Island 2 demand (MWh requirement)",
    "demand_total": "Total system demand across three islands (MWh requirement)",
    "equip_cong_any": "Equipment congestion indicator (1 = any congested resource in island)",
    "equip_excess_pct_sum": "Sum of equipment overload margins (sum of PCT_MW - 100, pct points)",
    "demand_island": "Island demand (MWh requirement)",
    "equipment_cong_bin_cluz": "Luzon equipment congestion indicator (1 = any congested resource)",
    "equipment_cong_bin_cvis": "Visayas equipment congestion indicator (1 = any congested resource)",
    "equipment_cong_bin_cmin": "Mindanao equipment congestion indicator (1 = any congested resource)",
    "equipment_cong_pct_cluz": "Luzon overload margin sum (sum of PCT_MW - 100, pct points)",
    "equipment_cong_pct_cvis": "Visayas overload margin sum (sum of PCT_MW - 100, pct points)",
    "equipment_cong_pct_cmin": "Mindanao overload margin sum (sum of PCT_MW - 100, pct points)",
    "demand_island_1": "Focal island demand (MWh requirement)",
}

SPECIFICATIONS = {
    "direct_pair_binary": {
        "title": "Specification 1: Direct Pair Panel, Binary Equipment Congestion",
        "panel_path_key": "direct",
        "dep_var": DIRECT_PAIR_DEP,
        "dependent_label_html": "|P<sub>i,t</sub> - P<sub>j,t</sub>|",
        "dependent_row_html": "|P<sub>i,t</sub> - P<sub>j,t</sub>| (PHP/MWh)",
        "dependent_description": "Absolute price gap between the two directly connected islands at 5-minute interval t (PHP/MWh).",
        "unit_of_observation": "Island-pair by 5-minute interval",
        "rhs_terms": [
            "link_congested_any",
            "equip_cong_any_1",
            "equip_cong_any_2",
            "demand_1",
            "demand_2",
            "demand_total",
        ],
        "always_terms": ["C(pair_key)"],
        "spec_description": "Uses binary congestion indicators for island-specific equipment congestion together with a binary inter-island link congestion indicator. Pair fixed effects are included in every column.",
        "formula_html": (
            "<em>|P<sub>i,t</sub> - P<sub>j,t</sub>|</em> = "
            "&beta;<sub>1</sub> Link congestion"
            " + &beta;<sub>2</sub> Equipment congestion (island 1)"
            " + &beta;<sub>3</sub> Equipment congestion (island 2)"
            " + &gamma;<sub>1</sub> Demand (island 1)"
            " + &gamma;<sub>2</sub> Demand (island 2)"
            " + &gamma;<sub>3</sub> Total demand"
            " + pair fixed effects + calendar fixed effects + &epsilon;<sub>i,j,t</sub>"
        ),
    },
    "direct_pair_pct": {
        "title": "Specification 2: Direct Pair Panel, Percent Equipment Congestion",
        "panel_path_key": "direct",
        "dep_var": DIRECT_PAIR_DEP,
        "dependent_label_html": "|P<sub>i,t</sub> - P<sub>j,t</sub>|",
        "dependent_row_html": "|P<sub>i,t</sub> - P<sub>j,t</sub>| (PHP/MWh)",
        "dependent_description": "Absolute price gap between the two directly connected islands at 5-minute interval t (PHP/MWh).",
        "unit_of_observation": "Island-pair by 5-minute interval",
        "rhs_terms": [
            "link_congested_any",
            "equip_excess_pct_sum_1",
            "equip_excess_pct_sum_2",
            "demand_1",
            "demand_2",
            "demand_total",
        ],
        "always_terms": ["C(pair_key)"],
        "spec_description": "Replaces binary equipment congestion indicators with the summed overload margin on each side of the pair, measured in percentage points above 100% loading. Pair fixed effects are included in every column.",
        "formula_html": (
            "<em>|P<sub>i,t</sub> - P<sub>j,t</sub>|</em> = "
            "&beta;<sub>1</sub> Link congestion"
            " + &beta;<sub>2</sub> Overload margin sum (island 1)"
            " + &beta;<sub>3</sub> Overload margin sum (island 2)"
            " + &gamma;<sub>1</sub> Demand (island 1)"
            " + &gamma;<sub>2</sub> Demand (island 2)"
            " + &gamma;<sub>3</sub> Total demand"
            " + pair fixed effects + calendar fixed effects + &epsilon;<sub>i,j,t</sub>"
        ),
    },
    "island_system_binary": {
        "title": "Specification 3: Island-System Panel, Binary Equipment Congestion",
        "panel_path_key": "island",
        "dep_var": ISLAND_SYSTEM_DEP,
        "dependent_label_html": "|P<sub>i,t</sub> - P<sub>sys,t</sub>|",
        "dependent_row_html": "|P<sub>i,t</sub> - P<sub>sys,t</sub>| (PHP/MWh)",
        "dependent_description": "Absolute deviation between an island price and the demand-weighted system price at 5-minute interval t (PHP/MWh).",
        "unit_of_observation": "Island by 5-minute interval",
        "rhs_terms": [
            "equip_cong_any",
            "demand_island",
            "demand_total",
        ],
        "always_terms": ["C(island_code)"],
        "spec_description": "Uses a binary indicator for whether any mapped congested equipment appears in the island during the interval. Island fixed effects are included in every column.",
        "formula_html": (
            "<em>|P<sub>i,t</sub> - P<sub>sys,t</sub>|</em> = "
            "&beta;<sub>1</sub> Equipment congestion"
            " + &gamma;<sub>1</sub> Island demand"
            " + &gamma;<sub>2</sub> Total demand"
            " + island fixed effects + calendar fixed effects + &epsilon;<sub>i,t</sub>"
        ),
    },
    "island_system_pct": {
        "title": "Specification 4: Island-System Panel, Percent Equipment Congestion",
        "panel_path_key": "island",
        "dep_var": ISLAND_SYSTEM_DEP,
        "dependent_label_html": "|P<sub>i,t</sub> - P<sub>sys,t</sub>|",
        "dependent_row_html": "|P<sub>i,t</sub> - P<sub>sys,t</sub>| (PHP/MWh)",
        "dependent_description": "Absolute deviation between an island price and the demand-weighted system price at 5-minute interval t (PHP/MWh).",
        "unit_of_observation": "Island by 5-minute interval",
        "rhs_terms": [
            "equip_excess_pct_sum",
            "demand_island",
            "demand_total",
        ],
        "always_terms": ["C(island_code)"],
        "spec_description": "Uses the summed equipment overload margin in the island, measured as the interval sum of PCT_MW - 100 across mapped congested resources. Island fixed effects are included in every column.",
        "formula_html": (
            "<em>|P<sub>i,t</sub> - P<sub>sys,t</sub>|</em> = "
            "&beta;<sub>1</sub> Overload margin sum"
            " + &gamma;<sub>1</sub> Island demand"
            " + &gamma;<sub>2</sub> Total demand"
            " + island fixed effects + calendar fixed effects + &epsilon;<sub>i,t</sub>"
        ),
    },
    "island_price_pct": {
        "title": "Specification 5: Focal-Island Price Panel, Percent Equipment Congestion",
        "panel_path_key": "island_congestion",
        "dep_var": ISLAND_PRICE_DEP,
        "dependent_label_html": "P<sub>i,t</sub>",
        "dependent_row_html": "P<sub>i,t</sub> (PHP/MWh)",
        "dependent_description": "Price in the focal island at 5-minute interval t (PHP/MWh).",
        "unit_of_observation": "Focal island by 5-minute interval",
        "rhs_terms": [
            "equipment_cong_pct_cluz",
            "equipment_cong_pct_cvis",
            "equipment_cong_pct_cmin",
            "demand_island_1",
        ],
        "always_terms": ["C(island_1)"],
        "spec_description": "Relates focal-island price to overload margin sums in Luzon, Visayas, and Mindanao, plus focal-island demand. Island fixed effects are included in every column.",
        "formula_html": (
            "<em>P<sub>i,t</sub></em> = "
            "&beta;<sub>1</sub> Luzon overload margin sum"
            " + &beta;<sub>2</sub> Visayas overload margin sum"
            " + &beta;<sub>3</sub> Mindanao overload margin sum"
            " + &gamma;<sub>1</sub> Focal-island demand"
            " + island fixed effects + calendar fixed effects + &epsilon;<sub>i,t</sub>"
        ),
    },
    "island_price_pct_by_island": {
        "title": "Specification 6: Focal-Island Price Panel, Separate by Focal Island",
        "panel_path_key": "island_congestion",
        "dep_var": ISLAND_PRICE_DEP,
        "dependent_label_html": "P<sub>i,t</sub>",
        "dependent_row_html": "P<sub>i,t</sub> (PHP/MWh)",
        "dependent_description": "Price in the focal island at 5-minute interval t (PHP/MWh).",
        "unit_of_observation": "Focal island by 5-minute interval",
        "rhs_terms": [
            "equipment_cong_pct_cluz",
            "equipment_cong_pct_cvis",
            "equipment_cong_pct_cmin",
            "demand_island_1",
        ],
        "calendar_fe_suffix": " + C(fe_day)",
        "spec_description": "Runs the same percent-congestion specification separately for Luzon, Visayas, and Mindanao focal-island rows, using day fixed effects within each island-specific sample.",
        "formula_html": (
            "<em>P<sub>i,t</sub></em> = "
            "&beta;<sub>1</sub> Luzon overload margin sum"
            " + &beta;<sub>2</sub> Visayas overload margin sum"
            " + &beta;<sub>3</sub> Mindanao overload margin sum"
            " + &gamma;<sub>1</sub> Focal-island demand"
            " + day fixed effects + &epsilon;<sub>i,t</sub>"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline OLS regressions on the RTD pair and island-system panels."
    )
    parser.add_argument(
        "--direct-pair-panel",
        default=str(DEFAULT_DIRECT_PAIR_PANEL),
        help="Direct-pair panel parquet path.",
    )
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
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory for regression tables and coefficient exports.",
    )
    return parser.parse_args()


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


def load_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["time_interval"] = pd.to_datetime(frame["time_interval"])
    return frame


def format_estimate_cell(result: object, term: str) -> str:
    coef = float(result.params[term])
    se = float(result.bse[term])
    pvalue = float(result.pvalues[term]) if term in result.pvalues.index else float("nan")
    if coef == 0.0 and se == 0.0 and pd.isna(pvalue):
        return "No overloads in sample"
    return f"{format_number(coef)}{significance_stars(pvalue)}<br>({format_number(se)})"


def fit_models(
    frame: pd.DataFrame,
    dep_var: str,
    rhs_terms: list[str],
    always_terms: list[str] | None = None,
) -> dict[str, object]:
    rhs_parts = [*rhs_terms, *(always_terms or [])]
    rhs = " + ".join(rhs_parts)
    models: dict[str, object] = {}
    for label, fe_suffix in FE_OPTIONS.items():
        formula = f"{dep_var} ~ {rhs}{fe_suffix}"
        models[label] = smf.ols(formula=formula, data=frame).fit(cov_type="HC1")
    return models


def fit_models_by_island(
    frame: pd.DataFrame,
    dep_var: str,
    rhs_terms: list[str],
    calendar_fe_suffix: str,
) -> dict[str, object]:
    models: dict[str, object] = {}
    rhs = " + ".join(rhs_terms)
    for island_code, island_label in [("CLUZ", "Luzon"), ("CVIS", "Visayas"), ("CMIN", "Mindanao")]:
        subset = frame.loc[frame["island_1"] == island_code].copy()
        formula = f"{dep_var} ~ {rhs}{calendar_fe_suffix}"
        models[island_label] = smf.ols(formula=formula, data=subset).fit(cov_type="HC1")
    return models


def tidy_results(spec_name: str, dep_var: str, rhs_terms: list[str], models: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fe_label, result in models.items():
        for term in ["Intercept", *rhs_terms]:
            rows.append(
                {
                    "specification": spec_name,
                    "fe_option": fe_label,
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


def build_display_table(rhs_terms: list[str], models: dict[str, object], dependent_row_html: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for term in rhs_terms:
        row = {"Variable": TERM_LABELS.get(term, term)}
        for fe_label, result in models.items():
            row[fe_label] = format_estimate_cell(result, term)
        rows.append(row)

    entity_fe_label = "No"
    first_model = next(iter(models.values()))
    exog_names = getattr(first_model.model, "exog_names", [])
    if any(name.startswith("C(pair_key)") for name in exog_names):
        entity_fe_label = "Pair FE"
    elif any(name.startswith("C(island_code)") or name.startswith("C(island_1)") for name in exog_names):
        entity_fe_label = "Island FE"

    stats_rows = [
        ("Observations", lambda result, fe_label: f"{int(result.nobs):,}"),
        ("R-squared", lambda result, fe_label: format_number(float(result.rsquared), digits=3)),
        ("Dependent variable", lambda result, fe_label: dependent_row_html),
        ("Entity FE", lambda result, fe_label: entity_fe_label),
        ("Calendar FE", lambda result, fe_label: fe_label),
        ("Robust SE", lambda result, fe_label: "HC1"),
    ]
    for label, formatter in stats_rows:
        row = {"Variable": label}
        for fe_label, result in models.items():
            row[fe_label] = formatter(result, fe_label)
        rows.append(row)
    return pd.DataFrame(rows)


def build_by_island_display_table(
    rhs_terms: list[str],
    models: dict[str, object],
    dependent_row_html: str,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for term in rhs_terms:
        row = {"Variable": TERM_LABELS.get(term, term)}
        for island_label, result in models.items():
            row[island_label] = format_estimate_cell(result, term)
        rows.append(row)

    stats_rows = [
        ("Observations", lambda result, island_label: f"{int(result.nobs):,}"),
        ("R-squared", lambda result, island_label: format_number(float(result.rsquared), digits=3)),
        ("Dependent variable", lambda result, island_label: dependent_row_html),
        ("Entity FE", lambda result, island_label: "No"),
        ("Calendar FE", lambda result, island_label: "Day FE"),
        ("Sample", lambda result, island_label: island_label),
        ("Robust SE", lambda result, island_label: "HC1"),
    ]
    for label, formatter in stats_rows:
        row = {"Variable": label}
        for island_label, result in models.items():
            row[island_label] = formatter(result, island_label)
        rows.append(row)
    return pd.DataFrame(rows)


def dataframe_to_html_table(frame: pd.DataFrame) -> str:
    table = frame.copy()
    return table.to_html(index=False, escape=False, classes=["reg-table"])


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
    island_congestion_panel_path: Path,
    spec_tables: dict[str, pd.DataFrame],
    tidy_results_by_spec: dict[str, pd.DataFrame],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    coeff_path = output_root / "panel_regression_coefficients.csv"
    pd.concat(list(tidy_results_by_spec.values()), ignore_index=True).to_csv(coeff_path, index=False)

    html_path = output_root / "panel_regression_tables.html"
    markdown_path = output_root / "panel_regression_tables.md"
    if markdown_path.exists():
        markdown_path.unlink()

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
.reg-table td:first-child { font-weight: 600; width: 340px; background: #e6ecf2; color: #0b1f33; }
.reg-table tr:nth-child(even) td:not(:first-child) { background: #eef3f7; }
.notes { color: #243b53; font-size: 14px; background: #e6ecf2; border-radius: 12px; padding: 16px 18px; margin-top: 28px; }
"""
    spec_sections = [
        render_spec_section(SPECIFICATIONS["direct_pair_binary"], direct_panel_path, spec_tables["direct_pair_binary"]),
        render_spec_section(SPECIFICATIONS["direct_pair_pct"], direct_panel_path, spec_tables["direct_pair_pct"]),
        render_spec_section(SPECIFICATIONS["island_system_binary"], island_panel_path, spec_tables["island_system_binary"]),
        render_spec_section(SPECIFICATIONS["island_system_pct"], island_panel_path, spec_tables["island_system_pct"]),
        render_spec_section(
            SPECIFICATIONS["island_price_pct"],
            island_congestion_panel_path,
            spec_tables["island_price_pct"],
        ),
        render_spec_section(
            SPECIFICATIONS["island_price_pct_by_island"],
            island_congestion_panel_path,
            spec_tables["island_price_pct_by_island"],
        ),
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
  <p class="lead">These regressions use <code>statsmodels</code> OLS with <code>HC1</code> robust standard errors and no 5-minute slot fixed effects. Specifications 1 through 5 report the same congestion design under four calendar fixed-effect options: none, month, week, and day. Specification 6 instead reports separate day-FE regressions for Luzon, Visayas, and Mindanao focal-island samples.</p>

  {''.join(spec_sections)}

  <div class="notes">
    <p>FE variants shown are <code>No FE</code>, <code>Month FE</code>, <code>Week FE</code>, and <code>Day FE</code> for Specifications 1 through 5. Specification 6 is estimated separately by focal island with <code>Day FE</code> in each island-specific sample.</p>
    <p>Significance stars: <code>* p&lt;0.10</code>, <code>** p&lt;0.05</code>, <code>*** p&lt;0.01</code>.</p>
    <p>Full tidy coefficient export: <code>{html.escape(str(coeff_path))}</code></p>
  </div>
</body>
</html>
"""
    html_path.write_text(html_body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    direct_panel_path = Path(args.direct_pair_panel)
    island_panel_path = Path(args.island_system_panel)
    island_congestion_panel_path = Path(args.island_congestion_panel)
    output_root = Path(args.output_root)

    direct_frame = load_panel(direct_panel_path)
    island_frame = load_panel(island_panel_path)
    island_congestion_frame = load_panel(island_congestion_panel_path)

    frames = {"direct": direct_frame, "island": island_frame, "island_congestion": island_congestion_frame}
    spec_tables: dict[str, pd.DataFrame] = {}
    tidy_results_by_spec: dict[str, pd.DataFrame] = {}
    for spec_key, spec in SPECIFICATIONS.items():
        frame = frames[spec["panel_path_key"]]
        if spec_key == "island_price_pct_by_island":
            models = fit_models_by_island(frame, spec["dep_var"], spec["rhs_terms"], spec["calendar_fe_suffix"])
            tidy_results_by_spec[spec_key] = tidy_results(spec_key, spec["dep_var"], spec["rhs_terms"], models)
            spec_tables[spec_key] = build_by_island_display_table(
                spec["rhs_terms"],
                models,
                spec["dependent_row_html"],
            )
        else:
            models = fit_models(frame, spec["dep_var"], spec["rhs_terms"], spec.get("always_terms"))
            tidy_results_by_spec[spec_key] = tidy_results(spec_key, spec["dep_var"], spec["rhs_terms"], models)
            spec_tables[spec_key] = build_display_table(spec["rhs_terms"], models, spec["dependent_row_html"])

    write_report(
        output_root,
        direct_panel_path,
        island_panel_path,
        island_congestion_panel_path,
        spec_tables,
        tidy_results_by_spec,
    )

    print(f"Wrote {output_root / 'panel_regression_tables.html'}")
    print(f"Wrote {output_root / 'panel_regression_coefficients.csv'}")


if __name__ == "__main__":
    main()
