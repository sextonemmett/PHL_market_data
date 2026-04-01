#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DEFAULT_OUTPUT_HTML = Path("regressions/luzon_visayas_price_pooled_ols.html")
DEFAULT_OUTPUT_CSV = Path("regressions/luzon_visayas_price_pooled_ols_coefficients.csv")
TIMESTAMP_TOKEN_RE = re.compile(r"(\d{8,12})")
BASE_VARIABLE_SPECS = [
    {"key": "own_equip_congestion", "label": "Own island, equipment congestion", "reporting_delta": 1.0},
    {"key": "other_equip_congestion", "label": "Other island, equipment congestion", "reporting_delta": 1.0},
    {"key": "own_demand", "label": "Own island, demand (+100 MW)", "reporting_delta": 100.0},
    {"key": "other_demand", "label": "Other island, demand (+100 MW)", "reporting_delta": 100.0},
]
SPLIT_VARIABLE_SPECS = [
    {"key": "own_congested_overloaded", "label": "Own island, congested and overloaded", "reporting_delta": 1.0},
    {"key": "own_congested_no_overload", "label": "Own island, congested, no overload", "reporting_delta": 1.0},
    {"key": "other_congested_overloaded", "label": "Other island, congested and overloaded", "reporting_delta": 1.0},
    {"key": "other_congested_no_overload", "label": "Other island, congested, no overload", "reporting_delta": 1.0},
    {"key": "own_demand", "label": "Own island, demand (+100 MW)", "reporting_delta": 100.0},
    {"key": "other_demand", "label": "Other island, demand (+100 MW)", "reporting_delta": 100.0},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit pooled day-FE OLS models for Luzon and Visayas prices with interactions "
            "for the Luzon-Visayas congestion regime, then write an HTML table."
        )
    )
    parser.add_argument("--direct-pair-panel", help="Direct-pair panel parquet path.")
    parser.add_argument("--regional-parquet", help="RTDREG parquet path.")
    parser.add_argument(
        "--output-html",
        default=str(DEFAULT_OUTPUT_HTML),
        help="Output HTML path.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Output CSV path.",
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


def load_luzon_visayas_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["time_interval"] = pd.to_datetime(frame["time_interval"])
    frame = frame.loc[frame["pair_key"] == "CLUZ_CVIS"].copy()
    frame["luz_vis_link_congestion"] = frame["link_congested_any"].astype(int)
    return frame


def load_pair_demand(path: Path) -> pd.DataFrame:
    regional = pd.read_parquet(path).copy()
    regional["TIME_INTERVAL"] = pd.to_datetime(regional["TIME_INTERVAL"])
    regional = regional.loc[
        (regional["COMMODITY_TYPE"] == "En") & regional["REGION_NAME"].isin(["CLUZ", "CVIS"]),
        ["TIME_INTERVAL", "REGION_NAME", "MKT_REQT"],
    ].copy()
    duplicates = regional.duplicated(subset=["TIME_INTERVAL", "REGION_NAME"], keep=False)
    if duplicates.any():
        sample = regional.loc[duplicates, ["TIME_INTERVAL", "REGION_NAME"]].head(10).to_dict("records")
        raise ValueError(f"RTDREG demand inputs are not unique by interval and region: {sample}")

    demand = (
        regional.pivot(index="TIME_INTERVAL", columns="REGION_NAME", values="MKT_REQT")
        .rename_axis(index="time_interval", columns=None)
        .sort_index()
        .rename(columns={"CLUZ": "Luzon_demand", "CVIS": "Visayas_demand"})
    )
    return demand.reset_index()


def build_analysis_frame(panel_path: Path, regional_path: Path) -> pd.DataFrame:
    pair_frame = load_luzon_visayas_panel(panel_path)
    demand_frame = load_pair_demand(regional_path)
    return pair_frame.merge(demand_frame, on="time_interval", how="inner", validate="one_to_one")


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


def format_number(value: float, digits: int = 4) -> str:
    return f"{value:,.{digits}f}"


def pretty_column_title(dependent_variable: str, congested: bool) -> str:
    regime = "Link Congested" if congested else "Link Uncongested"
    return f"{dependent_variable} Price | {regime}"


def pooled_rhs(include_day_fe: bool, variable_keys: list[str]) -> str:
    interaction_block = "luz_vis_link_congestion * (" + " + ".join(variable_keys) + ")"
    return f"C(fe_day) + {interaction_block}" if include_day_fe else interaction_block


def prepare_model_frame(frame: pd.DataFrame, perspective: str, split_equipment_terms: bool) -> pd.DataFrame:
    model_frame = frame.copy()
    if perspective == "luzon":
        own_cong = "equip_cong_any_1"
        own_overload = "equip_overload_any_1"
        other_cong = "equip_cong_any_2"
        other_overload = "equip_overload_any_2"
        own_demand = "Luzon_demand"
        other_demand = "Visayas_demand"
    else:
        own_cong = "equip_cong_any_2"
        own_overload = "equip_overload_any_2"
        other_cong = "equip_cong_any_1"
        other_overload = "equip_overload_any_1"
        own_demand = "Visayas_demand"
        other_demand = "Luzon_demand"

    if split_equipment_terms:
        model_frame["own_congested_overloaded"] = (
            (model_frame[own_cong] == 1) & (model_frame[own_overload] == 1)
        ).astype(int)
        model_frame["own_congested_no_overload"] = (
            (model_frame[own_cong] == 1) & (model_frame[own_overload] == 0)
        ).astype(int)
        model_frame["other_congested_overloaded"] = (
            (model_frame[other_cong] == 1) & (model_frame[other_overload] == 1)
        ).astype(int)
        model_frame["other_congested_no_overload"] = (
            (model_frame[other_cong] == 1) & (model_frame[other_overload] == 0)
        ).astype(int)
    else:
        model_frame["own_equip_congestion"] = model_frame[own_cong]
        model_frame["other_equip_congestion"] = model_frame[other_cong]

    model_frame["own_demand"] = model_frame[own_demand]
    model_frame["other_demand"] = model_frame[other_demand]
    return model_frame


def fit_models(
    frame: pd.DataFrame,
    include_day_fe: bool,
    section_key: str,
    section_title: str,
    variable_specs: list[dict[str, object]],
    split_equipment_terms: bool,
) -> list[dict[str, object]]:
    variable_keys = [str(spec["key"]) for spec in variable_specs]
    rhs = pooled_rhs(include_day_fe, variable_keys)
    model_specs = [
        {
            "model_key": "luzon_price",
            "pooled_label": f"Luzon pooled model ({section_title})",
            "dep_var": "price_1",
            "perspective": "luzon",
        },
        {
            "model_key": "visayas_price",
            "pooled_label": f"Visayas pooled model ({section_title})",
            "dep_var": "price_2",
            "perspective": "visayas",
        },
    ]

    fitted: list[dict[str, object]] = []
    for spec in model_specs:
        model_frame = prepare_model_frame(frame, perspective=str(spec["perspective"]), split_equipment_terms=split_equipment_terms)
        formula = f"{spec['dep_var']} ~ {rhs}"
        result = smf.ols(formula=formula, data=model_frame).fit(cov_type="HC1")
        fitted.append(
            {
                **spec,
                "formula": formula,
                "result": result,
                "section_key": section_key,
                "section_title": section_title,
                "include_day_fe": include_day_fe,
                "variable_specs": variable_specs,
                "split_equipment_terms": split_equipment_terms,
            }
        )
    return fitted


def linear_combo_stats(result: object, terms: list[str]) -> tuple[float, float, float]:
    exog_names = list(result.params.index)
    weights = np.zeros(len(exog_names))
    for term in terms:
        if term not in result.params.index:
            raise KeyError(f"Missing term {term} in fitted model.")
        weights[exog_names.index(term)] += 1.0
    test = result.t_test(weights)
    effect = float(np.squeeze(test.effect))
    std_err = float(np.squeeze(test.sd))
    pvalue = float(np.squeeze(test.pvalue))
    return effect, std_err, pvalue


def interaction_term_name(base_term: str) -> str:
    return f"luz_vis_link_congestion:{base_term}"


def format_linear_combo_cell(result: object, terms: list[str], reporting_delta: float) -> str:
    effect, std_err, pvalue = linear_combo_stats(result, terms)
    return (
        f"{format_number(effect * reporting_delta)}{significance_stars(pvalue)}"
        f"<br>({format_number(std_err * reporting_delta)})"
    )


def build_display_specs(models: list[dict[str, object]], variable_specs: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {model["model_key"]: model for model in models}
    return [
        {
            "title": pretty_column_title("Luzon", congested=False),
            "pooled_label": by_key["luzon_price"]["pooled_label"],
            "result": by_key["luzon_price"]["result"],
            "display_terms": {
                str(spec["key"]): [str(spec["key"])] for spec in variable_specs
            },
            "column_interpretation": "Reported effect when the Luzon-Visayas link is uncongested",
            "dependent_variable": "Luzon price",
            "formula": by_key["luzon_price"]["formula"],
            "section_key": by_key["luzon_price"]["section_key"],
            "section_title": by_key["luzon_price"]["section_title"],
            "include_day_fe": by_key["luzon_price"]["include_day_fe"],
            "variable_specs": variable_specs,
        },
        {
            "title": pretty_column_title("Luzon", congested=True),
            "pooled_label": by_key["luzon_price"]["pooled_label"],
            "result": by_key["luzon_price"]["result"],
            "display_terms": {
                str(spec["key"]): [str(spec["key"]), interaction_term_name(str(spec["key"]))] for spec in variable_specs
            },
            "column_interpretation": "Reported effect when the Luzon-Visayas link is congested",
            "dependent_variable": "Luzon price",
            "formula": by_key["luzon_price"]["formula"],
            "section_key": by_key["luzon_price"]["section_key"],
            "section_title": by_key["luzon_price"]["section_title"],
            "include_day_fe": by_key["luzon_price"]["include_day_fe"],
            "variable_specs": variable_specs,
        },
        {
            "title": pretty_column_title("Visayas", congested=False),
            "pooled_label": by_key["visayas_price"]["pooled_label"],
            "result": by_key["visayas_price"]["result"],
            "display_terms": {
                str(spec["key"]): [str(spec["key"])] for spec in variable_specs
            },
            "column_interpretation": "Reported effect when the Luzon-Visayas link is uncongested",
            "dependent_variable": "Visayas price",
            "formula": by_key["visayas_price"]["formula"],
            "section_key": by_key["visayas_price"]["section_key"],
            "section_title": by_key["visayas_price"]["section_title"],
            "include_day_fe": by_key["visayas_price"]["include_day_fe"],
            "variable_specs": variable_specs,
        },
        {
            "title": pretty_column_title("Visayas", congested=True),
            "pooled_label": by_key["visayas_price"]["pooled_label"],
            "result": by_key["visayas_price"]["result"],
            "display_terms": {
                str(spec["key"]): [str(spec["key"]), interaction_term_name(str(spec["key"]))] for spec in variable_specs
            },
            "column_interpretation": "Reported effect when the Luzon-Visayas link is congested",
            "dependent_variable": "Visayas price",
            "formula": by_key["visayas_price"]["formula"],
            "section_key": by_key["visayas_price"]["section_key"],
            "section_title": by_key["visayas_price"]["section_title"],
            "include_day_fe": by_key["visayas_price"]["include_day_fe"],
            "variable_specs": variable_specs,
        },
    ]


def build_table(display_specs: list[dict[str, object]]) -> pd.DataFrame:
    headers = ["Variable", *[str(spec["title"]) for spec in display_specs]]
    variable_specs = display_specs[0]["variable_specs"]

    rows: list[dict[str, str]] = []
    for variable_spec in variable_specs:
        key = str(variable_spec["key"])
        row = {"Variable": str(variable_spec["label"])}
        for spec in display_specs:
            row[str(spec["title"])] = format_linear_combo_cell(
                spec["result"],
                spec["display_terms"][key],
                float(variable_spec["reporting_delta"]),
            )
        rows.append(row)

    stats_rows = [
        ("Day FE", lambda spec: "Yes" if spec["include_day_fe"] else "No"),
        ("Observations", lambda spec: f"{int(spec['result'].nobs):,}"),
        ("R²", lambda spec: format_number(float(spec["result"].rsquared), 3)),
        ("Dependent variable", lambda spec: str(spec["dependent_variable"])),
        ("Reported coefficient", lambda spec: str(spec["column_interpretation"])),
        ("Underlying pooled model", lambda spec: str(spec["pooled_label"])),
        ("Robust SE", lambda spec: "HC1"),
    ]
    for label, value_fn in stats_rows:
        row = {"Variable": label}
        for spec in display_specs:
            row[str(spec["title"])] = value_fn(spec)
        rows.append(row)

    return pd.DataFrame(rows, columns=headers)


def build_tidy_rows(display_specs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    variable_specs = display_specs[0]["variable_specs"]
    for spec in display_specs:
        result = spec["result"]
        for variable_spec in variable_specs:
            key = str(variable_spec["key"])
            display_label = str(variable_spec["label"])
            estimate, std_err, pvalue = linear_combo_stats(result, spec["display_terms"][key])
            reporting_delta = float(variable_spec["reporting_delta"])
            rows.append(
                {
                    "column_title": spec["title"],
                    "underlying_pooled_model": spec["pooled_label"],
                    "formula": spec["formula"],
                    "display_term": display_label,
                    "estimate": estimate,
                    "std_err": std_err,
                    "pvalue": pvalue,
                    "reported_estimate": estimate * reporting_delta,
                    "reported_std_err": std_err * reporting_delta,
                    "reporting_delta": reporting_delta,
                    "nobs": int(result.nobs),
                    "r_squared": float(result.rsquared),
                    "adj_r_squared": float(result.rsquared_adj),
                    "dependent_variable": spec["dependent_variable"],
                    "column_interpretation": spec["column_interpretation"],
                    "section_key": spec["section_key"],
                    "section_title": spec["section_title"],
                    "day_fe": bool(spec["include_day_fe"]),
                }
            )
    return pd.DataFrame(rows)


def dataframe_to_html_table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, escape=False, classes=["reg-table"])


def polished_formula_html(dependent_variable: str, include_day_fe: bool, split_equipment_terms: bool) -> str:
    outcome = "P<sub>L,t</sub>" if dependent_variable == "price_1" else "P<sub>V,t</sub>"
    day_fe_term = " + &alpha;<sub>d</sub>" if include_day_fe else ""
    if split_equipment_terms:
        main_terms = (
            "&beta;<sub>1</sub> OwnCongOverload<sub>t</sub> + "
            "&beta;<sub>2</sub> OwnCongNoOverload<sub>t</sub> + "
            "&beta;<sub>3</sub> OtherCongOverload<sub>t</sub> + "
            "&beta;<sub>4</sub> OtherCongNoOverload<sub>t</sub> + "
            "&beta;<sub>5</sub> OwnDemand<sub>t</sub> + "
            "&beta;<sub>6</sub> OtherDemand<sub>t</sub>"
        )
        interaction_terms = (
            "&gamma;<sub>0</sub> LinkCong<sub>t</sub> + "
            "&gamma;<sub>1</sub>(LinkCong<sub>t</sub> &times; OwnCongOverload<sub>t</sub>) + "
            "&gamma;<sub>2</sub>(LinkCong<sub>t</sub> &times; OwnCongNoOverload<sub>t</sub>) + "
            "&gamma;<sub>3</sub>(LinkCong<sub>t</sub> &times; OtherCongOverload<sub>t</sub>) + "
            "&gamma;<sub>4</sub>(LinkCong<sub>t</sub> &times; OtherCongNoOverload<sub>t</sub>) + "
            "&gamma;<sub>5</sub>(LinkCong<sub>t</sub> &times; OwnDemand<sub>t</sub>) + "
            "&gamma;<sub>6</sub>(LinkCong<sub>t</sub> &times; OtherDemand<sub>t</sub>)"
        )
    else:
        main_terms = (
            "&beta;<sub>1</sub> OwnCong<sub>t</sub> + "
            "&beta;<sub>2</sub> OtherCong<sub>t</sub> + "
            "&beta;<sub>3</sub> OwnDemand<sub>t</sub> + "
            "&beta;<sub>4</sub> OtherDemand<sub>t</sub>"
        )
        interaction_terms = (
            "&gamma;<sub>0</sub> LinkCong<sub>t</sub> + "
            "&gamma;<sub>1</sub>(LinkCong<sub>t</sub> &times; OwnCong<sub>t</sub>) + "
            "&gamma;<sub>2</sub>(LinkCong<sub>t</sub> &times; OtherCong<sub>t</sub>) + "
            "&gamma;<sub>3</sub>(LinkCong<sub>t</sub> &times; OwnDemand<sub>t</sub>) + "
            "&gamma;<sub>4</sub>(LinkCong<sub>t</sub> &times; OtherDemand<sub>t</sub>)"
        )
    return (
        f"<em>{outcome}</em> = "
        f"{main_terms} + "
        f"{interaction_terms}"
        f"{day_fe_term} + &varepsilon;<sub>t</sub>"
    )


def render_section(
    title: str,
    description: str,
    formulas_html: str,
    table: pd.DataFrame,
) -> str:
    return f"""
  <section class="spec-card">
    <h2>{html.escape(title)}</h2>
    <p>{html.escape(description)}</p>
    <div class="formula-box">
      <div class="formula-label">Underlying Pooled Formulas</div>
      <div class="formula">{formulas_html}</div>
    </div>
    {dataframe_to_html_table(table)}
  </section>
"""


def build_html(
    panel_path: Path,
    regional_path: Path,
    output_csv: Path,
    frame: pd.DataFrame,
    sections: list[dict[str, object]],
) -> str:
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
.formula-box { background: #102a43; color: #fdfdfd; border-radius: 12px; padding: 14px 16px; margin: 0 0 18px; }
.formula-label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.85; margin-bottom: 8px; }
.formula { font-size: 16px; line-height: 1.5; }
.reg-table { border-collapse: collapse; width: 100%; margin: 18px 0 10px; font-size: 14px; box-shadow: 0 8px 24px rgba(16, 42, 67, 0.10); }
.reg-table th { background: #0b1f33; color: #fdfdfd; padding: 11px 12px; text-align: center; border: 1px solid #102a43; }
.reg-table td { border: 1px solid #bcccdc; padding: 9px 12px; vertical-align: top; background: #fffdf8; color: #102a43; }
.reg-table td:first-child { font-weight: 600; width: 320px; background: #e6ecf2; color: #0b1f33; }
.reg-table tr:nth-child(even) td:not(:first-child) { background: #eef3f7; }
.notes { color: #243b53; font-size: 14px; background: #e6ecf2; border-radius: 12px; padding: 16px 18px; margin-top: 28px; }
"""
    section_html = []
    for section in sections:
        formulas_html = "<br><br>".join(
            [
                (
                    f"<strong>{html.escape(model['pooled_label'])}</strong><br>"
                    f"{polished_formula_html(str(model['dep_var']), bool(model['include_day_fe']), bool(model['split_equipment_terms']))}"
                )
                for model in section["models"]
            ]
        )
        section_html.append(
            render_section(
                str(section["title"]),
                str(section["description"]),
                formulas_html,
                section["table"],
            )
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Luzon-Visayas Price Pooled OLS</title>
  <style>{css}</style>
</head>
<body>
  <h1>Luzon-Visayas Pooled OLS Regression Table</h1>
  <p class="lead">These columns come from pooled models that use the full Luzon-Visayas sample and interact the key regressors with the Luzon-Visayas congestion indicator. Equipment-congestion rows report the 0-to-1 effect of the indicator. Demand rows report the implied effect of a <code>+100 MW</code> increase in <code>MKT_REQT</code>.</p>

  <section class="spec-card">
    <h2>Specification Summary</h2>
    <div class="meta-grid">
      <div><span class="meta-label">Direct-pair source</span><code>{html.escape(str(panel_path))}</code></div>
      <div><span class="meta-label">Demand source</span><code>{html.escape(str(regional_path))}</code></div>
      <div><span class="meta-label">Sample</span><span>Luzon-Visayas pair only</span></div>
      <div><span class="meta-label">Merged observations</span><span>{len(frame):,}</span></div>
      <div><span class="meta-label">Equipment congestion</span><span><code>equip_cong_any</code> indicators from the direct-pair panel</span></div>
      <div><span class="meta-label">Demand measure</span><span><code>MKT_REQT</code> from <code>RTDREG</code> where <code>COMMODITY_TYPE == "En"</code></span></div>
    </div>
  </section>

  {''.join(section_html)}

  <div class="notes">
    <p>Both tables use pooled Luzon and Visayas regressions with interactions by <code>luz_vis_link_congestion</code>.</p>
    <p>The no-congestion columns report baseline coefficients from the pooled model when <code>luz_vis_link_congestion = 0</code>.</p>
    <p>The with-congestion columns report linear combinations of baseline and interaction terms when <code>luz_vis_link_congestion = 1</code>.</p>
    <p><code>MKT_REQT</code> is treated as demand in MW, so demand coefficients are reported for a <code>+100 MW</code> change rather than a per-1-MW change.</p>
    <p>For Luzon-price columns, “own” refers to Luzon and “other” refers to Visayas. For Visayas-price columns, “own” refers to Visayas and “other” refers to Luzon.</p>
    <p>Significance stars: <code>* p&lt;0.10</code>, <code>** p&lt;0.05</code>, <code>*** p&lt;0.01</code>.</p>
    <p>Tidy displayed-coefficient export: <code>{html.escape(str(output_csv))}</code></p>
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    panel_path = Path(args.direct_pair_panel) if args.direct_pair_panel else latest_matching_file(
        Path("data/panels"),
        "RTD_DIRECT_PAIR_PANEL_*.parquet",
    )
    regional_path = Path(args.regional_parquet) if args.regional_parquet else latest_matching_file(
        Path("data/rtdreg/combined"),
        "RTDREG_*.parquet",
    )
    output_html = Path(args.output_html)
    output_csv = Path(args.output_csv)

    frame = build_analysis_frame(panel_path, regional_path)
    sections: list[dict[str, object]] = []
    tidy_frames: list[pd.DataFrame] = []
    for include_day_fe, section_key, section_title, description, variable_specs, split_equipment_terms in [
        (
            True,
            "with_day_fe",
            "Pooled Model With Day FE",
            "This version uses common day fixed effects across the full sample and reports implied coefficients for the no-congestion and with-congestion regimes.",
            BASE_VARIABLE_SPECS,
            False,
        ),
        (
            True,
            "with_day_fe_split_congestion",
            "Pooled Model With Day FE and Split Equipment Congestion",
            "This version keeps common day fixed effects and separates each island's congestion effect into congested-and-overloaded versus congested-without-overload indicators before interacting them with the Luzon-Visayas link-congestion regime.",
            SPLIT_VARIABLE_SPECS,
            True,
        ),
        (
            False,
            "without_day_fe",
            "Pooled Model Without Day FE",
            "This version keeps the same pooled interaction structure but removes day fixed effects, so the regime-specific columns come from a pooled model without daily intercept controls.",
            BASE_VARIABLE_SPECS,
            False,
        ),
    ]:
        models = fit_models(
            frame,
            include_day_fe=include_day_fe,
            section_key=section_key,
            section_title=section_title,
            variable_specs=variable_specs,
            split_equipment_terms=split_equipment_terms,
        )
        display_specs = build_display_specs(models, variable_specs=variable_specs)
        sections.append(
            {
                "title": section_title,
                "description": description,
                "models": models,
                "table": build_table(display_specs),
            }
        )
        tidy_frames.append(build_tidy_rows(display_specs))
    tidy = pd.concat(tidy_frames, ignore_index=True)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        build_html(panel_path, regional_path, output_csv, frame, sections),
        encoding="utf-8",
    )
    tidy.to_csv(output_csv, index=False)

    print(f"Wrote {output_html}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
