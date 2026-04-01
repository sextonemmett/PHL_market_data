#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path
import re

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
SPLIT_NO_DEMAND_VARIABLE_SPECS = [
    {"key": "own_congested_overloaded", "label": "Own island, congested and overloaded", "reporting_delta": 1.0},
    {"key": "own_congested_no_overload", "label": "Own island, congested, no overload", "reporting_delta": 1.0},
    {"key": "other_congested_overloaded", "label": "Other island, congested and overloaded", "reporting_delta": 1.0},
    {"key": "other_congested_no_overload", "label": "Other island, congested, no overload", "reporting_delta": 1.0},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit separate-panel OLS models for the four Luzon/Visayas price columns "
            "across three specifications, then write HTML regression tables."
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


def section_rhs(include_day_fe: bool, variable_keys: list[str]) -> str:
    rhs = " + ".join(variable_keys)
    return f"{rhs} + C(fe_day)" if include_day_fe else rhs


def prepare_model_frame(frame: pd.DataFrame, perspective: str, split_equipment_terms: bool) -> pd.DataFrame:
    model_frame = frame.copy()
    if perspective == "luzon":
        own_cong = "equip_cong_any_1"
        own_overload = "equip_overload_any_1"
        other_cong = "equip_cong_any_2"
        other_overload = "equip_overload_any_2"
        own_demand_col = "Luzon_demand"
        other_demand_col = "Visayas_demand"
    else:
        own_cong = "equip_cong_any_2"
        own_overload = "equip_overload_any_2"
        other_cong = "equip_cong_any_1"
        other_overload = "equip_overload_any_1"
        own_demand_col = "Visayas_demand"
        other_demand_col = "Luzon_demand"

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

    model_frame["own_demand"] = model_frame[own_demand_col]
    model_frame["other_demand"] = model_frame[other_demand_col]
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
    rhs = section_rhs(include_day_fe, variable_keys)
    model_specs = [
        {
            "title": pretty_column_title("Luzon", congested=False),
            "dep_var": "price_1",
            "perspective": "luzon",
            "subsample_value": 0,
            "subsample_label": "Link uncongested",
            "dependent_variable": "Luzon price",
        },
        {
            "title": pretty_column_title("Luzon", congested=True),
            "dep_var": "price_1",
            "perspective": "luzon",
            "subsample_value": 1,
            "subsample_label": "Link congested",
            "dependent_variable": "Luzon price",
        },
        {
            "title": pretty_column_title("Visayas", congested=False),
            "dep_var": "price_2",
            "perspective": "visayas",
            "subsample_value": 0,
            "subsample_label": "Link uncongested",
            "dependent_variable": "Visayas price",
        },
        {
            "title": pretty_column_title("Visayas", congested=True),
            "dep_var": "price_2",
            "perspective": "visayas",
            "subsample_value": 1,
            "subsample_label": "Link congested",
            "dependent_variable": "Visayas price",
        },
    ]

    fitted: list[dict[str, object]] = []
    for spec in model_specs:
        sample = frame.loc[frame["luz_vis_link_congestion"] == spec["subsample_value"]].copy()
        model_frame = prepare_model_frame(sample, perspective=str(spec["perspective"]), split_equipment_terms=split_equipment_terms)
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


def format_cell(result: object, term: str, reporting_delta: float) -> str:
    if term not in result.params.index:
        return ""
    coef = float(result.params[term]) * reporting_delta
    std_err = float(result.bse[term]) * reporting_delta
    pvalue = float(result.pvalues[term])
    return f"{format_number(coef)}{significance_stars(pvalue)}<br>({format_number(std_err)})"


def build_table(models: list[dict[str, object]], variable_specs: list[dict[str, object]]) -> pd.DataFrame:
    headers = ["Variable", *[str(model["title"]) for model in models]]
    rows: list[dict[str, str]] = []

    for variable_spec in variable_specs:
        key = str(variable_spec["key"])
        row = {"Variable": str(variable_spec["label"])}
        for model in models:
            row[str(model["title"])] = format_cell(model["result"], key, float(variable_spec["reporting_delta"]))
        rows.append(row)

    stats_rows = [
        ("Day FE", lambda model: "Yes" if model["include_day_fe"] else "No"),
        ("Observations", lambda model: f"{int(model['result'].nobs):,}"),
        ("R²", lambda model: format_number(float(model["result"].rsquared), 3)),
        ("Dependent variable", lambda model: str(model["dependent_variable"])),
        ("Subsample", lambda model: str(model["subsample_label"])),
        ("Estimation", lambda model: "Separate panel OLS"),
        ("Robust SE", lambda model: "HC1"),
    ]
    for label, value_fn in stats_rows:
        row = {"Variable": label}
        for model in models:
            row[str(model["title"])] = value_fn(model)
        rows.append(row)

    return pd.DataFrame(rows, columns=headers)


def build_tidy_rows(models: list[dict[str, object]], variable_specs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in models:
        result = model["result"]
        for variable_spec in variable_specs:
            key = str(variable_spec["key"])
            if key not in result.params.index:
                continue
            reporting_delta = float(variable_spec["reporting_delta"])
            estimate = float(result.params[key])
            std_err = float(result.bse[key])
            pvalue = float(result.pvalues[key])
            rows.append(
                {
                    "column_title": model["title"],
                    "formula": model["formula"],
                    "display_term": str(variable_spec["label"]),
                    "term": key,
                    "estimate": estimate,
                    "std_err": std_err,
                    "pvalue": pvalue,
                    "reported_estimate": estimate * reporting_delta,
                    "reported_std_err": std_err * reporting_delta,
                    "reporting_delta": reporting_delta,
                    "nobs": int(result.nobs),
                    "r_squared": float(result.rsquared),
                    "adj_r_squared": float(result.rsquared_adj),
                    "dependent_variable": model["dependent_variable"],
                    "subsample_label": model["subsample_label"],
                    "estimation": "separate_panel",
                    "section_key": model["section_key"],
                    "section_title": model["section_title"],
                    "day_fe": bool(model["include_day_fe"]),
                    "split_equipment_terms": bool(model["split_equipment_terms"]),
                }
            )
    return pd.DataFrame(rows)


def dataframe_to_html_table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, escape=False, classes=["reg-table"])


def polished_term_html(variable_key: str) -> str:
    mapping = {
        "own_equip_congestion": "OwnCong<sub>t</sub>",
        "other_equip_congestion": "OtherCong<sub>t</sub>",
        "own_congested_overloaded": "OwnCongOverload<sub>t</sub>",
        "own_congested_no_overload": "OwnCongNoOverload<sub>t</sub>",
        "other_congested_overloaded": "OtherCongOverload<sub>t</sub>",
        "other_congested_no_overload": "OtherCongNoOverload<sub>t</sub>",
        "own_demand": "OwnDemand<sub>t</sub>",
        "other_demand": "OtherDemand<sub>t</sub>",
    }
    if variable_key not in mapping:
        raise KeyError(f"Unsupported display variable key: {variable_key}")
    return mapping[variable_key]


def polished_formula_html(
    dependent_variable: str,
    include_day_fe: bool,
    variable_keys: list[str],
    subsample_label: str,
) -> str:
    outcome = "P<sub>L,t</sub>" if dependent_variable == "price_1" else "P<sub>V,t</sub>"
    sample_condition = "LinkCong<sub>t</sub> = 1" if subsample_label == "Link congested" else "LinkCong<sub>t</sub> = 0"
    day_fe_term = " + &alpha;<sub>d</sub>" if include_day_fe else ""
    main_terms = " + ".join(
        f"&beta;<sub>{index}</sub> {polished_term_html(variable_key)}"
        for index, variable_key in enumerate(variable_keys, start=1)
    )
    return (
        f"<strong>Estimated on subsample:</strong> {sample_condition}<br>"
        f"<em>{outcome}</em> = {main_terms}{day_fe_term} + &varepsilon;<sub>t</sub>"
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
      <div class="formula-label">Separate Panel Formulas</div>
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
                    f"<strong>{html.escape(str(model['title']))}</strong><br>"
                    f"{polished_formula_html(str(model['dep_var']), bool(model['include_day_fe']), [str(spec['key']) for spec in model['variable_specs']], str(model['subsample_label']))}"
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
  <title>Luzon-Visayas Price OLS Tables</title>
  <style>{css}</style>
</head>
<body>
  <h1>Luzon-Visayas Separate-Panel OLS Regression Tables</h1>
  <p class="lead">This report keeps the same four displayed columns as the old pooled output, but each column is now estimated from its own Luzon-Visayas subsample instead of being recovered from pooled interaction terms. Equipment-congestion rows report the 0-to-1 effect of the indicator. In specifications that include demand controls, demand rows report the implied effect of a <code>+100 MW</code> increase in <code>MKT_REQT</code>.</p>

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
    <p>Each displayed column is estimated from a separate regression fitted only on its matching Luzon-Visayas congestion regime.</p>
    <p>The uncongested columns use rows where <code>luz_vis_link_congestion = 0</code>; the congested columns use rows where <code>luz_vis_link_congestion = 1</code>.</p>
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
    section_specs = [
        {
            "key": "with_day_fe",
            "title": "Separate Panels With Day FE",
            "description": (
                "This version estimates the four displayed columns as separate panel regressions, "
                "using day fixed effects within each congestion regime."
            ),
            "include_day_fe": True,
            "variable_specs": BASE_VARIABLE_SPECS,
            "split_equipment_terms": False,
        },
        {
            "key": "with_day_fe_split_equipment",
            "title": "Separate Panels With Day FE and Split Equipment Congestion",
            "description": (
                "This version still estimates the four columns separately, but splits each island's "
                "equipment effect into congested-and-overloaded versus congested-without-overload indicators."
            ),
            "include_day_fe": True,
            "variable_specs": SPLIT_VARIABLE_SPECS,
            "split_equipment_terms": True,
        },
        {
            "key": "with_day_fe_split_equipment_no_demand",
            "title": "Separate Panels With Day FE and Split Equipment Congestion, No Demand Controls",
            "description": (
                "This version keeps the split congestion-versus-overload indicators and day fixed effects, "
                "but removes the own-island and other-island demand controls."
            ),
            "include_day_fe": True,
            "variable_specs": SPLIT_NO_DEMAND_VARIABLE_SPECS,
            "split_equipment_terms": True,
        },
        {
            "key": "without_day_fe",
            "title": "Separate Panels Without Day FE",
            "description": (
                "This version estimates the same four separate panels without day fixed effects."
            ),
            "include_day_fe": False,
            "variable_specs": BASE_VARIABLE_SPECS,
            "split_equipment_terms": False,
        },
    ]

    sections: list[dict[str, object]] = []
    tidy_frames: list[pd.DataFrame] = []
    for section_spec in section_specs:
        models = fit_models(
            frame=frame,
            include_day_fe=bool(section_spec["include_day_fe"]),
            section_key=str(section_spec["key"]),
            section_title=str(section_spec["title"]),
            variable_specs=list(section_spec["variable_specs"]),
            split_equipment_terms=bool(section_spec["split_equipment_terms"]),
        )
        table = build_table(models, list(section_spec["variable_specs"]))
        tidy_frames.append(build_tidy_rows(models, list(section_spec["variable_specs"])))
        sections.append(
            {
                "title": section_spec["title"],
                "description": section_spec["description"],
                "models": models,
                "table": table,
            }
        )

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
