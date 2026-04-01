#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path
import re

import pandas as pd
import statsmodels.formula.api as smf

DEFAULT_OUTPUT_HTML = Path("regressions/luzon_visayas_price_split_ols.html")
DEFAULT_OUTPUT_CSV = Path("regressions/luzon_visayas_price_split_ols_coefficients.csv")
TIMESTAMP_TOKEN_RE = re.compile(r"(\d{8,12})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit four day-FE OLS specifications for Luzon and Visayas prices split by "
            "whether the Luzon-Visayas link is congested, then write an HTML regression table."
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


def format_cell(result: object, term: str, digits: int = 4) -> str:
    if term not in result.params.index:
        return ""
    coef = float(result.params[term])
    se = float(result.bse[term])
    pvalue = float(result.pvalues[term])
    return f"{format_number(coef, digits)}{significance_stars(pvalue)}<br>({format_number(se, digits)})"


def fit_models(frame: pd.DataFrame) -> list[dict[str, object]]:
    rhs = "own_equip_congestion + other_equip_congestion + own_demand + other_demand + C(fe_day)"
    model_specs = [
        {
            "title": "Luzon price (no luz_vis link congestion)",
            "dep_var": "price_1",
            "subsample_value": 0,
            "subsample_label": "No Luzon-Visayas link congestion",
            "rename_map": {
                "own_equip_congestion": "equip_cong_any_1",
                "other_equip_congestion": "equip_cong_any_2",
                "own_demand": "Luzon_demand",
                "other_demand": "Visayas_demand",
            },
        },
        {
            "title": "Luzon price (with luz_vis link congestion)",
            "dep_var": "price_1",
            "subsample_value": 1,
            "subsample_label": "With Luzon-Visayas link congestion",
            "rename_map": {
                "own_equip_congestion": "equip_cong_any_1",
                "other_equip_congestion": "equip_cong_any_2",
                "own_demand": "Luzon_demand",
                "other_demand": "Visayas_demand",
            },
        },
        {
            "title": "Visayas price (no luz_vis link congestion)",
            "dep_var": "price_2",
            "subsample_value": 0,
            "subsample_label": "No Luzon-Visayas link congestion",
            "rename_map": {
                "own_equip_congestion": "equip_cong_any_2",
                "other_equip_congestion": "equip_cong_any_1",
                "own_demand": "Visayas_demand",
                "other_demand": "Luzon_demand",
            },
        },
        {
            "title": "Visayas price (with luz_vis link congestion)",
            "dep_var": "price_2",
            "subsample_value": 1,
            "subsample_label": "With Luzon-Visayas link congestion",
            "rename_map": {
                "own_equip_congestion": "equip_cong_any_2",
                "other_equip_congestion": "equip_cong_any_1",
                "own_demand": "Visayas_demand",
                "other_demand": "Luzon_demand",
            },
        },
    ]

    fitted: list[dict[str, object]] = []
    for spec in model_specs:
        model_frame = frame.loc[frame["luz_vis_link_congestion"] == spec["subsample_value"]].copy()
        for target_name, source_name in spec["rename_map"].items():
            model_frame[target_name] = model_frame[source_name]
        formula = f"{spec['dep_var']} ~ {rhs}"
        result = smf.ols(formula=formula, data=model_frame).fit(cov_type="HC1")
        fitted.append({**spec, "formula": formula, "result": result, "n_sample_rows": int(len(model_frame))})
    return fitted


def build_tidy_rows(models: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in models:
        result = model["result"]
        for term in result.params.index:
            rows.append(
                {
                    "model": model["title"],
                    "formula": model["formula"],
                    "term": term,
                    "coef": float(result.params[term]),
                    "std_err": float(result.bse[term]),
                    "pvalue": float(result.pvalues[term]),
                    "nobs": int(result.nobs),
                    "r_squared": float(result.rsquared),
                    "adj_r_squared": float(result.rsquared_adj),
                    "subsample_label": model["subsample_label"],
                    "dependent_variable": model["dep_var"],
                }
            )
    return pd.DataFrame(rows)


def build_table(models: list[dict[str, object]]) -> pd.DataFrame:
    headers = ["Variable", *[str(model["title"]) for model in models]]
    variable_rows = [
        ("Own island, equipment congestion", "own_equip_congestion"),
        ("Other island, equipment congestion", "other_equip_congestion"),
        ("Own island, demand", "own_demand"),
        ("Other island, demand", "other_demand"),
    ]

    rows: list[dict[str, str]] = []
    for label, term in variable_rows:
        row = {"Variable": label}
        for model in models:
            row[str(model["title"])] = format_cell(model["result"], term)
        rows.append(row)

    stats_rows = [
        ("Day FE", lambda model: "Yes"),
        ("Observations", lambda model: f"{int(model['result'].nobs):,}"),
        ("R²", lambda model: format_number(float(model["result"].rsquared), 3)),
        ("Dependent variable", lambda model: "Luzon price" if model["dep_var"] == "price_1" else "Visayas price"),
        ("Subsample", lambda model: str(model["subsample_label"])),
        ("Robust SE", lambda model: "HC1"),
    ]
    for label, value_fn in stats_rows:
        row = {"Variable": label}
        for model in models:
            row[str(model["title"])] = value_fn(model)
        rows.append(row)

    return pd.DataFrame(rows, columns=headers)


def dataframe_to_html_table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, escape=False, classes=["reg-table"])


def build_html(
    panel_path: Path,
    regional_path: Path,
    output_csv: Path,
    frame: pd.DataFrame,
    models: list[dict[str, object]],
    table: pd.DataFrame,
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
    formulas_html = "<br>".join(
        [
            (
                f"<strong>{html.escape(model['title'])}</strong>: "
                f"<code>{html.escape(model['formula'])}</code>"
            )
            for model in models
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Luzon-Visayas Price Split OLS</title>
  <style>{css}</style>
</head>
<body>
  <h1>Luzon-Visayas Price Split OLS Regression Table</h1>
  <p class="lead">Four OLS specifications explain Luzon and Visayas RTD price levels separately, splitting the sample by whether the Luzon-Visayas link is congested. Each model includes day fixed effects and <code>HC1</code> robust standard errors.</p>

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
    <div class="formula-box">
      <div class="formula-label">Model Formulas</div>
      <div class="formula">{formulas_html}</div>
    </div>
    {dataframe_to_html_table(table)}
  </section>

  <div class="notes">
    <p>The no-congestion columns use rows where <code>link_congested_any == 0</code>; the with-congestion columns use rows where <code>link_congested_any == 1</code>.</p>
    <p>For Luzon-price columns, “own” refers to Luzon and “other” refers to Visayas. For Visayas-price columns, “own” refers to Visayas and “other” refers to Luzon.</p>
    <p>Significance stars: <code>* p&lt;0.10</code>, <code>** p&lt;0.05</code>, <code>*** p&lt;0.01</code>.</p>
    <p>Tidy coefficient export: <code>{html.escape(str(output_csv))}</code></p>
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
    models = fit_models(frame)
    table = build_table(models)
    tidy = build_tidy_rows(models)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        build_html(panel_path, regional_path, output_csv, frame, models, table),
        encoding="utf-8",
    )
    tidy.to_csv(output_csv, index=False)

    print(f"Wrote {output_html}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
