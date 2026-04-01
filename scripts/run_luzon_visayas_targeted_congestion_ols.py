#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path
import re

import pandas as pd
import statsmodels.formula.api as smf

DEFAULT_OUTPUT_HTML = Path("regressions/luzon_visayas_targeted_congestion_ols.html")
DEFAULT_OUTPUT_CSV = Path("regressions/luzon_visayas_targeted_congestion_ols_coefficients.csv")
TIMESTAMP_TOKEN_RE = re.compile(r"(\d{8,12})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit targeted Luzon-Visayas congestion regressions and write an HTML table."
        )
    )
    parser.add_argument("--direct-pair-panel", help="Direct-pair panel parquet path.")
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
    frame["link_cong"] = frame["link_congested_any"].astype(int)
    frame["L_equip_cong"] = frame["equip_cong_any_1"].astype(int)
    frame["V_equip_cong"] = frame["equip_cong_any_2"].astype(int)
    frame["V_equip_cong_w_overload"] = (
        (frame["equip_cong_any_2"] == 1) & (frame["equip_overload_any_2"] == 1)
    ).astype(int)
    frame["V_equip_cong_no_overload"] = (
        (frame["equip_cong_any_2"] == 1) & (frame["equip_overload_any_2"] == 0)
    ).astype(int)
    frame["abs_price_diff"] = frame["dep_abs_price_gap"]
    return frame


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


def format_cell(result: object, term: str) -> str:
    if term not in result.params.index:
        return ""
    coef = float(result.params[term])
    se = float(result.bse[term])
    pvalue = float(result.pvalues[term])
    return f"{format_number(coef)}{significance_stars(pvalue)}<br>({format_number(se)})"


def fit_models(frame: pd.DataFrame) -> list[dict[str, object]]:
    specs = [
        {
            "title": "Luzon price ~ own_equip_cong | full sample",
            "dep_var": "price_1",
            "sample_filter": "full sample",
            "sample_value": None,
            "rhs_terms": ["L_equip_cong"],
            "dependent_label": "Luzon price",
        },
        {
            "title": "Luzon price ~ own_equip_cong | link_cong == 1",
            "dep_var": "price_1",
            "sample_filter": "link_cong == 1",
            "sample_value": 1,
            "rhs_terms": ["L_equip_cong"],
            "dependent_label": "Luzon price",
        },
        {
            "title": "Vis price ~ own_equip_cong | link_cong == 1",
            "dep_var": "price_2",
            "sample_filter": "link_cong == 1",
            "sample_value": 1,
            "rhs_terms": ["V_equip_cong"],
            "dependent_label": "Visayas price",
        },
        {
            "title": "Vis price ~ equip_cong_no_overload + equip_cong_w_overload | link_cong == 1",
            "dep_var": "price_2",
            "sample_filter": "link_cong == 1",
            "sample_value": 1,
            "rhs_terms": ["V_equip_cong_no_overload", "V_equip_cong_w_overload"],
            "dependent_label": "Visayas price",
        },
        {
            "title": "abs(price diff) ~ L_equip_cong | link_cong == 0",
            "dep_var": "abs_price_diff",
            "sample_filter": "link_cong == 0",
            "sample_value": 0,
            "rhs_terms": ["L_equip_cong"],
            "dependent_label": "|Luzon price - Visayas price|",
        },
        {
            "title": "abs(price diff) ~ V_equip_cong | link_cong == 0",
            "dep_var": "abs_price_diff",
            "sample_filter": "link_cong == 0",
            "sample_value": 0,
            "rhs_terms": ["V_equip_cong"],
            "dependent_label": "|Luzon price - Visayas price|",
        },
        {
            "title": "abs(price diff) ~ L_equip_cong + V_equip_cong | link_cong == 0",
            "dep_var": "abs_price_diff",
            "sample_filter": "link_cong == 0",
            "sample_value": 0,
            "rhs_terms": ["L_equip_cong", "V_equip_cong"],
            "dependent_label": "|Luzon price - Visayas price|",
        },
    ]

    fitted: list[dict[str, object]] = []
    for spec in specs:
        if spec["sample_value"] is None:
            sample = frame.copy()
        else:
            sample = frame.loc[frame["link_cong"] == spec["sample_value"]].copy()
        formula = f"{spec['dep_var']} ~ {' + '.join(spec['rhs_terms'])}"
        result = smf.ols(formula=formula, data=sample).fit(cov_type="HC1")
        fitted.append(
            {
                **spec,
                "formula": formula,
                "result": result,
                "n_sample_rows": int(len(sample)),
            }
        )
    return fitted


def build_table(models: list[dict[str, object]]) -> pd.DataFrame:
    headers = ["Variable", *[str(model["title"]) for model in models]]
    variable_rows = [
        ("Constant", "Intercept"),
        ("L_equip_cong", "L_equip_cong"),
        ("V_equip_cong", "V_equip_cong"),
        ("V_equip_cong_no_overload", "V_equip_cong_no_overload"),
        ("V_equip_cong_w_overload", "V_equip_cong_w_overload"),
    ]

    rows: list[dict[str, str]] = []
    for label, term in variable_rows:
        row = {"Variable": label}
        for model in models:
            row[str(model["title"])] = format_cell(model["result"], term)
        rows.append(row)

    stats_rows = [
        ("Observations", lambda model: f"{int(model['result'].nobs):,}"),
        ("R²", lambda model: format_number(float(model["result"].rsquared), 3)),
        ("Dependent variable", lambda model: str(model["dependent_label"])),
        ("Sample filter", lambda model: str(model["sample_filter"])),
        ("Robust SE", lambda model: "HC1"),
    ]
    for label, value_fn in stats_rows:
        row = {"Variable": label}
        for model in models:
            row[str(model["title"])] = value_fn(model)
        rows.append(row)

    return pd.DataFrame(rows, columns=headers)


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
                    "dependent_variable": model["dependent_label"],
                    "sample_filter": model["sample_filter"],
                }
            )
    return pd.DataFrame(rows)


def dataframe_to_html_table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, escape=False, classes=["reg-table"])


def build_html(
    panel_path: Path,
    output_csv: Path,
    frame: pd.DataFrame,
    models: list[dict[str, object]],
    table: pd.DataFrame,
) -> str:
    css = """
body { font-family: Georgia, "Times New Roman", serif; margin: 32px auto; max-width: 1600px; color: #102a43; line-height: 1.55; background: #f4f1ea; }
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
.reg-table td:first-child { font-weight: 600; width: 220px; background: #e6ecf2; color: #0b1f33; }
.reg-table tr:nth-child(even) td:not(:first-child) { background: #eef3f7; }
.notes { color: #243b53; font-size: 14px; background: #e6ecf2; border-radius: 12px; padding: 16px 18px; margin-top: 28px; }
"""
    formulas_html = "<br>".join(
        [
            (
                f"<strong>{html.escape(str(model['title']))}</strong>: "
                f"<code>{html.escape(str(model['formula']))}</code>"
            )
            for model in models
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Luzon-Visayas Targeted Congestion OLS</title>
  <style>{css}</style>
</head>
<body>
  <h1>Luzon-Visayas Targeted Congestion OLS</h1>
  <p class="lead">This report collects five targeted HC1-robust OLS regressions on the CLUZ-CVIS direct-pair sample, covering congested-sample price levels and uncongested-sample absolute price gaps.</p>

  <section class="spec-card">
    <h2>Specification Summary</h2>
    <div class="meta-grid">
      <div><span class="meta-label">Direct-pair source</span><code>{html.escape(str(panel_path))}</code></div>
      <div><span class="meta-label">Sample</span><span>Luzon-Visayas pair only</span></div>
      <div><span class="meta-label">Merged observations</span><span>{len(frame):,}</span></div>
      <div><span class="meta-label">Outcome definitions</span><span><code>price_1</code>, <code>price_2</code>, and <code>dep_abs_price_gap = |price_1 - price_2|</code></span></div>
      <div><span class="meta-label">Congestion indicators</span><span><code>L_equip_cong = equip_cong_any_1</code>, <code>V_equip_cong = equip_cong_any_2</code></span></div>
      <div><span class="meta-label">Link filter</span><span><code>link_cong = link_congested_any</code></span></div>
    </div>
    <div class="formula-box">
      <div class="formula-label">Model Formulas</div>
      <div class="formula">{formulas_html}</div>
    </div>
    {dataframe_to_html_table(table)}
  </section>

  <div class="notes">
    <p>The first Luzon-price regression uses the full CLUZ-CVIS sample.</p>
    <p>The next two price-level regressions use the congested subsample where <code>link_cong == 1</code>.</p>
    <p>The three absolute-gap regressions use the uncongested subsample where <code>link_cong == 0</code>.</p>
    <p>For the CLUZ-CVIS pair, <code>L_equip_cong</code> maps to Luzon equipment congestion and <code>V_equip_cong</code> maps to Visayas equipment congestion.</p>
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
    output_html = Path(args.output_html)
    output_csv = Path(args.output_csv)

    frame = load_luzon_visayas_panel(panel_path)
    models = fit_models(frame)
    table = build_table(models)
    tidy = build_tidy_rows(models)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        build_html(panel_path, output_csv, frame, models, table),
        encoding="utf-8",
    )
    tidy.to_csv(output_csv, index=False)

    print(f"Wrote {output_html}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
