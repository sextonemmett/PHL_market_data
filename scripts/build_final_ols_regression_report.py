#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_HTML = Path("regressions/final_ols_regression_report.html")
DEFAULT_DIRECT_CLEAN_CSV = Path("regressions/direct_pair_ols_clean_base_coefficients.csv")
DEFAULT_DIRECT_SEASON_CSV = Path("regressions/direct_pair_ols_seasonality_fe_coefficients.csv")
DEFAULT_PROGRESSIVE_CSV = Path("regressions/luzon_visayas_ols_progressive_coefficients.csv")
DEFAULT_PRICE_LEVEL_CSV = Path("regressions/luzon_visayas_price_pooled_ols_coefficients.csv")
DEFAULT_TARGETED_CSV = Path("regressions/luzon_visayas_targeted_congestion_ols_coefficients.csv")
DEFAULT_FULL_SAMPLE_IMAGE = Path("regressions/direct_pair_ols_clean_base_visual_full_sample.png")
DEFAULT_WINSOR_IMAGE = Path("regressions/direct_pair_ols_clean_base_visual_winsor_99.png")
DEFAULT_SEASON_IMAGE = Path("regressions/direct_pair_ols_seasonality_fe_visual.png")


TERM_LABELS = {
    "Intercept": "Constant",
    "link_congestion": "Direct Luzon-Visayas link congested",
    "total_demand": "Luzon plus Visayas demand, +100 MW",
    "link_congestion:total_demand": "Link congested x total demand, +100 MW",
    "L_equip_cong": "Luzon equipment congested",
    "V_equip_cong": "Visayas equipment congested",
    "V_equip_cong_no_overload": "Visayas equipment congested, no overload",
    "V_equip_cong_w_overload": "Visayas equipment congested, with overload",
}

SECTION_TITLE_ORDER = [
    "Separate Panels With Day FE",
    "Separate Panels With Day FE and Split Equipment Congestion",
    "Separate Panels With Day FE and Split Equipment Congestion, No Demand Controls",
    "Separate Panels Without Day FE",
]

COLUMN_LABELS = {
    "Luzon Price | Link Uncongested": "Luzon price, link uncongested",
    "Luzon Price | Link Congested": "Luzon price, link congested",
    "Visayas Price | Link Uncongested": "Visayas price, link uncongested",
    "Visayas Price | Link Congested": "Visayas price, link congested",
}

TARGETED_MODEL_LABELS = {
    "Luzon price ~ own_equip_cong | full sample": "Luzon price on Luzon equipment congestion, all intervals",
    "Luzon price ~ own_equip_cong | link_cong == 1": "Luzon price on Luzon equipment congestion, link-congested intervals",
    "Vis price ~ own_equip_cong | link_cong == 1": "Visayas price on Visayas equipment congestion, link-congested intervals",
    "Vis price ~ equip_cong_no_overload + equip_cong_w_overload | link_cong == 1": (
        "Visayas price on Visayas congestion with and without overload, link-congested intervals"
    ),
    "abs(price diff) ~ L_equip_cong | link_cong == 0": (
        "Absolute Luzon-Visayas price gap on Luzon equipment congestion, link-uncongested intervals"
    ),
    "abs(price diff) ~ V_equip_cong | link_cong == 0": (
        "Absolute Luzon-Visayas price gap on Visayas equipment congestion, link-uncongested intervals"
    ),
    "abs(price diff) ~ L_equip_cong + V_equip_cong | link_cong == 0": (
        "Absolute Luzon-Visayas price gap on both island congestion indicators, link-uncongested intervals"
    ),
    "abs(price diff) ~ L_equip_cong + V_equip_cong_no_overload + V_equip_cong_w_overload | link_cong == 0": (
        "Absolute Luzon-Visayas price gap on Luzon congestion and split Visayas congestion, link-uncongested intervals"
    ),
    "abs(price diff) ~ L_equip_cong | link_cong == 1": (
        "Absolute Luzon-Visayas price gap on Luzon equipment congestion, link-congested intervals"
    ),
    "abs(price diff) ~ V_equip_cong | link_cong == 1": (
        "Absolute Luzon-Visayas price gap on Visayas equipment congestion, link-congested intervals"
    ),
    "abs(price diff) ~ L_equip_cong + V_equip_cong | link_cong == 1": (
        "Absolute Luzon-Visayas price gap on both island congestion indicators, link-congested intervals"
    ),
    "abs(price diff) ~ L_equip_cong + V_equip_cong_no_overload + V_equip_cong_w_overload | link_cong == 1": (
        "Absolute Luzon-Visayas price gap on Luzon congestion and split Visayas congestion, link-congested intervals"
    ),
}

SAMPLE_LABELS = {
    "full sample": "All Luzon-Visayas intervals",
    "link_cong == 1": "Luzon-Visayas link-congested intervals only",
    "link_cong == 0": "Luzon-Visayas link-uncongested intervals only",
    "Link uncongested": "Luzon-Visayas link-uncongested intervals only",
    "Link congested": "Luzon-Visayas link-congested intervals only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final OLS-only econometric HTML report.")
    parser.add_argument("--output-html", default=str(DEFAULT_OUTPUT_HTML), help="Output HTML path.")
    return parser.parse_args()


def stars(pvalue: float) -> str:
    if pd.isna(pvalue):
        return ""
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.10:
        return "*"
    return ""


def fmt(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{value:,.{digits}f}"


def fmt_int(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{int(value):,}"


def estimate_cell(estimate: float, std_err: float, pvalue: float, digits: int = 2) -> str:
    return f"{fmt(estimate, digits)}{stars(pvalue)}<br><span class=\"se\">({fmt(std_err, digits)})</span>"


def pct_cell(estimate: float, std_err: float, pvalue: float) -> str:
    return f"{fmt(estimate, 1)}%{stars(pvalue)}<br><span class=\"se\">({fmt(std_err, 1)}%)</span>"


def html_table(frame: pd.DataFrame, classes: str = "reg-table") -> str:
    return frame.to_html(index=False, escape=False, classes=[classes])


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def section(title: str, body: str, section_id: str | None = None) -> str:
    id_attr = f' id="{html.escape(section_id)}"' if section_id else ""
    return f"""
  <section class="report-section"{id_attr}>
    <h2>{html.escape(title)}</h2>
    {body}
  </section>
"""


def meta_grid(items: list[tuple[str, str]]) -> str:
    cells = "\n".join(
        f'<div><span class="meta-label">{html.escape(label)}</span><span>{value}</span></div>'
        for label, value in items
    )
    return f'<div class="meta-grid">{cells}</div>'


def build_progressive_section(path: Path) -> str:
    frame = pd.read_csv(path)
    rows = [
        ("Constant", "Intercept", 1.0),
        ("Direct Luzon-Visayas link congested", "link_congestion", 1.0),
        ("Luzon plus Visayas demand, +100 MW", "total_demand", 100.0),
        ("Link congested x total demand, +100 MW", "link_congestion:total_demand", 100.0),
    ]
    models = ["(1)", "(2)", "(3)"]
    out_rows: list[dict[str, str]] = []
    for label, term, scale in rows:
        row = {"Regressor": label}
        for model in models:
            subset = frame.loc[(frame["model"] == model) & (frame["term"] == term)]
            if subset.empty:
                row[model] = ""
            else:
                value = subset.iloc[0]
                row[model] = estimate_cell(
                    float(value["coef"]) * scale,
                    float(value["std_err"]) * scale,
                    float(value["pvalue"]),
                )
        out_rows.append(row)

    for label, key in [
        ("Fixed effects", None),
        ("Observations", "nobs"),
        ("R-squared", "r_squared"),
        ("Dependent variable", None),
        ("Standard errors", None),
    ]:
        row = {"Regressor": label}
        for model in models:
            model_frame = frame.loc[frame["model"] == model]
            first = model_frame.iloc[0]
            if label == "Fixed effects":
                row[model] = "None"
            elif label == "Dependent variable":
                row[model] = "Absolute Luzon-Visayas price gap"
            elif label == "Standard errors":
                row[model] = "HC1 robust"
            elif key == "nobs":
                row[model] = fmt_int(float(first[key]))
            else:
                row[model] = fmt(float(first[key]), 3)
        out_rows.append(row)

    body = f"""
    <p class="plain">These three models provide the simplest bridge from the raw direct-link congestion indicator to the Luzon-Visayas price gap. The outcome is the absolute difference between the Luzon and Visayas real-time prices in each five-minute interval. Demand is the sum of Luzon and Visayas market requirement from the energy-market regional data; demand rows are reported for a 100 MW increase.</p>
    {meta_grid([
        ("Estimator", "Ordinary least squares"),
        ("Outcome construction", "Absolute value of Luzon price minus Visayas price"),
        ("Congestion construction", "Indicator equal to one when the observed Luzon-Visayas inter-island link is congested"),
        ("Fixed effects", "<strong>None.</strong> These are cross-interval OLS models with only the listed regressors."),
    ])}
    <div class="formula-box">
      <div class="formula-label">Model sequence</div>
      <div class="formula">Model 1 includes only direct-link congestion. Model 2 adds total Luzon plus Visayas demand. Model 3 adds the interaction between direct-link congestion and total demand.</div>
    </div>
    {html_table(pd.DataFrame(out_rows))}
"""
    return section("1. Progressive Luzon-Visayas Gap Models", body, "progressive")


def clean_base_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["pair_title", "sample_label"], observed=True)
        .agg(
            clean_pair_base=("clean_pair_base", "first"),
            clean_rows=("clean_rows", "first"),
            nobs=("nobs", "first"),
        )
        .reset_index()
    )
    summary = summary.rename(
        columns={
            "pair_title": "Pair",
            "sample_label": "Outcome sample",
            "clean_pair_base": "Clean pair-average price",
            "clean_rows": "Rows used for clean base",
            "nobs": "Regression observations",
        }
    )
    summary["Clean pair-average price"] = summary["Clean pair-average price"].map(lambda x: fmt(float(x), 0))
    summary["Rows used for clean base"] = summary["Rows used for clean base"].map(lambda x: fmt_int(float(x)))
    summary["Regression observations"] = summary["Regression observations"].map(lambda x: fmt_int(float(x)))
    return summary


def direct_gap_table(frame: pd.DataFrame, value_label: str = "Effect on absolute gap") -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    ordered = frame.copy()
    if "sample_key" not in ordered.columns:
        ordered["sample_key"] = ordered.get("seasonality_key", "full_sample")
    if "sample_label" not in ordered.columns:
        ordered["sample_label"] = ordered.get("seasonality_label", "Full sample")
    term_order = {
        "link_congested_any": 0,
        "equip_cong_any_1": 1,
        "equip_overload_any_1": 2,
        "equip_cong_any_2": 3,
        "equip_overload_any_2": 4,
    }
    ordered = ordered.assign(_order=ordered["term"].map(term_order)).sort_values(["pair_title", "sample_key", "_order"])
    for _, row in ordered.iterrows():
        rows.append(
            {
                "Pair": html.escape(str(row["pair_title"])),
                "Outcome sample": html.escape(str(row.get("sample_label", "Full sample"))),
                "Regressor": html.escape(str(row["term_label"])),
                value_label: estimate_cell(float(row["coef"]), float(row["std_err"]), float(row["pvalue"])),
                "Percent of clean price base": pct_cell(
                    float(row["scaled_pct_of_clean_base"]),
                    float(row["scaled_se_pct_of_clean_base"]),
                    float(row["pvalue"]),
                ),
            }
        )
    return pd.DataFrame(rows)


def build_direct_gap_section(clean_path: Path, full_image: Path, winsor_image: Path) -> str:
    frame = pd.read_csv(clean_path)
    body = f"""
    <p class="plain">These models estimate the direct-pair absolute price gap separately for Luzon-Visayas and Visayas-Mindanao. The reported congestion coefficients are shown both in price units and as a share of a clean pair-average price. The clean base is computed within each pair from intervals where the direct link is not congested, neither island has equipment congestion, and neither island has overload.</p>
    {meta_grid([
        ("Estimator", "Pair-specific ordinary least squares"),
        ("Outcome construction", "Absolute value of the first island price minus the second island price in the pair"),
        ("Displayed regressors", "Direct-link congestion, equipment congestion on each side, and overload on each side"),
        ("Additional controls", "Losses, generation, market imports, and market exports for each side of the pair"),
        ("Fixed effects", "<strong>Calendar-day fixed effects included.</strong> No pair fixed effects are needed because each regression is estimated separately by pair."),
        ("Winsorized outcome", "For the winsorized sample, the absolute price gap is capped at that pair's 99th percentile before estimation."),
    ])}
    <div class="figure-grid">
      <figure>
        <img src="{image_data_uri(full_image)}" alt="Full-sample direct-pair OLS stacked effects">
        <figcaption>Full-sample estimates, scaled by the clean pair-average price.</figcaption>
      </figure>
      <figure>
        <img src="{image_data_uri(winsor_image)}" alt="Winsorized direct-pair OLS stacked effects">
        <figcaption>Same specification after capping each pair's absolute gap at its 99th percentile.</figcaption>
      </figure>
    </div>
    <h3>Clean Price Bases</h3>
    {html_table(clean_base_summary(frame), "compact-table")}
    <h3>Displayed Congestion Coefficients</h3>
    {html_table(direct_gap_table(frame))}
"""
    return section("2. Direct-Pair Gap Models With Day Fixed Effects", body, "direct-gap")


def build_seasonality_section(path: Path, image_path: Path) -> str:
    frame = pd.read_csv(path)
    body = f"""
    <p class="plain">This is the same direct-pair absolute-gap exercise, but the calendar controls are coarser seasonal fixed effects rather than calendar-day fixed effects. It is useful for seeing whether the direct-pair congestion story depends on absorbing every individual day.</p>
    {meta_grid([
        ("Estimator", "Pair-specific ordinary least squares"),
        ("Outcome construction", "Absolute direct-pair price gap"),
        ("Displayed regressors", "Direct-link congestion, equipment congestion on each side, and overload on each side"),
        ("Additional controls", "Losses, generation, market imports, and market exports for each side of the pair"),
        ("Fixed effects", "<strong>Month, ISO-week, and day-of-week fixed effects included.</strong> Calendar-day fixed effects are not included in this specification."),
    ])}
    <figure class="wide-figure">
      <img src="{image_data_uri(image_path)}" alt="Seasonality fixed-effect direct-pair OLS stacked effects">
      <figcaption>Direct-pair congestion coefficients with month, ISO-week, and day-of-week fixed effects.</figcaption>
    </figure>
    {html_table(direct_gap_table(frame, value_label="Effect on absolute gap"))}
"""
    return section("3. Direct-Pair Gap Models With Seasonal Fixed Effects", body, "seasonality")


def price_level_table(section_frame: pd.DataFrame) -> pd.DataFrame:
    columns = [COLUMN_LABELS[column] for column in section_frame["column_title"].drop_duplicates()]
    display_terms = section_frame["display_term"].drop_duplicates().tolist()
    out_rows: list[dict[str, str]] = []
    for display_term in display_terms:
        row = {"Regressor": html.escape(str(display_term))}
        for raw_column, pretty_column in COLUMN_LABELS.items():
            subset = section_frame.loc[
                (section_frame["column_title"] == raw_column) & (section_frame["display_term"] == display_term)
            ]
            if subset.empty:
                row[pretty_column] = ""
            else:
                value = subset.iloc[0]
                row[pretty_column] = estimate_cell(
                    float(value["reported_estimate"]),
                    float(value["reported_std_err"]),
                    float(value["pvalue"]),
                )
        out_rows.append(row)

    for label in ["Fixed effects", "Observations", "R-squared", "Dependent variable", "Sample", "Standard errors"]:
        row = {"Regressor": label}
        for raw_column, pretty_column in COLUMN_LABELS.items():
            subset = section_frame.loc[section_frame["column_title"] == raw_column]
            if subset.empty:
                row[pretty_column] = ""
                continue
            first = subset.iloc[0]
            if label == "Fixed effects":
                row[pretty_column] = "Calendar-day FE" if bool(first["day_fe"]) else "None"
            elif label == "Observations":
                row[pretty_column] = fmt_int(float(first["nobs"]))
            elif label == "R-squared":
                row[pretty_column] = fmt(float(first["r_squared"]), 3)
            elif label == "Dependent variable":
                row[pretty_column] = html.escape(str(first["dependent_variable"]))
            elif label == "Sample":
                row[pretty_column] = html.escape(SAMPLE_LABELS.get(str(first["subsample_label"]), str(first["subsample_label"])))
            else:
                row[pretty_column] = "HC1 robust"
        out_rows.append(row)
    return pd.DataFrame(out_rows, columns=["Regressor", *columns])


def build_price_level_section(path: Path) -> str:
    frame = pd.read_csv(path)
    section_blocks: list[str] = []
    for title in SECTION_TITLE_ORDER:
        part = frame.loc[frame["section_title"] == title].copy()
        if part.empty:
            continue
        day_fe = bool(part["day_fe"].iloc[0])
        split = bool(part["split_equipment_terms"].iloc[0])
        if title == "Separate Panels With Day FE":
            description = (
                "Own-island and other-island equipment congestion enter as broad indicators, with own-demand and other-demand controls."
            )
        elif title == "Separate Panels With Day FE and Split Equipment Congestion":
            description = (
                "Equipment congestion is split into congestion with overload and congestion without overload; demand controls remain included."
            )
        elif title == "Separate Panels With Day FE and Split Equipment Congestion, No Demand Controls":
            description = (
                "The split congestion indicators are kept, but own-demand and other-demand controls are removed."
            )
        else:
            description = (
                "The broad equipment-congestion and demand specification is re-estimated without fixed effects."
            )
        fe_text = (
            "<strong>Calendar-day fixed effects included.</strong>"
            if day_fe
            else "<strong>No fixed effects included.</strong>"
        )
        split_text = (
            "Congestion indicators distinguish overloaded from non-overloaded congestion episodes."
            if split
            else "Congestion indicators do not split overload from non-overload episodes."
        )
        section_blocks.append(
            f"""
      <div class="subsection">
        <h3>{html.escape(title)}</h3>
        <p class="plain">{html.escape(description)} {split_text} {fe_text}</p>
        {html_table(price_level_table(part))}
      </div>
"""
        )

    body = f"""
    <p class="plain">These models move from price gaps to price levels. Each displayed column is estimated as its own OLS regression: Luzon price in uncongested intervals, Luzon price in congested intervals, Visayas price in uncongested intervals, and Visayas price in congested intervals. For Luzon-price regressions, "own island" means Luzon and "other island" means Visayas. For Visayas-price regressions, "own island" means Visayas and "other island" means Luzon.</p>
    {meta_grid([
        ("Estimator", "Separate-panel ordinary least squares"),
        ("Outcome construction", "The real-time energy price for the island named in the column heading"),
        ("Sample construction", "Columns are split by whether the Luzon-Visayas direct link is congested in the interval"),
        ("Equipment construction", "Own-island and other-island indicators mark whether that side has equipment congestion in the direct-pair panel"),
        ("Demand construction", "Own-island and other-island demand come from regional energy market requirement; reported for +100 MW when included"),
        ("Fixed effects", "Stated separately below for each specification block."),
    ])}
    {''.join(section_blocks)}
"""
    return section("4. Luzon-Visayas Price-Level Models", body, "price-level")


def targeted_term_label(term: str) -> str:
    return TERM_LABELS.get(term, term)


def targeted_wide_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_order = frame["model"].drop_duplicates().tolist()
    model_columns = {model: f"({index})" for index, model in enumerate(model_order, start=1)}
    term_order = [
        "Intercept",
        "L_equip_cong",
        "V_equip_cong",
        "V_equip_cong_no_overload",
        "V_equip_cong_w_overload",
    ]
    rows: list[dict[str, str]] = []
    for term in term_order:
        row = {"Regressor": html.escape(targeted_term_label(term))}
        for model in model_order:
            subset = frame.loc[(frame["model"] == model) & (frame["term"] == term)]
            if subset.empty:
                row[model_columns[model]] = ""
            else:
                value = subset.iloc[0]
                row[model_columns[model]] = estimate_cell(
                    float(value["coef"]),
                    float(value["std_err"]),
                    float(value["pvalue"]),
                )
        rows.append(row)

    stat_rows = [
        ("Dependent variable", lambda value: html.escape(str(value["dependent_variable"]))),
        ("Sample", lambda value: html.escape(SAMPLE_LABELS.get(str(value["sample_filter"]), str(value["sample_filter"])))),
        ("Fixed effects", lambda value: "None"),
        ("Observations", lambda value: fmt_int(float(value["nobs"]))),
        ("R-squared", lambda value: fmt(float(value["r_squared"]), 3)),
        ("Standard errors", lambda value: "HC1 robust"),
    ]
    for label, value_fn in stat_rows:
        row = {"Regressor": label}
        for model in model_order:
            first = frame.loc[frame["model"] == model].iloc[0]
            row[model_columns[model]] = value_fn(first)
        rows.append(row)

    definitions = pd.DataFrame(
        [
            {
                "Column": model_columns[model],
                "Model": html.escape(TARGETED_MODEL_LABELS.get(str(model), str(model))),
            }
            for model in model_order
        ]
    )
    wide = pd.DataFrame(rows, columns=["Regressor", *[model_columns[model] for model in model_order]])
    return wide, definitions


def build_targeted_section(path: Path) -> str:
    frame = pd.read_csv(path)
    wide, definitions = targeted_wide_table(frame)

    body = f"""
    <p class="plain">These narrower checks isolate specific Luzon-Visayas congestion questions. They are intentionally sparse: each regression uses only the listed congestion indicator or indicators and no fixed effects. That makes them useful as transparent diagnostic contrasts against the richer price-level and direct-gap specifications above.</p>
    {meta_grid([
        ("Estimator", "Ordinary least squares"),
        ("Outcome construction", "Either the Luzon price, the Visayas price, or the absolute Luzon-Visayas price gap"),
        ("Sample construction", "Full sample, link-congested intervals only, or link-uncongested intervals only"),
        ("Congestion construction", "Equipment-congestion indicators identify whether Luzon or Visayas has equipment congestion; Visayas congestion is also split by overload status in one model"),
        ("Fixed effects", "<strong>None in every targeted model.</strong>"),
    ])}
    <div class="table-scroll">
      {html_table(wide, "target-table")}
    </div>
    <h3>Column Definitions</h3>
    {html_table(definitions, "compact-table")}
"""
    return section("5. Targeted Luzon-Visayas Congestion Checks", body, "targeted")


def build_html() -> str:
    sections = [
        build_progressive_section(DEFAULT_PROGRESSIVE_CSV),
        build_direct_gap_section(DEFAULT_DIRECT_CLEAN_CSV, DEFAULT_FULL_SAMPLE_IMAGE, DEFAULT_WINSOR_IMAGE),
        build_seasonality_section(DEFAULT_DIRECT_SEASON_CSV, DEFAULT_SEASON_IMAGE),
        build_price_level_section(DEFAULT_PRICE_LEVEL_CSV),
        build_targeted_section(DEFAULT_TARGETED_CSV),
    ]
    css = """
:root {
  --ink: #17202a;
  --muted: #52616f;
  --line: #c8d2dc;
  --panel: #ffffff;
  --band: #edf2f7;
  --accent: #244c66;
  --accent-2: #8f3f2b;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #f6f4ef;
  color: var(--ink);
  font-family: Georgia, "Times New Roman", serif;
  line-height: 1.5;
}
.page {
  max-width: 1500px;
  margin: 0 auto;
  padding: 34px 34px 56px;
}
header {
  border-bottom: 3px solid var(--accent);
  padding-bottom: 20px;
  margin-bottom: 26px;
}
h1 {
  margin: 0 0 12px;
  font-size: 36px;
  line-height: 1.1;
  color: #0e2330;
}
h2 {
  margin: 0 0 14px;
  font-size: 25px;
  color: #0e2330;
}
h3 {
  margin: 24px 0 10px;
  font-size: 18px;
  color: #16384f;
}
.lead {
  max-width: 1120px;
  margin: 0;
  font-size: 17px;
  color: #2d3d4a;
}
.plain {
  max-width: 1180px;
  margin: 0 0 16px;
}
.report-section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 24px;
  margin: 24px 0;
  box-shadow: 0 10px 24px rgba(23, 32, 42, 0.06);
}
.meta-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin: 18px 0;
}
.meta-grid div {
  background: var(--band);
  border: 1px solid #dbe3eb;
  border-radius: 6px;
  padding: 10px 12px;
}
.meta-label {
  display: block;
  font-size: 11px;
  font-family: Arial, sans-serif;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
  margin-bottom: 4px;
}
.formula-box {
  background: #132f42;
  color: #f8fbfd;
  border-radius: 6px;
  padding: 13px 15px;
  margin: 16px 0 18px;
}
.formula-label {
  font-size: 11px;
  font-family: Arial, sans-serif;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  opacity: 0.8;
  margin-bottom: 5px;
}
.figure-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
  margin: 18px 0;
}
figure {
  margin: 0;
}
img {
  width: 100%;
  height: auto;
  display: block;
  border: 1px solid var(--line);
  background: #ffffff;
}
figcaption {
  color: var(--muted);
  font-size: 13px;
  margin-top: 7px;
}
.wide-figure {
  max-width: 1000px;
  margin: 18px 0;
}
.reg-table, .compact-table, .target-table {
  width: 100%;
  border-collapse: collapse;
  margin: 14px 0 6px;
  font-size: 13.5px;
}
.table-scroll {
  width: 100%;
  overflow-x: auto;
}
.target-table {
  min-width: 1320px;
  font-size: 12.5px;
}
.reg-table th, .compact-table th, .target-table th {
  background: #16384f;
  color: #ffffff;
  border: 1px solid #102a3d;
  padding: 9px 10px;
  text-align: center;
  vertical-align: bottom;
}
.reg-table td, .compact-table td, .target-table td {
  border: 1px solid var(--line);
  padding: 8px 10px;
  vertical-align: top;
  background: #ffffff;
}
.reg-table td:first-child, .compact-table td:first-child, .target-table td:first-child {
  background: #edf2f7;
  font-weight: 600;
  color: #132f42;
}
.target-table td:first-child { min-width: 210px; }
.se {
  color: var(--muted);
}
.notes {
  color: #33434f;
  font-size: 14px;
  margin-top: 20px;
}
@media (max-width: 900px) {
  .page { padding: 20px; }
  .meta-grid, .figure-grid { grid-template-columns: 1fr; }
  .report-section { padding: 18px; }
}
"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OLS Congestion Regressions</title>
  <style>{css}</style>
</head>
<body>
  <main class="page">
    <header>
      <h1>OLS Evidence on Philippine Inter-Island Congestion and Prices</h1>
      <p class="lead">This report packages the OLS regressions into a single econometric reading file. It excludes the multiplicative count-style specifications and excludes descriptive optional material. Every table uses plain-language labels, reports HC1 robust standard errors in parentheses, and states explicitly whether fixed effects are included.</p>
    </header>

    {''.join(sections)}

    <div class="notes">
      <p>Significance stars: * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01. Unless a row says otherwise, reported estimates are in price units. Demand rows in the progressive and price-level tables are scaled to a 100 MW change.</p>
    </div>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(build_html(), encoding="utf-8")
    print(f"Wrote {output_html}")


if __name__ == "__main__":
    main()
