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

DEFAULT_OUTPUT_HTML = Path("regressions/large_effect_diagnostics.html")
DEFAULT_OUTPUT_CSV = Path("regressions/large_effect_diagnostics.csv")
DEFAULT_COEFF_CSV = Path("regressions/panel_regression_coefficients.csv")
TIMESTAMP_TOKEN_RE = re.compile(r"(\d{8,12})")
CONTROL_COLUMNS = ("losses", "generation", "mkt_import", "mkt_export")
DIRECT_RHS_TERMS = [
    "link_congested_any",
    "equip_cong_any_1",
    "equip_overload_any_1",
    "equip_cong_any_2",
    "equip_overload_any_2",
    *[f"{control}_{side}" for control in CONTROL_COLUMNS for side in ("1", "2")],
]
ISLAND_RHS_TERMS = [
    "interlink_congested_any",
    "equip_cong_any",
    "equip_overload_any",
    *[f"{control}_island" for control in CONTROL_COLUMNS],
]
PAIR_LABELS = {"pooled": "Pooled (Both Pairs)", "cluz_cvis": "Luzon-Visayas", "cvis_cmin": "Visayas-Mindanao"}
ISLAND_LABELS = {
    "pooled": "Pooled (All Islands)",
    "cluz": "Luzon vs System",
    "cvis": "Visayas vs System",
    "cmin": "Mindanao vs System",
}
TERM_LABELS = {
    "link_congested_any": "Pair-Specific Link Congestion",
    "interlink_congested_any": "Any Connected Inter-Link Congestion",
    "equip_cong_any_1": "Equipment Congestion: Island 1",
    "equip_overload_any_1": "Overload > 0: Island 1",
    "equip_cong_any_2": "Equipment Congestion: Island 2",
    "equip_overload_any_2": "Overload > 0: Island 2",
    "equip_cong_any": "Equipment Congestion: Focal Island",
    "equip_overload_any": "Overload > 0: Focal Island",
    "losses_1": "Losses: Island 1 (+100 MW)",
    "losses_2": "Losses: Island 2 (+100 MW)",
    "generation_1": "Generation: Island 1 (+100 MW)",
    "generation_2": "Generation: Island 2 (+100 MW)",
    "mkt_import_1": "Market Imports: Island 1 (+100 MW)",
    "mkt_import_2": "Market Imports: Island 2 (+100 MW)",
    "mkt_export_1": "Market Exports: Island 1 (+100 MW)",
    "mkt_export_2": "Market Exports: Island 2 (+100 MW)",
    "losses_island": "Losses: Focal Island (+100 MW)",
    "generation_island": "Generation: Focal Island (+100 MW)",
    "mkt_import_island": "Market Imports: Focal Island (+100 MW)",
    "mkt_export_island": "Market Exports: Focal Island (+100 MW)",
}
PRIMARY_VARIANTS = (
    "winsor_p99_ppml",
    "trim_top1_ppml",
    "scarcity_excluded_ppml",
    "ols_levels",
    "ols_log1p",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose very large PPML effects from the current regression coefficient export "
            "using support, tail, specification, and fixed-effect-identification checks."
        )
    )
    parser.add_argument("--threshold-pct", type=float, default=1000.0, help="Absolute reported-effect threshold.")
    parser.add_argument("--direct-pair-panel", help="Direct-pair panel parquet path.")
    parser.add_argument("--island-system-panel", help="Island-vs-system panel parquet path.")
    parser.add_argument("--coeff-csv", default=str(DEFAULT_COEFF_CSV), help="Coefficient export CSV path.")
    parser.add_argument("--output-html", default=str(DEFAULT_OUTPUT_HTML), help="Output HTML path.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Output CSV path.")
    parser.add_argument(
        "--winsor-quantiles",
        default="0.99,0.995",
        help="Comma-separated upper-tail winsor quantiles (for example 0.99,0.995).",
    )
    parser.add_argument("--scarcity-cap", type=float, default=30_000.0, help="Scarcity upper price cutoff.")
    parser.add_argument("--scarcity-floor", type=float, default=-9_000.0, help="Scarcity lower price cutoff.")
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


def parse_winsor_quantiles(raw: str) -> list[float]:
    values = sorted({float(value.strip()) for value in raw.split(",") if value.strip()})
    if len(values) < 2:
        raise ValueError("Pass at least two winsor quantiles, for example 0.99,0.995.")
    if any(value <= 0 or value >= 1 for value in values):
        raise ValueError("Winsor quantiles must be strictly between 0 and 1.")
    return values


def load_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["time_interval"] = pd.to_datetime(frame["time_interval"])
    return frame


def format_number(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{value:,.{digits}f}"


def dataframe_to_html_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda x: format_number(float(x), float_digits) if not pd.isna(x) else "")
    return display.to_html(index=False, escape=False, classes=["report-table"])


def latest_panel_path(kind: str) -> Path:
    if kind == "direct":
        return latest_matching_file(Path("data/panels"), "RTD_DIRECT_PAIR_PANEL_*.parquet")
    if kind == "island":
        return latest_matching_file(Path("data/panels"), "RTD_ISLAND_SYSTEM_PANEL_*.parquet")
    raise ValueError(f"Unknown panel kind: {kind}")


def is_binary_term(term: str) -> bool:
    return term in {
        "link_congested_any",
        "interlink_congested_any",
        "equip_cong_any_1",
        "equip_overload_any_1",
        "equip_cong_any_2",
        "equip_overload_any_2",
        "equip_cong_any",
        "equip_overload_any",
    }


def reporting_delta(term: str) -> float | None:
    if is_binary_term(term):
        return 1.0
    if term.startswith(("losses_", "generation_", "mkt_import_", "mkt_export_")):
        return 100.0
    return None


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


def transform_effect(coef: float, std_err: float, term: str) -> tuple[float, float, str]:
    delta = reporting_delta(term)
    if delta is None:
        return np.nan, np.nan, "level_coefficient"
    exponent = float(np.clip(coef * delta, -700, 700))
    exp_term = float(np.exp(exponent))
    effect_pct = 100.0 * (exp_term - 1.0)
    effect_se_pct = 100.0 * exp_term * delta * std_err
    return effect_pct, effect_se_pct, "percent_effect"


def sign_of_value(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def fit_ppml(frame: pd.DataFrame, dep_var: str, rhs_terms: list[str], fixed_effect_terms: list[str]) -> object:
    formula = f"{dep_var} ~ {' + '.join([*rhs_terms, *fixed_effect_terms])}"
    return smf.glm(formula=formula, data=frame, family=sm.families.Poisson()).fit(cov_type="HC1", maxiter=200)


def fit_ols(frame: pd.DataFrame, dep_var: str, rhs_terms: list[str], fixed_effect_terms: list[str]) -> object:
    formula = f"{dep_var} ~ {' + '.join([*rhs_terms, *fixed_effect_terms])}"
    return smf.ols(formula=formula, data=frame).fit(cov_type="HC1")


def build_context(
    flagged_row: pd.Series,
    direct_frame: pd.DataFrame,
    island_frame: pd.DataFrame,
) -> dict[str, object]:
    section = str(flagged_row["section"])
    column_key = str(flagged_row["column_key"])
    if section == "direct_pair":
        if column_key == "pooled":
            frame = direct_frame.copy()
            fixed_effect_terms = ["C(pair_key)", "C(fe_day)"]
            entity_var = "pair_key"
            entity_fe_in_model = True
        else:
            frame = direct_frame.loc[direct_frame["pair_key"] == column_key.upper()].copy()
            fixed_effect_terms = ["C(fe_day)"]
            entity_var = "pair_key"
            entity_fe_in_model = False
        return {
            "section": section,
            "section_title": "Direct Pair Price Gap",
            "column_key": column_key,
            "column_title": PAIR_LABELS[column_key],
            "frame": frame,
            "dep_var": "dep_abs_price_gap",
            "dep_label": "|P_i,t - P_j,t|",
            "rhs_terms": DIRECT_RHS_TERMS,
            "fixed_effect_terms": fixed_effect_terms,
            "entity_var": entity_var,
            "entity_fe_in_model": entity_fe_in_model,
            "day_var": "fe_day",
            "price_cols": ["price_1", "price_2"],
            "congestion_terms": [
                "link_congested_any",
                "equip_cong_any_1",
                "equip_overload_any_1",
                "equip_cong_any_2",
                "equip_overload_any_2",
            ],
            "near_zero_threshold": 0.0,
        }
    if section == "island_vs_system":
        if column_key == "pooled":
            frame = island_frame.copy()
            fixed_effect_terms = ["C(island_code)", "C(fe_day)"]
            entity_var = "island_code"
            entity_fe_in_model = True
        else:
            frame = island_frame.loc[island_frame["island_code"] == column_key.upper()].copy()
            fixed_effect_terms = ["C(fe_day)"]
            entity_var = "island_code"
            entity_fe_in_model = False
        return {
            "section": section,
            "section_title": "Island Against System",
            "column_key": column_key,
            "column_title": ISLAND_LABELS[column_key],
            "frame": frame,
            "dep_var": "dep_price_minus_sys",
            "dep_label": "|P_i,t - P_sys,t|",
            "rhs_terms": ISLAND_RHS_TERMS,
            "fixed_effect_terms": fixed_effect_terms,
            "entity_var": entity_var,
            "entity_fe_in_model": entity_fe_in_model,
            "day_var": "fe_day",
            "price_cols": ["price_island", "price_sys_dw"],
            "congestion_terms": ["interlink_congested_any", "equip_cong_any", "equip_overload_any"],
            "near_zero_threshold": 1.0,
        }
    raise ValueError(f"Unsupported section: {section}")


def summarize_binary_support(frame: pd.DataFrame, dep_var: str, term: str, near_zero_threshold: float) -> tuple[pd.DataFrame, dict[str, float]]:
    summary = (
        frame.groupby(term, observed=True)[dep_var]
        .agg(
            rows="size",
            mean="mean",
            median="median",
            p90=lambda s: float(s.quantile(0.9)),
            p99=lambda s: float(s.quantile(0.99)),
            max="max",
            zero_share=lambda s: float((s == 0).mean()),
            near_zero_share=lambda s: float((s <= near_zero_threshold).mean()),
        )
        .reset_index()
        .rename(columns={term: "state"})
    )
    prevalence = float(frame[term].mean())
    untreated = summary.loc[summary["state"] == 0]
    treated = summary.loc[summary["state"] == 1]
    metrics = {
        "prevalence": prevalence,
        "untreated_mean_outcome": float(untreated["mean"].iloc[0]) if not untreated.empty else np.nan,
        "untreated_zero_share": float(untreated["zero_share"].iloc[0]) if not untreated.empty else np.nan,
        "untreated_near_zero_share": float(untreated["near_zero_share"].iloc[0]) if not untreated.empty else np.nan,
        "treated_mean_outcome": float(treated["mean"].iloc[0]) if not treated.empty else np.nan,
    }
    return summary, metrics


def summarize_continuous_support(frame: pd.DataFrame, dep_var: str, term: str) -> tuple[pd.DataFrame, dict[str, float]]:
    threshold = float(frame[term].quantile(0.9))
    high_flag = frame[term] >= threshold
    summary = pd.DataFrame(
        [
            {
                "metric": "rows",
                "value": float(len(frame)),
            },
            {"metric": "mean", "value": float(frame[term].mean())},
            {"metric": "median", "value": float(frame[term].median())},
            {"metric": "p90", "value": threshold},
            {"metric": "p99", "value": float(frame[term].quantile(0.99))},
            {"metric": "max", "value": float(frame[term].max())},
            {"metric": "zero_share", "value": float((frame[term] == 0).mean())},
            {"metric": "top_decile_share", "value": float(high_flag.mean())},
        ]
    )
    outcome_table = (
        frame.assign(high_regressor=high_flag.astype(int))
        .groupby("high_regressor", observed=True)[dep_var]
        .agg(rows="size", mean="mean", median="median", p90=lambda s: float(s.quantile(0.9)), p99=lambda s: float(s.quantile(0.99)))
        .reset_index()
    )
    metrics = {
        "prevalence": float(high_flag.mean()),
        "untreated_mean_outcome": float(outcome_table.loc[outcome_table["high_regressor"] == 0, "mean"].iloc[0]),
        "untreated_zero_share": np.nan,
        "untreated_near_zero_share": np.nan,
        "treated_mean_outcome": float(outcome_table.loc[outcome_table["high_regressor"] == 1, "mean"].iloc[0]),
        "top_decile_cutoff": threshold,
    }
    return pd.concat([summary, outcome_table], axis=0, ignore_index=True), metrics


def overlap_table_binary(frame: pd.DataFrame, term: str, congestion_terms: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for other in congestion_terms:
        if other == term:
            continue
        rows.append(
            {
                "other_indicator": TERM_LABELS.get(other, other),
                "share_when_term_0": float(frame.loc[frame[term] == 0, other].mean()) if (frame[term] == 0).any() else np.nan,
                "share_when_term_1": float(frame.loc[frame[term] == 1, other].mean()) if (frame[term] == 1).any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def overlap_table_continuous(frame: pd.DataFrame, term: str, congestion_terms: list[str]) -> pd.DataFrame:
    threshold = float(frame[term].quantile(0.9))
    high_flag = frame[term] >= threshold
    rows: list[dict[str, object]] = []
    for other in congestion_terms:
        rows.append(
            {
                "indicator": TERM_LABELS.get(other, other),
                "share_when_regressor_below_top_decile": float(frame.loc[~high_flag, other].mean()) if (~high_flag).any() else np.nan,
                "share_when_regressor_in_top_decile": float(frame.loc[high_flag, other].mean()) if high_flag.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def tail_diagnostics_binary(frame: pd.DataFrame, dep_var: str, term: str, winsor_quantiles: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    overall_share = float(frame[term].mean())
    for quantile in winsor_quantiles:
        cutoff = float(frame[dep_var].quantile(quantile))
        tail_flag = frame[dep_var] >= cutoff
        rows.append(
            {
                "tail": f"Top {(1 - quantile) * 100:.1f}%",
                "outcome_cutoff": cutoff,
                "tail_rows": int(tail_flag.sum()),
                "indicator_share_in_tail": float(frame.loc[tail_flag, term].mean()) if tail_flag.any() else np.nan,
                "indicator_share_overall": overall_share,
            }
        )

    primary_q = winsor_quantiles[0]
    trim_cutoff = float(frame[dep_var].quantile(primary_q))
    tail_summary_rows: list[dict[str, object]] = []
    for variant_name, variant_frame, column_name in (
        ("Original", frame, dep_var),
        (f"Winsor @ {winsor_quantiles[0]:.3f}", frame.assign(dep_variant=frame[dep_var].clip(upper=float(frame[dep_var].quantile(winsor_quantiles[0])))), "dep_variant"),
        (f"Winsor @ {winsor_quantiles[1]:.3f}", frame.assign(dep_variant=frame[dep_var].clip(upper=float(frame[dep_var].quantile(winsor_quantiles[1])))), "dep_variant"),
        ("Trim Top 1%", frame.loc[frame[dep_var] < trim_cutoff].copy(), dep_var),
    ):
        grouped = (
            variant_frame.groupby(term, observed=True)[column_name]
            .agg(mean="mean", median="median", p90=lambda s: float(s.quantile(0.9)), p99=lambda s: float(s.quantile(0.99)))
            .reset_index()
            .rename(columns={term: "state"})
        )
        grouped["variant"] = variant_name
        tail_summary_rows.extend(grouped.to_dict("records"))
    return pd.DataFrame(rows), pd.DataFrame(tail_summary_rows)


def tail_diagnostics_continuous(frame: pd.DataFrame, dep_var: str, term: str, winsor_quantiles: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    reg_top_decile_cutoff = float(frame[term].quantile(0.9))
    high_reg = frame[term] >= reg_top_decile_cutoff
    for quantile in winsor_quantiles:
        cutoff = float(frame[dep_var].quantile(quantile))
        tail_flag = frame[dep_var] >= cutoff
        rows.append(
            {
                "tail": f"Top {(1 - quantile) * 100:.1f}%",
                "outcome_cutoff": cutoff,
                "tail_rows": int(tail_flag.sum()),
                "share_in_regressor_top_decile": float(high_reg.loc[tail_flag].mean()) if tail_flag.any() else np.nan,
                "overall_top_decile_share": float(high_reg.mean()),
                "mean_regressor_in_tail": float(frame.loc[tail_flag, term].mean()) if tail_flag.any() else np.nan,
            }
        )

    primary_q = winsor_quantiles[0]
    trim_cutoff = float(frame[dep_var].quantile(primary_q))
    variant_rows: list[dict[str, object]] = []
    for variant_name, variant_frame, column_name in (
        ("Original", frame, dep_var),
        (f"Winsor @ {winsor_quantiles[0]:.3f}", frame.assign(dep_variant=frame[dep_var].clip(upper=float(frame[dep_var].quantile(winsor_quantiles[0])))), "dep_variant"),
        (f"Winsor @ {winsor_quantiles[1]:.3f}", frame.assign(dep_variant=frame[dep_var].clip(upper=float(frame[dep_var].quantile(winsor_quantiles[1])))), "dep_variant"),
        ("Trim Top 1%", frame.loc[frame[dep_var] < trim_cutoff].copy(), dep_var),
    ):
        variant_rows.append(
            {
                "variant": variant_name,
                "outcome_mean": float(variant_frame[column_name].mean()),
                "outcome_median": float(variant_frame[column_name].median()),
                "outcome_p99": float(variant_frame[column_name].quantile(0.99)),
                "regressor_mean": float(variant_frame[term].mean()),
                "regressor_p99": float(variant_frame[term].quantile(0.99)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(variant_rows)


def fixed_effect_support(frame: pd.DataFrame, term: str, entity_var: str, day_var: str) -> pd.DataFrame:
    if is_binary_term(term):
        day_share = float(frame.groupby(day_var, observed=True)[term].nunique().ge(2).mean())
        entity_share = float(frame.groupby(entity_var, observed=True)[term].nunique().ge(2).mean())
        cell_share = float(frame.groupby([entity_var, day_var], observed=True)[term].nunique().ge(2).mean())
        return pd.DataFrame(
            [
                {"support_check": "Day FE cells with both states", "share": day_share},
                {"support_check": "Entity cells with both states", "share": entity_share},
                {"support_check": "Entity-by-day cells with both states", "share": cell_share},
            ]
        )

    day_share = float(frame.groupby(day_var, observed=True)[term].nunique().gt(1).mean())
    entity_share = float(frame.groupby(entity_var, observed=True)[term].nunique().gt(1).mean())
    cell_share = float(frame.groupby([entity_var, day_var], observed=True)[term].nunique().gt(1).mean())
    return pd.DataFrame(
        [
            {"support_check": "Day FE cells with variation", "share": day_share},
            {"support_check": "Entity cells with variation", "share": entity_share},
            {"support_check": "Entity-by-day cells with variation", "share": cell_share},
        ]
    )


def fit_variant(
    context: dict[str, object],
    term: str,
    variant_name: str,
    frame: pd.DataFrame,
    dep_var: str,
    model_kind: str,
) -> dict[str, object]:
    rhs_terms = select_estimable_terms(frame, list(context["rhs_terms"]))
    included = term in rhs_terms
    nobs = int(len(frame))
    if nobs == 0:
        return {
            "variant_name": variant_name,
            "model_kind": model_kind,
            "nobs": 0,
            "term_in_model": False,
            "coef": np.nan,
            "std_err": np.nan,
            "pvalue": np.nan,
            "effect_pct": np.nan,
            "effect_se_pct": np.nan,
            "estimate_unit": "missing",
            "sign": "missing",
        }

    if model_kind == "ppml":
        result = fit_ppml(frame, dep_var, rhs_terms, list(context["fixed_effect_terms"]))
    elif model_kind == "ols_levels":
        result = fit_ols(frame, dep_var, rhs_terms, list(context["fixed_effect_terms"]))
    elif model_kind == "ols_log1p":
        work = frame.copy()
        work["log1p_dep_variant"] = np.log1p(work[dep_var].astype(float))
        result = fit_ols(work, "log1p_dep_variant", rhs_terms, list(context["fixed_effect_terms"]))
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    if included:
        coef = float(result.params[term])
        std_err = float(result.bse[term])
        pvalue = float(result.pvalues[term])
        if model_kind == "ols_levels":
            effect_pct = np.nan
            effect_se_pct = np.nan
            estimate_unit = "level_coefficient"
        else:
            effect_pct, effect_se_pct, estimate_unit = transform_effect(coef, std_err, term)
        sign = sign_of_value(coef)
    else:
        coef = np.nan
        std_err = np.nan
        pvalue = np.nan
        effect_pct = np.nan
        effect_se_pct = np.nan
        estimate_unit = "dropped"
        sign = "missing"

    return {
        "variant_name": variant_name,
        "model_kind": model_kind,
        "nobs": int(result.nobs),
        "term_in_model": included,
        "coef": coef,
        "std_err": std_err,
        "pvalue": pvalue,
        "effect_pct": effect_pct,
        "effect_se_pct": effect_se_pct,
        "estimate_unit": estimate_unit,
        "sign": sign,
    }


def variant_display(variant_row: dict[str, object]) -> str:
    if not variant_row["term_in_model"]:
        return "Dropped"
    if variant_row["estimate_unit"] == "percent_effect":
        return f"{format_number(float(variant_row['effect_pct']), 2)}%"
    return f"{format_number(float(variant_row['coef']), 4)}"


def classify_effect(
    term: str,
    baseline_variant: dict[str, object],
    variant_rows: list[dict[str, object]],
    support_metrics: dict[str, float],
    fe_support: pd.DataFrame,
) -> tuple[str, bool, int]:
    baseline_sign = str(baseline_variant["sign"])
    primary_rows = [row for row in variant_rows if row["variant_name"] in PRIMARY_VARIANTS]
    sign_matches = sum(row["sign"] == baseline_sign for row in primary_rows if row["sign"] != "missing")
    sign_flip = any(row["sign"] not in {baseline_sign, "missing"} for row in primary_rows)
    fe_cell_share = float(fe_support.loc[fe_support["support_check"].str.contains("Entity-by-day"), "share"].iloc[0])

    support_warning = False
    if is_binary_term(term):
        untreated_mean = float(support_metrics["untreated_mean_outcome"])
        untreated_zero_share = float(support_metrics["untreated_zero_share"])
        support_warning = untreated_mean < 1.0 or untreated_zero_share > 0.95 or fe_cell_share < 0.10
    else:
        support_warning = fe_cell_share < 0.10

    baseline_effect = float(baseline_variant["effect_pct"])
    tail_sensitive = False
    if not pd.isna(baseline_effect) and baseline_effect != 0:
        for row in variant_rows:
            if row["variant_name"] not in {"winsor_p99_ppml", "winsor_p995_ppml", "trim_top1_ppml", "scarcity_excluded_ppml"}:
                continue
            if row["sign"] != baseline_sign or pd.isna(row["effect_pct"]):
                continue
            if abs(float(row["effect_pct"])) < 0.25 * abs(baseline_effect):
                tail_sensitive = True
                break

    if sign_flip:
        return "Not robust", support_warning, sign_matches
    if support_warning:
        return "Sparse-support", support_warning, sign_matches
    if tail_sensitive:
        return "Tail-sensitive", support_warning, sign_matches
    if sign_matches >= 4:
        return "Stable", support_warning, sign_matches
    return "Not robust", support_warning, sign_matches


def derive_mechanism(
    term: str,
    classification: str,
    support_metrics: dict[str, float],
    tail_table: pd.DataFrame,
) -> str:
    if is_binary_term(term):
        untreated_mean = float(support_metrics["untreated_mean_outcome"])
        untreated_zero_share = float(support_metrics["untreated_zero_share"])
        if untreated_mean < 1.0 or untreated_zero_share > 0.95:
            return (
                "The untreated baseline is nearly degenerate, so even moderate treated-state gaps "
                "translate into extremely large percent effects."
            )
        if "indicator_share_in_tail" in tail_table.columns and not tail_table.empty:
            top_share = float(tail_table["indicator_share_in_tail"].max())
            overall_share = float(tail_table["indicator_share_overall"].max())
            if top_share > 0.9 and top_share > overall_share:
                return (
                    "The indicator is heavily concentrated in the extreme right tail of the outcome, "
                    "so the estimate is largely tracking scarcity-like tail episodes."
                )
    else:
        if "share_in_regressor_top_decile" in tail_table.columns and not tail_table.empty:
            top_share = float(tail_table["share_in_regressor_top_decile"].max())
            overall_share = float(tail_table["overall_top_decile_share"].max())
            if top_share > overall_share + 0.3:
                return (
                    "The large continuous-control effect is tied to the outcome tail: extreme gaps are "
                    "disproportionately concentrated when the regressor is in its upper tail."
                )

    if classification == "Tail-sensitive":
        return "The large effect weakens sharply once the top tail or scarcity-style intervals are handled differently."
    if classification == "Not robust":
        return "The sign or inclusion of the coefficient is unstable across alternative specifications."
    if classification == "Stable":
        return "The large effect remains directionally consistent across the main robustness checks."
    return "The effect appears to be supported by only a thin slice of untreated or within-cell variation."


def build_specification_rows(
    context: dict[str, object],
    term: str,
    winsor_quantiles: list[float],
    scarcity_cap: float,
    scarcity_floor: float,
) -> list[dict[str, object]]:
    frame = context["frame"]
    dep_var = str(context["dep_var"])
    rows: list[dict[str, object]] = []

    baseline = fit_variant(context, term, "baseline_ppml", frame.copy(), dep_var, "ppml")
    rows.append(baseline)

    for quantile in winsor_quantiles:
        work = frame.copy()
        cutoff = float(work[dep_var].quantile(quantile))
        work[dep_var] = work[dep_var].clip(upper=cutoff)
        suffix = "99" if np.isclose(quantile, 0.99) else "995" if np.isclose(quantile, 0.995) else str(quantile).replace(".", "")
        variant_name = f"winsor_p{suffix}_ppml"
        rows.append(fit_variant(context, term, variant_name, work, dep_var, "ppml"))

    trim_cutoff = float(frame[dep_var].quantile(winsor_quantiles[0]))
    trimmed = frame.loc[frame[dep_var] < trim_cutoff].copy()
    rows.append(fit_variant(context, term, "trim_top1_ppml", trimmed, dep_var, "ppml"))

    scarcity_mask = np.zeros(len(frame), dtype=bool)
    for price_col in context["price_cols"]:
        scarcity_mask |= (frame[price_col] >= scarcity_cap) | (frame[price_col] <= scarcity_floor)
    scarcity_excluded = frame.loc[~scarcity_mask].copy()
    rows.append(fit_variant(context, term, "scarcity_excluded_ppml", scarcity_excluded, dep_var, "ppml"))

    rows.append(fit_variant(context, term, "ols_levels", frame.copy(), dep_var, "ols_levels"))
    rows.append(fit_variant(context, term, "ols_log1p", frame.copy(), dep_var, "ols_log1p"))
    return rows


def specification_table(variant_rows: list[dict[str, object]], baseline_sign: str) -> pd.DataFrame:
    label_map = {
        "baseline_ppml": "Baseline PPML",
        "winsor_p990_ppml": "Winsorized PPML @ p99",
        "winsor_p995_ppml": "Winsorized PPML @ p99.5",
        "trim_top1_ppml": "Trim Top 1% PPML",
        "scarcity_excluded_ppml": "Scarcity-Excluded PPML",
        "ols_levels": "OLS Levels",
        "ols_log1p": "OLS log(1 + y)",
    }
    rows: list[dict[str, object]] = []
    for row in variant_rows:
        rows.append(
            {
                "variant": label_map.get(row["variant_name"], row["variant_name"]),
                "term_in_model": "Yes" if row["term_in_model"] else "No",
                "estimate": variant_display(row),
                "estimate_unit": row["estimate_unit"],
                "coef": row["coef"],
                "std_err": row["std_err"],
                "pvalue": row["pvalue"],
                "nobs": row["nobs"],
                "sign": row["sign"],
                "matches_baseline_sign": "Yes" if row["sign"] == baseline_sign else "No",
            }
        )
    return pd.DataFrame(rows)


def flagged_summary_table(flagged: pd.DataFrame, diagnostics: list[dict[str, object]]) -> pd.DataFrame:
    diag_lookup = {(row["section"], row["column_key"], row["term"]): row for row in diagnostics}
    rows: list[dict[str, object]] = []
    for _, flagged_row in flagged.iterrows():
        key = (str(flagged_row["section"]), str(flagged_row["column_key"]), str(flagged_row["term"]))
        diag = diag_lookup[key]
        rows.append(
            {
                "specification": f"{flagged_row['section']} / {flagged_row['column_key']}",
                "term": TERM_LABELS.get(str(flagged_row["term"]), str(flagged_row["term"])),
                "baseline_effect_pct": float(flagged_row["reported_effect_pct"]),
                "classification": diag["classification"],
                "mechanism": diag["mechanism"],
                "prevalence_or_top_decile_share": diag["support_metrics"]["prevalence"],
                "untreated_mean_outcome": diag["support_metrics"]["untreated_mean_outcome"],
            }
        )
    return pd.DataFrame(rows)


def render_html(
    output_path: Path,
    flagged: pd.DataFrame,
    diagnostics: list[dict[str, object]],
    summary_table: pd.DataFrame,
    threshold_pct: float,
    winsor_quantiles: list[float],
) -> None:
    css = """
body { font-family: Georgia, "Times New Roman", serif; margin: 32px auto; max-width: 1440px; color: #102a43; line-height: 1.55; background: #f4f1ea; }
h1, h2, h3 { color: #0b1f33; }
h1 { margin-bottom: 10px; }
h2 { margin: 22px 0 12px; }
h3 { margin: 18px 0 10px; }
p { margin: 0 0 14px; }
code { background: #dde7f0; color: #0b1f33; padding: 2px 5px; border-radius: 4px; }
.lead { font-size: 17px; color: #243b53; margin-bottom: 22px; }
.summary-card, .diag-card { background: #faf7f1; border: 1px solid #d9e2ec; border-radius: 14px; padding: 22px 24px; margin: 22px 0 28px; box-shadow: 0 10px 28px rgba(16, 42, 67, 0.08); }
.meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 20px; margin: 0 0 14px; }
.meta-grid div { background: #eef3f7; border-radius: 10px; padding: 10px 12px; }
.meta-label { display: block; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #486581; margin-bottom: 4px; }
.verdict { font-size: 16px; color: #102a43; background: #e6ecf2; border-left: 6px solid #486581; padding: 12px 14px; border-radius: 10px; margin-bottom: 18px; }
.report-table { border-collapse: collapse; width: 100%; margin: 14px 0 18px; font-size: 14px; box-shadow: 0 8px 24px rgba(16, 42, 67, 0.10); }
.report-table th { background: #0b1f33; color: #fdfdfd; padding: 10px 12px; text-align: center; border: 1px solid #102a43; }
.report-table td { border: 1px solid #bcccdc; padding: 8px 12px; vertical-align: top; background: #fffdf8; color: #102a43; }
.report-table td:first-child { font-weight: 600; background: #e6ecf2; color: #0b1f33; }
.report-table tr:nth-child(even) td:not(:first-child) { background: #eef3f7; }
"""
    cards: list[str] = []
    for diag in diagnostics:
        support_html = dataframe_to_html_table(diag["support_table"], float_digits=4)
        overlap_html = dataframe_to_html_table(diag["overlap_table"], float_digits=4)
        tail_html = dataframe_to_html_table(diag["tail_table"], float_digits=4)
        tail_outcome_html = dataframe_to_html_table(diag["tail_outcome_table"], float_digits=4)
        spec_html = dataframe_to_html_table(diag["spec_table"], float_digits=4)
        fe_html = dataframe_to_html_table(diag["fe_support"], float_digits=4)
        cards.append(
            f"""
  <section class="diag-card">
    <h2>{html.escape(diag['display_title'])}</h2>
    <div class="verdict"><strong>{html.escape(diag['classification'])}</strong>: {html.escape(diag['mechanism'])}</div>
    <div class="meta-grid">
      <div><span class="meta-label">Baseline effect</span><span>{html.escape(format_number(float(diag['baseline_effect_pct']), 2))}%</span></div>
      <div><span class="meta-label">Sample</span><span>{html.escape(diag['sample_title'])}</span></div>
      <div><span class="meta-label">Outcome</span><span>{html.escape(diag['dep_label'])}</span></div>
      <div><span class="meta-label">Sign matches</span><span>{html.escape(str(diag['sign_match_count']))} of 5 primary variants</span></div>
    </div>
    <h3>Support Diagnostics</h3>
    {support_html}
    <h3>Overlap With Other Congestion Indicators</h3>
    {overlap_html}
    <h3>Tail Diagnostics</h3>
    {tail_html}
    <h3>Outcome After Winsorizing / Trimming</h3>
    {tail_outcome_html}
    <h3>Specification Diagnostics</h3>
    {spec_html}
    <h3>Fixed-Effect Identification Diagnostics</h3>
    {fe_html}
  </section>
"""
        )

    html_body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Large Effect Diagnostics</title>
  <style>{css}</style>
</head>
<body>
  <h1>Large Effect Diagnostics</h1>
  <p class="lead">This report flags coefficients from <code>{html.escape(str(DEFAULT_COEFF_CSV))}</code> whose absolute reported effect exceeds <code>{html.escape(format_number(threshold_pct, 0))}%</code>, then diagnoses whether the effect is coming from sparse support, tail concentration, scarcity-style intervals, or a more stable pattern. Winsor checks use <code>{html.escape(', '.join(str(q) for q in winsor_quantiles))}</code>.</p>

  <section class="summary-card">
    <h2>Flagged Coefficients</h2>
    <p>{len(flagged):,} coefficients exceeded the threshold out of {flagged['total_coefficients'].iloc[0]:,} non-intercept reported rows.</p>
    {dataframe_to_html_table(summary_table, float_digits=4)}
  </section>

  {''.join(cards)}
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    winsor_quantiles = parse_winsor_quantiles(args.winsor_quantiles)
    direct_panel_path = Path(args.direct_pair_panel) if args.direct_pair_panel else latest_panel_path("direct")
    island_panel_path = Path(args.island_system_panel) if args.island_system_panel else latest_panel_path("island")
    coeff_path = Path(args.coeff_csv)
    output_html = Path(args.output_html)
    output_csv = Path(args.output_csv)

    direct_frame = load_panel(direct_panel_path)
    island_frame = load_panel(island_panel_path)
    coeff = pd.read_csv(coeff_path)

    non_intercept = coeff.loc[coeff["term"] != "Intercept"].copy()
    total_coefficients = int(len(non_intercept))
    flagged = non_intercept.loc[non_intercept["reported_effect_pct"].abs() > float(args.threshold_pct)].copy()
    if flagged.empty:
        raise ValueError(f"No coefficients exceeded the threshold of {args.threshold_pct} percent.")
    flagged["total_coefficients"] = total_coefficients
    flagged = flagged.sort_values(["section", "column_key", "term"]).reset_index(drop=True)

    diagnostics: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []

    for _, flagged_row in flagged.iterrows():
        context = build_context(flagged_row, direct_frame, island_frame)
        term = str(flagged_row["term"])
        sample_frame = context["frame"]
        display_title = f"{context['section_title']} / {context['column_title']} / {TERM_LABELS.get(term, term)}"

        if is_binary_term(term):
            support_table, support_metrics = summarize_binary_support(
                sample_frame,
                str(context["dep_var"]),
                term,
                float(context["near_zero_threshold"]),
            )
            overlap_table = overlap_table_binary(sample_frame, term, list(context["congestion_terms"]))
            tail_table, tail_outcome_table = tail_diagnostics_binary(sample_frame, str(context["dep_var"]), term, winsor_quantiles)
        else:
            support_table, support_metrics = summarize_continuous_support(sample_frame, str(context["dep_var"]), term)
            overlap_table = overlap_table_continuous(sample_frame, term, list(context["congestion_terms"]))
            tail_table, tail_outcome_table = tail_diagnostics_continuous(sample_frame, str(context["dep_var"]), term, winsor_quantiles)

        fe_support = fixed_effect_support(sample_frame, term, str(context["entity_var"]), str(context["day_var"]))
        variant_rows = build_specification_rows(
            context,
            term,
            winsor_quantiles,
            float(args.scarcity_cap),
            float(args.scarcity_floor),
        )
        baseline_variant = next(row for row in variant_rows if row["variant_name"] == "baseline_ppml")
        classification, support_warning, sign_match_count = classify_effect(
            term,
            baseline_variant,
            variant_rows,
            support_metrics,
            fe_support,
        )
        mechanism = derive_mechanism(term, classification, support_metrics, tail_table)
        spec_table = specification_table(variant_rows, str(baseline_variant["sign"]))

        diagnostics.append(
            {
                "section": str(flagged_row["section"]),
                "column_key": str(flagged_row["column_key"]),
                "term": term,
                "display_title": display_title,
                "sample_title": str(context["column_title"]),
                "dep_label": str(context["dep_label"]),
                "classification": classification,
                "mechanism": mechanism,
                "support_metrics": support_metrics,
                "baseline_effect_pct": float(flagged_row["reported_effect_pct"]),
                "support_table": support_table,
                "overlap_table": overlap_table,
                "tail_table": tail_table,
                "tail_outcome_table": tail_outcome_table,
                "fe_support": fe_support,
                "spec_table": spec_table,
                "sign_match_count": sign_match_count,
            }
        )

        fe_map = {row["support_check"]: float(row["share"]) for _, row in fe_support.iterrows()}
        for variant_row in variant_rows:
            csv_rows.append(
                {
                    "section": flagged_row["section"],
                    "column_key": flagged_row["column_key"],
                    "term": term,
                    "sample_title": context["column_title"],
                    "classification": classification,
                    "mechanism": mechanism,
                    "support_warning": support_warning,
                    "variant_name": variant_row["variant_name"],
                    "model_kind": variant_row["model_kind"],
                    "nobs": variant_row["nobs"],
                    "term_in_model": variant_row["term_in_model"],
                    "coef": variant_row["coef"],
                    "std_err": variant_row["std_err"],
                    "pvalue": variant_row["pvalue"],
                    "effect_pct": variant_row["effect_pct"],
                    "effect_se_pct": variant_row["effect_se_pct"],
                    "estimate_unit": variant_row["estimate_unit"],
                    "sign": variant_row["sign"],
                    "matches_baseline_sign": variant_row["sign"] == baseline_variant["sign"],
                    "baseline_effect_pct": float(flagged_row["reported_effect_pct"]),
                    "baseline_sign": baseline_variant["sign"],
                    "prevalence_or_top_decile_share": support_metrics["prevalence"],
                    "untreated_mean_outcome": support_metrics["untreated_mean_outcome"],
                    "untreated_zero_share": support_metrics["untreated_zero_share"],
                    "untreated_near_zero_share": support_metrics["untreated_near_zero_share"],
                    "treated_mean_outcome": support_metrics["treated_mean_outcome"],
                    "day_support_share": fe_map.get("Day FE cells with both states", fe_map.get("Day FE cells with variation", np.nan)),
                    "entity_support_share": fe_map.get("Entity cells with both states", fe_map.get("Entity cells with variation", np.nan)),
                    "entity_day_support_share": fe_map.get(
                        "Entity-by-day cells with both states",
                        fe_map.get("Entity-by-day cells with variation", np.nan),
                    ),
                    "tail_top1_share": (
                        float(tail_table.loc[tail_table["tail"] == "Top 1.0%", "indicator_share_in_tail"].iloc[0])
                        if "indicator_share_in_tail" in tail_table.columns and (tail_table["tail"] == "Top 1.0%").any()
                        else float(tail_table.loc[tail_table["tail"] == "Top 1.0%", "share_in_regressor_top_decile"].iloc[0])
                        if "share_in_regressor_top_decile" in tail_table.columns and (tail_table["tail"] == "Top 1.0%").any()
                        else np.nan
                    ),
                    "tail_top0_5_share": (
                        float(tail_table.loc[tail_table["tail"] == "Top 0.5%", "indicator_share_in_tail"].iloc[0])
                        if "indicator_share_in_tail" in tail_table.columns and (tail_table["tail"] == "Top 0.5%").any()
                        else float(tail_table.loc[tail_table["tail"] == "Top 0.5%", "share_in_regressor_top_decile"].iloc[0])
                        if "share_in_regressor_top_decile" in tail_table.columns and (tail_table["tail"] == "Top 0.5%").any()
                        else np.nan
                    ),
                }
            )

    summary_table = flagged_summary_table(flagged, diagnostics)
    csv_frame = pd.DataFrame(csv_rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_frame.to_csv(output_csv, index=False)
    render_html(output_html, flagged, diagnostics, summary_table, float(args.threshold_pct), winsor_quantiles)

    print(f"Wrote {output_html}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
