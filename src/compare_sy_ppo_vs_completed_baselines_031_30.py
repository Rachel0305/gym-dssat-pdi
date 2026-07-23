from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_30_sy_ppo_vs_completed_baselines.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def df_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return ""
    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows)
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def idxmax_row(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    idx = values.idxmax()
    return df.loc[idx]


def baseline_envelope(baseline: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, group in baseline.groupby("year"):
        group = group.copy()
        yield_row = idxmax_row(group, "grain_yield_kg_ha")
        wp_row = idxmax_row(group, "WP_ET_kg_m3")
        pfp_group = group[pd.to_numeric(group["PFP_N_kg_kg"], errors="coerce").notna()].copy()
        pfp_row = idxmax_row(pfp_group, "PFP_N_kg_kg") if not pfp_group.empty else pd.Series(dtype=object)
        official = group[group["scenario"].eq("official_extension_expert")]
        recorded_templates = group[group["scenario"].astype(str).str.startswith("recorded_farmer_template_")]
        best_recorded = idxmax_row(recorded_templates, "grain_yield_kg_ha") if not recorded_templates.empty else pd.Series(dtype=object)
        rows.append(
            {
                "year": int(year),
                "baseline_count": int(len(group)),
                "max_baseline_yield": float(yield_row["grain_yield_kg_ha"]),
                "max_baseline_yield_scenario": str(yield_row["scenario"]),
                "max_baseline_wp_et": float(wp_row["WP_ET_kg_m3"]),
                "max_baseline_wp_et_scenario": str(wp_row["scenario"]),
                "max_baseline_pfp_n": float(pfp_row["PFP_N_kg_kg"]) if not pfp_row.empty else np.nan,
                "max_baseline_pfp_n_scenario": str(pfp_row["scenario"]) if not pfp_row.empty else "",
                "official_expert_irrigation": float(official["actual_irrigation_mm"].iloc[0]) if not official.empty else np.nan,
                "official_expert_n": float(official["actual_nitrogen_kg_ha"].iloc[0]) if not official.empty else np.nan,
                "best_recorded_template_scenario": str(best_recorded["scenario"]) if not best_recorded.empty else "",
                "best_recorded_template_yield": float(best_recorded["grain_yield_kg_ha"]) if not best_recorded.empty else np.nan,
                "best_recorded_template_irrigation": float(best_recorded["actual_irrigation_mm"]) if not best_recorded.empty else np.nan,
                "best_recorded_template_n": float(best_recorded["actual_nitrogen_kg_ha"]) if not best_recorded.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def compare(candidates: pd.DataFrame, envelope: pd.DataFrame) -> pd.DataFrame:
    merged = candidates.merge(envelope, on="year", how="left", validate="many_to_one")
    merged["yield_delta_vs_best_baseline"] = pd.to_numeric(merged["final_grain_kg_ha"], errors="coerce") - merged["max_baseline_yield"]
    merged["wp_et_delta_vs_best_baseline"] = pd.to_numeric(merged["wp_et_kg_m3"], errors="coerce") - merged["max_baseline_wp_et"]
    merged["pfp_n_delta_vs_best_baseline"] = pd.to_numeric(merged["pfp_n_kg_kg"], errors="coerce") - merged["max_baseline_pfp_n"]
    merged["yield_winner"] = merged["yield_delta_vs_best_baseline"] > 0
    merged["wp_et_winner"] = merged["wp_et_delta_vs_best_baseline"] > 0
    merged["pfp_n_winner"] = merged["pfp_n_delta_vs_best_baseline"] > 0
    merged["any_metric_winner"] = merged[["yield_winner", "wp_et_winner", "pfp_n_winner"]].any(axis=1)
    merged["all_three_winner"] = merged[["yield_winner", "wp_et_winner", "pfp_n_winner"]].all(axis=1)
    merged["n_winning_metrics"] = merged[["yield_winner", "wp_et_winner", "pfp_n_winner"]].sum(axis=1).astype(int)
    merged["water_saving_vs_official_expert"] = merged["official_expert_irrigation"] - pd.to_numeric(merged["total_irrigation"], errors="coerce")
    merged["n_saving_vs_official_expert"] = merged["official_expert_n"] - pd.to_numeric(merged["total_n"], errors="coerce")
    merged["water_saving_vs_best_recorded_template"] = merged["best_recorded_template_irrigation"] - pd.to_numeric(merged["total_irrigation"], errors="coerce")
    merged["n_saving_vs_best_recorded_template"] = merged["best_recorded_template_n"] - pd.to_numeric(merged["total_n"], errors="coerce")
    return merged


def summarize_by_seed(compared: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, group in compared.groupby("seed"):
        rows.append(
            {
                "seed": int(seed),
                "year_count": int(group["year"].nunique()),
                "any_metric_winner_years": int(group["any_metric_winner"].sum()),
                "yield_winner_years": int(group["yield_winner"].sum()),
                "wp_et_winner_years": int(group["wp_et_winner"].sum()),
                "pfp_n_winner_years": int(group["pfp_n_winner"].sum()),
                "all_three_winner_years": int(group["all_three_winner"].sum()),
                "mean_yield_delta_vs_best_baseline": float(group["yield_delta_vs_best_baseline"].mean()),
                "mean_wp_et_delta_vs_best_baseline": float(group["wp_et_delta_vs_best_baseline"].mean()),
                "mean_pfp_n_delta_vs_best_baseline": float(group["pfp_n_delta_vs_best_baseline"].mean()),
                "mean_water_saving_vs_official_expert": float(group["water_saving_vs_official_expert"].mean()),
                "mean_n_saving_vs_official_expert": float(group["n_saving_vs_official_expert"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("seed")


def best_candidate_by_year(compared: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sort_cols = ["n_winning_metrics", "any_metric_winner", "yield_delta_vs_best_baseline", "wp_et_delta_vs_best_baseline", "pfp_n_delta_vs_best_baseline"]
    for year, group in compared.groupby("year"):
        best = group.sort_values(sort_cols, ascending=[False, False, False, False, False]).iloc[0]
        rows.append(best)
    return pd.DataFrame(rows).sort_values("year")


def main() -> None:
    cfg = load_yaml(CONFIG)
    out = ROOT / cfg["output_root"]
    if out.exists() and any(out.glob("evaluation/*.csv")):
        raise FileExistsError(f"Existing 031_30 outputs found, refusing overwrite: {out}")
    (out / "evaluation").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs").mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(ROOT / cfg["candidate_summary_csv"], keep_default_na=False)
    baselines = pd.read_csv(ROOT / cfg["baseline_summary_csv"], keep_default_na=False)
    if len(candidates) != int(cfg["expected_candidate_rows"]):
        raise RuntimeError(f"Expected {cfg['expected_candidate_rows']} candidate rows, got {len(candidates)}")
    if len(baselines) != int(cfg["expected_baseline_rows"]):
        raise RuntimeError(f"Expected {cfg['expected_baseline_rows']} baseline rows, got {len(baselines)}")
    env = baseline_envelope(baselines)
    compared = compare(candidates, env)
    seed_summary = summarize_by_seed(compared)
    best_year = best_candidate_by_year(compared)
    env.to_csv(out / "evaluation" / "031_30_baseline_envelope_by_year.csv", index=False, encoding="utf-8-sig")
    compared.to_csv(out / "evaluation" / "031_30_candidate_vs_baseline_envelope.csv", index=False, encoding="utf-8-sig")
    seed_summary.to_csv(out / "evaluation" / "031_30_seed_level_summary.csv", index=False, encoding="utf-8-sig")
    best_year.to_csv(out / "evaluation" / "031_30_year_level_best_candidate.csv", index=False, encoding="utf-8-sig")
    year_success = {
        "years_with_any_seed_any_metric_winner": int(best_year["any_metric_winner"].sum()),
        "years_with_best_seed_yield_winner": int(best_year["yield_winner"].sum()),
        "years_with_best_seed_wp_et_winner": int(best_year["wp_et_winner"].sum()),
        "years_with_best_seed_pfp_n_winner": int(best_year["pfp_n_winner"].sum()),
        "years_with_best_seed_all_three_winner": int(best_year["all_three_winner"].sum()),
        "total_years": int(best_year["year"].nunique()),
    }
    (out / "evaluation" / "031_30_result.json").write_text(json.dumps(year_success, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# 031_30 SY PPO versus completed baselines record",
        "",
        "## Scope",
        "",
        "- No training and no DSSAT rerun.",
        "- 031_27 PPO candidates are compared against the 031_29 generated baseline envelope.",
        "- `recorded_farmer_template_*` scenarios are treated as counterfactual historical-management transfers, not real recorded-farmer observations for the target year.",
        "",
        "## Row-count checks",
        "",
        f"- Candidate rows: {len(candidates)} / expected {cfg['expected_candidate_rows']}.",
        f"- Baseline rows: {len(baselines)} / expected {cfg['expected_baseline_rows']}.",
        "",
        "## Best-candidate year-level outcome",
        "",
        df_to_markdown(pd.DataFrame([year_success]), 10),
        "",
        "## Seed-level summary",
        "",
        df_to_markdown(seed_summary, 20),
        "",
        "## Best candidate by year",
        "",
        df_to_markdown(
            best_year[
                [
                    "year",
                    "seed",
                    "checkpoint_step",
                    "final_grain_kg_ha",
                    "wp_et_kg_m3",
                    "pfp_n_kg_kg",
                    "total_irrigation",
                    "total_n",
                    "n_winning_metrics",
                    "yield_winner",
                    "wp_et_winner",
                    "pfp_n_winner",
                    "yield_delta_vs_best_baseline",
                    "wp_et_delta_vs_best_baseline",
                    "pfp_n_delta_vs_best_baseline",
                ]
            ],
            30,
        ),
    ]
    doc = ROOT / "docs" / "031_30_sy_ppo_vs_completed_baselines_record.md"
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", **year_success}, ensure_ascii=False))


if __name__ == "__main__":
    main()
