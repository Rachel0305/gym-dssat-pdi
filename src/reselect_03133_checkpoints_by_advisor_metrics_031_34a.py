from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "031_34a_reselect_03133_checkpoints_by_advisor_metrics"
DOC = ROOT / "docs" / "031_34a_reselect_03133_checkpoints_by_advisor_metrics_record.md"
CKPT_EVAL = ROOT / "benchmark_results" / "031_33_four_site_free_timing_maskableppo_checkpoint_selection" / "evaluation" / "031_33_checkpoint_eval_summary.csv"
BASELINE = ROOT / "benchmark_results" / "028_12_screened_year_representative_advisor_package" / "028_12_all_representative_five_scenario_summary.csv"


STATION_TO_SITE = {"HLA": "HLA", "FQA": "FQ", "LCA": "LC", "YCA": "YC"}
BASELINE_SCENARIOS = ["null", "recorded_farmer", "dssat_auto", "official_extension_expert"]


def ensure_dirs() -> None:
    (OUT / "evaluation").mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def baseline_envelope(selected: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(BASELINE, keep_default_na=False)
    base = base[base["scenario"].isin(BASELINE_SCENARIOS)].copy()
    base = to_num(base, ["year", "final_grain_kg_ha", "irrigation_event_total_mm", "nitrogen_event_total_kg_ha", "wp_et_kg_m3", "pfp_n_kg_kg"])
    wanted = set(zip(selected["site"], selected["year"].astype(int)))
    base = base[[((row.site, int(row.year)) in wanted) for row in base.itertuples(index=False)]].copy()
    rows: list[dict[str, Any]] = []
    for (site, year), group in base.groupby(["site", "year"]):
        g = group.copy()
        if len(g) < 4:
            raise RuntimeError(f"Baseline rows incomplete for {site}{year}: {len(g)}")

        def max_metric(metric: str) -> tuple[float, str]:
            vals = pd.to_numeric(g[metric], errors="coerce")
            idx = vals.idxmax()
            return float(vals.loc[idx]), str(g.loc[idx, "scenario"])

        max_y, max_y_s = max_metric("final_grain_kg_ha")
        max_wp, max_wp_s = max_metric("wp_et_kg_m3")
        pfp_valid = g[pd.to_numeric(g["pfp_n_kg_kg"], errors="coerce").notna()].copy()
        if pfp_valid.empty:
            max_pfp, max_pfp_s = math.nan, ""
        else:
            idx = pd.to_numeric(pfp_valid["pfp_n_kg_kg"], errors="coerce").idxmax()
            max_pfp, max_pfp_s = float(pfp_valid.loc[idx, "pfp_n_kg_kg"]), str(pfp_valid.loc[idx, "scenario"])
        expert = g[g["scenario"].eq("official_extension_expert")].iloc[0]
        rows.append(
            {
                "site": site,
                "year": int(year),
                "baseline_max_yield": max_y,
                "baseline_max_yield_scenario": max_y_s,
                "baseline_max_wp_et": max_wp,
                "baseline_max_wp_et_scenario": max_wp_s,
                "baseline_max_pfp_n": max_pfp,
                "baseline_max_pfp_n_scenario": max_pfp_s,
                "expert_yield": float(expert["final_grain_kg_ha"]),
                "expert_irrigation": float(expert["irrigation_event_total_mm"]),
                "expert_n": float(expert["nitrogen_event_total_kg_ha"]),
                "expert_wp_et": float(expert["wp_et_kg_m3"]),
                "expert_pfp_n": float(expert["pfp_n_kg_kg"]) if pd.notna(expert["pfp_n_kg_kg"]) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def add_gaps(eval_df: pd.DataFrame, env: pd.DataFrame) -> pd.DataFrame:
    df = eval_df.copy()
    df["site"] = df["station_code"].map(STATION_TO_SITE)
    df = to_num(df, ["year", "seed", "checkpoint_step", "final_grnwt", "total_irrigation", "total_n", "PFP_N", "literature_reward_sum"])
    out = df.merge(env, on=["site", "year"], how="left")
    out["gap_yield"] = out["final_grnwt"] - out["baseline_max_yield"]
    out["gap_pfp_n"] = out["PFP_N"] - out["baseline_max_pfp_n"]
    out["yield_win"] = out["gap_yield"] > 0
    out["pfp_n_win"] = out["gap_pfp_n"] > 0
    out["known_any_metric_win"] = out["yield_win"] | out["pfp_n_win"]
    out["gap_irrigation_vs_expert"] = out["total_irrigation"] - out["expert_irrigation"]
    out["gap_n_vs_expert"] = out["total_n"] - out["expert_n"]
    out["known_winning_metrics"] = out.apply(
        lambda r: ";".join([name for name, flag in [("yield", bool(r["yield_win"])), ("PFP_N", bool(r["pfp_n_win"]))] if flag]),
        axis=1,
    )
    y_score = out["gap_yield"] / out["baseline_max_yield"]
    pfp_score = out["gap_pfp_n"] / out["baseline_max_pfp_n"]
    out["advisor_known_metric_score"] = pd.concat([y_score, pfp_score], axis=1).max(axis=1)
    return out


def select_by_advisor(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    usable = df[df["run_status"].astype(str).str.startswith("ok")].copy()
    for (station, seed), group in usable.groupby(["station_code", "seed"]):
        g = group.copy()
        g = g.sort_values(
            by=[
                "known_any_metric_win",
                "advisor_known_metric_score",
                "gap_yield",
                "total_n",
                "total_irrigation",
                "checkpoint_step",
            ],
            ascending=[False, False, False, True, True, True],
        )
        row = g.iloc[0].to_dict()
        row["advisor_reselection_reason"] = (
            "advisor_known_metric_winner" if bool(row["known_any_metric_win"]) else "no_known_metric_win_highest_known_score"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_record(all_df: pd.DataFrame, selected: pd.DataFrame, station_counts: pd.DataFrame, env: pd.DataFrame) -> None:
    cols = [
        "station_code",
        "year",
        "seed",
        "checkpoint_step",
        "final_grnwt",
        "baseline_max_yield",
        "gap_yield",
        "total_irrigation",
        "total_n",
        "PFP_N",
        "baseline_max_pfp_n",
        "gap_pfp_n",
        "known_any_metric_win",
        "known_winning_metrics",
        "advisor_reselection_reason",
        "action_sequence",
    ]
    lines = [
        "# 031_34a Reselect 031_33 checkpoints by advisor metrics",
        "",
        "## Scope",
        "",
        "- No training.",
        "- No DSSAT rerun.",
        "- Re-screen all 031_33 checkpoints using advisor-facing known metrics.",
        "- Known metrics here are yield and PFP_N only; WP_ET is unavailable in 031_33 because ETCP/snapshot was not saved.",
        "",
        "## Baseline envelope",
        "",
        env.to_string(index=False),
        "",
        "## Advisor-reselected checkpoint per station-seed",
        "",
        selected[[c for c in cols if c in selected.columns]].to_string(index=False) if not selected.empty else "No selected rows.",
        "",
        "## Station-level counts",
        "",
        station_counts.to_string(index=False) if not station_counts.empty else "No counts.",
        "",
        "## Interpretation",
        "",
        "- If a checkpoint wins in this audit, 031_33 already produced a candidate but reward-sum selection may not have selected the most advisor-aligned checkpoint.",
        "- If no checkpoint wins across all checkpoints for a station, the current frozen PPO configuration did not produce a training-year known-metric winner for that station.",
        "- WP_ET must be handled in 031_34 by rerunning frozen selected checkpoints with ETCP/snapshot outputs.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    eval_df = pd.read_csv(CKPT_EVAL, keep_default_na=False)
    eval_df = eval_df[eval_df["station_code"].isin(STATION_TO_SITE)].copy()
    eval_df["site"] = eval_df["station_code"].map(STATION_TO_SITE)
    env = baseline_envelope(eval_df)
    all_gaps = add_gaps(eval_df, env)
    selected = select_by_advisor(all_gaps)
    station_counts = (
        all_gaps.groupby("station_code")
        .agg(
            checkpoint_count=("checkpoint_step", "count"),
            known_metric_winning_checkpoints=("known_any_metric_win", "sum"),
            yield_winning_checkpoints=("yield_win", "sum"),
            pfp_n_winning_checkpoints=("pfp_n_win", "sum"),
        )
        .reset_index()
    )
    selected_counts = (
        selected.groupby("station_code")
        .agg(
            selected_seed_count=("seed", "count"),
            selected_known_metric_winning_seeds=("known_any_metric_win", "sum"),
            selected_yield_winning_seeds=("yield_win", "sum"),
            selected_pfp_n_winning_seeds=("pfp_n_win", "sum"),
        )
        .reset_index()
    )
    station_counts = station_counts.merge(selected_counts, on="station_code", how="left")

    all_path = OUT / "evaluation" / "031_34a_all_03133_checkpoint_advisor_gaps.csv"
    sel_path = OUT / "evaluation" / "031_34a_advisor_reselected_checkpoints.csv"
    cnt_path = OUT / "evaluation" / "031_34a_station_level_counts.csv"
    env_path = OUT / "evaluation" / "031_34a_four_baseline_envelope.csv"
    all_gaps.to_csv(all_path, index=False, encoding="utf-8-sig")
    selected.to_csv(sel_path, index=False, encoding="utf-8-sig")
    station_counts.to_csv(cnt_path, index=False, encoding="utf-8-sig")
    env.to_csv(env_path, index=False, encoding="utf-8-sig")
    write_record(all_gaps, selected, station_counts, env)
    result = {
        "task": "031_34a_reselect_03133_checkpoints_by_advisor_metrics",
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "all_gaps": str(all_path.relative_to(ROOT)).replace("\\", "/"),
        "selected": str(sel_path.relative_to(ROOT)).replace("\\", "/"),
        "station_counts": str(cnt_path.relative_to(ROOT)).replace("\\", "/"),
        "baseline_envelope": str(env_path.relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "031_34a_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(selected[[
        "station_code", "year", "seed", "checkpoint_step", "final_grnwt", "gap_yield", "PFP_N", "gap_pfp_n", "known_any_metric_win", "known_winning_metrics"
    ]].sort_values(["station_code", "seed"]).to_string(index=False))
    print(station_counts.to_string(index=False))


if __name__ == "__main__":
    main()
