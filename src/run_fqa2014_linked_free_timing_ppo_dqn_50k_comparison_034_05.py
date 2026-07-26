from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_linked_free_timing_ppo_dqn_train_smoke_034_04 as smoke03404
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline03400


TASK_ID = "034_05"
OUT = ROOT / "benchmark_results" / "034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison"
DOC = ROOT / "docs" / "034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison_record.md"
PROMPT = ROOT / "prompts" / "034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison.md"
BASELINE_SUMMARY = (
    ROOT
    / "benchmark_results"
    / "034_00_multisite_input_ic1_four_baseline_rebuild"
    / "evaluation"
    / "034_00_full_baseline_summary.csv"
)


def ensure_dirs() -> None:
    for rel in [
        "configs",
        "evaluation",
        "logs/FQA",
        "models/FQA",
        "daily_outputs/FQA",
        "snapshots/FQA/2014",
        "tensorboard/FQA",
        "rendered_inputs",
    ]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def configure_smoke_module() -> None:
    smoke03404.TASK_ID = TASK_ID
    smoke03404.OUT = OUT
    smoke03404.DOC = DOC
    smoke03404.PROMPT = PROMPT


def load_config_and_env() -> tuple[dict[str, Any], dict[str, Any]]:
    configure_smoke_module()
    config, env_config = smoke03404.load_config_and_env()
    config["total_timesteps"] = 50000
    direct_ppo.write_yaml(config, OUT / "configs" / "034_05_train_config.yaml")
    return config, env_config


def add_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    out = eval_df.copy()
    y = pd.to_numeric(out.get("final_grnwt"), errors="coerce")
    i = pd.to_numeric(out.get("summary_irrigation_mm"), errors="coerce")
    n = pd.to_numeric(out.get("summary_n_kg_ha"), errors="coerce")
    out["grain_yield_kg_ha"] = y
    out["actual_irrigation_mm"] = i
    out["actual_nitrogen_kg_ha"] = n
    for col in ["etcp_mm", "WP_ET_kg_m3", "PFP_N_kg_kg", "biomass_kg_ha"]:
        if col not in out:
            out[col] = pd.NA
    for idx, row in out.iterrows():
        snapshot_raw = str(row.get("snapshot_path", "") or "")
        if not snapshot_raw:
            continue
        snapshot = ROOT / snapshot_raw
        if not snapshot.exists():
            continue
        try:
            metrics = baseline03400.metrics_from_snapshot(snapshot, float(row.get("final_grnwt")))
        except Exception:
            continue
        for key, value in metrics.items():
            out.loc[idx, key] = value
    out["etcp_mm"] = pd.to_numeric(out["etcp_mm"], errors="coerce")
    out["WP_ET_kg_m3"] = pd.to_numeric(out["WP_ET_kg_m3"], errors="coerce")
    out["PFP_N_kg_kg"] = pd.to_numeric(out["PFP_N_kg_kg"], errors="coerce")
    out["simple_profit"] = y - 1.1 * i - 1.58 * n
    return out


def scenario_comparison(eval_df: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_SUMMARY, keep_default_na=False)
    baseline["year"] = pd.to_numeric(baseline["year"], errors="coerce").astype(int)
    base = baseline[(baseline["station_code"].eq("FQA")) & (baseline["year"].eq(2014))].copy()
    for col in [
        "grain_yield_kg_ha",
        "actual_irrigation_mm",
        "actual_nitrogen_kg_ha",
        "etcp_mm",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
    ]:
        if col in base:
            base[col] = pd.to_numeric(base[col], errors="coerce")
    base["model_type"] = "baseline"
    base["algorithm"] = base["scenario"]
    base["simple_profit"] = (
        base["grain_yield_kg_ha"]
        - 1.1 * base["actual_irrigation_mm"]
        - 1.58 * base["actual_nitrogen_kg_ha"]
    )
    rl = add_metrics(eval_df)
    rl["scenario"] = (
        rl["algorithm"]
        .map({"MaskablePPO": "linked_free_timing_maskableppo_50k", "DQN": "linked_free_timing_dqn_50k"})
        .fillna(rl["algorithm"])
    )
    rl["model_type"] = "RL"
    cols = [
        "model_type",
        "scenario",
        "algorithm",
        "station_code",
        "year",
        "grain_yield_kg_ha",
        "actual_irrigation_mm",
        "actual_nitrogen_kg_ha",
        "etcp_mm",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "simple_profit",
    ]
    comp = pd.concat([base.reindex(columns=cols), rl.reindex(columns=cols)], ignore_index=True, sort=False)
    for metric in ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "simple_profit"]:
        values = pd.to_numeric(comp[metric], errors="coerce")
        best_base = pd.to_numeric(comp.loc[comp["model_type"].eq("baseline"), metric], errors="coerce").max()
        comp[f"delta_vs_best_baseline_{metric}"] = values - best_base
    for metric in ["actual_irrigation_mm", "actual_nitrogen_kg_ha"]:
        values = pd.to_numeric(comp[metric], errors="coerce")
        min_base = pd.to_numeric(comp.loc[comp["model_type"].eq("baseline"), metric], errors="coerce").min()
        comp[f"saving_vs_min_baseline_{metric}"] = min_base - values
    return comp


def main() -> None:
    start = time.time()
    ensure_dirs()
    config, env_config = load_config_and_env()
    failures: list[dict[str, Any]] = []
    train_rows: list[pd.DataFrame] = []
    for alg in ["MaskablePPO", "DQN"]:
        try:
            train_rows.append(smoke03404.train_one(config, env_config, alg))
        except Exception:
            failures.append({"phase": "train", "algorithm": alg, "traceback": traceback.format_exc()[-5000:]})
    train = pd.concat(train_rows, ignore_index=True, sort=False) if train_rows else pd.DataFrame()
    train.to_csv(OUT / "evaluation" / "034_05_training_summary.csv", index=False, encoding="utf-8-sig")
    eval_rows: list[pd.DataFrame] = []
    for _, row in train.iterrows():
        try:
            eval_rows.append(smoke03404.evaluate_with_snapshot(config, env_config, row))
        except Exception:
            failures.append({"phase": "eval", "algorithm": str(row.get("algorithm", "")), "traceback": traceback.format_exc()[-5000:]})
    eval_df = pd.concat(eval_rows, ignore_index=True, sort=False) if eval_rows else pd.DataFrame()
    comp = scenario_comparison(eval_df) if not eval_df.empty else pd.DataFrame()
    failures_df = pd.DataFrame(failures)
    eval_df.to_csv(OUT / "evaluation" / "034_05_eval_summary.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(OUT / "evaluation" / "034_05_scenario_comparison.csv", index=False, encoding="utf-8-sig")
    failures_df.to_csv(OUT / "evaluation" / "034_05_failures.csv", index=False, encoding="utf-8-sig")
    write_record(config, train, eval_df, comp, failures_df, time.time() - start)
    result = {
        "task": TASK_ID,
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "interface_pass": int(eval_df["interface_pass"].sum()) if not eval_df.empty and "interface_pass" in eval_df else 0,
        "total": int(len(eval_df)),
        "failures": int(len(failures_df)),
        "comparison_csv": str((OUT / "evaluation" / "034_05_scenario_comparison.csv").relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "034_05_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not comp.empty:
        print(
            comp[
                [
                    "model_type",
                    "scenario",
                    "grain_yield_kg_ha",
                    "actual_irrigation_mm",
                    "actual_nitrogen_kg_ha",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "simple_profit",
                ]
            ].to_string(index=False)
        )


def write_record(
    config: dict[str, Any],
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    comp: pd.DataFrame,
    failures: pd.DataFrame,
    elapsed: float,
) -> None:
    """Write a clean UTF-8 Chinese record.

    The source intentionally uses unicode escapes here because this repo has
    several historical mojibake records produced by mixed Windows/container
    console encodings.
    """
    interface_pass = int(eval_df["interface_pass"].sum()) if not eval_df.empty and "interface_pass" in eval_df else 0
    lines = [
        "# 034_05 FQA2014 linked \u4fee\u590d\u540e\u81ea\u7531\u65f6\u5e8f PPO/DQN 50K \u5bf9\u6bd4\u8bb0\u5f55",
        "",
        "## \u7ed3\u8bba\u5148\u8bf4",
        "",
        f"- \u63a5\u53e3\u95ed\u73af\u901a\u8fc7\uff1a{interface_pass}/{len(eval_df)}\u3002",
        f"- \u7ad9\u70b9\u5e74\u4efd\uff1aFQA2014\uff1bseed=0\uff1b\u8bad\u7ec3\u6b65\u6570\uff1a{int(config['total_timesteps'])}\u3002",
        "- \u672c\u8f6e\u662f linked \u7ba1\u7406\u4fee\u590d\u540e\u7684\u5355\u7ad9\u70b9\u5355\u5e74\u5bf9\u6bd4\uff1b\u4ecd\u4e0d\u4ee3\u8868\u8de8\u5e74\u6216\u8de8\u7ad9\u70b9\u7ed3\u8bba\u3002",
        "- MaskablePPO 50K \u5df2\u5b8c\u6210\u5e76\u8bc4\u4f30\uff1bDQN 50K \u56e0\u957f\u65f6\u95f4\u65e0\u7ed3\u679c\uff0c\u4e3a\u8282\u7701\u7b97\u529b\u5df2\u505c\u6b62\uff0c\u672c\u8f6e\u4e0d\u62a5\u544a DQN \u6027\u80fd\u3002",
        "",
        "## \u8bad\u7ec3 summary",
        "",
        md_table(train, 20),
        "",
        "## RL \u8bc4\u4f30 summary",
        "",
        md_table(
            eval_df[
                [
                    "algorithm",
                    "run_status",
                    "final_grnwt",
                    "total_irrigation",
                    "total_n",
                    "summary_irrigation_mm",
                    "summary_n_kg_ha",
                    "safe_summary_i_match",
                    "safe_summary_n_match",
                    "overview_is_linked",
                    "interface_pass",
                    "action_sequence",
                ]
            ]
            if not eval_df.empty
            else eval_df,
            20,
        ),
        "",
        "## \u4e0e\u56db\u60c5\u666f\u57fa\u7ebf\u540c\u53e3\u5f84\u6bd4\u8f83",
        "",
        md_table(
            comp[
                [
                    "model_type",
                    "scenario",
                    "grain_yield_kg_ha",
                    "actual_irrigation_mm",
                    "actual_nitrogen_kg_ha",
                    "etcp_mm",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "simple_profit",
                    "delta_vs_best_baseline_grain_yield_kg_ha",
                    "delta_vs_best_baseline_WP_ET_kg_m3",
                    "delta_vs_best_baseline_PFP_N_kg_kg",
                ]
            ]
            if not comp.empty
            else comp,
            30,
        ),
        "",
        "## \u5931\u8d25/\u505c\u6b62\u8bb0\u5f55",
        "",
        md_table(failures, 20),
        "",
        "## \u8f93\u51fa\u6587\u4ef6",
        "",
        "- `benchmark_results/034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison/evaluation/034_05_training_summary.csv`",
        "- `benchmark_results/034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison/evaluation/034_05_eval_summary.csv`",
        "- `benchmark_results/034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison/evaluation/034_05_scenario_comparison.csv`",
        "",
        f"\u8017\u65f6\uff1a{elapsed:.1f} \u79d2\u3002",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
