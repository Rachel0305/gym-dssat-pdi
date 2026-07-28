from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "035_01"
OUT = ROOT / "benchmark_results" / "035_01_fqa2014_reward_metric_alignment_audit"
DOC = ROOT / "docs" / "035_01_fqa2014_reward_metric_alignment_audit_record.md"
PROMPT = ROOT / "prompts" / "035_01_fqa2014_reward_metric_alignment_audit.md"
RULE_SUMMARY = ROOT / "benchmark_results" / "035_00_fqa2014_linked_free_timing_rule_probe" / "evaluation" / "035_00_rule_summary.csv"
SCENARIO_COMP = ROOT / "benchmark_results" / "035_00_fqa2014_linked_free_timing_rule_probe" / "evaluation" / "035_00_scenario_comparison.csv"
PPO_EVAL = ROOT / "benchmark_results" / "034_05_fqa2014_linked_free_timing_ppo_dqn_50k_comparison" / "evaluation" / "034_05_eval_summary.csv"


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        (OUT / "configs" / PROMPT.name).write_text(PROMPT.read_text(encoding="utf-8"), encoding="utf-8")


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = pd.read_csv(RULE_SUMMARY)
    rules = numeric(
        rules,
        [
            "grain_yield_kg_ha",
            "summary_irrigation_mm",
            "summary_nitrogen_kg_ha",
            "WP_ET_kg_m3",
            "PFP_N_kg_kg",
            "simple_profit",
            "reward_sum",
            "stress_relief_bonus_sum_unscaled",
        ],
    )
    rules["source"] = "linked_rule_03500"
    rules["name"] = rules["rule"]

    ppo = pd.read_csv(PPO_EVAL)
    ppo = numeric(
        ppo,
        [
            "final_grnwt",
            "summary_irrigation_mm",
            "summary_n_kg_ha",
            "profit_simple",
            "PFP_N",
            "reward_stress_aware_sum",
            "stress_relief_bonus_sum_unscaled",
        ],
    )
    ppo_rows = pd.DataFrame(
        {
            "source": ["rl_03405"],
            "name": ["linked_free_timing_maskableppo_50k"],
            "grain_yield_kg_ha": [float(ppo["final_grnwt"].iloc[0])],
            "summary_irrigation_mm": [float(ppo["summary_irrigation_mm"].iloc[0])],
            "summary_nitrogen_kg_ha": [float(ppo["summary_n_kg_ha"].iloc[0])],
            "WP_ET_kg_m3": [np.nan],
            "PFP_N_kg_kg": [float(ppo["PFP_N"].iloc[0])],
            "simple_profit": [float(ppo["profit_simple"].iloc[0])],
            "reward_sum": [float(ppo["reward_stress_aware_sum"].iloc[0])],
            "stress_relief_bonus_sum_unscaled": [float(ppo["stress_relief_bonus_sum_unscaled"].iloc[0])],
            "action_sequence": [str(ppo.get("action_sequence", pd.Series([""])).iloc[0])],
        }
    )

    comp = pd.read_csv(SCENARIO_COMP)
    ppo_comp = comp[comp["name"].astype(str).eq("linked_free_timing_maskableppo_50k")]
    if len(ppo_comp) and "WP_ET_kg_m3" in ppo_comp:
        ppo_rows.loc[0, "WP_ET_kg_m3"] = pd.to_numeric(ppo_comp["WP_ET_kg_m3"], errors="coerce").iloc[0]

    candidates = pd.concat(
        [
            rules[
                [
                    "source",
                    "name",
                    "grain_yield_kg_ha",
                    "summary_irrigation_mm",
                    "summary_nitrogen_kg_ha",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "simple_profit",
                    "reward_sum",
                    "stress_relief_bonus_sum_unscaled",
                    "action_sequence",
                ]
            ],
            ppo_rows,
        ],
        ignore_index=True,
        sort=False,
    )

    scenario = pd.read_csv(SCENARIO_COMP)
    scenario = numeric(
        scenario,
        [
            "grain_yield_kg_ha",
            "actual_irrigation_mm",
            "actual_nitrogen_kg_ha",
            "WP_ET_kg_m3",
            "PFP_N_kg_kg",
            "simple_profit",
        ],
    )
    return candidates, scenario


def add_ranks(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    for metric in ["reward_sum", "grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "simple_profit"]:
        out[f"rank_{metric}"] = pd.to_numeric(out[metric], errors="coerce").rank(ascending=False, method="min")
    for metric in ["summary_irrigation_mm", "summary_nitrogen_kg_ha"]:
        out[f"rank_low_{metric}"] = pd.to_numeric(out[metric], errors="coerce").rank(ascending=True, method="min")
    return out.sort_values("rank_reward_sum").reset_index(drop=True)


def correlations(ranked: pd.DataFrame) -> pd.DataFrame:
    metrics = ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "simple_profit", "summary_irrigation_mm", "summary_nitrogen_kg_ha"]
    rows = []
    reward = pd.to_numeric(ranked["reward_sum"], errors="coerce")
    for metric in metrics:
        value = pd.to_numeric(ranked[metric], errors="coerce")
        rows.append(
            {
                "metric": metric,
                "spearman_vs_reward": reward.corr(value, method="spearman"),
                "pearson_vs_reward": reward.corr(value, method="pearson"),
                "n_non_na": int(pd.concat([reward, value], axis=1).dropna().shape[0]),
            }
        )
    return pd.DataFrame(rows)


def expert_deltas(ranked: pd.DataFrame, scenario: pd.DataFrame) -> pd.DataFrame:
    expert = scenario[scenario["name"].astype(str).eq("official_extension_expert")]
    if expert.empty:
        raise ValueError("未找到 official_extension_expert")
    e = expert.iloc[0]
    out = ranked.copy()
    out["delta_yield_vs_expert"] = out["grain_yield_kg_ha"] - float(e["grain_yield_kg_ha"])
    out["delta_i_vs_expert"] = out["summary_irrigation_mm"] - float(e["actual_irrigation_mm"])
    out["delta_n_vs_expert"] = out["summary_nitrogen_kg_ha"] - float(e["actual_nitrogen_kg_ha"])
    out["delta_wp_vs_expert"] = out["WP_ET_kg_m3"] - float(e["WP_ET_kg_m3"])
    out["delta_pfp_vs_expert"] = out["PFP_N_kg_kg"] - float(e["PFP_N_kg_kg"])
    out["delta_profit_vs_expert"] = out["simple_profit"] - float(e["simple_profit"])
    out["beats_expert_yield"] = out["delta_yield_vs_expert"] > 0
    out["beats_expert_wp"] = out["delta_wp_vs_expert"] > 0
    out["beats_expert_pfp"] = out["delta_pfp_vs_expert"] > 0
    out["beats_expert_profit"] = out["delta_profit_vs_expert"] > 0
    out["advisor_one_metric_pass"] = out[["beats_expert_yield", "beats_expert_wp", "beats_expert_pfp"]].any(axis=1)
    return out


def recommend_guardrail(deltas: pd.DataFrame) -> pd.DataFrame:
    work = deltas.copy()
    work["guardrail_candidate_yield_not_lower"] = work["delta_yield_vs_expert"] >= 0
    work["guardrail_candidate_profit_not_lower"] = work["delta_profit_vs_expert"] >= 0
    work["guardrail_candidate_any_main_metric"] = work["advisor_one_metric_pass"]
    work["guardrail_score"] = (
        work["guardrail_candidate_yield_not_lower"].astype(int)
        + work["guardrail_candidate_profit_not_lower"].astype(int)
        + work["guardrail_candidate_any_main_metric"].astype(int)
    )
    return work.sort_values(["guardrail_score", "simple_profit"], ascending=[False, False]).reset_index(drop=True)


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


def write_record(ranked: pd.DataFrame, corr: pd.DataFrame, deltas: pd.DataFrame, guard: pd.DataFrame) -> None:
    reward_best = ranked.iloc[0].to_dict()
    profit_best = ranked.sort_values("simple_profit", ascending=False).iloc[0].to_dict()
    ppo = ranked[ranked["name"].eq("linked_free_timing_maskableppo_50k")]
    ppo_reward_rank = int(ppo["rank_reward_sum"].iloc[0]) if len(ppo) else -1
    ppo_profit_rank = int(ppo["rank_simple_profit"].iloc[0]) if len(ppo) else -1
    lines = [
        "# 035_01 FQA2014 reward-指标对齐审计记录",
        "",
        "## 结论先说",
        "",
        f"- 当前 reward 排名第一：{reward_best.get('name')}；simple_profit 排名第一：{profit_best.get('name')}。",
        f"- 034_05 PPO 50K 的 reward 排名：{ppo_reward_rank}；simple_profit 排名：{ppo_profit_rank}。",
        "- 当前 reward 与最终指标并不等价；只用 reward 选 checkpoint 会有明显风险。",
        "- PPO 50K 既不是 reward 最优，也不是指标最优；下一步要同时处理训练优化不足和选择指标错位。",
        "",
        "## reward 排序表",
        "",
        md_table(
            ranked[
                [
                    "source",
                    "name",
                    "reward_sum",
                    "rank_reward_sum",
                    "grain_yield_kg_ha",
                    "rank_grain_yield_kg_ha",
                    "WP_ET_kg_m3",
                    "rank_WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "rank_PFP_N_kg_kg",
                    "simple_profit",
                    "rank_simple_profit",
                    "summary_irrigation_mm",
                    "summary_nitrogen_kg_ha",
                ]
            ],
            30,
        ),
        "",
        "## reward 与指标相关性",
        "",
        md_table(corr, 20),
        "",
        "## 相对 expert 差值",
        "",
        md_table(
            deltas[
                [
                    "name",
                    "delta_yield_vs_expert",
                    "delta_i_vs_expert",
                    "delta_n_vs_expert",
                    "delta_wp_vs_expert",
                    "delta_pfp_vs_expert",
                    "delta_profit_vs_expert",
                    "advisor_one_metric_pass",
                ]
            ],
            40,
        ),
        "",
        "## guardrail 候选排序",
        "",
        md_table(
            guard[
                [
                    "name",
                    "guardrail_score",
                    "grain_yield_kg_ha",
                    "summary_irrigation_mm",
                    "summary_nitrogen_kg_ha",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "simple_profit",
                    "reward_sum",
                ]
            ],
            40,
        ),
        "",
        "## 下一步建议",
        "",
        "1. 不要继续只按当前 reward 选择 checkpoint。",
        "2. 下一轮 linked PPO/DQN 训练必须保存中间 checkpoint，并用外部指标 guardrail 复评。",
        "3. 最低 guardrail 建议：产量不低于 expert，且产量/WP_ET/PFP_N 至少一个超过 expert；同分时优先 simple_profit。",
        "4. 若模型训练 reward 高但不满足 guardrail，不应作为成功策略。",
        "5. 若模型无法产生 guardrail 候选，应优先改 reward 或训练稳定性，而不是扩大全站点。",
        "",
    ]
    text = "\n".join(lines)
    DOC.write_text(text + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text(text + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    candidates, scenario = load_candidates()
    ranked = add_ranks(candidates)
    corr = correlations(ranked)
    deltas = expert_deltas(ranked, scenario)
    guard = recommend_guardrail(deltas)
    ranked.to_csv(OUT / "evaluation" / "035_01_ranked_candidates.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(OUT / "evaluation" / "035_01_reward_metric_correlations.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(OUT / "evaluation" / "035_01_expert_deltas.csv", index=False, encoding="utf-8-sig")
    guard.to_csv(OUT / "evaluation" / "035_01_guardrail_candidates.csv", index=False, encoding="utf-8-sig")
    write_record(ranked, corr, deltas, guard)
    result = {
        "task": TASK_ID,
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "reward_best": str(ranked.iloc[0]["name"]),
        "profit_best": str(ranked.sort_values("simple_profit", ascending=False).iloc[0]["name"]),
        "ranked_csv": str((OUT / "evaluation" / "035_01_ranked_candidates.csv").relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "035_01_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(ranked[["source", "name", "reward_sum", "rank_reward_sum", "grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "simple_profit", "rank_simple_profit"]].to_string(index=False))


if __name__ == "__main__":
    main()

