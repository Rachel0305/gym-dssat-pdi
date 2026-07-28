from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "035_03"
OUT = ROOT / "benchmark_results" / "035_03_fqa2014_reward_v2_alignment_audit"
DOC = ROOT / "docs" / "035_03_fqa2014_reward_v2_alignment_audit_record.md"
PROMPT = ROOT / "prompts" / "035_03_fqa2014_reward_v2_alignment_audit.md"
RULE_SUMMARY = ROOT / "benchmark_results" / "035_00_fqa2014_linked_free_timing_rule_probe" / "evaluation" / "035_00_rule_summary.csv"
SCENARIO_COMP = ROOT / "benchmark_results" / "035_00_fqa2014_linked_free_timing_rule_probe" / "evaluation" / "035_00_scenario_comparison.csv"


def ensure_dirs() -> None:
    (OUT / "evaluation").mkdir(parents=True, exist_ok=True)
    (OUT / "configs").mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        (OUT / "configs" / PROMPT.name).write_text(PROMPT.read_text(encoding="utf-8"), encoding="utf-8")


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_candidates() -> tuple[pd.DataFrame, pd.Series]:
    if not RULE_SUMMARY.exists():
        raise FileNotFoundError(RULE_SUMMARY)
    if not SCENARIO_COMP.exists():
        raise FileNotFoundError(SCENARIO_COMP)

    rules = pd.read_csv(RULE_SUMMARY)
    rules = to_num(
        rules,
        [
            "grain_yield_kg_ha",
            "summary_irrigation_mm",
            "summary_nitrogen_kg_ha",
            "WP_ET_kg_m3",
            "PFP_N_kg_kg",
            "simple_profit",
            "reward_sum",
            "irrigation_event_count",
            "nitrogen_event_count",
            "first_irrigation_dap",
            "first_n_dap",
            "swfac_stress_days_gt_0p05",
            "nstres_days_gt_0p05",
            "max_swfac",
            "max_nstres",
        ],
    )
    rules["source"] = "linked_rule_03500"
    rules["name"] = rules["rule"].astype(str)

    scenario = pd.read_csv(SCENARIO_COMP)
    scenario = to_num(
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
    expert = scenario[scenario["name"].astype(str).eq("official_extension_expert")]
    if expert.empty:
        raise ValueError("official_extension_expert not found in 035_00 scenario comparison.")
    expert_row = expert.iloc[0]

    cols = [
        "source",
        "name",
        "grain_yield_kg_ha",
        "summary_irrigation_mm",
        "summary_nitrogen_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "simple_profit",
        "reward_sum",
        "irrigation_event_count",
        "nitrogen_event_count",
        "first_irrigation_dap",
        "first_n_dap",
        "swfac_stress_days_gt_0p05",
        "nstres_days_gt_0p05",
        "max_swfac",
        "max_nstres",
        "action_sequence",
    ]
    candidates = rules[[c for c in cols if c in rules.columns]].copy()
    candidates = candidates.rename(
        columns={
            "summary_irrigation_mm": "irrigation_mm",
            "summary_nitrogen_kg_ha": "nitrogen_kg_ha",
            "reward_sum": "logged_reward_sum",
            "simple_profit": "csv_project_simple_profit",
        }
    )
    return candidates, expert_row


def add_scores(candidates: pd.DataFrame, expert: pd.Series) -> pd.DataFrame:
    out = candidates.copy()
    expert_yield = float(expert["grain_yield_kg_ha"])
    expert_i = float(expert["actual_irrigation_mm"])
    expert_n = float(expert["actual_nitrogen_kg_ha"])
    expert_wp = float(expert["WP_ET_kg_m3"])
    expert_pfp = float(expert["PFP_N_kg_kg"])

    y = pd.to_numeric(out["grain_yield_kg_ha"], errors="coerce")
    i = pd.to_numeric(out["irrigation_mm"], errors="coerce")
    n = pd.to_numeric(out["nitrogen_kg_ha"], errors="coerce")
    wp = pd.to_numeric(out["WP_ET_kg_m3"], errors="coerce")
    pfp = pd.to_numeric(out["PFP_N_kg_kg"], errors="coerce")

    # Project simple_profit used by 034/035 code:
    # final_yield - 1.1 * irrigation - 1.58 * nitrogen.
    out["score_project_simple_profit"] = y - 1.1 * i - 1.58 * n
    out["score_project_profit_csv_diff"] = out["score_project_simple_profit"] - pd.to_numeric(
        out.get("csv_project_simple_profit", np.nan), errors="coerce"
    )

    # Historical proxy often discussed in earlier tasks. It is only a candidate score here.
    out["score_old_proxy_y_i_5n"] = y - i - 5.0 * n
    out["score_yield_gate_project_profit_1620"] = out["score_project_simple_profit"] + np.where(
        y >= expert_yield, 1620.0, 0.0
    )
    out["score_yield_shortfall_project_penalty_x5"] = out["score_project_simple_profit"] - 5.0 * np.maximum(
        0.0, expert_yield - y
    )

    out["delta_yield_vs_expert"] = y - expert_yield
    out["delta_i_vs_expert"] = i - expert_i
    out["delta_n_vs_expert"] = n - expert_n
    out["delta_wp_vs_expert"] = wp - expert_wp
    out["delta_pfp_vs_expert"] = pfp - expert_pfp
    out["beats_expert_yield"] = out["delta_yield_vs_expert"] > 0
    out["beats_expert_wp"] = out["delta_wp_vs_expert"] > 0
    out["beats_expert_pfp"] = out["delta_pfp_vs_expert"] > 0
    out["advisor_one_metric_pass"] = out[["beats_expert_yield", "beats_expert_wp", "beats_expert_pfp"]].any(axis=1)
    out["strict_resource_gate_pass"] = (y >= expert_yield) & (i <= expert_i) & (n <= expert_n)
    out["yield_not_lower_than_expert"] = y >= expert_yield
    out["resource_not_higher_than_expert"] = (i <= expert_i) & (n <= expert_n)
    return out


def make_rankings(scored: pd.DataFrame) -> pd.DataFrame:
    score_cols = [
        "logged_reward_sum",
        "score_project_simple_profit",
        "score_old_proxy_y_i_5n",
        "score_yield_gate_project_profit_1620",
        "score_yield_shortfall_project_penalty_x5",
    ]
    rows = []
    for score_col in score_cols:
        tmp = scored.copy()
        tmp["rank"] = pd.to_numeric(tmp[score_col], errors="coerce").rank(ascending=False, method="min")
        tmp = tmp.sort_values(["rank", "score_project_simple_profit"], ascending=[True, False])
        for _, row in tmp.iterrows():
            rows.append(
                {
                    "score_name": score_col,
                    "rank": int(row["rank"]) if pd.notna(row["rank"]) else np.nan,
                    "name": row["name"],
                    "score_value": row[score_col],
                    "grain_yield_kg_ha": row["grain_yield_kg_ha"],
                    "irrigation_mm": row["irrigation_mm"],
                    "nitrogen_kg_ha": row["nitrogen_kg_ha"],
                    "WP_ET_kg_m3": row["WP_ET_kg_m3"],
                    "PFP_N_kg_kg": row["PFP_N_kg_kg"],
                    "score_project_simple_profit": row["score_project_simple_profit"],
                    "score_old_proxy_y_i_5n": row["score_old_proxy_y_i_5n"],
                    "advisor_one_metric_pass": row["advisor_one_metric_pass"],
                    "yield_not_lower_than_expert": row["yield_not_lower_than_expert"],
                }
            )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def summarize_score(ranking: pd.DataFrame, score_name: str) -> dict[str, object]:
    sub = ranking[ranking["score_name"].eq(score_name)].sort_values("rank")
    top = sub.iloc[0]
    bad_names = {"noop", "early_dump_cap", "ppo_03405_replay", "linked_free_timing_maskableppo_50k"}
    top3 = sub.head(3)
    return {
        "score_name": score_name,
        "top1": top["name"],
        "top1_one_metric_pass": bool(top["advisor_one_metric_pass"]),
        "top3_names": "; ".join(top3["name"].astype(str).tolist()),
        "bad_candidate_in_top3": bool(top3["name"].isin(bad_names).any()),
    }


def write_record(scored: pd.DataFrame, ranking: pd.DataFrame, expert: pd.Series) -> None:
    score_names = [
        "logged_reward_sum",
        "score_project_simple_profit",
        "score_old_proxy_y_i_5n",
        "score_yield_gate_project_profit_1620",
        "score_yield_shortfall_project_penalty_x5",
    ]
    score_summary = pd.DataFrame([summarize_score(ranking, name) for name in score_names])
    top_by_project_profit = scored.sort_values("score_project_simple_profit", ascending=False).iloc[0]
    top_by_logged = ranking[ranking["score_name"].eq("logged_reward_sum")].sort_values("rank").iloc[0]
    max_csv_diff = pd.to_numeric(scored["score_project_profit_csv_diff"], errors="coerce").abs().max()

    lines = [
        "# 035_03 FQA2014 自由时序 reward v2 对齐审计记录",
        "",
        "## 结论先说",
        "",
        f"- 当前 logged reward 排名第一是 `{top_by_logged['name']}`，这不是我们希望强化学习优先学习的节水节氮高产策略。",
        f"- 项目现有 `simple_profit = yield - 1.1 * irrigation - 1.58 * nitrogen` 排名第一是 `{top_by_project_profit['name']}`。",
        f"- 重新计算的项目 simple_profit 与 035_00 CSV 原列最大差异为 `{max_csv_diff:.6g}`，说明本次公式和既有结果口径一致。",
        "- 在 035_00 已有候选中，项目 simple_profit 及其产量门槛变体能把 water_saving_n160 / split_moderate_n160 / critical_i90_n200 排到前列。",
        "- 旧代理 `Y - I - 5N` 会把 no-op 排在第一，不适合作为当前 FQA2014 自由时序训练主目标。",
        "- 因此 035_02 的失败更像是当前 logged reward 与最终指标不对齐，而不是单纯训练步数不够。",
        "- 本任务不启动训练；若继续，应另开任务测试项目 simple_profit 或其带产量 guardrail 的变体。",
        "",
        "## expert 参照值",
        "",
        md_table(
            pd.DataFrame(
                [
                    {
                        "expert_yield": expert["grain_yield_kg_ha"],
                        "expert_irrigation": expert["actual_irrigation_mm"],
                        "expert_nitrogen": expert["actual_nitrogen_kg_ha"],
                        "expert_WP_ET": expert["WP_ET_kg_m3"],
                        "expert_PFP_N": expert["PFP_N_kg_kg"],
                        "expert_simple_profit": expert["simple_profit"],
                    }
                ]
            )
        ),
        "",
        "## 各候选分数的一句话审计",
        "",
        md_table(score_summary, 20),
        "",
        "## 候选策略完整分数表",
        "",
        md_table(
            scored[
                [
                    "name",
                    "grain_yield_kg_ha",
                    "irrigation_mm",
                    "nitrogen_kg_ha",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "logged_reward_sum",
                    "score_project_simple_profit",
                    "score_old_proxy_y_i_5n",
                    "score_yield_gate_project_profit_1620",
                    "score_yield_shortfall_project_penalty_x5",
                    "delta_yield_vs_expert",
                    "delta_i_vs_expert",
                    "delta_n_vs_expert",
                    "advisor_one_metric_pass",
                ]
            ].sort_values("score_project_simple_profit", ascending=False),
            40,
        ),
        "",
        "## 按分数展开的排名",
        "",
        md_table(
            ranking[
                [
                    "score_name",
                    "rank",
                    "name",
                    "score_value",
                    "grain_yield_kg_ha",
                    "irrigation_mm",
                    "nitrogen_kg_ha",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "advisor_one_metric_pass",
                ]
            ],
            100,
        ),
        "",
        "## 下一步建议",
        "",
        "035_04 不应继续沿用当前 logged reward 直接加步数。建议只改 reward / 选模目标，优先测试：",
        "",
        "1. 训练即时 reward 改为与项目综合指标一致的 `delta_GRNWT - 1.1I - 1.58N`；",
        "2. checkpoint 选择使用统一 DSSAT 回放后的项目 simple_profit 与 expert guardrail；",
        "3. 仍先只做 FQA2014 单站点单年，不扩展全站点；",
        "4. 若仍打满 I150/N240，再判断是 reward 传播问题还是动作空间/约束仍不足。",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    candidates, expert = load_candidates()
    scored = add_scores(candidates, expert)
    ranking = make_rankings(scored)

    scored_path = OUT / "evaluation" / "035_03_candidate_scores.csv"
    ranking_path = OUT / "evaluation" / "035_03_score_rankings.csv"
    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    ranking.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    write_record(scored, ranking, expert)

    compact = pd.DataFrame(
        [
            summarize_score(ranking, "logged_reward_sum"),
            summarize_score(ranking, "score_project_simple_profit"),
            summarize_score(ranking, "score_old_proxy_y_i_5n"),
            summarize_score(ranking, "score_yield_gate_project_profit_1620"),
            summarize_score(ranking, "score_yield_shortfall_project_penalty_x5"),
        ]
    )
    print(
        {
            "task": TASK_ID,
            "candidates": int(len(scored)),
            "scored_csv": str(scored_path.relative_to(ROOT)),
            "ranking_csv": str(ranking_path.relative_to(ROOT)),
            "record_md": str(DOC.relative_to(ROOT)),
        }
    )
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
