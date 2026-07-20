from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark_results" / "027_05"
OUT = ROOT / "benchmark_results" / "027_06_current_five_station_rl_overview"
DOC = ROOT / "docs" / "2026-07-17_027_06_current_five_station_rl_overview.md"


SELECTED = {
    "HLA": {
        "algorithm": "MaskablePPO",
        "checkpoint": 180,
        "protocol_budget": 240,
        "step_unit": "stage transitions",
        "evidence": "current PPO candidate; seed0 positive, but HLA primary replication only 1/3 seeds",
        "replication": "1/3 primary under the original HLA preregistered criterion",
    },
    "YC": {
        "algorithm": "DQN",
        "checkpoint": 5000,
        "protocol_budget": np.nan,
        "step_unit": "environment steps",
        "evidence": "historical provisional DQN candidate; old exploration-schedule protocol",
        "replication": "representative seed0 only; no current formal cross-seed claim",
    },
    "FQ": {
        "algorithm": "DQN",
        "checkpoint": 30000,
        "protocol_budget": np.nan,
        "step_unit": "environment steps",
        "evidence": "historical provisional DQN candidate; old exploration-schedule protocol",
        "replication": "representative seed1 only; no current formal cross-seed claim",
    },
    "LC": {
        "algorithm": "DQN",
        "checkpoint": 5000,
        "protocol_budget": np.nan,
        "step_unit": "environment steps (smoke candidate)",
        "evidence": "historical provisional 5K DQN smoke candidate",
        "replication": "seed0 smoke only; not a formal cross-seed result",
    },
    "SY": {
        "algorithm": "MaskablePPO",
        "checkpoint": 120,
        "protocol_budget": 240,
        "step_unit": "stage transitions",
        "evidence": "current PPO candidate with IC=2 frozen reevaluation",
        "replication": "3/3 selected checkpoints primary; trajectory persistence 2/3 seeds",
    },
}


def numeric(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def percent_margin(candidate: float, benchmark: float) -> float:
    if not np.isfinite(candidate) or not np.isfinite(benchmark) or benchmark == 0:
        return np.nan
    return 100.0 * (candidate - benchmark) / benchmark


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)

    dqn = pd.read_csv(SOURCE / "027_05_dqn_five_scenario_summary.csv", keep_default_na=False)
    ppo = pd.read_csv(SOURCE / "027_05_ppo_five_scenario_summary.csv", keep_default_na=False)
    parsed = pd.read_csv(SOURCE / "027_05_five_scenario_summary.csv", keep_default_na=False)
    daily = pd.read_csv(SOURCE / "027_05_daily_values.csv", keep_default_na=False)

    rows: list[dict[str, object]] = []
    for site, spec in SELECTED.items():
        endpoints = ppo[ppo["site"].eq(site)].copy() if spec["algorithm"] == "MaskablePPO" else dqn[dqn["site"].eq(site)].copy()
        if set(endpoints["scenario"]) != {"null", "recorded_farmer", "dssat_auto", "official_extension_expert", "rl_candidate"}:
            raise RuntimeError(f"{site}: incomplete five-scenario endpoint evidence")
        candidate = endpoints[endpoints["scenario"].eq("rl_candidate")].iloc[0]
        others = endpoints[~endpoints["scenario"].eq("rl_candidate")].copy()

        process = parsed[
            parsed["site"].eq(site)
            & parsed["algorithm"].eq(spec["algorithm"])
            & parsed["scenario"].eq("rl_candidate")
        ]
        if len(process) != 1:
            raise RuntimeError(f"{site}: expected one process summary row, got {len(process)}")
        process_row = process.iloc[0]
        trajectory = daily[
            daily["site"].eq(site)
            & daily["algorithm"].eq(spec["algorithm"])
            & daily["scenario"].eq("rl_candidate")
        ].sort_values("dap")
        if trajectory.empty:
            raise RuntimeError(f"{site}: missing candidate daily trajectory")

        yield_value = numeric(candidate["final_grain_kg_ha"])
        wue_value = numeric(candidate["wp_et_kg_m3"])
        nue_value = numeric(candidate["pfp_n_kg_kg"])
        best_other_yield = pd.to_numeric(others["final_grain_kg_ha"], errors="coerce").max()
        best_other_wue = pd.to_numeric(others["wp_et_kg_m3"], errors="coerce").max()
        finite_other_nue = pd.to_numeric(others["pfp_n_kg_kg"], errors="coerce").dropna()
        best_other_nue = finite_other_nue.max() if not finite_other_nue.empty else np.nan

        yield_winner = bool(yield_value > best_other_yield)
        wue_winner = bool(wue_value > best_other_wue)
        # PFP_N is not defined for zero-N scenarios.  It is therefore reported
        # only as a winner among scenarios with positive N and finite PFP_N.
        comparable_nue_winner = bool(np.isfinite(nue_value) and np.isfinite(best_other_nue) and nue_value > best_other_nue)
        strict_nue_all_four_comparable = bool(len(finite_other_nue) == 4)
        strict_any_all_four = yield_winner or wue_winner or (strict_nue_all_four_comparable and comparable_nue_winner)
        comparable_any = yield_winner or wue_winner or comparable_nue_winner
        winning = []
        if yield_winner:
            winning.append("yield")
        if wue_winner:
            winning.append("WP_ET")
        if comparable_nue_winner:
            winning.append("PFP_N(comparable positive-N scenarios)")

        rows.append({
            "site": site,
            "station": candidate["station"],
            "year": int(numeric(candidate["year"])),
            "representative_model": spec["algorithm"],
            "seed": int(numeric(candidate["seed"])),
            "selected_checkpoint_step": int(spec["checkpoint"]),
            "protocol_training_budget_step": spec["protocol_budget"],
            "step_unit": spec["step_unit"],
            "grain_yield_kg_ha": yield_value,
            "WP_ET_kg_m3": wue_value,
            "PFP_N_kg_kg": nue_value,
            "nitrogen_total_kg_ha": numeric(candidate["nitrogen_event_total_kg_ha"]),
            "irrigation_total_mm": numeric(candidate["irrigation_event_total_mm"]),
            "max_water_stress_WSPD": numeric(process_row["max_water_stress_wspd"]),
            "max_nitrogen_stress_NSTD": numeric(process_row["max_nitrogen_stress_nstd"]),
            "final_soil_water_SWTD_mm": numeric(trajectory.iloc[-1]["soil_water_mm"]),
            "best_other_yield_kg_ha": best_other_yield,
            "yield_margin_vs_best_other_pct": percent_margin(yield_value, best_other_yield),
            "best_other_WP_ET_kg_m3": best_other_wue,
            "WP_ET_margin_vs_best_other_pct": percent_margin(wue_value, best_other_wue),
            "best_other_comparable_PFP_N_kg_kg": best_other_nue,
            "PFP_N_margin_vs_best_comparable_pct": percent_margin(nue_value, best_other_nue),
            "finite_other_PFP_N_scenario_count": int(len(finite_other_nue)),
            "strict_any_metric_above_all_four": strict_any_all_four,
            "comparable_any_metric_winner": comparable_any,
            "winning_metric": "; ".join(winning) if winning else "none",
            "evidence_status": spec["evidence"],
            "seed_replication_status": spec["replication"],
        })

    overview = pd.DataFrame(rows)
    overview.to_csv(OUT / "027_06_current_five_station_rl_overview.csv", index=False, encoding="utf-8-sig")

    audit = {
        "status": "completed",
        "training_steps": 0,
        "dssat_runs": 0,
        "sites": len(overview),
        "strict_any_metric_above_all_four_count": int(overview["strict_any_metric_above_all_four"].sum()),
        "comparable_any_metric_winner_count": int(overview["comparable_any_metric_winner"].sum()),
        "universal_framework_verified_all_five": False,
        "current_stage_maskableppo_formally_trained_sites": ["SY", "HLA"],
        "current_stage_maskableppo_not_yet_formally_trained_sites": ["YC", "FQ", "LC"],
    }
    (OUT / "027_06_result.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    display_cols = [
        "site", "year", "representative_model", "selected_checkpoint_step", "step_unit",
        "grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "nitrogen_total_kg_ha",
        "irrigation_total_mm", "max_water_stress_WSPD", "max_nitrogen_stress_NSTD",
        "final_soil_water_SWTD_mm", "winning_metric", "comparable_any_metric_winner",
    ]
    markdown_table = overview[display_cols].round(3).to_markdown(index=False)
    lines = [
        "# 027_06 五站点当前强化学习结果总览",
        "",
        "## 结论先行",
        "",
        "当前没有证据证明存在一套完全相同的参数或同一套模型权重，可使五个站点各自训练后都达到“至少一个指标严格高于另外四情景”。",
        "",
        "阶段型 MaskablePPO 已在 SY 与 HLA 使用同一算法和同一组核心超参数；SY 明确通过产量与 WP_ET 两个全五情景可比指标，HLA 仅在正施氮情景可比的 PFP_N 上排名第一。YC/FQ/LC 尚未正式运行这套阶段型 MaskablePPO，因此不能把历史 DQN 候选当成该统一框架已经覆盖五站点的证据。",
        "",
        "## 当前每站点代表候选",
        "",
        markdown_table,
        "",
        "## 判定口径",
        "",
        "- 产量和 WP_ET：候选必须严格高于另外四情景才算全五情景第一。",
        "- PFP_N：施氮量为0时数学上未定义，不能把 NA 当作0，也不能声称严格超过四个数；仅另报在正施氮、PFP_N可定义情景中的排名。",
        "- WSPD/NSTD 为季内最大胁迫指数，0表示无胁迫；SWTD为收获时土壤剖面水量，不是越高越好，需与胁迫和投入联合解释。",
        "- “接近超过”尚无导师给定数值阈值，因此表中给出相对最佳其他情景的百分比差值，不擅自判定接近与否。",
        "",
        "## 统一框架判断",
        "",
        "1. 同一模型权重跨站点直接应用：当前证据不支持。",
        "2. 完全相同的全部参数：当前也不成立，因为每站点至少需要自己的输入文件、IC、物候/阶段环境和观测scaler。",
        "3. 同一算法代码和核心超参数、每站点独立训练：这是当前最可行的统一框架。SY/HLA已使用 MaskablePPO `[32,32]`、lr=3e-4、gamma=1、GAE lambda=1、n_steps=60、batch=30、n_epochs=5、240阶段步；但只验证了两个站点，尚不能称五站点通用。",
        "4. 历史DQN五站点候选受021_05探索率日程问题影响，且YC/FQ/LC在当前五情景表中没有任何一个全五情景可比指标严格第一，不能作为通用成功框架的正式证据。",
        "",
        "## 下一项最小实验",
        "",
        "保持SY/HLA已经冻结的阶段型MaskablePPO算法、核心超参数和240阶段步协议不变，按027_00顺序在YC2014、FQ2016、LC2010分别做站点专属输入/scaler准备和三seed训练。只有三站也达到预注册跨seed门槛后，才能回答“同一训练框架是否适用于五站点”。在此之前不应改PPO参数或宣称已有万能参数。",
        "",
        "## 来源",
        "",
        "- `benchmark_results/027_05/027_05_dqn_five_scenario_summary.csv`",
        "- `benchmark_results/027_05/027_05_ppo_five_scenario_summary.csv`",
        "- `benchmark_results/027_05/027_05_five_scenario_summary.csv`",
        "- `benchmark_results/027_05/027_05_daily_values.csv`",
        "- `docs/2026-07-17_026_04_sy2014_stage_maskable_ppo_seed2_three_seed_confirmation.md`",
        "- `docs/2026-07-17_027_03_hla2010_stage_maskable_ppo_three_seed_replication.md`",
        "- `prompts/027_00_five_station_site_specific_stage_maskable_ppo_protocol.md`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
