from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/yc_random_weather_ppo/004_05"
OUT = ROOT / "results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/best_seed_seed3"
BASELINE = ROOT / "benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/055_03_yca_five_scenario_season_summary.csv"
EPISODES = RESULTS / "all_evaluation_episode_level_0_7.csv"
EVENTS = OUT / "seed3_observed_management_events.csv"
MANIFEST = OUT / "seed3_diagnostic_manifest.json"
CSV_OUT = OUT / "seed3_fixed_validation_year_comparison.csv"
REPORT_OUT = ROOT / "docs/yc_random_weather_seed3_fixed_validation_year_assessment.md"

YEARS = list(range(2014, 2024))
BASELINE_SCENARIOS = {
    "null": "Null",
    "dssat_auto_external_n": "DSSAT auto + external N",
    "official_extension_expert": "Official expert",
    "recorded_farmer_template": "Recorded farmer",
}
PPO_SCENARIO = "Random-weather PPO seed 3"
COMMON_REWARD = "common_reward_05503"
COMMON_FORMULA = "0.158 * yield - 1.1 * irrigation - 1.58 * nitrogen"
LOWER_BETTER = {"irrigation", "nitrogen"}


def dominates(left: pd.Series, right: pd.Series) -> bool:
    objectives = ["yield", "irrigation", "nitrogen", COMMON_REWARD]
    left_values = [left[k] * (-1 if k in LOWER_BETTER else 1) for k in objectives]
    right_values = [right[k] * (-1 if k in LOWER_BETTER else 1) for k in objectives]
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def main() -> None:
    episodes = pd.read_csv(EPISODES)
    episodes = episodes[
        (episodes["training_regime"] == "RANDOM_WEATHER_WGEN")
        & (episodes["ppo_seed"] == 3)
        & (episodes["evaluation_weather_type"] == "observed_weather")
    ].copy()
    episodes["year"] = episodes["evaluation_weather_year"].astype(int)
    if sorted(episodes["year"].tolist()) != YEARS:
        raise ValueError("seed 3 must have exactly one observed-weather episode per validation year")
    if episodes["year"].duplicated().any():
        raise ValueError("duplicate seed 3 episodes found for a validation year")

    baselines = pd.read_csv(BASELINE, keep_default_na=False)
    baselines = baselines[
        (baselines["site"] == "YC") & baselines["scenario"].isin(BASELINE_SCENARIOS)
    ].copy()
    numeric_columns = [
        "grain_yield_kg_ha",
        "actual_irrigation_mm",
        "actual_nitrogen_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "unified_reward",
    ]
    for column in numeric_columns:
        baselines[column] = pd.to_numeric(baselines[column], errors="coerce")
    if len(baselines) != len(YEARS) * len(BASELINE_SCENARIOS):
        raise ValueError("expected exactly four YC baseline scenarios for each of ten years")
    if baselines.duplicated(["year", "scenario"]).any():
        raise ValueError("duplicate baseline year/scenario rows found")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("selected_seed") != 3:
        raise ValueError("seed selection manifest does not identify seed 3")

    events = pd.read_csv(EVENTS)
    events["year"] = events["year"].astype(int)
    events["irrigation_action_mm"] = pd.to_numeric(events["irrigation_action_mm"], errors="coerce").fillna(0)
    events["fertilizer_action_kgN_ha"] = pd.to_numeric(events["fertilizer_action_kgN_ha"], errors="coerce").fillna(0)
    event_stats = {}
    for year, group in events.groupby("year"):
        active = (group["irrigation_action_mm"] > 0) | (group["fertilizer_action_kgN_ha"] > 0)
        event_stats[int(year)] = {
            "active_action_days": int(active.sum()),
            "irrigation_action_days": int((group["irrigation_action_mm"] > 0).sum()),
            "nitrogen_action_days": int((group["fertilizer_action_kgN_ha"] > 0).sum()),
            "event_irrigation_mm": float(group["irrigation_action_mm"].sum()),
            "event_nitrogen_kg_ha": float(group["fertilizer_action_kgN_ha"].sum()),
        }

    records = []
    annual_meta = []
    for year in YEARS:
        ppo = episodes.loc[episodes["year"] == year].iloc[0]
        episode_irrigation = float(ppo["total_irrigation"])
        episode_nitrogen = float(ppo["total_fertilizer"])
        episode_yield = float(ppo["yield"])
        baseline_year = baselines[baselines["year"] == year]
        rows = []
        for _, base in baseline_year.iterrows():
            irrigation = float(base["actual_irrigation_mm"])
            nitrogen = float(base["actual_nitrogen_kg_ha"])
            grain_yield = float(base["grain_yield_kg_ha"])
            pfp = grain_yield / nitrogen if nitrogen > 0 else float("nan")
            rows.append(
                {
                    "year": year,
                    "scenario": BASELINE_SCENARIOS[base["scenario"]],
                    "is_ppo": False,
                    "yield": grain_yield,
                    "irrigation": irrigation,
                    "nitrogen": nitrogen,
                    "pfp_n": pfp,
                    "wp_et": float(base["WP_ET_kg_m3"]) if pd.notna(base["WP_ET_kg_m3"]) else float("nan"),
                    COMMON_REWARD: float(base["unified_reward"]),
                    "canonical_reward": float("nan"),
                    "active_action_days": float("nan"),
                }
            )
        ppo_pfp = episode_yield / episode_nitrogen if episode_nitrogen > 0 else float("nan")
        rows.append(
            {
                "year": year,
                "scenario": PPO_SCENARIO,
                "is_ppo": True,
                "yield": episode_yield,
                "irrigation": episode_irrigation,
                "nitrogen": episode_nitrogen,
                "pfp_n": ppo_pfp,
                "wp_et": float("nan"),
                COMMON_REWARD: 0.158 * episode_yield - 1.1 * episode_irrigation - 1.58 * episode_nitrogen,
                "canonical_reward": float(ppo["reward"]),
                "active_action_days": event_stats.get(year, {}).get("active_action_days", float("nan")),
            }
        )
        year_frame = pd.DataFrame(rows)
        if len(year_frame) != 5:
            raise ValueError(f"year {year} does not have five scenarios")

        for metric in ["yield", "irrigation", "nitrogen", "pfp_n", "wp_et", COMMON_REWARD]:
            ascending = metric in LOWER_BETTER
            year_frame[f"{metric}_rank"] = year_frame[metric].rank(method="min", ascending=ascending)

        frontier = []
        for i, row in year_frame.iterrows():
            if not any(dominates(other, row) for j, other in year_frame.iterrows() if j != i):
                frontier.append(row["scenario"])
        year_frame["partial_pareto_frontier"] = year_frame["scenario"].isin(frontier)
        year_frame["pareto_dimensions"] = "yield:max; irrigation:min; nitrogen:min; common_reward_05503:max"
        year_frame["WP_ET_kg_m3"] = year_frame["wp_et"]
        year_frame["PFP_N_kg_kg"] = year_frame["pfp_n"]
        year_frame["canonical_reward_note"] = "PPO canonical reward only; not ranked against baseline common reward"
        records.extend(year_frame.to_dict(orient="records"))

        ppo_row = year_frame[year_frame["is_ppo"]].iloc[0]
        best_yield = year_frame[~year_frame["is_ppo"]].loc[lambda frame: frame["yield"].idxmax()]
        expert = year_frame[year_frame["scenario"] == "Official expert"].iloc[0]
        farmer = year_frame[year_frame["scenario"] == "Recorded farmer"].iloc[0]
        events_y = event_stats.get(year, {})
        event_agree = (
            abs(events_y.get("event_irrigation_mm", float("nan")) - episode_irrigation) < 1e-8
            and abs(events_y.get("event_nitrogen_kg_ha", float("nan")) - episode_nitrogen) < 1e-8
        )
        annual_meta.append(
            {
                "year": year,
                "ppo_yield": episode_yield,
                "best_yield_scenario": best_yield["scenario"],
                "best_yield": float(best_yield["yield"]),
                "ppo_yield_gap_to_best": episode_yield - float(best_yield["yield"]),
                "ppo_irrigation": episode_irrigation,
                "ppo_nitrogen": episode_nitrogen,
                "ppo_pfp_n": ppo_pfp,
                "ppo_common_reward": float(ppo_row[COMMON_REWARD]),
                "ppo_canonical_reward": float(ppo["reward"]),
                "ppo_partial_pareto": bool(ppo_row["partial_pareto_frontier"]),
                "frontier": ", ".join(frontier),
                "yield_rank": int(ppo_row["yield_rank"]),
                "irrigation_rank": int(ppo_row["irrigation_rank"]),
                "nitrogen_rank": int(ppo_row["nitrogen_rank"]),
                "pfp_n_rank": int(ppo_row["pfp_n_rank"]) if pd.notna(ppo_row["pfp_n_rank"]) else None,
                "wp_et_rank": None,
                "common_reward_rank": int(ppo_row[f"{COMMON_REWARD}_rank"]),
                "irrigation_saved_vs_expert": float(expert["irrigation"] - episode_irrigation),
                "nitrogen_saved_vs_expert": float(expert["nitrogen"] - episode_nitrogen),
                "irrigation_saved_vs_farmer": float(farmer["irrigation"] - episode_irrigation),
                "nitrogen_saved_vs_farmer": float(farmer["nitrogen"] - episode_nitrogen),
                "active_action_days": events_y.get("active_action_days"),
                "irrigation_action_days": events_y.get("irrigation_action_days"),
                "nitrogen_action_days": events_y.get("nitrogen_action_days"),
                "event_totals_match_episode": event_agree,
            }
        )

    result = pd.DataFrame(records)
    ordered = [
        "year", "scenario", "is_ppo", "yield", "yield_rank", "irrigation", "irrigation_rank",
        "nitrogen", "nitrogen_rank", "pfp_n", "pfp_n_rank", "wp_et", "wp_et_rank",
        COMMON_REWARD, f"{COMMON_REWARD}_rank", "canonical_reward", "partial_pareto_frontier",
        "pareto_dimensions", "WP_ET_kg_m3", "PFP_N_kg_kg", "active_action_days", "canonical_reward_note",
    ]
    result[ordered].to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

    annual = pd.DataFrame(annual_meta)
    ppo_frontier_years = int(annual["ppo_partial_pareto"].sum())
    yield_first = int((annual["yield_rank"] == 1).sum())
    irrigation_first = int((annual["irrigation_rank"] == 1).sum())
    nitrogen_first = int((annual["nitrogen_rank"] == 1).sum())
    pfp_first = int((annual["pfp_n_rank"] == 1).sum())
    common_reward_first = int((annual["common_reward_rank"] == 1).sum())
    any_year_beats_best_yield = int((annual["ppo_yield_gap_to_best"] > 0).sum())
    mean_yield = annual["ppo_yield"].mean()
    mean_i = annual["ppo_irrigation"].mean()
    mean_n = annual["ppo_nitrogen"].mean()
    mean_pfp = annual["ppo_pfp_n"].mean()
    formula_check = baselines.assign(
        formula_reward=0.158 * baselines["grain_yield_kg_ha"]
        - 1.1 * baselines["actual_irrigation_mm"]
        - 1.58 * baselines["actual_nitrogen_kg_ha"]
    )
    max_reward_rounding_delta = float(
        (formula_check["formula_reward"] - formula_check["unified_reward"]).abs().max()
    )
    annual_lines = []
    for row in annual.itertuples(index=False):
        annual_lines.append(
            f"| {row.year} | {row.ppo_yield:.0f} | {row.best_yield_scenario} ({row.best_yield:.0f}) | "
            f"{row.ppo_yield_gap_to_best:+.0f} | {row.ppo_irrigation:.0f} | {row.ppo_nitrogen:.0f} | "
            f"{row.ppo_pfp_n:.1f} | {row.ppo_common_reward:.1f} | "
            f"{'是' if row.ppo_partial_pareto else '否'} | {row.frontier} |"
        )

    report = f"""# YC 固定 PPO seed 3：2014–2023 验证年五情景审计

## 目的与边界

本次把同一个已训练的 Random-weather PPO seed 3 固定用于 2014–2023 十个验证年，逐年对照 Null、DSSAT auto + external N、Official expert、Recorded farmer 四个基准。没有逐年换模型或拼接年度冠军，也没有重训、重跑 DSSAT 或修改任何实验设置。

seed 3 是此前按 observed 与 heldout WGEN canonical reward 等权平均事后选出的候选（均值 0.763），选择使用了评估结果，因此存在选择偏差。本报告是描述性复核，不是独立确认性结论。

## 指标口径

- 五情景共同 reward 按 `{COMMON_FORMULA}` 计算，所有情景同口径。
- 基准 common reward 使用 055_03 season summary 的正式 `unified_reward`；用已舍入的 yield/水氮字段重算，与正式值最大差异为 **{max_reward_rounding_delta:.3f}**。PPO common reward 使用同一公式和 episode 精度。
- PPO canonical reward 单独列示，不与基准 `common_reward_05503` 混排。
- `PFP_N = yield / N`，仅在 N > 0 时定义；N=0 为 N/A。
- PPO 没有匹配的 ETCP replay，`WP_ET` 保留缺失。因此不能给 PPO 做 WP_ET 排名，也不能据此宣称完整农业效率 Pareto 最优。
- 下文的“部分 Pareto 前沿”仅使用五情景都有值的四维指标：yield 最大、灌溉最小、N 最小、common reward 最大。common reward 本身由前三项公式构成，非独立的新证据。

## 数据质量与管理闭合

- seed 3 observed episode：{len(episodes)} 行，覆盖 {min(YEARS)}–{max(YEARS)}，每年唯一一行。
- 四基准：{len(baselines)} 行，十年每年恰好四个情景；站点限定为 YC。
- PPO 管理事件与正式 episode 的总灌溉、施氮总量逐年一致：{'是' if all(x['event_totals_match_episode'] for x in annual_meta) else '否'}。
- 源文件：`{EPISODES.relative_to(ROOT)}`、`{BASELINE.relative_to(ROOT)}`、`{EVENTS.relative_to(ROOT)}`、`{MANIFEST.relative_to(ROOT)}`。

## 逐年表现

| 年份 | PPO yield | 当年最高产量基准 | PPO-最高基准 | 灌溉 mm | N kg/ha | PFP_N | common reward | PPO在部分前沿 | 当年部分前沿情景 |
|---:|---:|---|---:|---:|---:|---:|---:|:---:|---|
{chr(10).join(annual_lines)}

## 汇总判断

- 同一个 seed 3 完成十年评估。平均产量 **{mean_yield:.1f} kg/ha**，平均灌溉 **{mean_i:.1f} mm**，平均施氮 **{mean_n:.1f} kg/ha**，平均 `PFP_N` **{mean_pfp:.1f} kg grain/kg N**。
- 逐年排名第一次数（可比范围）：yield **{yield_first}/10**，灌溉最低 **{irrigation_first}/10**，N最低 **{nitrogen_first}/10**，`PFP_N` **{pfp_first}/10**，common reward **{common_reward_first}/10**。`WP_ET` 对 PPO 缺数，**0/10 可排名**。
- PPO 在四指标部分 Pareto 前沿中的年份数：**{ppo_frontier_years}/10**。该结果不能替代包含 WP_ET 的完整农业效率判定。
- PPO 超过当年四基准最高产量的年份数：**{any_year_beats_best_yield}/10**；其余年份需以表中产量差阅读，不能把“接近最高基准”说成胜出。
- 相对官方专家，seed 3 每年灌溉节省约 **{annual['irrigation_saved_vs_expert'].mean():.1f} mm**、施氮节省约 **{annual['nitrogen_saved_vs_expert'].mean():.1f} kg/ha**；相对记录农户模板，分别节省 **{annual['irrigation_saved_vs_farmer'].mean():.1f} mm** 和 **{annual['nitrogen_saved_vs_farmer'].mean():.1f} kg/ha**。节省投入并不自动等于综合改进，需与产量和效率并列判断。
- 管理事件表显示 PPO 多年重复使用相同灌溉总量或少数氮剂量档位；逐年措施数量和汇总值见 CSV。该行为适应性有限，不能仅由 episode return 较高推断策略已找到逐年精细最优措施。

seed 3 的管理画像：2014–2023 年灌溉总量均为 **75 mm**；施氮在 **80 或 120 kg/ha** 两档，正动作日为 **{min(x['active_action_days'] for x in annual_meta)}–{max(x['active_action_days'] for x in annual_meta)} 天/季**（十年合计 {sum(x['active_action_days'] for x in annual_meta)} 个动作日）。这说明当前固定策略虽给出一致的低灌溉方案，但年度动作响应较粗，2014/2019 的低产尤其需要逐日状态及 ETCP replay 复核。

## 结论与下一步

seed 3 是一个已训练、固定并实际用于十个验证年的候选，不是逐年挑选的不同策略。它呈现低于官方专家/农户的水氮投入和较高 `PFP_N` 的权衡，但产量并未稳定达到当年基准最高值；`WP_ET` 缺失又使综合效率结论不完整。因此当前证据支持“存在值得核验的资源效率权衡”，不支持“已经在每个验证年综合胜出”或“已获得完整 Pareto 最优策略”。

低算力下一步：优先为这个固定 seed 3 补做**evaluation-only、精确 ETCP/WP_ET replay**并核验同年基准口径；若 replay 通过，再按逐年完整指标重算前沿。完成前不扩大训练、不增加 seed、不按年挑选模型。

可复现命令：`python src/run_004_07_yc_fixed_seed3_validation_pareto_audit.py`

输出 CSV：`{CSV_OUT.relative_to(ROOT)}`。未 commit、未 push。
"""
    REPORT_OUT.write_text(report, encoding="utf-8")
    print(f"CSV: {CSV_OUT}")
    print(f"REPORT: {REPORT_OUT}")
    print(
        json.dumps(
            {
                "episodes": len(episodes),
                "baseline_rows": len(baselines),
                "event_totals_match_all_years": all(x["event_totals_match_episode"] for x in annual_meta),
                "ppo_frontier_years_partial": ppo_frontier_years,
                "ppo_yield_first_years": yield_first,
                "ppo_common_reward_first_years": common_reward_first,
                "ppo_years_above_best_baseline_yield": any_year_beats_best_yield,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
