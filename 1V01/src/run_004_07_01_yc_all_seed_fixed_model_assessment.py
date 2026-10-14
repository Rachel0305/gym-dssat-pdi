from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EPISODES = ROOT / "results/yc_random_weather_ppo/004_05/all_evaluation_episode_level_0_7.csv"
BASELINE = ROOT / "benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/055_03_yca_five_scenario_season_summary.csv"
OUT = ROOT / "results/yc_random_weather_ppo/004_05_review_figures/all_seed_fixed_model_assessment"
YEAR_CSV = OUT / "seed_year_comparison.csv"
LONG_CSV = OUT / "five_scenario_long.csv"
PROFILE_CSV = OUT / "seed_profile_summary.csv"
REPORT = ROOT / "docs/yc_random_weather_004_05_all_seed_fixed_model_assessment.md"

YEARS = list(range(2014, 2024))
SEEDS = list(range(8))
BASE_SCENARIOS = {
    "null": "Null",
    "dssat_auto_external_n": "DSSAT auto + external N",
    "official_extension_expert": "Official expert",
    "recorded_farmer_template": "Recorded farmer",
}
PPO_LABEL = "Random-weather PPO seed {seed}"
COMMON = "common_reward_05503"


def pareto_front(frame: pd.DataFrame) -> set[str]:
    # Common reward is reported/ranked, but the partial frontier uses the three
    # independent objectives because common reward is derived from them.
    front = set()
    for _, candidate in frame.iterrows():
        dominated = False
        for _, other in frame.iterrows():
            if other["scenario"] == candidate["scenario"]:
                continue
            no_worse = (
                other["yield"] >= candidate["yield"]
                and other["irrigation"] <= candidate["irrigation"]
                and other["nitrogen"] <= candidate["nitrogen"]
            )
            strictly_better = (
                other["yield"] > candidate["yield"]
                or other["irrigation"] < candidate["irrigation"]
                or other["nitrogen"] < candidate["nitrogen"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.add(candidate["scenario"])
    return front


def build_base_rows(baselines: pd.DataFrame, year: int) -> list[dict]:
    rows = []
    for _, base in baselines[baselines["year"] == year].iterrows():
        y = float(base["grain_yield_kg_ha"])
        irrigation = float(base["actual_irrigation_mm"])
        nitrogen = float(base["actual_nitrogen_kg_ha"])
        rows.append(
            {
                "year": year,
                "scenario": BASE_SCENARIOS[base["scenario"]],
                "ppo_seed": None,
                "yield": y,
                "irrigation": irrigation,
                "nitrogen": nitrogen,
                "pfp_n": y / nitrogen if nitrogen > 0 else float("nan"),
                "wp_et": float(base["WP_ET_kg_m3"]) if pd.notna(base["WP_ET_kg_m3"]) else float("nan"),
                COMMON: float(base["unified_reward"]),
                "canonical_reward": float("nan"),
                "action_events": float("nan"),
                "irrigation_event_count": float("nan"),
                "fertilizer_event_count": float("nan"),
                "is_ppo": False,
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = pd.read_csv(EPISODES)
    episodes = episodes[
        (episodes["training_regime"] == "RANDOM_WEATHER_WGEN")
        & (episodes["evaluation_weather_type"] == "observed_weather")
        & episodes["ppo_seed"].isin(SEEDS)
    ].copy()
    episodes["year"] = episodes["evaluation_weather_year"].astype(int)
    expected = pd.MultiIndex.from_product([SEEDS, YEARS], names=["ppo_seed", "year"])
    actual = pd.MultiIndex.from_frame(episodes[["ppo_seed", "year"]])
    if len(episodes) != 80 or actual.has_duplicates or set(actual) != set(expected):
        raise ValueError("expected exactly one observed episode for every seed 0-7 and year 2014-2023")

    baseline = pd.read_csv(BASELINE, keep_default_na=False)
    baseline = baseline[(baseline["site"] == "YC") & baseline["scenario"].isin(BASE_SCENARIOS)].copy()
    numeric = [
        "grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha",
        "WP_ET_kg_m3", "PFP_N_kg_kg", "unified_reward",
    ]
    for col in numeric:
        baseline[col] = pd.to_numeric(baseline[col], errors="coerce")
    if len(baseline) != 40 or baseline.duplicated(["year", "scenario"]).any():
        raise ValueError("expected exactly four unique YC baseline rows per validation year")

    all_year_rows = []
    long_rows = []
    for year in YEARS:
        rows = build_base_rows(baseline, year)
        for _, ep in episodes[episodes["year"] == year].sort_values("ppo_seed").iterrows():
            y = float(ep["yield"])
            irrigation = float(ep["total_irrigation"])
            nitrogen = float(ep["total_fertilizer"])
            rows.append(
                {
                    "year": year,
                    "scenario": PPO_LABEL.format(seed=int(ep["ppo_seed"])),
                    "ppo_seed": int(ep["ppo_seed"]),
                    "yield": y,
                    "irrigation": irrigation,
                    "nitrogen": nitrogen,
                    "pfp_n": y / nitrogen if nitrogen > 0 else float("nan"),
                    "wp_et": float("nan"),
                    COMMON: 0.158 * y - 1.1 * irrigation - 1.58 * nitrogen,
                    "canonical_reward": float(ep["reward"]),
                    "action_events": float(ep["action_events"]),
                    "irrigation_event_count": float(ep["irrigation_event_count"]),
                    "fertilizer_event_count": float(ep["fertilizer_event_count"]),
                    "is_ppo": True,
                }
            )

        year_frame = pd.DataFrame(rows)
        if len(year_frame) != 12:
            raise ValueError(f"year {year} expected four baselines and eight PPO models")
        for metric, ascending in [
            ("yield", False), ("irrigation", True), ("nitrogen", True),
            ("pfp_n", False), ("wp_et", False), (COMMON, False),
        ]:
            year_frame[f"{metric}_rank_all_candidates"] = year_frame[metric].rank(method="min", ascending=ascending)
        front_all = pareto_front(year_frame)
        year_frame["partial_pareto_all_candidates"] = year_frame["scenario"].isin(front_all)
        year_frame["wp_et_note"] = "PPO WP_ET unavailable: no exact seed-year ETCP replay"
        long_rows.extend(year_frame.to_dict(orient="records"))

        baseline_year = year_frame[~year_frame["is_ppo"]]
        base_max_yield = float(baseline_year["yield"].max())
        for seed in SEEDS:
            ppo_row = year_frame[year_frame["ppo_seed"] == seed].iloc[0]
            rank_columns = ["scenario", "yield", "irrigation", "nitrogen", "pfp_n", COMMON]
            rank_rows = baseline_year[rank_columns].to_dict(orient="records")
            rank_rows.append(ppo_row[rank_columns].to_dict())
            scenario5 = pd.DataFrame(rank_rows)
            for metric, ascending in [
                ("yield", False), ("irrigation", True), ("nitrogen", True),
                ("pfp_n", False), (COMMON, False),
            ]:
                scenario5[f"{metric}_rank_five"] = scenario5[metric].rank(method="min", ascending=ascending)
            front5 = pareto_front(scenario5)
            selected = scenario5[scenario5["scenario"] == PPO_LABEL.format(seed=seed)].iloc[0]
            all_year_rows.append(
                {
                    "year": year,
                    "ppo_seed": seed,
                    "yield": float(selected["yield"]),
                    "yield_rank_vs_four_baselines": float(selected["yield_rank_five"]),
                    "irrigation_mm": float(selected["irrigation"]),
                    "irrigation_rank_vs_four_baselines": float(selected["irrigation_rank_five"]),
                    "nitrogen_kg_ha": float(selected["nitrogen"]),
                    "nitrogen_rank_vs_four_baselines": float(selected["nitrogen_rank_five"]),
                    "pfp_n": float(selected["pfp_n"]) if pd.notna(selected["pfp_n"]) else float("nan"),
                    "pfp_n_rank_vs_four_baselines": float(selected["pfp_n_rank_five"]) if pd.notna(selected["pfp_n_rank_five"]) else float("nan"),
                    "wp_et": float("nan"),
                    "common_reward_05503": float(selected[COMMON]),
                    "common_reward_rank_vs_four_baselines": float(selected[f"{COMMON}_rank_five"]),
                    "canonical_reward": float(ppo_row["canonical_reward"]),
                    "partial_pareto_vs_four_baselines": selected["scenario"] in front5,
                    "yield_gap_vs_best_baseline": float(selected["yield"]) - base_max_yield,
                    "action_events": float(ppo_row["action_events"]),
                    "irrigation_event_count": float(ppo_row["irrigation_event_count"]),
                    "fertilizer_event_count": float(ppo_row["fertilizer_event_count"]),
                }
            )

    year_df = pd.DataFrame(all_year_rows)
    long_df = pd.DataFrame(long_rows)
    year_df.to_csv(YEAR_CSV, index=False, encoding="utf-8-sig")
    long_df.to_csv(LONG_CSV, index=False, encoding="utf-8-sig")

    profiles = []
    for seed, group in year_df.groupby("ppo_seed"):
        profiles.append(
            {
                "ppo_seed": int(seed),
                "mean_yield": group["yield"].mean(),
                "sd_yield": group["yield"].std(ddof=1),
                "mean_irrigation_mm": group["irrigation_mm"].mean(),
                "mean_nitrogen_kg_ha": group["nitrogen_kg_ha"].mean(),
                "mean_pfp_n": group["pfp_n"].mean(),
                "mean_canonical_reward": group["canonical_reward"].mean(),
                "mean_common_reward_05503": group["common_reward_05503"].mean(),
                "mean_yield_rank": group["yield_rank_vs_four_baselines"].mean(),
                "yield_first_years": int((group["yield_rank_vs_four_baselines"] == 1).sum()),
                "irrigation_min_years": int((group["irrigation_rank_vs_four_baselines"] == 1).sum()),
                "nitrogen_min_years": int((group["nitrogen_rank_vs_four_baselines"] == 1).sum()),
                "pfp_n_first_years": int((group["pfp_n_rank_vs_four_baselines"] == 1).sum()),
                "common_reward_first_years": int((group["common_reward_rank_vs_four_baselines"] == 1).sum()),
                "partial_pareto_years": int(group["partial_pareto_vs_four_baselines"].sum()),
                "years_above_best_baseline_yield": int((group["yield_gap_vs_best_baseline"] > 0).sum()),
                "mean_yield_gap_vs_best_baseline": group["yield_gap_vs_best_baseline"].mean(),
                "worst_yield_gap_vs_best_baseline": group["yield_gap_vs_best_baseline"].min(),
                "mean_action_events": group["action_events"].mean(),
                "mean_irrigation_events": group["irrigation_event_count"].mean(),
                "mean_nitrogen_events": group["fertilizer_event_count"].mean(),
            }
        )
    profile = pd.DataFrame(profiles).sort_values("ppo_seed")
    profile.to_csv(PROFILE_CSV, index=False, encoding="utf-8-sig")

    baseline_check = baseline.assign(
        formula_reward=0.158 * baseline["grain_yield_kg_ha"]
        - 1.1 * baseline["actual_irrigation_mm"]
        - 1.58 * baseline["actual_nitrogen_kg_ha"]
    )
    max_reward_delta = float((baseline_check["formula_reward"] - baseline_check["unified_reward"]).abs().max())
    best_mean_yield = int(profile.loc[profile["mean_yield"].idxmax(), "ppo_seed"])
    best_common = int(profile.loc[profile["mean_common_reward_05503"].idxmax(), "ppo_seed"])
    best_pfp_value = profile["mean_pfp_n"].max()
    best_pfp_seeds = ", ".join(
        str(int(seed)) for seed in profile.loc[profile["mean_pfp_n"] == best_pfp_value, "ppo_seed"]
    )
    low_input = profile.sort_values(["mean_irrigation_mm", "mean_nitrogen_kg_ha"]).iloc[0]

    profile_lines = []
    for row in profile.itertuples(index=False):
        profile_lines.append(
            f"| {row.ppo_seed} | {row.mean_yield:.0f} | {row.mean_irrigation_mm:.1f} | {row.mean_nitrogen_kg_ha:.1f} | "
            f"{row.mean_pfp_n:.1f} | {row.mean_common_reward_05503:.1f} | {row.yield_first_years}/10 | "
            f"{row.pfp_n_first_years}/10 | {row.common_reward_first_years}/10 | {row.partial_pareto_years}/10 | "
            f"{row.years_above_best_baseline_yield}/10 | {row.mean_yield_gap_vs_best_baseline:+.0f} | {row.mean_action_events:.1f} |"
        )

    seed3 = profile[profile["ppo_seed"] == 3].iloc[0]
    report = f"""# YC 004_05：全部 Random-weather PPO seed 的逐年固定模型对照

## 评估设计

对 8 个已训练 seed 分别固定模型，各自使用已有正式 episode 覆盖 2014–2023 十个验证年；每个 seed 独立与同年四基准比较。没有逐年换 seed，也没有拼接不同 seed 的年度最佳结果。没有训练、DSSAT replay 或天气生成。

seed 3 是此前按 observed/heldout canonical reward 等权平均事后选出的，不应因此忽略其余 seed；但现在再按十年验证数据比较并挑候选仍带有选择偏差，所以本报告只能用于探索性筛查，不能当作未见数据上的确认结果。

## 数据质量与口径

- PPO observed episodes：**{len(episodes)} / 80**，完整包含 seed 0–7 × 2014–2023，每个 seed-year 唯一。
- YC 四基准：**{len(baseline)} / 40**，每年四个情景完整且唯一。
- PPO `WP_ET`：全部缺失，未做推算；`PFP_N` 在 N>0 时按 yield/N 计算，N=0 记 N/A。
- PPO common reward 按 `0.158*yield - 1.1*irrigation - 1.58*N` 计算；基准使用正式 `unified_reward`。基准已舍入列重新计算的最大差值为 **{max_reward_delta:.3f}**。PPO canonical reward 与其分列。
- 每个 seed-year 的 Pareto 是与四基准组成的五情景部分前沿，使用 yield 最大、水和 N 最小三项独立指标。common reward 单独排名；它是水、N、yield 的派生式，不重复充当 Pareto 维度。

## 八个固定模型汇总

| Seed | 年均产量 kg/ha | 灌溉 mm | N kg/ha | PFP_N | common reward | Yield 第一 | PFP_N 第一 | Reward 第一 | 部分前沿年数 | 超过年度最高基准产量 | 对年度最高基准平均差 | 动作日/季 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(profile_lines)}

## 结果解读

- 八个 seed 的侧重点不同，不存在可以只凭一个均值说“全面最好”的模型。按当前表格的边际摘要，年均产量最高的是 seed **{best_mean_yield}**；年均 common reward 最高的是 seed **{best_common}**；年均 `PFP_N` 最高的是 seed **{best_pfp_seeds}**；平均灌溉最低的是 seed **{int(low_input['ppo_seed'])}**（其 N 均值 {low_input['mean_nitrogen_kg_ha']:.1f} kg/ha）。这些是不同优化维度，不构成统一冠军。
- seed 3 的位置：均产 {seed3['mean_yield']:.0f} kg/ha，灌溉 {seed3['mean_irrigation_mm']:.1f} mm，施氮 {seed3['mean_nitrogen_kg_ha']:.1f} kg/ha，`PFP_N` {seed3['mean_pfp_n']:.1f}，common reward {seed3['mean_common_reward_05503']:.1f}；yield 第一 {int(seed3['yield_first_years'])}/10、common reward 第一 {int(seed3['common_reward_first_years'])}/10、部分前沿 {int(seed3['partial_pareto_years'])}/10。它是有吸引力的低投入候选，但不能代表其他 seed。
- 表内“超过年度最高基准产量”与“平均产量最高”是两种不同证据；逐年差值保存在 `seed_year_comparison.csv`。不能按年挑 seed 后拼成一个实际 PPO 模型。
- 原始配对实验的预设成功判据仍为 **0/8**（`results/yc_random_weather_ppo/004_05/multi_seed_decision.json`）；在相同验证结果中筛出表现较好的 seed，不能证明 weather augmentation 对 PPO 模型族整体有效。
- `PFP_N` 与用氮量并列阅读；没有 PPO `WP_ET` 就无法完成水分效率评价。`common_reward` 与 canonical PPO reward 不是一个标度。

## 建议

不启动新训练。先从 `seed_profile_summary.csv` 和 `seed_year_comparison.csv` 选出少量**不同权衡类型**的固定候选（高产、低投入、高 PFP_N），再做精确、evaluation-only `Summary.OUT/ETCP` replay 补齐这些候选十年的 `WP_ET`。根据完整逐年指标重新判断 Pareto 支配关系；如果要做确认性结论，另留未参与 seed 选择的新天气/年份验证。

脚本：`src/run_004_07_01_yc_all_seed_fixed_model_assessment.py`

输出：
- `results/yc_random_weather_ppo/004_05_review_figures/all_seed_fixed_model_assessment/seed_year_comparison.csv`（80 行）
- `results/yc_random_weather_ppo/004_05_review_figures/all_seed_fixed_model_assessment/five_scenario_long.csv`（120 行扩展候选池）
- `results/yc_random_weather_ppo/004_05_review_figures/all_seed_fixed_model_assessment/seed_profile_summary.csv`（8 行）

运行命令：`python src/run_004_07_01_yc_all_seed_fixed_model_assessment.py`。未 commit、未 push。
"""
    REPORT.write_text(report, encoding="utf-8")
    print(f"seed_year: {YEAR_CSV}")
    print(f"candidate_long: {LONG_CSV}")
    print(f"seed_profile: {PROFILE_CSV}")
    print(f"report: {REPORT}")
    print(profile[["ppo_seed", "mean_yield", "mean_irrigation_mm", "mean_nitrogen_kg_ha", "mean_pfp_n", "mean_common_reward_05503", "yield_first_years", "common_reward_first_years", "partial_pareto_years", "years_above_best_baseline_yield"]].to_string(index=False))


if __name__ == "__main__":
    main()
