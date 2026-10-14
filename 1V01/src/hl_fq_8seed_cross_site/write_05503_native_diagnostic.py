"""Summarize the frozen native-auto/corrected-2018 comparison with QA evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hl_fq_8seed_cross_site/055_03_native_auto_corrected2018"
ORIGINAL = ROOT / "results/hl_fq_8seed_cross_site/055_03_five_scenario"


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def main() -> None:
    summary = []
    corrected2018 = []
    closure_issues = []
    png_count = 0
    for site, count in (("HL", 8), ("FQ", 7)):
        for seed in range(count):
            prefix = f"{site.lower()}_seed{seed}"
            folder = OUT / site / f"seed{seed}"
            tables = folder / "tables"
            metrics = pd.read_csv(tables / f"{prefix}_yearly_five_scenario_metrics.csv", keep_default_na=False)
            daily = pd.read_csv(tables / f"{prefix}_five_scenario_daily.csv", keep_default_na=False)
            figures = list((folder / "figures").glob("*.png"))
            png_count += len(figures)
            if len(metrics) != 50 or len(figures) != 23:
                raise ValueError(f"Coverage failure {site} seed{seed}")
            auto = metrics.loc[metrics.scenario.eq("dssat_auto")]
            if len(auto) != 10 or not num(auto.total_n_kg_ha).eq(0).all():
                raise ValueError(f"Native auto N failure {site} seed{seed}")
            for row in metrics.itertuples():
                subset = daily.loc[daily.year.eq(row.year) & daily.scenario.eq(row.scenario)]
                if subset.empty:
                    closure_issues.append((site, seed, row.year, row.scenario, "missing daily"))
                    continue
                yield_diff = float(subset.grain_yield_kg_ha.iloc[-1]) - float(row.grain_yield_kg_ha)
                i_diff = float(num(subset.irrigation_executed_mm).sum()) - float(row.total_irrigation_mm)
                n_diff = float(num(subset.nitrogen_executed_kg_ha).sum()) - float(row.total_n_kg_ha)
                if abs(yield_diff) > 1 or abs(i_diff) > 0.6 or abs(n_diff) > 0.6:
                    closure_issues.append((site, seed, row.year, row.scenario, yield_diff, i_diff, n_diff))
            ppo = metrics.loc[metrics.scenario.eq(f"ppo_seed_{seed}")].set_index("year")
            expert = metrics.loc[metrics.scenario.eq("official_extension_expert")].set_index("year")
            null = metrics.loc[metrics.scenario.eq("null")].set_index("year")
            audit_path = ORIGINAL / site / f"best_seed_seed{seed}" / "tables" / f"{prefix}_action_sequence_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            summary.append({
                "site": site, "seed": seed,
                "mean_ppo_yield_kg_ha": num(ppo.grain_yield_kg_ha).mean(),
                "mean_expert_yield_kg_ha": num(expert.grain_yield_kg_ha).mean(),
                "mean_ppo_minus_expert_yield_kg_ha": num(ppo.grain_yield_kg_ha).mean() - num(expert.grain_yield_kg_ha).mean(),
                "mean_null_yield_kg_ha": num(null.grain_yield_kg_ha).mean(),
                "mean_native_auto_yield_kg_ha": num(auto.grain_yield_kg_ha).mean(),
                "mean_native_auto_irrigation_mm": num(auto.total_irrigation_mm).mean(),
                "mean_native_auto_n_kg_ha": num(auto.total_n_kg_ha).mean(),
                "mean_ppo_irrigation_mm": num(ppo.total_irrigation_mm).mean(),
                "mean_expert_irrigation_mm": num(expert.total_irrigation_mm).mean(),
                "mean_ppo_n_kg_ha": num(ppo.total_n_kg_ha).mean(),
                "mean_expert_n_kg_ha": num(expert.total_n_kg_ha).mean(),
                "mean_ppo_WP_ET_kg_m3": num(ppo.WP_ET_kg_m3).mean(),
                "mean_expert_WP_ET_kg_m3": num(expert.WP_ET_kg_m3).mean(),
                "mean_ppo_PFP_N_kg_kg": num(ppo.PFP_N_kg_grain_per_kg_N).mean(),
                "mean_expert_PFP_N_kg_kg": num(expert.PFP_N_kg_grain_per_kg_N).mean(),
                "ppo_yield_above_expert_years": int(num(ppo.grain_yield_kg_ha).gt(num(expert.grain_yield_kg_ha)).sum()),
                "ppo_yield_below_expert_by_over_5pct_years": int((num(ppo.grain_yield_kg_ha) / num(expert.grain_yield_kg_ha)).lt(0.95).sum()),
                "original_input_unique_positive_action_sequences": audit["unique_yearly_action_sequence_count"],
            })
            if site == "FQ":
                for row in metrics.loc[metrics.year.eq(2018)].itertuples():
                    corrected2018.append({
                        "seed": seed, "scenario": row.scenario,
                        "yield_kg_ha": row.grain_yield_kg_ha,
                        "irrigation_mm": row.total_irrigation_mm,
                        "n_kg_ha": row.total_n_kg_ha,
                        "WP_ET_kg_m3": row.WP_ET_kg_m3,
                        "PFP_N_kg_kg": row.PFP_N_kg_grain_per_kg_N,
                    })
    if closure_issues:
        raise ValueError(f"Season/daily closure failed: {closure_issues[:8]}")
    if png_count != 345:
        raise ValueError(f"Expected 345 figures, found {png_count}")
    frame = pd.DataFrame(summary)
    frame.to_csv(OUT / "seed_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(corrected2018).to_csv(OUT / "fq2018_corrected_all_seed_summary.csv", index=False, encoding="utf-8-sig")
    h5 = frame.query("site == 'HL' and seed == 5").iloc[0]
    h7 = frame.query("site == 'HL' and seed == 7").iloc[0]
    f2 = frame.query("site == 'FQ' and seed == 2").iloc[0]
    f3 = frame.query("site == 'FQ' and seed == 3").iloc[0]
    f4 = frame.query("site == 'FQ' and seed == 4").iloc[0]
    f6 = frame.query("site == 'FQ' and seed == 6").iloc[0]
    if any(abs(f2[key] - f6[key]) > 1e-9 for key in ("mean_ppo_yield_kg_ha", "mean_ppo_irrigation_mm", "mean_ppo_n_kg_ha", "mean_ppo_WP_ET_kg_m3")):
        raise ValueError("Expected seed2/seed6 identity did not hold")
    weather_checks = []
    for seed in range(7):
        path = OUT / "FQ" / f"seed{seed}" / "tables" / f"fq_seed{seed}_five_scenario_daily.csv"
        daily = pd.read_csv(path, keep_default_na=False)
        null = daily.loc[daily.year.eq(2018) & daily.scenario.eq("null"), ["date", "rainfall_mm", "tmax_c", "tmin_c"]]
        ppo = daily.loc[daily.year.eq(2018) & daily.scenario.eq(f"ppo_seed_{seed}"), ["date", "rainfall_mm", "tmax_c", "tmin_c"]]
        merged = null.merge(ppo, on="date", suffixes=("_null", "_ppo"))
        if len(null) != 97 or len(ppo) != 97 or len(merged) != 97:
            raise ValueError(f"FQ2018 weather date closure failure seed{seed}")
        maximum = max(float((merged[f"{key}_null"] - merged[f"{key}_ppo"]).abs().max())
                      for key in ("rainfall_mm", "tmax_c", "tmin_c"))
        if maximum > 1e-6:
            raise ValueError(f"FQ2018 weather mismatch seed{seed}: {maximum}")
        weather_checks.append({"seed": seed, "matched_days": 97, "max_weather_difference": maximum})
    report = f"""# HL/FQ 五情景对照修订与可信度诊断

## 口径与文件

- 本版图集：`GALLERY.md`；每个 seed 23 张图，合计 {png_count} 张。旧版原始天气图位于 `../055_03_five_scenario/`，未覆盖。
- HL/FQ `dssat_auto` 均来自冻结的原生自动灌溉基线，外施 N 为 0；旧图所用 `dssat_auto_external_n` 已从新版五情景对照移除。HL 原生 auto 十年均值为 {h5.mean_native_auto_yield_kg_ha:.1f} kg/ha、{h5.mean_native_auto_irrigation_mm:.1f} mm、0 kg N/ha。
- FQ2018 使用 088 任务已有的隔离 Tmin 修正观测 `.WTH`，七个现有 100K PPO checkpoint 分别重新评估。模型、训练天气和其他年份均未改动；原始 2018 零产量结果保留在旧版。
- `seed_comparison.csv` 保存 15 个 seed 的十年均值、逐年胜场与序列统计；`fq2018_corrected_all_seed_summary.csv` 保存 FQ2018 全部 seed 与基线。

## HL seed5 与 seed7

| seed | 平均产量 | 专家产量 | 灌溉 / 专家 | N / 专家 | WP_ET / 专家 | PFP_N / 专家 | 超过专家的年份 | 原天气完整措施序列数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | {h5.mean_ppo_yield_kg_ha:.1f} | {h5.mean_expert_yield_kg_ha:.1f} | {h5.mean_ppo_irrigation_mm:.0f} / {h5.mean_expert_irrigation_mm:.0f} | {h5.mean_ppo_n_kg_ha:.0f} / {h5.mean_expert_n_kg_ha:.0f} | {h5.mean_ppo_WP_ET_kg_m3:.3f} / {h5.mean_expert_WP_ET_kg_m3:.3f} | {h5.mean_ppo_PFP_N_kg_kg:.1f} / {h5.mean_expert_PFP_N_kg_kg:.1f} | {h5.ppo_yield_above_expert_years}/10 | {h5.original_input_unique_positive_action_sequences} |
| 7 | {h7.mean_ppo_yield_kg_ha:.1f} | {h7.mean_expert_yield_kg_ha:.1f} | {h7.mean_ppo_irrigation_mm:.0f} / {h7.mean_expert_irrigation_mm:.0f} | {h7.mean_ppo_n_kg_ha:.0f} / {h7.mean_expert_n_kg_ha:.0f} | {h7.mean_ppo_WP_ET_kg_m3:.3f} / {h7.mean_expert_WP_ET_kg_m3:.3f} | {h7.mean_ppo_PFP_N_kg_kg:.1f} / {h7.mean_expert_PFP_N_kg_kg:.1f} | {h7.ppo_yield_above_expert_years}/10 | {h7.original_input_unique_positive_action_sequences} |

seed5 的十年水、N **总量**相同，但逐年完整事件日期有 6 种；不能称为十年完全相同的 action sequence。其平均产量只比专家高 {h5.mean_ppo_minus_expert_yield_kg_ha:.1f} kg/ha，属于几乎持平的模拟结果。seed7 的措施有 3 种，平均产量低 {abs(h7.mean_ppo_minus_expert_yield_kg_ha):.1f} kg/ha；有 {h7.ppo_yield_below_expert_by_over_5pct_years} 年低于专家超过 5%，因此是节水节氮但有产量下行风险的候选，不是稳健的全面等效。

## FQ seed2/3/4 的低投入现象

| seed | 十年产量 | 与专家差 | 灌溉 | N | WP_ET | PFP_N | 与原生 auto 比较 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | {f2.mean_ppo_yield_kg_ha:.1f} | {f2.mean_ppo_minus_expert_yield_kg_ha:+.1f} | {f2.mean_ppo_irrigation_mm:.0f} | {f2.mean_ppo_n_kg_ha:.0f} | {f2.mean_ppo_WP_ET_kg_m3:.3f} | {f2.mean_ppo_PFP_N_kg_kg:.1f} | {f2.mean_ppo_yield_kg_ha-f2.mean_native_auto_yield_kg_ha:+.1f} kg/ha |
| 3 | {f3.mean_ppo_yield_kg_ha:.1f} | {f3.mean_ppo_minus_expert_yield_kg_ha:+.1f} | {f3.mean_ppo_irrigation_mm:.0f} | {f3.mean_ppo_n_kg_ha:.0f} | {f3.mean_ppo_WP_ET_kg_m3:.3f} | {f3.mean_ppo_PFP_N_kg_kg:.1f} | {f3.mean_ppo_yield_kg_ha-f3.mean_native_auto_yield_kg_ha:+.1f} kg/ha |
| 4 | {f4.mean_ppo_yield_kg_ha:.1f} | {f4.mean_ppo_minus_expert_yield_kg_ha:+.1f} | {f4.mean_ppo_irrigation_mm:.0f} | {f4.mean_ppo_n_kg_ha:.0f} | {f4.mean_ppo_WP_ET_kg_m3:.3f} | {f4.mean_ppo_PFP_N_kg_kg:.1f} | {f4.mean_ppo_yield_kg_ha-f4.mean_native_auto_yield_kg_ha:+.1f} kg/ha |

- seed2/3 原天气九年及零产量异常年均为单次起始行动，分别是 45 mm + 40 kg N、45 mm + 80 kg N；修正 2018 的 DSSAT 事件仍各为一次。seed6 与 seed2 在这套验证下的行动与指标相同，不能当作独立的策略形态。seed4 并非十年完全一致：原天气审计有 {f4.original_input_unique_positive_action_sequences} 种序列，N 在 200/240 kg/ha 间变化。
- FQ2018 修正后：seed2={pd.DataFrame(corrected2018).query('seed == 2 and scenario == "ppo_seed_2"').iloc[0].yield_kg_ha:.0f}、seed3={pd.DataFrame(corrected2018).query('seed == 3 and scenario == "ppo_seed_3"').iloc[0].yield_kg_ha:.0f}、seed4={pd.DataFrame(corrected2018).query('seed == 4 and scenario == "ppo_seed_4"').iloc[0].yield_kg_ha:.0f} kg/ha，专家约 8282 kg/ha。此年低投入策略明显未达到专家产量。
- 原生自动灌溉基线十年均值为 {f2.mean_native_auto_yield_kg_ha:.1f} kg/ha、{f2.mean_native_auto_irrigation_mm:.0f} mm、0 kg 外施 N，平均产量高于 seed2/3/4；因此不能仅凭与高投入专家接近就称这些 PPO 策略遥遥领先。
- FQ 每个验证年独立重置相同的 originIC。原生 null 的 SoilNiBal.OUT 显示开季土壤 NO3=90.03、NH4=57.20，共 147.23 kg N/ha；2014 年另矿化约 60.6 kg N/ha。seed2 的 2014 快照显示确实施入 40 kg N，作物从土壤吸收约 171.5 kg N/ha。PFP_N=产量/外施 N，其高值并未计入土壤初值和矿化 N，不能直接作为长期氮效率或可持续性的证明。

## 验收和限制

- {15*50} 条季节情景记录与逐日末产量及水、N 事件合计闭合：产量容差 1 kg/ha，水/N 容差 0.6（Summary.OUT 舍入）。FQ2023 原生 auto 的 MgmtEvent.OUT 含多次内部运行；逐日图已改为只读取最后一次运行，38.9 mm 事件合计与季节表 39.0 mm 的舍入值闭合。
- 七个 FQ2018 PPO 回放与 null 基线在同一 97 个日期上的降雨、最高温、最低温逐日完全一致；均无 `slowed grain filling` 或 `precipitation data error` 警告。seed0 回放复现 088 既有结果：8307 kg/ha、210 mm、240 kg N/ha、WP_ET 2.31。
- 检查支持“模拟产物内部一致、未发现 seed2/3 的施肥或产量记录错误”，并不能证明独立逐年重置土壤 N 的低投入策略可持续。连续多年土壤 N 消耗、田间实测校准和不确定性仍需另行研究。
- 新版 FQ2018 的快照事件 DAP 口径与旧 PPO 逐日验证日志不完全相同，故新版行动审计未强行判定十年精确序列是否一致；旧天气十年序列审计完整保留。
"""
    (OUT / "DIAGNOSTIC.md").write_text(report, encoding="utf-8")
    (OUT / "qa_manifest.json").write_text(json.dumps({
        "seeds": 15, "figures": png_count, "season_scenario_rows_checked": 750,
        "closure_issues": closure_issues, "native_auto_external_n_zero": True,
        "fq2018_weather_checks": weather_checks, "corrected_fq2018_ppo_replays": 7,
    }, indent=2), encoding="utf-8")
    print(f"Wrote diagnostic report; {png_count} PNG, {len(closure_issues)} closure issues", flush=True)


if __name__ == "__main__":
    main()
