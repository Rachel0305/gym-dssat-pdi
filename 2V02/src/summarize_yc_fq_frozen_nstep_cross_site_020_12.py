from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "DSSAT_auto_validation" / "frozen_nstep_cross_site_020_12"
DOC = ROOT / "docs" / "2026-07-11_020_12_yc_fq_frozen_nstep_cross_site_final_record.md"

WATER_COST = 1.0
NITROGEN_COST = 5.0


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rendered: list[str] = []
        for value in row:
            if pd.isna(value):
                text = ""
            elif isinstance(value, (float, np.floating)):
                text = f"{float(value):.3f}"
            elif isinstance(value, (int, np.integer)):
                text = str(int(value))
            else:
                text = str(value)
            rendered.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def unified_reward(yield_value: float, null_yield: float, irrigation: float, nitrogen: float) -> float:
    return max(0.0, float(yield_value) - float(null_yield)) - WATER_COST * float(irrigation) - NITROGEN_COST * float(nitrogen)


def choose_best(path: Path, seed: int, site: str, irrigation_col: str, nitrogen_col: str) -> dict[str, object]:
    frame = pd.read_csv(path)
    selected = frame.sort_values(["total_reward", "checkpoint_step"], ascending=[False, True], kind="stable").iloc[0]
    return {
        "site": site,
        "seed": seed,
        "checkpoint_step": int(selected["checkpoint_step"]),
        "final_grain_kg_ha": float(selected["final_grain_kg_ha"]),
        "irrigation_mm": float(selected[irrigation_col]),
        "nitrogen_kg_ha": float(selected[nitrogen_col]),
        "total_reward": float(selected["total_reward"]),
        "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
    }


def baseline_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    yc_path = ROOT / "DSSAT_auto_validation" / "yc2014_formal_four_scenario_015_06" / "seed0_seed1_best" / "015_06_yc2014_formal_four_scenario_summary.csv"
    yc = pd.read_csv(yc_path, keep_default_na=False)
    yc_null = float(yc.loc[yc["scenario"].eq("null"), "final_grain_kg_ha"].iloc[0])
    for scenario in ("null", "recorded", "dssat_auto"):
        row = yc.loc[yc["scenario"].eq(scenario)].iloc[0]
        irrigation = float(row["irrigation_total_mm"])
        nitrogen = float(row["fertilizer_total_kg_ha"])
        yield_value = float(row["final_grain_kg_ha"])
        rows.append(
            {
                "site": "YC",
                "station": "Yucheng",
                "year": 2014,
                "scenario_family": scenario,
                "dqn_seed": np.nan,
                "checkpoint_step": np.nan,
                "final_grain_kg_ha": yield_value,
                "final_biomass_kg_ha": float(row["final_biomass_kg_ha"]),
                "irrigation_mm": irrigation,
                "nitrogen_kg_ha": nitrogen,
                "max_water_stress": float(row["max_water_stress"]),
                "max_nitrogen_stress": float(row["max_nitrogen_stress"]),
                "unified_reward": unified_reward(yield_value, yc_null, irrigation, nitrogen),
                "source_file": str(yc_path.relative_to(ROOT)).replace("\\", "/"),
            }
        )

    fq_path = ROOT / "DSSAT_auto_validation" / "fq2016_four_scenario_process_017_02" / "fq2016_four_scenario_summary.csv"
    fq = pd.read_csv(fq_path, keep_default_na=False)
    fq_null = float(fq.loc[fq["scenario"].eq("null_zero"), "final_grain_kg_ha"].iloc[0])
    mapping = {"null_zero": "null", "recorded_shifted": "recorded", "dssat_auto": "dssat_auto"}
    for source_scenario, scenario in mapping.items():
        row = fq.loc[fq["scenario"].eq(source_scenario)].iloc[0]
        irrigation = float(row["irrigation_total"])
        nitrogen = float(row["fertilizer_total"])
        yield_value = float(row["final_grain_kg_ha"])
        rows.append(
            {
                "site": "FQ",
                "station": "Fengqiu",
                "year": 2016,
                "scenario_family": scenario,
                "dqn_seed": np.nan,
                "checkpoint_step": np.nan,
                "final_grain_kg_ha": yield_value,
                "final_biomass_kg_ha": float(row["final_biomass_kg_ha"]),
                "irrigation_mm": irrigation,
                "nitrogen_kg_ha": nitrogen,
                "max_water_stress": float(row["max_water_stress"]),
                "max_nitrogen_stress": float(row["max_nitrogen_stress"]),
                "unified_reward": unified_reward(yield_value, fq_null, irrigation, nitrogen),
                "source_file": str(fq_path.relative_to(ROOT)).replace("\\", "/"),
            }
        )

    extension_path = ROOT / "DSSAT_auto_validation" / "extension_expert_baseline_018_03" / "018_03_extension_expert_summary.csv"
    extension = pd.read_csv(extension_path, keep_default_na=False)
    for site, year, station, null_yield in (("YC", 2014, "Yucheng", yc_null), ("FQ", 2016, "Fengqiu", fq_null)):
        row = extension.loc[extension["site"].eq(site) & extension["year"].eq(year)].iloc[0]
        irrigation = float(row["event_irrigation_total"])
        nitrogen = float(row["event_fertilizer_total"])
        yield_value = float(row["final_gwad"])
        rows.append(
            {
                "site": site,
                "station": station,
                "year": year,
                "scenario_family": "extension_expert",
                "dqn_seed": np.nan,
                "checkpoint_step": np.nan,
                "final_grain_kg_ha": yield_value,
                "final_biomass_kg_ha": float(row["final_cwad"]),
                "irrigation_mm": irrigation,
                "nitrogen_kg_ha": nitrogen,
                "max_water_stress": float(row["max_water_stress"]),
                "max_nitrogen_stress": float(row["max_nitrogen_stress"]),
                "unified_reward": unified_reward(yield_value, null_yield, irrigation, nitrogen),
                "source_file": str(extension_path.relative_to(ROOT)).replace("\\", "/"),
            }
        )

    for site, station, year, run_dir in (
        ("YC", "Yucheng", 2014, OUT / "YC2014" / "seed0_50000steps"),
        ("FQ", "Fengqiu", 2016, OUT / "FQ2016" / "seed0_50000steps"),
    ):
        row = pd.read_csv(run_dir / "selected_checkpoint_summary.csv").iloc[0]
        rows.append(
            {
                "site": site,
                "station": station,
                "year": year,
                "scenario_family": "frozen_nstep5_dqn",
                "dqn_seed": 0,
                "checkpoint_step": int(row["checkpoint_step"]),
                "final_grain_kg_ha": float(row["final_grain_kg_ha"]),
                "final_biomass_kg_ha": float(row["final_biomass_kg_ha"]),
                "irrigation_mm": float(row["action_irrigation_total_mm"]),
                "nitrogen_kg_ha": float(row["action_nitrogen_total_kg_ha"]),
                "max_water_stress": float(row["max_water_stress"]),
                "max_nitrogen_stress": float(row["max_nitrogen_stress"]),
                "unified_reward": float(row["total_reward"]),
                "source_file": str((run_dir / "selected_checkpoint_summary.csv").relative_to(ROOT)).replace("\\", "/"),
            }
        )
    return rows


def previous_dqn_rows() -> pd.DataFrame:
    sources = [
        choose_best(
            ROOT / "DSSAT_auto_validation" / "yc2014_baseline_relative_checkpoint_refresh_016_08" / "seed0" / "dqn_baseline_relative_checkpoint" / "015_10_yc2014_baseline_relative_checkpoint_summary.csv",
            0,
            "YC",
            "action_irrigation_total",
            "action_fertilizer_total",
        ),
        choose_best(
            ROOT / "DSSAT_auto_validation" / "yc2014_baseline_relative_checkpoint_refresh_016_08" / "seed1" / "dqn_baseline_relative_checkpoint" / "015_10_yc2014_baseline_relative_checkpoint_summary.csv",
            1,
            "YC",
            "action_irrigation_total",
            "action_fertilizer_total",
        ),
        choose_best(
            ROOT / "DSSAT_auto_validation" / "fq2016_baseline_relative_dqn_checkpoint_015_14" / "seed0_50000steps" / "checkpoint_summary.csv",
            0,
            "FQ",
            "action_irrigation_total",
            "action_fertilizer_total",
        ),
        choose_best(
            ROOT / "DSSAT_auto_validation" / "fq2016_baseline_relative_dqn_checkpoint_015_14" / "seed1_50000steps" / "checkpoint_summary.csv",
            1,
            "FQ",
            "action_irrigation_total",
            "action_fertilizer_total",
        ),
    ]
    frame = pd.DataFrame(sources)
    frame.insert(1, "framework", "previous_nstep1")
    frozen = pd.read_csv(OUT / "020_12_selected_cross_site_summary.csv", keep_default_na=False)
    frozen = frozen.loc[frozen["scenario_family"].eq("frozen_nstep5_dqn")].copy()
    frozen_rows = pd.DataFrame(
        {
            "site": frozen["site"],
            "framework": "frozen_nstep5",
            "seed": pd.to_numeric(frozen["dqn_seed"], errors="raise").astype(int),
            "checkpoint_step": pd.to_numeric(frozen["checkpoint_step"], errors="raise").astype(int),
            "final_grain_kg_ha": frozen["final_grain_kg_ha"],
            "irrigation_mm": frozen["irrigation_mm"],
            "nitrogen_kg_ha": frozen["nitrogen_kg_ha"],
            "total_reward": frozen["unified_reward"],
            "source_file": frozen["source_file"],
        }
    )
    return pd.concat([frame, frozen_rows], ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(baseline_rows())
    scenario_order = {"null": 0, "recorded": 1, "dssat_auto": 2, "extension_expert": 3, "frozen_nstep5_dqn": 4}
    comparison["_order"] = comparison["scenario_family"].map(scenario_order)
    comparison = comparison.sort_values(["site", "_order"]).drop(columns="_order").reset_index(drop=True)
    for site, group in comparison.groupby("site"):
        null_yield = float(group.loc[group["scenario_family"].eq("null"), "final_grain_kg_ha"].iloc[0])
        auto_yield = float(group.loc[group["scenario_family"].eq("dssat_auto"), "final_grain_kg_ha"].iloc[0])
        expert_yield = float(group.loc[group["scenario_family"].eq("extension_expert"), "final_grain_kg_ha"].iloc[0])
        index = group.index
        comparison.loc[index, "yield_diff_vs_null_kg_ha"] = comparison.loc[index, "final_grain_kg_ha"] - null_yield
        comparison.loc[index, "yield_diff_vs_auto_kg_ha"] = comparison.loc[index, "final_grain_kg_ha"] - auto_yield
        comparison.loc[index, "yield_diff_vs_extension_expert_kg_ha"] = comparison.loc[index, "final_grain_kg_ha"] - expert_yield
    comparison.to_csv(OUT / "020_12_selected_cross_site_summary.csv", index=False, encoding="utf-8-sig")

    history = previous_dqn_rows()
    history.to_csv(OUT / "020_12_frozen_nstep5_vs_previous_nstep1.csv", index=False, encoding="utf-8-sig")

    compact = comparison[[
        "site", "scenario_family", "final_grain_kg_ha", "irrigation_mm", "nitrogen_kg_ha",
        "unified_reward", "yield_diff_vs_auto_kg_ha", "yield_diff_vs_extension_expert_kg_ha",
    ]]
    dqn_compact = history[["site", "framework", "seed", "checkpoint_step", "final_grain_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "total_reward"]]
    lines = [
        "# 020_12 YC/FQ 冻结 n-step DQN 跨站点验证最终记录",
        "",
        "## 完成范围",
        "",
        "- YC2014 与 FQ2016 均使用同一冻结训练框架重新训练50K；不是直接迁移HLA模型权重。",
        "- 两站均使用各自同输入null产量，奖励、9动作、I120/N300、7 DAP间隔、n_steps=5和其余DQN超参数不变。",
        "- 当前只有seed0，不能据此宣称跨seed稳定。",
        "",
        "## 与基线的统一比较",
        "",
        markdown_table(compact),
        "",
        "## 与旧 n-step=1 DQN 的对照",
        "",
        markdown_table(dqn_compact),
        "",
        "## 结论",
        "",
        "- YC2014：冻结n-step5在35K选中GWAD 8659、I60、N0。比null高834 kg/ha，较auto低54 kg/ha但少26.5 mm水；明显低于recorded和官方推广expert的约9417–9418 kg/ha。因此这是低投入、近auto候选，不是全面优于五情景的成功案例。",
        "- FQ2016：冻结n-step5在50K选中GWAD 7985、I120、N50。比null高919 kg/ha，也略高于recorded与官方推广expert，但较auto低27 kg/ha，同时比auto多60.1 mm水和50 kg/ha氮；尚未达到产量与水氮效率同时超过auto的目标。",
        "- 同一冻结框架在两个目标站点都能学出显著优于null的策略，说明训练链路可跨站点使用；但两个seed0结果均未同时超过所有强基线，跨站点优越性尚未成立。",
        "- 下一步应先做seed1最小稳定性复核：YC检验低投入近auto能否复现，FQ检验是否仍落后auto。只有复现后才值得扩seed2或跨年份，不再修改奖励/约束。",
        "",
        "## QA",
        "",
        "- YC保存模型re-audit：10/10 checkpoint产量、水氮、奖励精确复现；按pre-action operation_dap复核后10/10运行审计通过。",
        "- FQ：10/10 checkpoint运行审计通过。",
        "- HLA冻结配置仍为本次唯一框架源；本轮没有重新训练HLA，也没有覆盖旧结果。",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"wrote: {DOC}")


if __name__ == "__main__":
    main()
