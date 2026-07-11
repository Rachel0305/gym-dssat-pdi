from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from summarize_yc_fq_frozen_nstep_cross_site_020_12 import baseline_rows, markdown_table
from run_yc_fq_frozen_nstep_cross_site_020_12 import SITE_SPECS, write_record


OUT = ROOT / "DSSAT_auto_validation" / "frozen_nstep_cross_site_020_12"
DOC = ROOT / "docs" / "2026-07-11_020_13_yc_fq_frozen_nstep_seed1_stability_record.md"


def selected_row(site: str, station: str, year: int, seed: int) -> dict[str, object]:
    run_dir = OUT / f"{site}{year}" / f"seed{seed}_50000steps"
    path = run_dir / "selected_checkpoint_summary.csv"
    row = pd.read_csv(path).iloc[0]
    return {
        "site": site,
        "station": station,
        "year": year,
        "scenario_family": "frozen_nstep5_dqn",
        "dqn_seed": seed,
        "checkpoint_step": int(row["checkpoint_step"]),
        "final_grain_kg_ha": float(row["final_grain_kg_ha"]),
        "final_biomass_kg_ha": float(row["final_biomass_kg_ha"]),
        "irrigation_mm": float(row["action_irrigation_total_mm"]),
        "nitrogen_kg_ha": float(row["action_nitrogen_total_kg_ha"]),
        "max_water_stress": float(row["max_water_stress"]),
        "max_nitrogen_stress": float(row["max_nitrogen_stress"]),
        "unified_reward": float(row["total_reward"]),
        "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
    }


def main() -> None:
    for site, year in (("YC", 2014), ("FQ", 2016)):
        for seed in (0, 1):
            run_dir = OUT / f"{site}{year}" / f"seed{seed}_50000steps"
            null_summary = pd.read_csv(run_dir / "null" / "null_summary.csv").iloc[0].to_dict()
            summaries = pd.read_csv(run_dir / "checkpoint_summary.csv")
            selected = pd.read_csv(run_dir / "selected_checkpoint_summary.csv").iloc[0].to_dict()
            write_record(
                SITE_SPECS[site],
                seed,
                50_000,
                5_000,
                run_dir,
                null_summary,
                summaries,
                selected,
            )

    rows = baseline_rows()
    # baseline_rows already contains frozen seed0; append only seed1.
    rows.extend(
        [
            selected_row("YC", "Yucheng", 2014, 1),
            selected_row("FQ", "Fengqiu", 2016, 1),
        ]
    )
    frame = pd.DataFrame(rows)
    scenario_order = {
        "null": 0,
        "recorded": 1,
        "dssat_auto": 2,
        "extension_expert": 3,
        "frozen_nstep5_dqn": 4,
    }
    frame["_order"] = frame["scenario_family"].map(scenario_order)
    frame = frame.sort_values(["site", "_order", "dqn_seed"], na_position="first").drop(columns="_order").reset_index(drop=True)
    for site, group in frame.groupby("site"):
        null_yield = float(group.loc[group["scenario_family"].eq("null"), "final_grain_kg_ha"].iloc[0])
        auto_yield = float(group.loc[group["scenario_family"].eq("dssat_auto"), "final_grain_kg_ha"].iloc[0])
        expert_yield = float(group.loc[group["scenario_family"].eq("extension_expert"), "final_grain_kg_ha"].iloc[0])
        index = group.index
        frame.loc[index, "yield_diff_vs_null_kg_ha"] = frame.loc[index, "final_grain_kg_ha"] - null_yield
        frame.loc[index, "yield_diff_vs_auto_kg_ha"] = frame.loc[index, "final_grain_kg_ha"] - auto_yield
        frame.loc[index, "yield_diff_vs_extension_expert_kg_ha"] = frame.loc[index, "final_grain_kg_ha"] - expert_yield
    frame.to_csv(OUT / "020_13_yc_fq_seed0_seed1_summary.csv", index=False, encoding="utf-8-sig")

    dqn = frame.loc[frame["scenario_family"].eq("frozen_nstep5_dqn")].copy()
    dqn["yield_range_within_site_kg_ha"] = dqn.groupby("site")["final_grain_kg_ha"].transform(lambda values: values.max() - values.min())
    dqn["irrigation_range_within_site_mm"] = dqn.groupby("site")["irrigation_mm"].transform(lambda values: values.max() - values.min())
    dqn["nitrogen_range_within_site_kg_ha"] = dqn.groupby("site")["nitrogen_kg_ha"].transform(lambda values: values.max() - values.min())
    dqn.to_csv(OUT / "020_13_yc_fq_cross_seed_dqn_comparison.csv", index=False, encoding="utf-8-sig")

    audit_rows = []
    for site, year in (("YC", 2014), ("FQ", 2016)):
        for seed in (0, 1):
            audit_path = OUT / f"{site}{year}" / f"seed{seed}_50000steps" / "runtime_audit_summary.csv"
            if site == "YC" and seed == 0:
                audit_path = OUT / "YC2014" / "seed0_50000steps" / "reaudit_operation_dap_020_12" / "corrected_runtime_audit_summary.csv"
            audit = pd.read_csv(audit_path)
            audit_rows.append(
                {
                    "site": site,
                    "year": year,
                    "seed": seed,
                    "checkpoint_count": len(audit),
                    "all_runtime_audits_passed": bool(audit["passed"].astype(str).str.lower().eq("true").all()),
                    "source_file": str(audit_path.relative_to(ROOT)).replace("\\", "/"),
                }
            )
    audit_frame = pd.DataFrame(audit_rows)
    audit_frame.to_csv(OUT / "020_13_yc_fq_cross_seed_runtime_audit.csv", index=False, encoding="utf-8-sig")

    compact = dqn[[
        "site", "dqn_seed", "checkpoint_step", "final_grain_kg_ha", "irrigation_mm", "nitrogen_kg_ha",
        "unified_reward", "yield_diff_vs_auto_kg_ha", "yield_diff_vs_extension_expert_kg_ha",
        "yield_range_within_site_kg_ha", "irrigation_range_within_site_mm", "nitrogen_range_within_site_kg_ha",
    ]]
    lines = [
        "# 020_13 YC/FQ 冻结 n-step DQN seed1 稳定性复核记录",
        "",
        "## 设计",
        "",
        "- YC2014、FQ2016均在020_12冻结框架下独立训练seed1 50K；相对seed0只更换模型随机种子。",
        "- 奖励、9动作、I120/N300、7 DAP间隔、n_steps=5、目标站点输入和本地null全部保持不变。",
        "- 每5K确定性评估，按total reward最大、并列取最早checkpoint。",
        "",
        "## DQN跨seed结果",
        "",
        markdown_table(compact),
        "",
        "## 运行审计",
        "",
        markdown_table(audit_frame),
        "",
        "## 结论",
        "",
        "- YC2014：seed0/seed1产量为8659/8676 kg/ha，差17 kg/ha；两者均N0，灌溉为60/90 mm。产量和不施氮方向可复现，但精确灌溉量尚不稳定。两者都接近但低于auto 8713，也明显低于recorded和官方推广expert约9417–9418。",
        "- FQ2016：两个seed产量均为7985 kg/ha；灌溉为120/105 mm，施氮为50/0 kg/ha。产量高度稳定、用水较接近，但是否施氮尚未稳定。两者均低于auto 8012，且用水高于auto 59.9 mm。",
        "- 因此可以说同一冻结框架在YC/FQ上跨seed复现了‘显著优于null、接近强基线’的产量水平；不能说已跨seed复现了完全相同的水氮动作，也不能说全面超过auto和官方expert。",
        "- 下一步若继续，应先由导师决定是否把‘接近auto且更省部分资源’作为可接受目标；在此之前不建议直接追加seed2或修改奖励。",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(dqn.to_string(index=False))
    print(audit_frame.to_string(index=False))
    print(f"wrote: {DOC}")


if __name__ == "__main__":
    main()
