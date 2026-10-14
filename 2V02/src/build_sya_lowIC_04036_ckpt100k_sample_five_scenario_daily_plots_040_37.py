from __future__ import annotations

from pathlib import Path

import build_sya_lowIC_04028_ckpt75k_sample_five_scenario_daily_plots_040_30 as plot04030


ROOT = Path(__file__).resolve().parents[1]
TASK = "040_37_sya_lowIC_04036_ckpt100k_sample_five_scenario_daily_plots"
OUT = ROOT / "benchmark_results" / TASK
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
SNAP_DIR = OUT / "snapshots"
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"


def main() -> int:
    plot04030.TASK = TASK
    plot04030.OUT = OUT
    plot04030.FIG_DIR = FIG_DIR
    plot04030.TAB_DIR = TAB_DIR
    plot04030.SNAP_DIR = SNAP_DIR
    plot04030.DOC = DOC
    plot04030.PROMPT = PROMPT
    plot04030.CHECKPOINT = 100_000
    plot04030.PPO04028_EVAL = (
        ROOT
        / "benchmark_results"
        / "040_36_sya_lowIC_ppo_late_irrigation_reserve_mask"
        / "evaluation"
        / "040_36_checkpoint_validation_summary.csv"
    )
    plot04030.LABELS["rl_candidate"] = "040_36 MaskablePPO ckpt100k"
    plot04030.PLOT_TAG = "040_36 MaskablePPO ckpt100k"
    plot04030.FILE_TAG = "040_37_lowIC_04036_ckpt100k"
    plot04030.TABLE_TAG = "040_37_sya_2014_2017_2022_lowIC_04036_ckpt100k"
    result = plot04030.main()

    # The reused 040_30 module writes a record with historical task wording.
    # Keep the generated tables/figures, but add a short clean Chinese index
    # for this task so later review does not depend on the inherited text.
    clean = [
        "# 040_37 SYA lowIC 040_36 checkpoint100k 五情景日过程图",
        "",
        "## 任务说明",
        "",
        "- 本任务不训练模型，只绘制 040_36 的 checkpoint100000 代表年份日过程图。",
        "- 代表年份：2014、2017、2022。",
        "- 四基线沿用 039_02 lowIC 快照；recorded farmer 沿用现有模板复用路径。",
        "- PPO 候选来自 040_36 validation daily CSV。",
        "",
        "## 输出图",
        "",
    ]
    for year in [2014, 2017, 2022]:
        png = FIG_DIR / f"040_37_lowIC_04036_ckpt100k_sy{year}_five_scenario_daily.png"
        clean.append(f"- `{png.relative_to(ROOT).as_posix()}`")
    clean.extend(
        [
            "",
            "## 输出表",
            "",
            f"- 日值表：`{(TAB_DIR / '040_37_sya_2014_2017_2022_lowIC_04036_ckpt100k_five_scenario_daily.csv').relative_to(ROOT).as_posix()}`",
            f"- 终值表：`{(TAB_DIR / '040_37_sya_2014_2017_2022_lowIC_04036_ckpt100k_five_scenario_summary.csv').relative_to(ROOT).as_posix()}`",
            f"- PPO 管理事件表：`{(TAB_DIR / '040_37_sya_2014_2017_2022_lowIC_04036_ckpt100k_ppo_management_events.csv').relative_to(ROOT).as_posix()}`",
            "",
            "## 解释边界",
            "",
            "- 本图用于人工检查 040_36 的灌溉/施氮措施是否比 040_33 更合理。",
            "- 不能根据本图现场更换 checkpoint 或新增约束；如需继续改，应另立任务。",
        ]
    )
    DOC.write_text("\n".join(clean) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
