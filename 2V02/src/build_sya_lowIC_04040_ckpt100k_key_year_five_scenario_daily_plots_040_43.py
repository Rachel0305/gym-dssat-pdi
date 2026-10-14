from __future__ import annotations

from pathlib import Path

import build_sya_lowIC_04028_ckpt75k_sample_five_scenario_daily_plots_040_30 as plot04030


ROOT = Path(__file__).resolve().parents[1]
TASK = "040_43_sya_lowIC_04040_ckpt100k_key_year_five_scenario_daily_plots"
OUT = ROOT / "benchmark_results" / TASK
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
SNAP_DIR = OUT / "snapshots"
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

YEARS = [2014, 2017, 2021, 2022, 2023]


def main() -> int:
    plot04030.TASK = TASK
    plot04030.OUT = OUT
    plot04030.FIG_DIR = FIG_DIR
    plot04030.TAB_DIR = TAB_DIR
    plot04030.SNAP_DIR = SNAP_DIR
    plot04030.DOC = DOC
    plot04030.PROMPT = PROMPT
    plot04030.YEARS = YEARS
    plot04030.CHECKPOINT = 100_000
    plot04030.PPO04028_EVAL = (
        ROOT
        / "benchmark_results"
        / "040_40_sya_lowIC_ppo_yield_guardrail_v3"
        / "evaluation"
        / "040_40_checkpoint_validation_summary.csv"
    )
    plot04030.LABELS["rl_candidate"] = "040_40 MaskablePPO ckpt100k"
    plot04030.PLOT_TAG = "040_40 MaskablePPO ckpt100k"
    plot04030.FILE_TAG = "040_43_lowIC_04040_ckpt100k"
    plot04030.TABLE_TAG = "040_43_sya_2014_2017_2021_2022_2023_lowIC_04040_ckpt100k"
    result = plot04030.main()

    clean = [
        "# 040_43 SYA lowIC 040_40 checkpoint100K 关键年份五情景日过程图记录",
        "",
        "## 任务说明",
        "",
        "- 本任务不训练模型，只绘制 040_40 checkpoint100000 的关键年份日过程图。",
        "- 固定年份：2014、2017、2021、2022、2023。",
        "- 目的：检查 040_40 的措施是否能解释 040_42 指标表现，尤其关注 2017 低产和 2022 WP_ET 不足。",
        "",
        "## 输出图",
        "",
    ]
    for year in YEARS:
        png = FIG_DIR / f"040_43_lowIC_04040_ckpt100k_sy{year}_five_scenario_daily.png"
        clean.append(f"- `{png.relative_to(ROOT).as_posix()}`")
    clean.extend(
        [
            "",
            "## 输出表",
            "",
            f"- 日值表：`{(TAB_DIR / '040_43_sya_2014_2017_2021_2022_2023_lowIC_04040_ckpt100k_five_scenario_daily.csv').relative_to(ROOT).as_posix()}`",
            f"- 终值表：`{(TAB_DIR / '040_43_sya_2014_2017_2021_2022_2023_lowIC_04040_ckpt100k_five_scenario_summary.csv').relative_to(ROOT).as_posix()}`",
            f"- PPO 管理事件表：`{(TAB_DIR / '040_43_sya_2014_2017_2021_2022_2023_lowIC_04040_ckpt100k_ppo_management_events.csv').relative_to(ROOT).as_posix()}`",
            "",
            "## 解释边界",
            "",
            "- 本图用于人工检查措施合理性，不用于现场更换 reward、checkpoint 或约束。",
            "- 若发现个别年份仍有问题，应另开后续任务诊断。",
        ]
    )
    DOC.write_text("\n".join(clean) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())

