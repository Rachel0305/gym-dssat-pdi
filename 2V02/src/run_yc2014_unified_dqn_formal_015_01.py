from __future__ import annotations

from pathlib import Path

import pandas as pd

import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_unified_dqn_formal_015_01"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_015_01_yc2014_unified_dqn_50k_seed0_record.md"

SCENARIO = "dqn_unified_9action_free_daily"
SEED = 0
TIMESTEPS = 50_000

ACTION_TABLE_9: dict[int, dict[str, float]] = {
    0: {"amir": 0.0, "anfer": 0.0},
    1: {"amir": 15.0, "anfer": 0.0},
    2: {"amir": 30.0, "anfer": 0.0},
    3: {"amir": 0.0, "anfer": 50.0},
    4: {"amir": 15.0, "anfer": 50.0},
    5: {"amir": 30.0, "anfer": 50.0},
    6: {"amir": 0.0, "anfer": 100.0},
    7: {"amir": 15.0, "anfer": 100.0},
    8: {"amir": 30.0, "anfer": 100.0},
}


def configure_yc_module() -> None:
    yc.OUT_DIR = OUT_DIR
    yc.DOC_PATH = DOC_PATH
    yc.SEED = SEED
    yc.TIMESTEPS = TIMESTEPS
    yc.WATER_COST = 1.0
    yc.NITROGEN_COST = 5.0
    yc.IRRIGATION_BUDGET = 120.0
    yc.NITROGEN_BUDGET = 300.0
    yc.DAILY_IRRIGATION_CAP = 30.0
    yc.DAILY_NITROGEN_CAP = 100.0
    yc.MIN_INTERVAL_DAYS = 7
    yc.ACTION_TABLE = ACTION_TABLE_9
    yc.SCENARIO_ORDER = [SCENARIO]
    yc.SCENARIO_LABELS = {SCENARIO: "Unified DQN 9-action seed0"}
    yc.SCENARIO_COLORS = {SCENARIO: "#386411"}


def write_record(summary: pd.DataFrame, daily_path: Path, summary_path: Path) -> None:
    headers = list(summary.columns)
    md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in summary.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.3f}" if not float(val).is_integer() else f"{int(val)}")
            else:
                vals.append(str(val))
        md_lines.append("| " + " | ".join(vals) + " |")
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 015_01 YC2014 统一 DQN 正式训练 seed0 记录",
        "",
        "## 设置",
        "",
        "- 站点年份：Yucheng 2014",
        f"- 情景：`{SCENARIO}`",
        f"- 算法：DQN, timesteps={TIMESTEPS}, seed={SEED}",
        "- 奖励：`max(0, ΔGRNWT) - 1.0 × irrigation - 5.0 × nitrogen`",
        "- 动作空间：9 动作，I∈{0,15,30} mm，N∈{0,50,100} kg/ha",
        "- 预算：I≤120 mm，N≤300 kg/ha",
        "- 最小操作间隔：7 days",
        "- 管理模式：IRRIG=L, FERTI=L，保留原管理表/指针以支持动态动作写入",
        "",
        "## 输出",
        "",
        f"- 日值：`{daily_path.relative_to(PROJECT_ROOT)}`",
        f"- 汇总：`{summary_path.relative_to(PROJECT_ROOT)}`",
        "",
        "## 结果汇总",
        "",
        "\n".join(md_lines),
        "",
        "## 初步判读",
        "",
        "- 本轮为统一正式框架的第一个 seed0 长训练结果。",
        "- 本轮统一 reward 与所有后续站点保持一致，不能再为不同站点单独更换 reward。",
        "- 是否进入正式论文结果，还需要与 null / recorded / DSSAT auto 统一图表对比，并继续 seed1/seed2 稳定性验证。",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_yc_module()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_out_dir = OUT_DIR / f"seed{SEED}"
    run_out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    daily, summary_dict = yc.run_dqn_smoke(SCENARIO, yc.FREE_DAILY_WINDOWS)
    daily = yc.build_plot_df(daily)
    summary = pd.DataFrame([summary_dict])

    daily_path = run_out_dir / f"015_01_yc2014_unified_dqn_50k_seed{SEED}_daily.csv"
    summary_path = run_out_dir / f"015_01_yc2014_unified_dqn_50k_seed{SEED}_summary.csv"
    fig_path = figures_dir / f"yc2014_unified_dqn_50k_seed{SEED}_process.png"

    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    yc.process_plot(daily, fig_path)
    write_record(summary, daily_path, summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
