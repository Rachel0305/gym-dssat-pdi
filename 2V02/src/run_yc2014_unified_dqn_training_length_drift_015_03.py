from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_unified_dqn_training_length_drift_015_03"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_015_03_yc2014_unified_dqn_training_length_drift_record.md"

SEED = 0
WATER_COST = 1.0
NITROGEN_COST = 5.0
NEW_TIMESTEPS = [10_000, 20_000]

EXISTING_5K_SUMMARY = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "yc2014_unified_dqn_ncost_sensitivity_015_02"
    / "seed0"
    / "015_02_yc2014_unified_dqn_ncost_sensitivity_summary.csv"
)
EXISTING_5K_DAILY = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "yc2014_unified_dqn_ncost_sensitivity_015_02"
    / "seed0"
    / "015_02_yc2014_unified_dqn_ncost_sensitivity_daily.csv"
)
EXISTING_50K_SUMMARY = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "yc2014_unified_dqn_formal_015_01"
    / "seed0"
    / "015_01_yc2014_unified_dqn_50k_seed0_summary.csv"
)
EXISTING_50K_DAILY = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "yc2014_unified_dqn_formal_015_01"
    / "seed0"
    / "015_01_yc2014_unified_dqn_50k_seed0_daily.csv"
)

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

COLORS = {
    5_000: "#386CB0",
    10_000: "#F0027F",
    20_000: "#BF5B17",
    50_000: "#666666",
}


def scenario_name(timesteps: int) -> str:
    return f"dqn_unified_ncost5_{timesteps // 1000}k"


def configure_yc_module(timesteps: int) -> str:
    scenario = scenario_name(timesteps)
    yc.OUT_DIR = OUT_DIR
    yc.DOC_PATH = DOC_PATH
    yc.SEED = SEED
    yc.TIMESTEPS = timesteps
    yc.WATER_COST = WATER_COST
    yc.NITROGEN_COST = NITROGEN_COST
    yc.IRRIGATION_BUDGET = 120.0
    yc.NITROGEN_BUDGET = 300.0
    yc.DAILY_IRRIGATION_CAP = 30.0
    yc.DAILY_NITROGEN_CAP = 100.0
    yc.MIN_INTERVAL_DAYS = 7
    yc.ACTION_TABLE = ACTION_TABLE_9
    yc.SCENARIO_ORDER = [scenario]
    yc.SCENARIO_LABELS = {scenario: f"{timesteps // 1000}K"}
    yc.SCENARIO_COLORS = {scenario: COLORS[timesteps]}
    return scenario


def load_existing_5k() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(EXISTING_5K_SUMMARY)
    row = summary.loc[np.isclose(summary["n_cost"], 5.0)].copy()
    row["timesteps"] = 5_000
    row["scenario"] = scenario_name(5_000)
    daily = pd.read_csv(EXISTING_5K_DAILY)
    daily = daily.loc[np.isclose(daily["n_cost"], 5.0)].copy()
    daily["timesteps"] = 5_000
    daily["scenario"] = scenario_name(5_000)
    return daily, row


def load_existing_50k() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(EXISTING_50K_SUMMARY)
    summary = summary.copy()
    summary["timesteps"] = 50_000
    summary["scenario"] = scenario_name(50_000)
    daily = pd.read_csv(EXISTING_50K_DAILY)
    daily = daily.copy()
    daily["timesteps"] = 50_000
    daily["scenario"] = scenario_name(50_000)
    return daily, summary


def run_new_timestep(timesteps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario = configure_yc_module(timesteps)
    print(f"[015_03] running {scenario} timesteps={timesteps}", flush=True)
    daily, summary = yc.run_dqn_smoke(scenario, yc.FREE_DAILY_WINDOWS)
    daily = yc.build_plot_df(daily)
    daily["timesteps"] = timesteps
    summary["timesteps"] = timesteps
    summary_df = pd.DataFrame([summary])
    run_out_dir = OUT_DIR / f"seed{SEED}"
    run_out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(run_out_dir / f"015_03_yc2014_unified_dqn_{timesteps // 1000}k_daily.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(run_out_dir / f"015_03_yc2014_unified_dqn_{timesteps // 1000}k_summary.csv", index=False, encoding="utf-8-sig")
    return daily, summary_df


def plot_drift(all_daily: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15.5, 13.2),
        sharex=False,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.0, 1.05], "hspace": 0.32},
    )
    for timesteps in [5_000, 10_000, 20_000, 50_000]:
        sub = all_daily[all_daily["timesteps"].eq(timesteps)].sort_values("dap")
        if sub.empty:
            continue
        color = COLORS[timesteps]
        label = f"{timesteps // 1000}K"
        axes[0].plot(sub["dap"], sub["grnwt"], color=color, lw=1.8, label=label)
        axes[1].plot(sub["dap"], sub["nstres"], color=color, lw=1.8, label=label)
        axes[2].plot(sub["dap"], sub["swfac"], color=color, lw=1.8, label=label)
        mg_i = sub[sub["irrigation_mm"].fillna(0) > 1e-8]
        mg_n = sub[sub["fertilizer_kg_ha"].fillna(0) > 1e-8]
        if not mg_i.empty:
            axes[3].vlines(mg_i["dap"], 0, mg_i["irrigation_mm"], colors=color, lw=1.7, alpha=0.9)
        if not mg_n.empty:
            axes[3].scatter(mg_n["dap"], mg_n["fertilizer_kg_ha"], marker="^", s=36, color=color, edgecolor="white", linewidth=0.5, zorder=3)

    summary = summary.sort_values("timesteps")
    x = np.arange(len(summary))
    axes[4].bar(x - 0.22, summary["action_irrigation_total"], width=0.32, color="#74A9CF", label="I total")
    axes[4].bar(x + 0.10, summary["action_fertilizer_total"], width=0.32, color="#A1D99B", label="N total")
    ax2 = axes[4].twinx()
    ax2.plot(x, summary["final_grain_kg_ha"], color="#CB181D", marker="o", lw=1.9, label="Grain")
    axes[4].set_xticks(x)
    axes[4].set_xticklabels([f"{int(v // 1000)}K" for v in summary["timesteps"]])
    axes[4].set_xlabel("Training timesteps")

    axes[0].set_ylabel("GRNWT\nkg/ha")
    axes[1].set_ylabel("N stress")
    axes[2].set_ylabel("Water stress")
    axes[3].set_ylabel("Mgmt\namount")
    axes[4].set_ylabel("I/N total")
    ax2.set_ylabel("Final grain kg/ha")
    axes[0].set_title("YC2014 unified DQN training-length drift diagnostic, N_COST=5", loc="left", fontsize=13)
    axes[3].set_title("Irrigation as vertical lines; fertilization as triangles", loc="left", fontsize=10)
    for ax in axes:
        ax.grid(True, color="#E6E8F0", linewidth=0.8, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    axes[0].legend(loc="upper left", ncol=4, frameon=False, fontsize=9)
    axes[4].legend(loc="upper left", frameon=False, fontsize=9)
    ax2.legend(loc="upper right", frameon=False, fontsize=9)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.3f}" if not float(val).is_integer() else f"{int(val)}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_record(summary: pd.DataFrame, daily_path: Path, summary_path: Path, fig_path: Path) -> None:
    cols = [
        "timesteps",
        "action_irrigation_total",
        "action_fertilizer_total",
        "final_grain_kg_ha",
        "final_biomass_kg_ha",
        "max_water_stress",
        "max_nitrogen_stress",
        "total_reward",
    ]
    view = summary[cols].sort_values("timesteps").reset_index(drop=True)
    lines = [
        "# 015_03 YC2014 统一 DQN 训练步长漂移诊断记录",
        "",
        "## 目的",
        "",
        "015_02 显示 N_COST=5 在 5K 下可以学到 I120/N250 高产策略，但 015_01 的 50K 结果变成只灌水、不施氮。本轮复用已有 5K/50K 结果，只新跑 10K 和 20K，判断是否存在长训练策略漂移。",
        "",
        "## 固定设置",
        "",
        "- 站点年份：YC2014",
        "- 算法：DQN, seed=0",
        "- 奖励：R_t = max(0, ΔGRNWT_t) - 1.0 × I_t - 5.0 × N_t",
        "- 动作空间：9 个离散动作，I∈{0,15,30} mm，N∈{0,50,100} kg/ha",
        "- 预算：I≤120 mm，N≤300 kg/ha",
        "- 最小操作间隔：7 days",
        "- 管理模式：IRRIG=L, FERTI=L",
        "",
        "## 输出文件",
        "",
        f"- 日值总表：`{daily_path.relative_to(PROJECT_ROOT)}`",
        f"- 汇总表：`{summary_path.relative_to(PROJECT_ROOT)}`",
        f"- 对比图：`{fig_path.relative_to(PROJECT_ROOT)}`",
        "",
        "## 汇总结果",
        "",
        markdown_table(view),
        "",
        "## 初步结论",
        "",
        "- 如果 10K/20K 与 5K 保持 I120/N250，而 50K 仍是 I90/N0，则优先怀疑 015_01 配置与本轮不一致，或 50K 单次运行存在训练偶然性。",
        "- 如果 10K/20K 逐步减少施氮，则支持长训练策略漂移假设。",
        "- 本轮仍然是诊断，不是正式论文结果；下一步应根据漂移模式决定是复查脚本配置，还是调整 DQN 训练机制。",
        "",
    ]
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_out_dir = OUT_DIR / f"seed{SEED}"
    run_out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = run_out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    daily_frames = []
    summary_frames = []
    d5, s5 = load_existing_5k()
    d50, s50 = load_existing_50k()
    daily_frames.append(d5)
    summary_frames.append(s5)
    for timesteps in NEW_TIMESTEPS:
        d, s = run_new_timestep(timesteps)
        daily_frames.append(d)
        summary_frames.append(s)
    daily_frames.append(d50)
    summary_frames.append(s50)

    all_daily = pd.concat(daily_frames, ignore_index=True, sort=False)
    summary = pd.concat(summary_frames, ignore_index=True, sort=False)
    summary = summary.sort_values("timesteps").reset_index(drop=True)
    daily_path = run_out_dir / "015_03_yc2014_unified_dqn_training_length_drift_daily.csv"
    summary_path = run_out_dir / "015_03_yc2014_unified_dqn_training_length_drift_summary.csv"
    fig_path = fig_dir / "yc2014_unified_dqn_training_length_drift.png"
    all_daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plot_drift(all_daily, summary, fig_path)
    write_record(summary, daily_path, summary_path, fig_path)
    print(summary[["timesteps", "action_irrigation_total", "action_fertilizer_total", "final_grain_kg_ha", "final_biomass_kg_ha", "max_water_stress", "max_nitrogen_stress", "total_reward"]].to_string(index=False))


if __name__ == "__main__":
    main()
