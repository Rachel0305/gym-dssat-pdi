from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_unified_dqn_ncost_sensitivity_015_02"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_015_02_yc2014_unified_dqn_ncost_sensitivity_record.md"

SEED = 0
TIMESTEPS = 5_000
WATER_COST = 1.0
N_COSTS = [0.5, 1.0, 2.0, 3.0, 5.0]

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
    0.5: "#006BA4",
    1.0: "#FF800E",
    2.0: "#ABABAB",
    3.0: "#595959",
    5.0: "#5F9ED1",
}


def scenario_name(n_cost: float) -> str:
    token = str(n_cost).replace(".", "p")
    return f"dqn_unified_ncost_{token}"


def configure_yc_module(n_cost: float) -> str:
    scenario = scenario_name(n_cost)
    yc.OUT_DIR = OUT_DIR
    yc.DOC_PATH = DOC_PATH
    yc.SEED = SEED
    yc.TIMESTEPS = TIMESTEPS
    yc.WATER_COST = WATER_COST
    yc.NITROGEN_COST = float(n_cost)
    yc.IRRIGATION_BUDGET = 120.0
    yc.NITROGEN_BUDGET = 300.0
    yc.DAILY_IRRIGATION_CAP = 30.0
    yc.DAILY_NITROGEN_CAP = 100.0
    yc.MIN_INTERVAL_DAYS = 7
    yc.ACTION_TABLE = ACTION_TABLE_9
    yc.SCENARIO_ORDER = [scenario]
    yc.SCENARIO_LABELS = {scenario: f"N cost={n_cost:g}"}
    yc.SCENARIO_COLORS = {scenario: COLORS[n_cost]}
    return scenario


def plot_sensitivity(all_daily: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15.5, 13.5),
        sharex=False,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.0, 1.05], "hspace": 0.33},
    )
    for n_cost in N_COSTS:
        scenario = scenario_name(n_cost)
        sub = all_daily[all_daily["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        color = COLORS[n_cost]
        label = f"N cost={n_cost:g}"
        axes[0].plot(sub["dap"], sub["grnwt"], color=color, lw=1.8, label=label)
        axes[1].plot(sub["dap"], sub["nstres"], color=color, lw=1.8, label=label)
        axes[2].plot(sub["dap"], sub["swfac"], color=color, lw=1.8, label=label)
        mg_i = sub[sub["irrigation_mm"].fillna(0) > 1e-8]
        mg_n = sub[sub["fertilizer_kg_ha"].fillna(0) > 1e-8]
        if not mg_i.empty:
            axes[3].vlines(mg_i["dap"], 0, mg_i["irrigation_mm"], colors=color, lw=1.8, alpha=0.9)
        if not mg_n.empty:
            axes[3].scatter(
                mg_n["dap"],
                mg_n["fertilizer_kg_ha"],
                marker="^",
                s=36,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

    summary_sorted = summary.sort_values("n_cost")
    x = np.arange(len(summary_sorted))
    axes[4].bar(x - 0.2, summary_sorted["action_irrigation_total"], width=0.4, color="#74A9CF", label="I total")
    axes[4].bar(x + 0.2, summary_sorted["action_fertilizer_total"], width=0.4, color="#A1D99B", label="N total")
    ax2 = axes[4].twinx()
    ax2.plot(x, summary_sorted["final_grain_kg_ha"], color="#CB181D", marker="o", lw=1.8, label="Grain")
    axes[4].set_xticks(x)
    axes[4].set_xticklabels([f"{v:g}" for v in summary_sorted["n_cost"]])
    axes[4].set_xlabel("Nitrogen cost coefficient")

    axes[0].set_ylabel("GRNWT\nkg/ha")
    axes[1].set_ylabel("N stress")
    axes[2].set_ylabel("Water stress")
    axes[3].set_ylabel("Mgmt\namount")
    axes[4].set_ylabel("I/N total")
    ax2.set_ylabel("Final grain kg/ha")
    axes[0].set_title("YC2014 unified DQN nitrogen-cost sensitivity, 5K seed0", loc="left", fontsize=13)
    axes[3].set_title("Irrigation as vertical lines; fertilization as triangles", loc="left", fontsize=10)

    for ax in axes:
        ax.grid(True, color="#E6E8F0", linewidth=0.8, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    axes[0].legend(loc="upper left", ncol=3, frameon=False, fontsize=9)
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
    best_yield = summary.loc[summary["final_grain_kg_ha"].idxmax()]
    best_reward = summary.loc[summary["total_reward"].idxmax()]
    lines = [
        "# 015_02 YC2014 统一 DQN 氮成本敏感性诊断记录",
        "",
        "## 目的",
        "",
        "015_01 的统一 DQN 50K seed0 在 YC2014 上只灌水、不施氮。本轮用 5K 低成本训练扫描统一奖励函数中的氮成本系数，判断是否是 N_COST=5.0 过强导致施氮被压制。",
        "",
        "## 固定设置",
        "",
        "- 站点年份：YC2014",
        f"- DQN timesteps={TIMESTEPS}, seed={SEED}",
        "- 动作空间：9 个离散动作，I∈{0,15,30} mm，N∈{0,50,100} kg/ha",
        "- 预算：I≤120 mm，N≤300 kg/ha",
        "- 最小操作间隔：7 days",
        "- 水成本：WATER_COST=1.0",
        "- 管理模式：IRRIG=L, FERTI=L",
        "",
        "## 扫描变量",
        "",
        "- N_COST = 0.5, 1.0, 2.0, 3.0, 5.0",
        "- 奖励：R_t = max(0, ΔGRNWT_t) - 1.0 × I_t - N_COST × N_t",
        "",
        "## 输出文件",
        "",
        f"- 日值总表：`{daily_path.relative_to(PROJECT_ROOT)}`",
        f"- 汇总表：`{summary_path.relative_to(PROJECT_ROOT)}`",
        f"- 对比图：`{fig_path.relative_to(PROJECT_ROOT)}`",
        "",
        "## 汇总结果",
        "",
        markdown_table(summary),
        "",
        "## 初步结论",
        "",
        f"- 最高产量候选：N_COST={best_yield['n_cost']:g}，产量={best_yield['final_grain_kg_ha']:.1f} kg/ha，I={best_yield['action_irrigation_total']:.1f} mm，N={best_yield['action_fertilizer_total']:.1f} kg/ha。",
        f"- 最高累积 reward 候选：N_COST={best_reward['n_cost']:g}，reward={best_reward['total_reward']:.1f}，I={best_reward['action_irrigation_total']:.1f} mm，N={best_reward['action_fertilizer_total']:.1f} kg/ha。",
        "- 本轮是奖励函数诊断，不是正式论文训练结果；不能据此直接宣布 DQN 成功或失败。",
        "- 如果低 N_COST 恢复施氮且产量明显上升，说明正式统一奖励应重新校准氮成本；如果所有 N_COST 仍不施氮，则需要改奖励结构而不是继续微调成本。",
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
    summaries = []
    for n_cost in N_COSTS:
        scenario = configure_yc_module(n_cost)
        print(f"[015_02] running {scenario} TIMESTEPS={TIMESTEPS} N_COST={n_cost}", flush=True)
        daily, summary = yc.run_dqn_smoke(scenario, yc.FREE_DAILY_WINDOWS)
        daily = yc.build_plot_df(daily)
        daily["n_cost"] = n_cost
        summary["n_cost"] = n_cost
        daily_frames.append(daily)
        summaries.append(summary)
        per_daily = run_out_dir / f"015_02_yc2014_ncost_{str(n_cost).replace('.', 'p')}_daily.csv"
        daily.to_csv(per_daily, index=False, encoding="utf-8-sig")

    all_daily = pd.concat(daily_frames, ignore_index=True, sort=False)
    summary_df = pd.DataFrame(summaries).sort_values("n_cost").reset_index(drop=True)
    daily_path = run_out_dir / "015_02_yc2014_unified_dqn_ncost_sensitivity_daily.csv"
    summary_path = run_out_dir / "015_02_yc2014_unified_dqn_ncost_sensitivity_summary.csv"
    fig_path = fig_dir / "yc2014_unified_dqn_ncost_sensitivity.png"
    all_daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plot_sensitivity(all_daily, summary_df, fig_path)
    write_record(summary_df, daily_path, summary_path, fig_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
