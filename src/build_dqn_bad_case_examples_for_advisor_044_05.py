from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "044_05_dqn_bad_case_examples_for_advisor"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
DOC = ROOT / "docs" / "044_05_dqn_bad_case_examples_for_advisor_record.md"


def setup() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    for font_path in [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=str(font_path)).get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load() -> dict[str, pd.DataFrame]:
    return {
        "dqn_summary": pd.read_csv(
            ROOT / "benchmark_results/040_01_sya_lowIC_free_timing_dqn/evaluation/040_01_validation_summary_by_checkpoint.csv"
        ),
        "dqn_detail": pd.read_csv(
            ROOT / "benchmark_results/040_01_sya_lowIC_free_timing_dqn/evaluation/040_01_checkpoint_validation_summary.csv"
        ),
        "strict_summary": pd.read_csv(
            ROOT / "benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/evaluation/040_02_validation_summary_by_checkpoint.csv"
        ),
        "strict_detail": pd.read_csv(
            ROOT / "benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/evaluation/040_02_checkpoint_validation_summary.csv"
        ),
        "demo_summary": pd.read_csv(
            ROOT / "benchmark_results/044_00_sya_lowIC_binary_forecast_demo_dqn_smoke_smoke2k/evaluation/044_00_validation_summary_by_checkpoint.csv"
        ),
        "demo_detail": pd.read_csv(
            ROOT / "benchmark_results/044_00_sya_lowIC_binary_forecast_demo_dqn_smoke_smoke2k/evaluation/044_00_checkpoint_validation_summary.csv"
        ),
        "q_online": pd.read_csv(
            ROOT / "benchmark_results/044_01_sya_lowIC_demo_dqn_q_ranking_audit/tables/044_01_q_ranking_by_checkpoint.csv"
        ),
        "q_demo_only": pd.read_csv(
            ROOT / "benchmark_results/044_02_sya_lowIC_demo_only_pretrain_dqn_audit/tables/044_02_q_ranking_by_checkpoint.csv"
        ),
    }


def best_checkpoint(summary: pd.DataFrame) -> int:
    return int(summary.sort_values("mean_final_grnwt", ascending=False).iloc[0]["checkpoint_step"])


def extract_events(detail: pd.DataFrame, checkpoint: int, method: str) -> pd.DataFrame:
    sub = detail.loc[detail["checkpoint_step"] == checkpoint].copy()
    rows = []
    for _, r in sub.iterrows():
        seq = str(r.get("action_sequence", ""))
        if not seq or seq == "nan":
            continue
        for item in seq.split(";"):
            item = item.strip()
            if not item.startswith("DAP"):
                continue
            try:
                dap_part, action_part = item.split(" ", 1)
                dap = int(float(dap_part.replace("DAP", "")))
                irrig_s = action_part.split("/")[0].replace("I", "")
                n_s = action_part.split("/")[1].replace("N", "")
                rows.append(
                    {
                        "method": method,
                        "year": int(r["year"]),
                        "checkpoint_step": checkpoint,
                        "dap": dap,
                        "irrigation": float(irrig_s),
                        "nitrogen": float(n_s),
                    }
                )
            except Exception:
                continue
    return pd.DataFrame(rows)


def plot_ordinary_dqn_fixed_strategy(data: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    ckpt = best_checkpoint(data["dqn_summary"])
    detail = data["dqn_detail"].loc[data["dqn_detail"]["checkpoint_step"] == ckpt].copy()
    events = extract_events(data["dqn_detail"], ckpt, "普通DQN")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    axes[0].bar(detail["year"], detail["final_grnwt"], color="#4c78a8")
    axes[0].axhline(detail["final_grnwt"].mean(), color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"普通DQN 040_01：最佳 checkpoint={ckpt}，平均产量仍偏低")
    axes[0].set_xlabel("验证年份")
    axes[0].set_ylabel("籽粒产量 kg/ha")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.25)

    if not events.empty:
        irrig = events.loc[events["irrigation"] > 0]
        nitro = events.loc[events["nitrogen"] > 0]
        axes[1].scatter(irrig["dap"], irrig["year"], s=irrig["irrigation"] * 4, label="灌溉事件", alpha=0.75)
        axes[1].scatter(nitro["dap"], nitro["year"], s=nitro["nitrogen"] * 1.1, marker="^", label="施氮事件", alpha=0.75)
    axes[1].set_title("同一套早期动作几乎重复用于不同天气年份")
    axes[1].set_xlabel("DAP")
    axes[1].set_ylabel("年份")
    axes[1].set_yticks(sorted(detail["year"].unique()))
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    out = FIGS / "044_05_bad_case_ordinary_dqn_fixed_strategy.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)

    sequence_counts = detail["action_sequence"].value_counts().reset_index()
    sequence_counts.columns = ["action_sequence", "year_count"]
    sequence_counts.to_csv(TABLES / "044_05_ordinary_dqn_action_sequence_counts.csv", index=False, encoding="utf-8-sig")
    return str(out.relative_to(ROOT)), sequence_counts


def plot_strict_dqn_collapse(data: dict[str, pd.DataFrame]) -> str:
    s = data["strict_summary"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].plot(s["checkpoint_step"], s["mean_final_grnwt"], marker="o", color="#f58518")
    axes[0].set_title("产量随训练加深下降")
    axes[0].set_ylabel("平均籽粒产量 kg/ha")
    axes[1].plot(s["checkpoint_step"], s["mean_total_irrigation"], marker="o", label="灌溉", color="#4c78a8")
    axes[1].plot(s["checkpoint_step"], s["mean_total_n"], marker="o", label="施氮", color="#54a24b")
    axes[1].set_title("资源投入不稳定/后期趋向少操作")
    axes[1].set_ylabel("投入总量")
    axes[1].legend()
    axes[2].plot(s["checkpoint_step"], s["mean_swfac_stress_days_gt_0p05"], marker="o", label="水分胁迫天数")
    axes[2].plot(s["checkpoint_step"], s["mean_nstres_days_gt_0p05"], marker="o", label="氮胁迫天数")
    axes[2].set_title("胁迫天数没有稳定改善")
    axes[2].set_ylabel("平均天数")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("checkpoint step")
        ax.grid(alpha=0.25)
    fig.suptitle("严格 MaskableDQN 040_02：mask 后仍出现训练退化", fontsize=15)
    out = FIGS / "044_05_bad_case_strict_maskable_dqn_training_collapse.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return str(out.relative_to(ROOT))


def plot_demo_dqn_noop(data: dict[str, pd.DataFrame]) -> str:
    detail = data["demo_detail"].copy()
    ckpt = 2000 if (detail["checkpoint_step"] == 2000).any() else int(detail["checkpoint_step"].max())
    sub = detail.loc[detail["checkpoint_step"] == ckpt]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].bar(sub["year"], sub["grain_yield_kg_ha"], color="#54a24b")
    axes[0].set_title(f"Demo-DQN/DQfD {ckpt}步：验证年产量极低")
    axes[0].set_ylabel("籽粒产量 kg/ha")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(sub["year"], sub["total_irrigation"], label="灌溉", color="#4c78a8")
    axes[1].bar(sub["year"], sub["total_nitrogen"], bottom=sub["total_irrigation"], label="施氮", color="#54a24b")
    axes[1].set_title("验证策略为 no-op：水氮投入均为0")
    axes[1].set_ylabel("投入量（堆叠显示）")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend()

    axes[2].bar(sub["year"], sub["max_swfac"], label="最大水分胁迫", color="#4c78a8")
    axes[2].bar(sub["year"], sub["max_nstres"], bottom=sub["max_swfac"], label="最大氮胁迫", color="#f58518")
    axes[2].set_title("不操作导致胁迫高")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("验证年份")
        ax.grid(axis="y", alpha=0.25)
    out = FIGS / "044_05_bad_case_demo_dqn_noop.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return str(out.relative_to(ROOT))


def plot_q_bad_case(data: dict[str, pd.DataFrame]) -> str:
    online = data["q_online"].copy()
    online["method"] = "在线Demo-DQN"
    demo = data["q_demo_only"].copy()
    demo["method"] = "Demo-only预训练"
    q = pd.concat([online, demo], ignore_index=True)
    q.to_csv(TABLES / "044_05_q_ranking_bad_case_table.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for method, sub in q.groupby("method"):
        axes[0].plot(sub["checkpoint_step"], sub["nonzero_teacher_argmax_rate"], marker="o", label=method)
        axes[1].plot(sub["checkpoint_step"], sub["mean_q_teacher_minus_noop"], marker="o", label=method)
    axes[0].set_title("非零 teacher 动作几乎不能成为 Q 最大动作")
    axes[0].set_ylabel("非零 teacher argmax 比例")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Q(teacher) - Q(no-op) 后期为负")
    axes[1].set_ylabel("平均 Q 差")
    for ax in axes:
        ax.set_xlabel("checkpoint / epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Demo-DQN / DQfD 坏例子：示范动作没有真正压过 no-op", fontsize=15)
    out = FIGS / "044_05_bad_case_demo_dqn_q_ranking.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return str(out.relative_to(ROOT))


def write_record(figures: list[str], sequence_counts: pd.DataFrame, data: dict[str, pd.DataFrame]) -> None:
    dqn_best = data["dqn_summary"].sort_values("mean_final_grnwt", ascending=False).iloc[0]
    strict_25 = data["strict_summary"].loc[data["strict_summary"]["checkpoint_step"] == 25000].iloc[0]
    strict_100 = data["strict_summary"].loc[data["strict_summary"]["checkpoint_step"] == 100000].iloc[0]
    demo_2k = data["demo_summary"].loc[data["demo_summary"]["checkpoint_step"] == 2000].iloc[0]
    lines = [
        "# 044_05 DQN 坏例子图：导师汇报补充材料",
        "",
        "## 任务性质",
        "",
        "本任务只读取已有 DQN / DQfD 结果，不重新训练，不重新运行 DSSAT。",
        "",
        "## 具体坏例子",
        "",
        "### 1. 普通 DQN 040_01：策略固定且平均产量偏低",
        "",
        f"- 最佳 checkpoint：{int(dqn_best['checkpoint_step'])}。",
        f"- 验证年平均产量：{dqn_best['mean_final_grnwt']:.1f} kg/ha。",
        f"- 平均总灌溉：{dqn_best['mean_total_irrigation']:.1f} mm；平均总施氮：{dqn_best['mean_total_n']:.1f} kg/ha。",
        f"- 动作序列类型数：{len(sequence_counts)}；最常见序列覆盖 {int(sequence_counts.iloc[0]['year_count'])} 个验证年。",
        "",
        "### 2. 严格 MaskableDQN 040_02：训练越久越退化",
        "",
        f"- 25K 平均产量：{strict_25['mean_final_grnwt']:.1f} kg/ha。",
        f"- 100K 平均产量：{strict_100['mean_final_grnwt']:.1f} kg/ha。",
        f"- 100K 平均施氮为 {strict_100['mean_total_n']:.1f} kg/ha，已明显偏向少操作/no-op。",
        "",
        "### 3. Demo-DQN / DQfD 044_00：加入示范经验后仍 no-op",
        "",
        f"- 2K smoke 平均产量：{demo_2k['mean_yield']:.1f} kg/ha。",
        f"- 2K smoke 平均灌溉：{demo_2k['mean_irrigation']:.1f} mm；平均施氮：{demo_2k['mean_nitrogen']:.1f} kg/ha。",
        "- Q 排序诊断显示，在线 Demo-DQN 的非零 teacher 动作 argmax 比例为 0；Demo-only 预训练后也降为 0。",
        "",
        "## 图件",
        "",
    ]
    for f in figures:
        lines.append(f"- `{f}`")
    lines.extend(
        [
            "",
            "## 汇报口径",
            "",
            "这组图可以用来说明：我们不是只说 DQN 不好，而是具体看到三类失败形态：普通 DQN 动作模板化且产量不够，严格 MaskableDQN 随训练步数增加退化，Demo-DQN/DQfD 虽然加入了优质示范经验，但非零示范动作没有真正成为 Q 值最高动作。因此当前阶段 DQN 更适合作为算法对照保留，主线继续优化 PPO。",
            "",
        ]
    )
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup()
    data = load()
    fig1, seq = plot_ordinary_dqn_fixed_strategy(data)
    figures = [
        fig1,
        plot_strict_dqn_collapse(data),
        plot_demo_dqn_noop(data),
        plot_q_bad_case(data),
    ]
    write_record(figures, seq, data)
    result = {
        "task": "044_05_dqn_bad_case_examples_for_advisor",
        "figures": figures,
        "record_md": str(DOC.relative_to(ROOT)),
        "tables": [
            str((TABLES / "044_05_ordinary_dqn_action_sequence_counts.csv").relative_to(ROOT)),
            str((TABLES / "044_05_q_ranking_bad_case_table.csv").relative_to(ROOT)),
        ],
    }
    (OUT / "044_05_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

