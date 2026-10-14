from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "benchmark_results/216HLA_hla_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k_ckpt200k_validation_replay"
OUT = ROOT / "benchmark_results/216HLA_hla_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k_ckpt200k_validation_replay"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)


def main():
    d = pd.read_csv(SRC / "logs/216HLA_ckpt200k_validation_rows.csv")
    x = d.year
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    panels = [
        ("grain_yield_kg_ha", "Grain yield (kg/ha)"),
        ("simple_profit", "Simple profit"),
        ("actual_irrigation_mm_daily_sum", "Irrigation input (mm)"),
        ("actual_nitrogen_kg_ha_daily_sum", "Nitrogen input (kg/ha)"),
        ("WP_ET_kg_m3", "WP_ET (kg/m3)"),
        ("PFP_N_kg_kg", "PFP_N (kg/kg)"),
    ]
    for ax, (col, label) in zip(axes.ravel(), panels):
        ax.plot(x, d[col], color="#2A9D55", marker="o", linewidth=2)
        ax.set_title(label); ax.grid(alpha=.25); ax.set_xticks(x)
    fig.suptitle("HLA 216HLA checkpoint-200K: 2014-2023 validation", y=1.01)
    fig.tight_layout(); fig.savefig(FIG / "216HLA_ckpt200k_yearly_metrics.png", dpi=220, bbox_inches="tight"); fig.savefig(FIG / "216HLA_ckpt200k_yearly_metrics.pdf", bbox_inches="tight"); plt.close(fig)

    rows = []
    for _, r in d.iterrows():
        rows.append({"year": int(r.year), "irrigation_events": int(r.irrigation_event_count), "nitrogen_events": int(r.nitrogen_event_count), "signature": r.action_signature})
    pd.DataFrame(rows).to_csv(OUT / "tables_216HLA_action_behavior.csv", index=False)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - .18, d.irrigation_event_count, .36, label="Irrigation events", color="#2A9D55")
    ax.bar(x + .18, d.nitrogen_event_count, .36, label="Nitrogen events", color="#D8A305")
    ax.set_xlabel("Validation year"); ax.set_ylabel("Number of positive events"); ax.set_xticks(x); ax.grid(axis="y", alpha=.25); ax.legend()
    ax.set_title("HLA 216HLA checkpoint-200K: management event counts")
    fig.tight_layout(); fig.savefig(FIG / "216HLA_ckpt200k_yearly_event_counts.png", dpi=220); plt.close(fig)

    result = json.loads((SRC / "216HLA_ckpt200k_validation_result.json").read_text(encoding="utf-8"))
    lines = [
        "# HLA 当前冻结结果：216HLA checkpoint-200K", "",
        "## 汇报结论", "",
        "本次冻结的是 216HLA 的 200K checkpoint 十年回放结果。原计划 500K 训练在约 224,352 步触发内存保护而停止，因此不能称为完成 500K 训练。",
        "",
        "该 checkpoint 在 2014–2023 原始天气上可稳定运行，但管理总量高度固定：每年灌溉 240 mm、施氮 240 kg/ha，十年年度动作签名只有 1 种。因此它可作为当前 HL 的稳定运行基线，不应表述为已经实现逐年水氮自适应。",
        "",
        "## 环境和训练条件", "",
        "- 站点：HLA / Hailun，lowIC。",
        "- 环境：gym-DSSAT，在线 DSSAT 作物模拟。",
        "- 算法：Stable-Baselines3 MaskablePPO + MlpPolicy。",
        "- 训练天气：2004–2013 共 10 年，每年 original、early_dry、mid_dry、late_dry、early_wet、mid_wet 六种天气，共 60 个训练情景。",
        "- 验证天气：2014–2023 原始 lowIC 天气。",
        "- 不使用天气预报、外部施氮或 DSSAT 原生自动灌溉。",
        "",
        "## 动作、奖励和约束", "",
        "- 动作：4 个灌溉水平（0/15/30/45 mm）× 4 个施氮水平（0/40/80/120 kg/ha），共 16 个动作。",
        "- 奖励：`grain_yield - 1.1*irrigation - 1.58*nitrogen`，再乘以 0.001。",
        "- 无水分胁迫奖励和氮胁迫奖励。",
        "- 每日灌溉上限 45 mm；每日施氮上限 120 kg/ha。",
        "- 灌溉和施肥最小间隔均为 7 天。",
        "- 灌溉允许 DAP 1–120，施肥允许 DAP 1–90。",
        "- 生育后期保留灌溉机会的 mask 约束保持启用。",
        "",
        "## PPO 参数", "",
        "- learning rate=0.0003；gamma=1.0；GAE lambda=1.0。",
        "- n_steps=144；batch_size=144；n_epochs=5。",
        "- entropy coefficient=0.01；clip range=0.2；网络 `[64, 64]`。",
        "- CPU 单进程训练，内存警戒 1536 MB，停止阈值 2048 MB。",
        "",
        "## 200K 回放指标（2014–2023 均值）", "",
        f"- 产量：{result['means']['grain_yield_kg_ha']:.1f} kg/ha。",
        f"- 简单利润：{result['means']['simple_profit']:.1f}。",
        f"- WP_ET：{result['means']['WP_ET_kg_m3']:.3f} kg/m3。",
        f"- PFP_N：{result['means']['PFP_N_kg_kg']:.2f} kg/kg。",
        f"- 灌溉：{result['means']['actual_irrigation_mm_daily_sum']:.1f} mm/年。",
        f"- 施氮：{result['means']['actual_nitrogen_kg_ha_daily_sum']:.1f} kg/ha/年。",
        f"- 灌溉事件：{result['means']['irrigation_event_count']:.1f} 次/年；施氮事件：{result['means']['nitrogen_event_count']:.1f} 次/年。",
        "",
        "## 文件", "",
        "- `216HLA_ckpt200k_validation_result.json`：机器可读结果。",
        "- `logs/216HLA_ckpt200k_validation_rows.csv`：逐年指标和动作签名。",
        "- `figures/216HLA_ckpt200k_yearly_metrics.png`：逐年指标图。",
        "- `figures/216HLA_ckpt200k_yearly_event_counts.png`：逐年管理事件数图。",
    ]
    (OUT / "216HLA_freeze_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote HLA freeze report, tables, and figures")


if __name__ == "__main__": main()
