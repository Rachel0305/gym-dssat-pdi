from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot"
DIAG = ROOT / "benchmark_results/219YCA_training_regression_diagnosis"


def main():
    new = pd.read_csv(OUT / "tables/219YCA_checkpoint_summary.csv")
    old = pd.read_csv(DIAG / "tables/episode_metrics.csv")
    old5raw = old[old["checkpoint"] == 5000]
    old5 = old5raw[["yield_kg_ha", "simple_profit_score", "WP_ET", "PFP_N", "irrigation", "nitrogen"]].mean()
    rows = []
    for _, r in new.iterrows():
        rows.append({
            "seed": r["seed"], "checkpoint": r["checkpoint_step"],
            "yield_new": r["mean_final_grnwt"],
            "yield_delta_vs_218_5k": r["mean_final_grnwt"] - old5["yield_kg_ha"],
            "profit_new": r["mean_simple_profit"],
            "profit_delta_vs_218_5k": r["mean_simple_profit"] - old5["simple_profit_score"],
            "WP_ET_new": r["mean_WP_ET_kg_m3"], "WP_ET_delta_vs_218_5k": r["mean_WP_ET_kg_m3"] - old5["WP_ET"],
            "PFP_N_new": r["mean_PFP_N"], "I_total": r["mean_total_irrigation"],
            "N_total": r["mean_total_n"], "unique_signatures": r["unique_action_signatures"],
            "unique_I": r["unique_total_irrigation"], "unique_N": r["unique_total_n"],
        })
    compare = pd.DataFrame(rows)
    compare.to_csv(OUT / "tables/219YCA_vs_218YCA5k_pairwise_summary.csv", index=False)
    pd.DataFrame([old5]).to_csv(OUT / "tables/219YCA_source_218YCA5k_summary.csv", index=False)

    final = new[new["checkpoint_step"] == 20160]
    lines = [
        "# 219YCA 观测归一化 pilot 对照结论", "",
        "## 一句话结论", "",
        "观测归一化降低了网络输入饱和，并且训练过程内存稳定；但没有解决 YC 的固定总投入问题。两个 seed 在 20160 步仍为 I=150 mm、N=240 kg/ha。",
        "",
        "## 与旧 218YCA-5K 的关系", "",
        f"旧 218YCA-5K：产量 {old5['yield_kg_ha']:.1f}、利润 {old5['simple_profit_score']:.1f}、WP_ET {old5['WP_ET']:.3f}、PFP_N {old5['PFP_N']:.2f}。",
        "新的归一化 pilot 在不同 seed/检查点间有波动；seed1-20160 的平均产量和 WP_ET 较高，但 N 总量固定为 240，PFP_N 没有优于旧结果，因此不能称为整体更优。",
        "",
        "## 关键判断", "",
        "- 这不是训练崩溃：动作时序仍有变化，且网络饱和明显下降。",
        "- 但也不是资源自适应成功：unique_I=1、unique_N=1，说明每年总投入没有随天气/年份改变。",
        "- 下一步应先做资源总量反事实实验，判断 I=150/N=240 是合理的 YC 通用投入，还是 PPO 学成了固定模板；不建议直接延长训练。",
        "",
        "## 20K 末期汇总", "",
        final[["seed", "mean_final_grnwt", "mean_simple_profit", "mean_WP_ET_kg_m3", "mean_PFP_N", "mean_total_irrigation", "mean_total_n", "unique_action_signatures", "unique_total_irrigation", "unique_total_n"]].to_markdown(index=False),
        "",
        "对照明细见 `tables/219YCA_vs_218YCA5k_pairwise_summary.csv`。",
    ]
    (OUT / "219YCA_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote comparison tables and interpretation")


if __name__ == "__main__":
    main()
