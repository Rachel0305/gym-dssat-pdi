from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "benchmark_results/218YCA_yca_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k_five_scenario_figures_ckpt5000_attempt2"
NEW = ROOT / "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot"


def main():
    five = pd.read_csv(OLD / "tables/218YCA_five_scenario_season_summary.csv", keep_default_na=False)
    five.to_csv(NEW / "tables/219YCA_pre_norm_218YCA5k_five_scenario_comparison.csv", index=False)

    old = five[five["scenario"].eq("rl_candidate")][["year", "grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]].rename(columns={
        "grain_yield_kg_ha": "pre_norm_yield", "actual_irrigation_mm": "pre_norm_I", "actual_nitrogen_kg_ha": "pre_norm_N", "WP_ET_kg_m3": "pre_norm_WP_ET", "PFP_N_kg_kg": "pre_norm_PFP_N"})
    new = pd.read_csv(NEW / "tables/219YCA_development_validation_episode_metrics.csv")
    new = new[(new["checkpoint_step"] == 20160) & (new["seed"] == 1)][["year", "final_grnwt", "total_irrigation", "total_n", "WP_ET_kg_m3", "PFP_N"]].rename(columns={
        "final_grnwt": "normobs_yield", "total_irrigation": "normobs_I", "total_n": "normobs_N", "WP_ET_kg_m3": "normobs_WP_ET", "PFP_N": "normobs_PFP_N"})
    cmp = old.merge(new, on="year", how="inner")
    for metric in ["yield", "I", "N", "WP_ET", "PFP_N"]:
        cmp[f"normobs_{metric}"] = pd.to_numeric(cmp[f"normobs_{metric}"], errors="coerce")
        cmp[f"pre_norm_{metric}"] = pd.to_numeric(cmp[f"pre_norm_{metric}"], errors="coerce")
        cmp[f"delta_normobs_minus_pre_{metric}"] = cmp[f"normobs_{metric}"] - cmp[f"pre_norm_{metric}"]
    cmp.to_csv(NEW / "tables/219YCA_normobs_vs_pre_norm_yearly_original_weather.csv", index=False)

    summary = cmp[[c for c in cmp.columns if c != "year"]].mean(numeric_only=True)
    lines = [
        "# YC 归一化前后对照表说明", "",
        "## 口径", "",
        "- 归一化前：218YCA-5K，已有 2014–2023 五情景 replay。",
        "- 归一化后：219YCA normobs-20K seed1 checkpoint 20160，目前只有 2014–2023 原始天气验证。",
        "- 因此第二张表是同一原始天气下的 PPO 前后对照，不是归一化后五情景 replay。",
        "",
        "## 十年原始天气平均对照", "",
        f"- 归一化前：产量 {summary['pre_norm_yield']:.1f}，灌溉 {summary['pre_norm_I']:.1f}，施氮 {summary['pre_norm_N']:.1f}，WP_ET {summary['pre_norm_WP_ET']:.3f}，PFP_N {summary['pre_norm_PFP_N']:.2f}。",
        f"- 归一化后：产量 {summary['normobs_yield']:.1f}，灌溉 {summary['normobs_I']:.1f}，施氮 {summary['normobs_N']:.1f}，WP_ET {summary['normobs_WP_ET']:.3f}，PFP_N {summary['normobs_PFP_N']:.2f}。",
        "",
        "完整五情景表和逐年 PPO 对照表分别见同目录下的两个 CSV。若需要严格的‘归一化后五情景对比表’，下一步需对 219YCA checkpoint 20160 做一次五情景 replay。",
    ]
    (NEW / "219YCA_comparison_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote YC comparison tables")


if __name__ == "__main__":
    main()
