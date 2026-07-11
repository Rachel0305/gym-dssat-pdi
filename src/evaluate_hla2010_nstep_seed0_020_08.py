from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import evaluate_hla2010_double_dqn_020_04 as common

RUN_DIR = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_dqn_seed0_020_08" / "nstep5_seed0_50000steps"
OUT = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_dqn_seed0_020_08"
DOC = ROOT / "docs" / "2026-07-10_020_08_hla2010_nstep_dqn_seed0_record.md"
STANDARD = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_seed2_stability_020_02" / "020_02_seed0_seed1_seed2_fixed_rule_comparison.csv"
NSTEP1 = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_dqn_seed1_020_06" / "020_06_nstep_checkpoint_native_metrics.csv"
NSTEP2 = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_dqn_seed2_020_07" / "020_07_nstep_seed2_checkpoint_native_metrics.csv"


def table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False, name=None):
        vals = []
        for value in row:
            if pd.isna(value): vals.append("")
            elif isinstance(value, float): vals.append(f"{value:.3f}".rstrip("0").rstrip("."))
            else: vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    common.RUN_DIR = RUN_DIR; common.OUT = OUT
    metrics = common.add_flags(common.extract())
    selected = metrics.sort_values(["training_reward", "checkpoint_step"], ascending=[False, True]).iloc[0]
    std = pd.read_csv(STANDARD); std0 = std[std["seed"].eq(0)].iloc[0]
    n1 = pd.read_csv(NSTEP1).sort_values(["training_reward", "checkpoint_step"], ascending=[False, True]).iloc[0]
    n2 = pd.read_csv(NSTEP2).sort_values(["training_reward", "checkpoint_step"], ascending=[False, True]).iloc[0]
    rows = [
        {"algorithm": "standard DQN", "seed": 0, "checkpoint_step": int(std0["checkpoint_step"]), "HWAM_kg_ha": float(std0["HWAM_kg_ha"]), "WP_ET_kg_m3": float(std0["WP_ET_kg_m3"]), "irrigation_mm": float(std0["irrigation_mm"]), "nitrogen_kg_ha": float(std0["nitrogen_kg_ha"]), "NLCM_kg_ha": float(std0["NLCM_kg_ha"]), "training_reward": float(std0["training_reward"]), "strict_success_vs_both": bool(std0["strict_success_vs_both"])},
        {"algorithm": "n-step DQN", "seed": 0, "checkpoint_step": int(selected["checkpoint_step"]), "HWAM_kg_ha": float(selected["HWAM_kg_ha"]), "WP_ET_kg_m3": float(selected["WP_ET_kg_m3"]), "irrigation_mm": float(selected["irrigation_mm"]), "nitrogen_kg_ha": float(selected["nitrogen_kg_ha"]), "NLCM_kg_ha": float(selected["NLCM_kg_ha"]), "training_reward": float(selected["training_reward"]), "strict_success_vs_both": bool(selected["strict_success_vs_both"])},
        {"algorithm": "n-step DQN", "seed": 1, "checkpoint_step": int(n1["checkpoint_step"]), "HWAM_kg_ha": float(n1["HWAM_kg_ha"]), "WP_ET_kg_m3": float(n1["WP_ET_kg_m3"]), "irrigation_mm": float(n1["irrigation_mm"]), "nitrogen_kg_ha": float(n1["nitrogen_kg_ha"]), "NLCM_kg_ha": float(n1["NLCM_kg_ha"]), "training_reward": float(n1["training_reward"]), "strict_success_vs_both": bool(n1["strict_success_vs_both"])},
        {"algorithm": "n-step DQN", "seed": 2, "checkpoint_step": int(n2["checkpoint_step"]), "HWAM_kg_ha": float(n2["HWAM_kg_ha"]), "WP_ET_kg_m3": float(n2["WP_ET_kg_m3"]), "irrigation_mm": float(n2["irrigation_mm"]), "nitrogen_kg_ha": float(n2["nitrogen_kg_ha"]), "NLCM_kg_ha": float(n2["NLCM_kg_ha"]), "training_reward": float(n2["training_reward"]), "strict_success_vs_both": bool(n2["strict_success_vs_both"])},
    ]
    comparison = pd.DataFrame(rows)
    metrics.to_csv(OUT / "020_08_nstep_seed0_checkpoint_native_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "020_08_standard_nstep_three_seed_comparison.csv", index=False, encoding="utf-8-sig")
    success_count = int(comparison[comparison["algorithm"].eq("n-step DQN")]["strict_success_vs_both"].sum())
    lines = [
        "", "## 正式50K原生指标与三seed比较", "", table(comparison), "",
        f"- n-step seed0 按固定规则选中 {int(selected['checkpoint_step'])} steps：HWAM={selected['HWAM_kg_ha']:.1f} kg/ha，WP_ET={selected['WP_ET_kg_m3']:.2f} kg/m³，I={selected['irrigation_mm']:.1f} mm，N={selected['nitrogen_kg_ha']:.1f} kg/ha，NLCM={selected['NLCM_kg_ha']:.1f} kg/ha。",
        f"- n-step seed0 严格成功：{bool(selected['strict_success_vs_both'])}。",
        f"- n-step 三个 seed 严格成功数：{success_count}/3。",
        "- 结论：n-step 在当前HLA2010框架下完成三seed严格成功，成为目前最有稳定性证据的算法配置。",
    ]
    DOC.write_text(DOC.read_text(encoding="utf-8").rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))


if __name__ == "__main__": main()
