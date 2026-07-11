from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import evaluate_hla2010_double_dqn_020_04 as common


RUN_DIR = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_double_dqn_seed2_020_05" / "double_dqn_seed2_50000steps"
OUT = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_double_dqn_seed2_020_05"
DOC = ROOT / "docs" / "2026-07-10_020_05_hla2010_double_dqn_seed2_record.md"
SEED1_DOUBLE = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_double_dqn_seed1_020_04" / "020_04_double_dqn_checkpoint_native_metrics.csv"
STANDARD = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_seed2_stability_020_02" / "020_02_seed0_seed1_seed2_fixed_rule_comparison.csv"


def table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False, name=None):
        vals = []
        for value in row:
            if pd.isna(value):
                vals.append("")
            elif isinstance(value, float):
                vals.append(f"{value:.3f}".rstrip("0").rstrip("."))
            else:
                vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    common.RUN_DIR = RUN_DIR
    common.OUT = OUT
    metrics = common.add_flags(common.extract())
    selected = metrics.sort_values(["training_reward", "checkpoint_step"], ascending=[False, True]).iloc[0]
    std = pd.read_csv(STANDARD)
    std1 = std[std["seed"].eq(1)].iloc[0]
    d1 = pd.read_csv(SEED1_DOUBLE)
    d1 = d1.sort_values(["training_reward", "checkpoint_step"], ascending=[False, True]).iloc[0]
    comparison = pd.DataFrame([
        {"algorithm": "standard DQN", "seed": 1, "checkpoint_step": int(std1["checkpoint_step"]), "HWAM_kg_ha": float(std1["HWAM_kg_ha"]), "WP_ET_kg_m3": float(std1["WP_ET_kg_m3"]), "irrigation_mm": float(std1["irrigation_mm"]), "nitrogen_kg_ha": float(std1["nitrogen_kg_ha"]), "NLCM_kg_ha": float(std1["NLCM_kg_ha"]), "training_reward": float(std1["training_reward"]), "strict_success_vs_both": bool(std1["strict_success_vs_both"])},
        {"algorithm": "Double DQN", "seed": 1, "checkpoint_step": int(d1["checkpoint_step"]), "HWAM_kg_ha": float(d1["HWAM_kg_ha"]), "WP_ET_kg_m3": float(d1["WP_ET_kg_m3"]), "irrigation_mm": float(d1["irrigation_mm"]), "nitrogen_kg_ha": float(d1["nitrogen_kg_ha"]), "NLCM_kg_ha": float(d1["NLCM_kg_ha"]), "training_reward": float(d1["training_reward"]), "strict_success_vs_both": bool(d1["strict_success_vs_both"])},
        {"algorithm": "Double DQN", "seed": 2, "checkpoint_step": int(selected["checkpoint_step"]), "HWAM_kg_ha": float(selected["HWAM_kg_ha"]), "WP_ET_kg_m3": float(selected["WP_ET_kg_m3"]), "irrigation_mm": float(selected["irrigation_mm"]), "nitrogen_kg_ha": float(selected["nitrogen_kg_ha"]), "NLCM_kg_ha": float(selected["NLCM_kg_ha"]), "training_reward": float(selected["training_reward"]), "strict_success_vs_both": bool(selected["strict_success_vs_both"])},
    ])
    metrics.to_csv(OUT / "020_05_double_dqn_seed2_checkpoint_native_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "020_05_standard_double_seed1_seed2_comparison.csv", index=False, encoding="utf-8-sig")
    lines = [
        "",
        "## 正式50K原生指标与比较",
        "",
        table(comparison),
        "",
        f"- Double DQN seed2 按固定规则选中 {int(selected['checkpoint_step'])} steps：HWAM={selected['HWAM_kg_ha']:.1f} kg/ha，WP_ET={selected['WP_ET_kg_m3']:.2f} kg/m³，I={selected['irrigation_mm']:.1f} mm，N={selected['nitrogen_kg_ha']:.1f} kg/ha，NLCM={selected['NLCM_kg_ha']:.1f} kg/ha。",
        f"- Double DQN seed2 严格成功：{bool(selected['strict_success_vs_both'])}。",
        "- 结论：Double DQN seed1 的改善没有在 seed2 复现；当前只能报告单seed算法改善，不能称跨seed稳定。",
    ]
    DOC.write_text(DOC.read_text(encoding="utf-8").rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
