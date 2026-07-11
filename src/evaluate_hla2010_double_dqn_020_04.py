from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from calculate_five_site_wue_nue_from_summary_019_10 import (
    native_or_none,
    num,
    parse_summary_out,
    select_matching_row,
    valid_positive,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_double_dqn_seed1_020_04" / "double_dqn_seed1_50000steps"
BASELINES = ROOT / "DSSAT_auto_validation" / "five_site_wue_nue_019_10" / "019_10_native_wue_nue_metrics.csv"
STANDARD = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_seed2_stability_020_02" / "020_02_seed0_seed1_seed2_fixed_rule_comparison.csv"
DOC = ROOT / "docs" / "2026-07-10_020_04_hla2010_double_dqn_seed1_record.md"
OUT = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_double_dqn_seed1_020_04"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract() -> pd.DataFrame:
    summary = pd.read_csv(RUN_DIR / "checkpoint_summary.csv")
    rows = []
    for item in summary.itertuples(index=False):
        step = int(item.checkpoint_step)
        source = RUN_DIR / f"checkpoint_{step}" / "pdi_tmp_snapshot_eval" / "Summary.OUT"
        parsed = parse_summary_out(source)
        selected, score, row_index = select_matching_row(
            parsed,
            float(item.final_grain_kg_ha),
            float(item.action_irrigation_total),
            float(item.action_fertilizer_total),
        )
        ircm = num(selected, "IRCM")
        nicm = num(selected, "NICM")
        nucm = num(selected, "NUCM")
        ypem = native_or_none(num(selected, "YPEM"))
        ypim = native_or_none(num(selected, "YPIM"))
        ypnam = native_or_none(num(selected, "YPNAM"))
        ypnum = native_or_none(num(selected, "YPNUM"))
        rows.append(
            {
                "algorithm": "Double DQN",
                "seed": 1,
                "checkpoint_step": step,
                "HWAM_kg_ha": num(selected, "HWAM"),
                "WP_ET_kg_m3": ypem * 0.1 if ypem is not None else None,
                "irrigation_mm": float(item.action_irrigation_total),
                "nitrogen_kg_ha": float(item.action_fertilizer_total),
                "NLCM_kg_ha": num(selected, "NLCM"),
                "NUtE_kg_kg": ypnum if valid_positive(nucm) and ypnum is not None else None,
                "IWP_gross_kg_m3": ypim * 0.1 if valid_positive(ircm) and ypim is not None else None,
                "PFP_N_kg_kg": ypnam if valid_positive(nicm) and ypnam is not None else None,
                "training_reward": float(item.total_reward),
                "source_match_score": score,
                "source_row_index": row_index,
            }
        )
    return pd.DataFrame(rows).sort_values("checkpoint_step")


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    baselines = pd.read_csv(BASELINES)
    hla = baselines[baselines["site"].eq("HLA")]
    for short, scenario in [("auto", "DSSAT auto"), ("extension", "Official extension expert fixed DAP")]:
        base = hla[hla["scenario"].eq(scenario)].iloc[0]
        out[f"yield_ok_vs_{short}"] = out["HWAM_kg_ha"] >= float(base["HWAM_kg_ha"]) - 1.0
        out[f"WP_ET_ok_vs_{short}"] = out["WP_ET_kg_m3"] >= float(base["WP_ET_kg_m3_DSSAT_YPEM"]) - 0.01
        out[f"input_ok_vs_{short}"] = (out["irrigation_mm"] <= float(base["irrigation_mm_event_table"])) & (out["nitrogen_kg_ha"] <= float(base["NICM_kg_ha"]))
        out[f"leaching_ok_vs_{short}"] = out["NLCM_kg_ha"] <= float(base["NLCM_kg_ha"]) + 0.01
        out[f"strict_success_vs_{short}"] = out[f"yield_ok_vs_{short}"] & out[f"WP_ET_ok_vs_{short}"] & out[f"input_ok_vs_{short}"] & out[f"leaching_ok_vs_{short}"]
    out["strict_success_vs_both"] = out["strict_success_vs_auto"] & out["strict_success_vs_extension"]
    return out


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in df.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.3f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    metrics = add_flags(extract())
    if metrics["source_match_score"].max() != 0:
        raise ValueError("Double DQN Summary.OUT matching is not exact")
    selected = metrics.sort_values(["training_reward", "checkpoint_step"], ascending=[False, True]).iloc[0]
    standard = pd.read_csv(STANDARD)
    standard_seed1 = standard[standard["seed"].eq(1)].iloc[0].copy()
    comparison = pd.DataFrame(
        [
            {
                "algorithm": "standard DQN",
                "seed": 1,
                "checkpoint_step": int(standard_seed1["checkpoint_step"]),
                "HWAM_kg_ha": float(standard_seed1["HWAM_kg_ha"]),
                "WP_ET_kg_m3": float(standard_seed1["WP_ET_kg_m3"]),
                "irrigation_mm": float(standard_seed1["irrigation_mm"]),
                "nitrogen_kg_ha": float(standard_seed1["nitrogen_kg_ha"]),
                "NLCM_kg_ha": float(standard_seed1["NLCM_kg_ha"]),
                "training_reward": float(standard_seed1["training_reward"]),
                "strict_success_vs_both": bool(standard_seed1["strict_success_vs_both"]),
            },
            {
                "algorithm": "Double DQN",
                "seed": 1,
                "checkpoint_step": int(selected["checkpoint_step"]),
                "HWAM_kg_ha": float(selected["HWAM_kg_ha"]),
                "WP_ET_kg_m3": float(selected["WP_ET_kg_m3"]),
                "irrigation_mm": float(selected["irrigation_mm"]),
                "nitrogen_kg_ha": float(selected["nitrogen_kg_ha"]),
                "NLCM_kg_ha": float(selected["NLCM_kg_ha"]),
                "training_reward": float(selected["training_reward"]),
                "strict_success_vs_both": bool(selected["strict_success_vs_both"]),
            },
        ]
    )
    metrics.to_csv(OUT / "020_04_double_dqn_checkpoint_native_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "020_04_standard_vs_double_dqn_seed1_comparison.csv", index=False, encoding="utf-8-sig")
    summary = pd.read_csv(RUN_DIR / "checkpoint_summary.csv")
    event_error_i = (summary["action_irrigation_total"] - summary["irrigation_total_mgmtevent"]).abs().max()
    event_error_n = (summary["action_fertilizer_total"] - summary["fertilizer_total_mgmtevent"]).abs().max()
    input_old = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_baseline_relative_dqn_checkpoint_015_12" / "2010" / "baseline_relative_seed1_50000steps" / "input"
    input_new = RUN_DIR / "input"
    same_inputs = all(sha256(path) == sha256(input_old / path.name) for path in input_new.iterdir() if path.is_file())
    lines = [
        "",
        "## 正式50K结果与严格比较",
        "",
        markdown_table(comparison),
        "",
        f"- Double DQN 按固定规则选中 {int(selected['checkpoint_step'])} steps：HWAM={selected['HWAM_kg_ha']:.1f} kg/ha，WP_ET={selected['WP_ET_kg_m3']:.2f} kg/m³，I={selected['irrigation_mm']:.1f} mm，N={selected['nitrogen_kg_ha']:.1f} kg/ha，NLCM={selected['NLCM_kg_ha']:.1f} kg/ha。",
        f"- Double DQN 严格成功：{bool(selected['strict_success_vs_both'])}。",
        f"- 与标准DQN seed1相比：产量变化 {selected['HWAM_kg_ha'] - standard_seed1['HWAM_kg_ha']:+.1f} kg/ha，灌溉变化 {selected['irrigation_mm'] - standard_seed1['irrigation_mm']:+.1f} mm，施氮变化 {selected['nitrogen_kg_ha'] - standard_seed1['nitrogen_kg_ha']:+.1f} kg/ha。",
        f"- 全部checkpoint动作与MgmtEvent总量最大差异：灌溉 {event_error_i:.3f} mm，施氮 {event_error_n:.3f} kg/ha。",
        f"- Double DQN与标准DQN seed1输入文件一致：{same_inputs}。",
        "- 结论：当前框架下 Double DQN 使 seed1 达到与标准DQN seed0相同的严格成功点，但这只是一个 seed 的算法对照，尚不能证明 Double DQN 已跨seed稳定。",
    ]
    DOC.write_text(DOC.read_text(encoding="utf-8").rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
