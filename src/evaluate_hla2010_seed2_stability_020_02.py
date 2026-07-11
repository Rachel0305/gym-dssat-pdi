from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from calculate_five_site_wue_nue_from_summary_019_10 import (
    native_or_none,
    num,
    parse_summary_out,
    select_matching_row,
    valid_positive,
)


RUN_DIR = (
    ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "hla2010_seed2_stability_020_02"
    / "2010"
    / "baseline_relative_seed2_50000steps"
)
OUT_DIR = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_seed2_stability_020_02"
BASELINES = (
    ROOT
    / "DSSAT_auto_validation"
    / "five_site_wue_nue_019_10"
    / "019_10_native_wue_nue_metrics.csv"
)
OLD_FLAGS = (
    ROOT
    / "DSSAT_auto_validation"
    / "five_site_checkpoint_pareto_020_01"
    / "020_01_checkpoint_baseline_flags.csv"
)
DOC = ROOT / "docs" / "2026-07-10_020_02_hla2010_seed2_stability_record.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_hash_audit() -> pd.DataFrame:
    old_dir = (
        ROOT / "DSSAT_auto_validation" / "HLA_2004"
        / "hla_baseline_relative_dqn_checkpoint_015_12" / "2010"
        / "baseline_relative_seed0_50000steps" / "input"
    )
    new_dir = RUN_DIR / "input"
    rows = []
    for new_path in sorted(new_dir.iterdir()):
        if not new_path.is_file():
            continue
        old_path = old_dir / new_path.name
        rows.append(
            {
                "file": new_path.name,
                "seed0_sha256": sha256(old_path) if old_path.exists() else None,
                "seed2_sha256": sha256(new_path),
                "identical": old_path.exists() and sha256(old_path) == sha256(new_path),
            }
        )
    return pd.DataFrame(rows)


def extract_seed2() -> pd.DataFrame:
    summary = pd.read_csv(RUN_DIR / "checkpoint_summary.csv")
    rows = []
    for checkpoint in summary.itertuples(index=False):
        step = int(checkpoint.checkpoint_step)
        source = RUN_DIR / f"checkpoint_{step}" / "pdi_tmp_snapshot_eval" / "Summary.OUT"
        parsed = parse_summary_out(source)
        selected, score, selected_index = select_matching_row(
            parsed,
            float(checkpoint.final_grain_kg_ha),
            float(checkpoint.action_irrigation_total),
            float(checkpoint.action_fertilizer_total),
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
                "site": "HLA",
                "seed": 2,
                "checkpoint_step": step,
                "HWAM_kg_ha": num(selected, "HWAM"),
                "WP_ET_kg_m3": ypem * 0.1 if ypem is not None else None,
                "irrigation_mm": float(checkpoint.action_irrigation_total),
                "nitrogen_kg_ha": float(checkpoint.action_fertilizer_total),
                "NLCM_kg_ha": num(selected, "NLCM"),
                "NUCM_kg_ha": nucm,
                "NUtE_kg_kg": ypnum if valid_positive(nucm) and ypnum is not None else None,
                "IWP_gross_kg_m3": ypim * 0.1 if valid_positive(ircm) and ypim is not None else None,
                "PFP_N_kg_kg": ypnam if valid_positive(nicm) and ypnam is not None else None,
                "training_reward": float(checkpoint.total_reward),
                "source_summary_out": source.relative_to(ROOT).as_posix(),
                "source_match_score": score,
                "source_row_index": selected_index,
            }
        )
    return pd.DataFrame(rows).sort_values("checkpoint_step")


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    baselines = pd.read_csv(BASELINES)
    hla = baselines[baselines["site"].eq("HLA")]
    for short, scenario in [
        ("auto", "DSSAT auto"),
        ("extension", "Official extension expert fixed DAP"),
    ]:
        base = hla[hla["scenario"].eq(scenario)].iloc[0]
        out[f"yield_ok_vs_{short}"] = out["HWAM_kg_ha"] >= float(base["HWAM_kg_ha"]) - 1.0
        out[f"WP_ET_ok_vs_{short}"] = out["WP_ET_kg_m3"] >= float(base["WP_ET_kg_m3_DSSAT_YPEM"]) - 0.01
        out[f"input_ok_vs_{short}"] = (
            (out["irrigation_mm"] <= float(base["irrigation_mm_event_table"]))
            & (out["nitrogen_kg_ha"] <= float(base["NICM_kg_ha"]))
        )
        out[f"leaching_ok_vs_{short}"] = out["NLCM_kg_ha"] <= float(base["NLCM_kg_ha"]) + 0.01
        out[f"strict_success_vs_{short}"] = (
            out[f"yield_ok_vs_{short}"]
            & out[f"WP_ET_ok_vs_{short}"]
            & out[f"input_ok_vs_{short}"]
            & out[f"leaching_ok_vs_{short}"]
        )
    out["strict_success_vs_both"] = (
        out["strict_success_vs_auto"] & out["strict_success_vs_extension"]
    )
    return out


def select_by_fixed_rule(df: pd.DataFrame) -> pd.Series:
    return df.sort_values(
        ["training_reward", "checkpoint_step"], ascending=[False, True]
    ).iloc[0]


def selected_seed_comparison(seed2: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(OLD_FLAGS)
    old = old[(old["site"].eq("HLA")) & (old["seed"].isin([0, 1]))]
    selected = []
    for seed, group in old.groupby("seed"):
        row = select_by_fixed_rule(group).copy()
        row["selection_rule"] = "max_reward_then_earliest"
        selected.append(row)
    row = select_by_fixed_rule(seed2).copy()
    row["selection_rule"] = "max_reward_then_earliest"
    selected.append(row)
    columns = [
        "site", "seed", "checkpoint_step", "HWAM_kg_ha", "WP_ET_kg_m3",
        "irrigation_mm", "nitrogen_kg_ha", "NLCM_kg_ha", "NUtE_kg_kg",
        "training_reward", "strict_success_vs_both", "selection_rule",
    ]
    return pd.DataFrame(selected)[columns].sort_values("seed")


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
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    required = [RUN_DIR / "checkpoint_summary.csv", BASELINES, OLD_FLAGS, DOC]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prior evidence:\n" + "\n".join(missing))

    seed2 = add_flags(extract_seed2())
    if seed2["source_match_score"].max() != 0:
        raise ValueError("Seed2 Summary.OUT matching is not exact")
    comparison = selected_seed_comparison(seed2)
    chosen = comparison[comparison["seed"].eq(2)].iloc[0]
    raw_summary = pd.read_csv(RUN_DIR / "checkpoint_summary.csv")
    max_irrigation_event_error = (
        raw_summary["action_irrigation_total"] - raw_summary["irrigation_total_mgmtevent"]
    ).abs().max()
    max_nitrogen_event_error = (
        raw_summary["action_fertilizer_total"] - raw_summary["fertilizer_total_mgmtevent"]
    ).abs().max()
    hashes = input_hash_audit()
    if not hashes["identical"].all():
        raise ValueError("Seed2 formal input differs from seed0 input")

    seed2.to_csv(OUT_DIR / "020_02_seed2_checkpoint_native_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT_DIR / "020_02_seed0_seed1_seed2_fixed_rule_comparison.csv", index=False, encoding="utf-8-sig")
    hashes.to_csv(OUT_DIR / "020_02_input_hash_audit.csv", index=False, encoding="utf-8-sig")

    original = DOC.read_text(encoding="utf-8")
    marker = "## DSSAT 原生指标与最终判定"
    if marker in original:
        original = original.split(marker, 1)[0].rstrip() + "\n\n"
    view = comparison[
        [
            "seed", "checkpoint_step", "HWAM_kg_ha", "WP_ET_kg_m3",
            "irrigation_mm", "nitrogen_kg_ha", "NLCM_kg_ha",
            "training_reward", "strict_success_vs_both",
        ]
    ]
    strict_seed_count = int(comparison["strict_success_vs_both"].astype(bool).sum())
    lines = [
        marker,
        "",
        markdown_table(view),
        "",
        f"- seed2 固定规则选点：{int(chosen['checkpoint_step'])} steps。",
        f"- seed2 原生指标：HWAM={chosen['HWAM_kg_ha']:.1f} kg/ha，WP_ET={chosen['WP_ET_kg_m3']:.2f} kg/m³，I={chosen['irrigation_mm']:.1f} mm，N={chosen['nitrogen_kg_ha']:.1f} kg/ha，NLCM={chosen['NLCM_kg_ha']:.1f} kg/ha。",
        f"- seed2 严格成功：{bool(chosen['strict_success_vs_both'])}。",
        f"- 三个 seed 中按同一固定选点规则严格成功的 seed 数：{strict_seed_count}/3。",
        "- 500-step smoke 正常结束；其结果仅用于检查流程，不用于策略判断。",
        f"- 所有正式 checkpoint 的动作与 MgmtEvent 总量最大差异：灌溉 {max_irrigation_event_error:.3f} mm，施氮 {max_nitrogen_event_error:.3f} kg/ha。",
        f"- 正式 seed2 与 seed0 的 {len(hashes)} 个输入文件 SHA-256 全部一致。",
        "- 结论只针对同一 HLA2010 训练年和当前固定框架，不外推为跨站点或跨年份稳定。",
    ]
    DOC.write_text(original + "\n".join(lines) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
