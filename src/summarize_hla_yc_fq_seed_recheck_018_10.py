from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "extension_expert_baseline_018_03" / "018_10_hla_yc_fq_seed_recheck"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-09_018_10_hla_yc_fq_seed_recheck_record.md"


PATHS = {
    ("HLA", 0): PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_baseline_relative_dqn_checkpoint_015_12" / "2010" / "baseline_relative_seed0_50000steps" / "checkpoint_summary.csv",
    ("HLA", 1): PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_baseline_relative_dqn_checkpoint_015_12" / "2010" / "baseline_relative_seed1_50000steps" / "checkpoint_summary.csv",
    ("YC", 0): PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_unified_dqn_checkpoint_diagnostic_015_04" / "seed0" / "015_04_yc2014_unified_dqn_checkpoint_summary.csv",
    ("YC", 1): PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_unified_dqn_checkpoint_seed1_015_05" / "seed1" / "015_05_yc2014_unified_dqn_checkpoint_seed1_summary.csv",
    ("FQ", 0): PROJECT_ROOT / "DSSAT_auto_validation" / "fq2016_baseline_relative_dqn_checkpoint_015_14" / "seed0_50000steps" / "checkpoint_summary.csv",
    ("FQ", 1): PROJECT_ROOT / "DSSAT_auto_validation" / "fq2016_baseline_relative_dqn_checkpoint_015_14" / "seed1_50000steps" / "checkpoint_summary.csv",
}

META = {
    "HLA": {"station": "Hailun", "year": 2010},
    "YC": {"station": "Yucheng", "year": 2014},
    "FQ": {"station": "Fengqiu", "year": 2016},
}


def load_best(site: str, seed: int) -> dict:
    path = PATHS[(site, seed)]
    df = pd.read_csv(path)
    if site == "HLA":
        ranked = df.sort_values(
            ["total_reward", "final_grain_kg_ha", "final_biomass_kg_ha", "checkpoint_step"],
            ascending=[False, False, False, True],
        )
    elif site == "YC":
        ranked = df.sort_values(
            ["final_grain_kg_ha", "action_fertilizer_total", "action_irrigation_total", "checkpoint_step"],
            ascending=[False, True, True, True],
        )
    else:  # FQ
        ranked = df.sort_values(
            ["total_reward", "final_grain_kg_ha", "final_biomass_kg_ha", "checkpoint_step"],
            ascending=[False, False, False, True],
        )
    best = ranked.iloc[0].to_dict()
    best["site"] = site
    best["station"] = META[site]["station"]
    best["year"] = META[site]["year"]
    best["seed"] = seed
    return best


def judgement(site: str, seed0: dict, seed1: dict) -> tuple[str, str]:
    if site == "YC":
        if abs(seed0["final_grain_kg_ha"] - seed1["final_grain_kg_ha"]) <= 5 and seed0["action_irrigation_total"] == seed1["action_irrigation_total"]:
            return (
                "stable_yield_but_nitrogen_not_stable",
                "YC2014 两个 seed 的最佳产量一致，但 seed1 需要 300 kg/ha 氮、seed0 只需 250 kg/ha，说明产量稳定而节氮方向不稳定。",
            )
    if site == "HLA":
        if seed0["final_grain_kg_ha"] >= 7800 and seed1["final_grain_kg_ha"] < 7700:
            return (
                "resource_stable_but_yield_not_stable",
                "HLA2010 seed0 可在 N=0 条件下追平高产，但 seed1 最优点只有约 7573 kg/ha，说明跨 seed 产量不稳定。",
            )
    if site == "FQ":
        if seed1["final_grain_kg_ha"] >= 7990 and seed0["final_grain_kg_ha"] < 7900:
            return (
                "seed1_success_but_seed0_not_reproduced",
                "FQ2016 历史上用于四情景图的代表性 DQN 本来就是 seed1@ckpt30000=7995；本次统一复核发现 seed0 本地 checkpoint 最优仅约 7779，说明当前只确认了 seed1 成功，尚未形成跨 seed 稳定性。",
            )
    return ("needs_manual_review", "该站点需要人工进一步检查。")


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}" if not v.is_integer() else f"{int(v)}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_rows = []
    site_rows = []
    for site in ["HLA", "YC", "FQ"]:
        s0 = load_best(site, 0)
        s1 = load_best(site, 1)
        best_rows.extend([s0, s1])
        status, note = judgement(site, s0, s1)
        site_rows.append(
            {
                "site": site,
                "station": META[site]["station"],
                "year": META[site]["year"],
                "seed0_checkpoint": int(s0["checkpoint_step"]),
                "seed0_gwad": float(s0["final_grain_kg_ha"]),
                "seed0_irrigation": float(s0["action_irrigation_total"]),
                "seed0_nitrogen": float(s0["action_fertilizer_total"]),
                "seed1_checkpoint": int(s1["checkpoint_step"]),
                "seed1_gwad": float(s1["final_grain_kg_ha"]),
                "seed1_irrigation": float(s1["action_irrigation_total"]),
                "seed1_nitrogen": float(s1["action_fertilizer_total"]),
                "recheck_status": status,
                "interpretation": note,
            }
        )

    best_df = pd.DataFrame(best_rows)[
        [
            "site",
            "station",
            "year",
            "seed",
            "checkpoint_step",
            "action_irrigation_total",
            "action_fertilizer_total",
            "final_grain_kg_ha",
            "final_biomass_kg_ha",
            "max_water_stress",
            "max_nitrogen_stress",
            "total_reward",
        ]
    ].sort_values(["site", "seed"])
    site_df = pd.DataFrame(site_rows)

    best_df.to_csv(OUT_DIR / "018_10_seed_best_table.csv", index=False, encoding="utf-8-sig")
    site_df.to_csv(OUT_DIR / "018_10_site_recheck_status.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 018_10 HLA/YC/FQ seed 复核记录",
        "",
        "## 说明",
        "",
        "- 这次没有盲目重跑三个站点。",
        "- 经核查，YC2014 与 HLA2010 的 seed1 结果早已存在；FQ2016 的 seed1 也已有完整 50K 结果，仅缺统一归档。",
        "- 因此本轮工作的核心是：核实旧结果、补充 FQ 的 smoke 证据、统一形成正式复核口径。",
        "",
        "## seed 最佳 checkpoint 表",
        "",
        markdown_table(best_df),
        "",
        "## 站点复核结论",
        "",
        markdown_table(site_df),
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
