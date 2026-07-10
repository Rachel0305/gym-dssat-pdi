from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "extension_expert_baseline_018_03"
OUT_DIR = BASE_DIR / "018_05_multisite_dqn_vs_extension_audit"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-09_018_05_multisite_dqn_vs_extension_expert_audit_record.md"
INPUT = BASE_DIR / "018_03_clean_multisite_comparison_with_extension_expert.csv"

YIELD_TOL_KG_HA = 20.0
RESOURCE_TOL = 1e-6


def norm_label(label: str) -> str:
    label = str(label)
    if label in {"DQN ckpt15000", "DQN best checkpoint"}:
        return "DQN best checkpoint"
    if label in {"Recorded expert", "Recorded/farmer practice"}:
        return "Recorded/farmer practice"
    return label


def get_row(group: pd.DataFrame, label: str) -> pd.Series | None:
    hit = group[group["scenario_label_norm"].eq(label)]
    if hit.empty:
        return None
    return hit.iloc[0]


def status_flag(dqn: float, ref: float, tolerance: float = YIELD_TOL_KG_HA) -> str:
    if not np.isfinite(dqn) or not np.isfinite(ref):
        return "missing"
    diff = dqn - ref
    if diff > tolerance:
        return "exceeds"
    if diff >= -tolerance:
        return "matches"
    return "lower"


def resource_flag(dqn: float, ref: float) -> str:
    if not np.isfinite(dqn) or not np.isfinite(ref):
        return "missing"
    diff = dqn - ref
    if diff < -RESOURCE_TOL:
        return "saves"
    if abs(diff) <= RESOURCE_TOL:
        return "same"
    return "uses_more"


def classify_site(row: dict) -> tuple[str, str, int]:
    auto_yield_ok = row["yield_status_vs_dssat_auto"] in {"matches", "exceeds"}
    ext_yield_ok = row["yield_status_vs_extension"] in {"matches", "exceeds"}
    auto_water_save = row["irrigation_status_vs_dssat_auto"] == "saves"
    ext_water_save = row["irrigation_status_vs_extension"] == "saves"
    auto_n_save = row["nitrogen_status_vs_dssat_auto"] == "saves"
    ext_n_save = row["nitrogen_status_vs_extension"] == "saves"

    resource_good_vs_any_key = auto_water_save or ext_water_save or auto_n_save or ext_n_save

    if auto_yield_ok and ext_yield_ok and resource_good_vs_any_key:
        return (
            "promising_but_needs_seed",
            "DQN 已追平/超过 DSSAT auto 和官方推广 expert，且至少在水或氮上有节约；但仍需跨 seed 复核后才能称为稳定成功。",
            1,
        )
    if (auto_yield_ok or ext_yield_ok) and resource_good_vs_any_key:
        return (
            "near_plateau_tradeoff",
            "DQN 接近关键产量平台，并在部分资源投入上有优势；适合保留为候选，但需看导师是否接受产量-资源权衡。",
            2,
        )
    return (
        "needs_reward_or_setup_review",
        "DQN 尚未同时满足产量追平关键基线和资源节约，需要检查奖励函数、动作空间、初始条件或训练稳定性。",
        3,
    )


def build_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["scenario_label_norm"] = df["scenario_label"].map(norm_label)
    for col in ["grain_yield_kg_ha", "biomass_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "max_water_stress", "max_nitrogen_stress"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    pairwise_rows = []
    site_rows = []
    for (site, station, year), group in df.groupby(["site", "station", "year"], dropna=False):
        dqn = get_row(group, "DQN best checkpoint")
        if dqn is None:
            continue
        refs = {
            "null": get_row(group, "Null"),
            "recorded_farmer": get_row(group, "Recorded/farmer practice"),
            "dssat_auto": get_row(group, "DSSAT auto"),
            "extension": get_row(group, "Official extension expert fixed DAP"),
        }
        base = {
            "site": site,
            "station": station,
            "year": int(year),
            "dqn_yield": float(dqn["grain_yield_kg_ha"]),
            "dqn_biomass": float(dqn["biomass_kg_ha"]),
            "dqn_irrigation": float(dqn["irrigation_mm"]),
            "dqn_nitrogen": float(dqn["nitrogen_kg_ha"]),
            "dqn_max_water_stress": float(dqn["max_water_stress"]),
            "dqn_max_nitrogen_stress": float(dqn["max_nitrogen_stress"]),
            "dqn_source": dqn["source_file"],
        }
        for ref_name, ref in refs.items():
            if ref is None:
                pairwise_rows.append({**base, "reference": ref_name, "reference_missing": True})
                continue
            yield_diff = float(dqn["grain_yield_kg_ha"] - ref["grain_yield_kg_ha"])
            biomass_diff = float(dqn["biomass_kg_ha"] - ref["biomass_kg_ha"])
            irrigation_diff = float(dqn["irrigation_mm"] - ref["irrigation_mm"])
            nitrogen_diff = float(dqn["nitrogen_kg_ha"] - ref["nitrogen_kg_ha"])
            pairwise_rows.append(
                {
                    **base,
                    "reference": ref_name,
                    "reference_missing": False,
                    "ref_yield": float(ref["grain_yield_kg_ha"]),
                    "ref_biomass": float(ref["biomass_kg_ha"]),
                    "ref_irrigation": float(ref["irrigation_mm"]),
                    "ref_nitrogen": float(ref["nitrogen_kg_ha"]),
                    "yield_diff_dqn_minus_ref": yield_diff,
                    "biomass_diff_dqn_minus_ref": biomass_diff,
                    "irrigation_diff_dqn_minus_ref": irrigation_diff,
                    "nitrogen_diff_dqn_minus_ref": nitrogen_diff,
                    "yield_status": status_flag(float(dqn["grain_yield_kg_ha"]), float(ref["grain_yield_kg_ha"])),
                    "irrigation_status": resource_flag(float(dqn["irrigation_mm"]), float(ref["irrigation_mm"])),
                    "nitrogen_status": resource_flag(float(dqn["nitrogen_kg_ha"]), float(ref["nitrogen_kg_ha"])),
                    "ref_source": ref["source_file"],
                }
            )

        lookup = {r["reference"]: r for r in pairwise_rows if r.get("site") == site and r.get("year") == int(year) and not r.get("reference_missing", False)}
        site_row = {
            **base,
            "yield_diff_vs_dssat_auto": lookup.get("dssat_auto", {}).get("yield_diff_dqn_minus_ref", np.nan),
            "irrigation_diff_vs_dssat_auto": lookup.get("dssat_auto", {}).get("irrigation_diff_dqn_minus_ref", np.nan),
            "nitrogen_diff_vs_dssat_auto": lookup.get("dssat_auto", {}).get("nitrogen_diff_dqn_minus_ref", np.nan),
            "yield_status_vs_dssat_auto": lookup.get("dssat_auto", {}).get("yield_status", "missing"),
            "irrigation_status_vs_dssat_auto": lookup.get("dssat_auto", {}).get("irrigation_status", "missing"),
            "nitrogen_status_vs_dssat_auto": lookup.get("dssat_auto", {}).get("nitrogen_status", "missing"),
            "yield_diff_vs_extension": lookup.get("extension", {}).get("yield_diff_dqn_minus_ref", np.nan),
            "irrigation_diff_vs_extension": lookup.get("extension", {}).get("irrigation_diff_dqn_minus_ref", np.nan),
            "nitrogen_diff_vs_extension": lookup.get("extension", {}).get("nitrogen_diff_dqn_minus_ref", np.nan),
            "yield_status_vs_extension": lookup.get("extension", {}).get("yield_status", "missing"),
            "irrigation_status_vs_extension": lookup.get("extension", {}).get("irrigation_status", "missing"),
            "nitrogen_status_vs_extension": lookup.get("extension", {}).get("nitrogen_status", "missing"),
            "yield_diff_vs_recorded": lookup.get("recorded_farmer", {}).get("yield_diff_dqn_minus_ref", np.nan),
            "irrigation_diff_vs_recorded": lookup.get("recorded_farmer", {}).get("irrigation_diff_dqn_minus_ref", np.nan),
            "nitrogen_diff_vs_recorded": lookup.get("recorded_farmer", {}).get("nitrogen_diff_dqn_minus_ref", np.nan),
        }
        cls, rationale, priority = classify_site(site_row)
        site_row["current_evidence_level"] = cls
        site_row["next_priority_rank"] = priority
        site_row["interpretation"] = rationale
        site_rows.append(site_row)

    pairwise = pd.DataFrame(pairwise_rows)
    site_audit = pd.DataFrame(site_rows).sort_values(["next_priority_rank", "site"])
    return site_audit, pairwise


def plot_audit(site_audit: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(site_audit))
    labels = site_audit["site"] + "\n" + site_audit["year"].astype(str)
    specs = [
        ("yield_diff_vs_extension", "Yield diff vs official expert", "kg/ha", "#4477AA"),
        ("irrigation_diff_vs_extension", "Irrigation diff vs official expert", "mm", "#228833"),
        ("nitrogen_diff_vs_extension", "Nitrogen diff vs official expert", "kg/ha", "#AA3377"),
    ]
    for ax, (col, title, ylabel, color) in zip(axes, specs):
        vals = pd.to_numeric(site_audit[col], errors="coerce")
        ax.axhline(0, color="#222222", lw=1)
        ax.bar(x, vals, color=color, edgecolor="#222222", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("018_05 DQN advantage relative to official extension expert", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    view = df[cols].copy()
    for c in view.columns:
        view[c] = view[c].map(lambda x: "" if pd.isna(x) else (f"{x:.3f}" if isinstance(x, (float, np.floating)) else str(x)))
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, r in view.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in view.columns) + " |")
    return "\n".join(lines)


def write_record(site_audit: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    key_cols = [
        "site",
        "station",
        "year",
        "dqn_yield",
        "dqn_irrigation",
        "dqn_nitrogen",
        "yield_diff_vs_dssat_auto",
        "yield_diff_vs_extension",
        "irrigation_diff_vs_extension",
        "nitrogen_diff_vs_extension",
        "current_evidence_level",
        "next_priority_rank",
        "interpretation",
    ]
    pair_cols = [
        "site",
        "year",
        "reference",
        "dqn_yield",
        "ref_yield",
        "yield_diff_dqn_minus_ref",
        "dqn_irrigation",
        "ref_irrigation",
        "irrigation_diff_dqn_minus_ref",
        "dqn_nitrogen",
        "ref_nitrogen",
        "nitrogen_diff_dqn_minus_ref",
        "yield_status",
        "irrigation_status",
        "nitrogen_status",
    ]
    lines = [
        "# 018_05 五站点 DQN vs 官方推广 expert 审计记录",
        "",
        "## 做了什么",
        "",
        "本轮没有训练、没有重跑 DSSAT、没有修改奖励函数。只读取 018_03/018_04 已整理的五站点代表年份合并表，计算 DQN 相对 DSSAT auto、官方推广 expert、recorded/farmer practice 的产量和水氮投入差异。",
        "",
        "## 站点级审计结果",
        "",
        md_table(site_audit, key_cols),
        "",
        "## DQN 相对各基线的逐项差异",
        "",
        md_table(pairwise[pairwise["reference"].isin(["dssat_auto", "extension", "recorded_farmer"])], pair_cols),
        "",
        "## 当前解释",
        "",
        "- 这些结果说明 DQN 的提升不是只相对旧 recorded/farmer practice 成立；在若干站点上，DQN 也能接近或达到官方推广 expert / DSSAT auto 的产量平台，并减少部分水氮投入。",
        "- 但这仍是代表年份和 best checkpoint 证据，不等于五站点全部跨 seed 稳定成功。",
        "- 下一步优先做 seed 稳定性复核，而不是马上改奖励函数。",
        "",
        "## 建议下一步",
        "",
        "1. 优先复核 LC2010：seed0 很漂亮，但 seed1 曾显示资源使用不稳。",
        "2. 复核 SY2014：DQN 产量高于官方推广 expert，但资源投入和 seed 稳定性需要确认。",
        "3. 整理 HLA2010 的 seed0/seed1 差异，确认是否只作为“高产平台节水候选”。",
        "4. YC2014/FQ2016 作为水氮权衡案例，暂不急着改奖励函数，先看导师是否接受“接近产量平台并节约部分资源”的叙事。",
        "",
        "## 输出文件",
        "",
        f"- site audit: `{(OUT_DIR / '018_05_site_level_audit.csv').relative_to(PROJECT_ROOT)}`",
        f"- pairwise: `{(OUT_DIR / '018_05_pairwise_dqn_advantage.csv').relative_to(PROJECT_ROOT)}`",
        f"- figure: `{(OUT_DIR / 'figures' / '018_05_dqn_advantage_summary.png').relative_to(PROJECT_ROOT)}`",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    site_audit, pairwise = build_audit(df)
    site_audit.to_csv(OUT_DIR / "018_05_site_level_audit.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(OUT_DIR / "018_05_pairwise_dqn_advantage.csv", index=False, encoding="utf-8-sig")
    plot_audit(site_audit, OUT_DIR / "figures" / "018_05_dqn_advantage_summary.png")
    write_record(site_audit, pairwise)
    print(site_audit[["site", "year", "dqn_yield", "dqn_irrigation", "dqn_nitrogen", "yield_diff_vs_dssat_auto", "yield_diff_vs_extension", "irrigation_diff_vs_extension", "nitrogen_diff_vs_extension", "current_evidence_level", "next_priority_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()

