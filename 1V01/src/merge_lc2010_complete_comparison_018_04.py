from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "extension_expert_baseline_018_03"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-09_018_04_lc2010_complete_four_scenario_merge_record.md"


def row(site, station, year, label, grain, biomass, irrigation, nitrogen, max_w, max_n, source, note):
    return {
        "site": site,
        "station": station,
        "year": year,
        "scenario_label": label,
        "scenario_group": label,
        "grain_yield_kg_ha": grain,
        "biomass_kg_ha": biomass,
        "irrigation_mm": irrigation,
        "nitrogen_kg_ha": nitrogen,
        "max_water_stress": max_w,
        "max_nitrogen_stress": max_n,
        "source_file": source,
        "note": note,
    }


def build_lc_complete() -> pd.DataFrame:
    rows = []
    lc_summary_path = PROJECT_ROOT / "DSSAT_auto_validation" / "lc_fixed_input_year_screening_017_11" / "017_11_lc_fixed_input_summary.csv"
    lc_summary = pd.read_csv(lc_summary_path)
    lc2010 = lc_summary[lc_summary["year"].eq(2010)].copy()
    label_map = {
        np.nan: "Null",
        "recorded": "Recorded/farmer practice",
        "dssat_auto": "DSSAT auto",
    }
    for _, r in lc2010.iterrows():
        scen = r["scenario"]
        label = "Null" if pd.isna(scen) else label_map.get(str(scen), str(scen))
        rows.append(
            row(
                "LC",
                "Luancheng",
                2010,
                label,
                float(r["final_gwad"]),
                float(r["final_cwad"]),
                float(r["event_irrigation_total"]),
                float(r["event_fertilizer_total"]),
                float(r["max_water_stress"]),
                float(r["max_nitrogen_stress"]),
                str(lc_summary_path.relative_to(PROJECT_ROOT)),
                "LC 017_11 fixed-input screening complete baseline",
            )
        )

    ckpt_path = PROJECT_ROOT / "DSSAT_auto_validation" / "lc2010_baseline_relative_dqn_smoke_017_12" / "seed0_5000steps" / "checkpoint_summary.csv"
    ckpt = pd.read_csv(ckpt_path)
    best = ckpt.sort_values("total_reward", ascending=False).iloc[0]
    rows.append(
        row(
            "LC",
            "Luancheng",
            2010,
            "DQN best checkpoint",
            float(best["final_grain_kg_ha"]),
            float(best["final_biomass_kg_ha"]),
            float(best["event_irrigation_total"]),
            float(best["event_fertilizer_total"]),
            float(best["max_water_stress"]),
            float(best["max_nitrogen_stress"]),
            str(ckpt_path.relative_to(PROJECT_ROOT)),
            f"LC DQN seed0 best reward checkpoint={int(best['checkpoint_step'])}",
        )
    )

    ext_path = OUT_DIR / "018_03_extension_expert_summary.csv"
    ext = pd.read_csv(ext_path)
    lc_ext = ext[(ext["site"].eq("LC")) & (ext["year"].eq(2010))].iloc[0]
    rows.append(
        row(
            "LC",
            "Luancheng",
            2010,
            "Official extension expert fixed DAP",
            float(lc_ext["final_gwad"]),
            float(lc_ext["final_cwad"]),
            float(lc_ext["irrigation_mm"]),
            float(lc_ext["nitrogen_kg_ha"]),
            float(lc_ext["max_water_stress"]),
            float(lc_ext["max_nitrogen_stress"]),
            str(ext_path.relative_to(PROJECT_ROOT)),
            "official extension schedule fixed DAP",
        )
    )
    return pd.DataFrame(rows)


def rebuild_multisite_clean(lc_complete: pd.DataFrame) -> pd.DataFrame:
    clean_path = OUT_DIR / "018_03_clean_multisite_comparison_with_extension_expert.csv"
    clean = pd.read_csv(clean_path)
    clean_no_lc = clean[~clean["site"].eq("LC")].copy()
    out = pd.concat([clean_no_lc, lc_complete], ignore_index=True)
    order = {
        "Null": 0,
        "Recorded expert": 1,
        "Recorded/farmer practice": 1,
        "DSSAT auto": 2,
        "DQN best checkpoint": 3,
        "DQN ckpt15000": 3,
        "Official extension expert fixed DAP": 4,
    }
    out["_order"] = out["scenario_label"].map(order).fillna(9)
    out = out.sort_values(["site", "year", "_order", "scenario_label"]).drop(columns=["_order"])
    return out


def md_table(df: pd.DataFrame) -> str:
    view = df.copy()
    for c in view.columns:
        view[c] = view[c].map(lambda x: "" if pd.isna(x) else (f"{x:.3f}" if isinstance(x, (float, np.floating)) else str(x)))
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, r in view.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in view.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    lc_complete = build_lc_complete()
    lc_path = OUT_DIR / "018_04_lc2010_complete_comparison.csv"
    lc_complete.to_csv(lc_path, index=False, encoding="utf-8-sig")

    multisite = rebuild_multisite_clean(lc_complete)
    clean_path = OUT_DIR / "018_03_clean_multisite_comparison_with_extension_expert.csv"
    multisite.to_csv(clean_path, index=False, encoding="utf-8-sig")

    lines = [
        "# 018_04 LC2010 完整四情景补齐记录",
        "",
        "## 做了什么",
        "",
        "018_03 中 LC2010 的 null / recorded / DSSAT auto 曾只使用旧脚本常量，缺少生物量、资源投入和胁迫字段。本轮不训练、不重跑 DSSAT，只从已经存在的 LC 017_11 完整筛选结果中抽取 LC2010 三个传统情景，并合并 LC DQN seed0 最佳奖励 checkpoint 与 018_03 官方推广 expert。",
        "",
        "## LC2010 完整对照表",
        "",
        md_table(lc_complete[["site", "station", "year", "scenario_label", "grain_yield_kg_ha", "biomass_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "max_water_stress", "max_nitrogen_stress", "source_file", "note"]]),
        "",
        "## 输出",
        "",
        f"- LC complete: `{lc_path.relative_to(PROJECT_ROOT)}`",
        f"- Updated multisite clean comparison: `{clean_path.relative_to(PROJECT_ROOT)}`",
        "",
        "## 结论",
        "",
        "LC2010 现在已经和其他站点一样具备完整字段。当前 LC2010 的 DQN seed0 最佳奖励 checkpoint 产量为 8739 kg/ha，灌溉 90 mm，施氮 0 kg/ha；与 DSSAT auto 8738 kg/ha 接近并略高，且比 recorded/farmer practice 8732 kg/ha 略高。官方推广 expert 产量 8739 kg/ha，但投入约 198.8 mm 灌溉和 247 kg/ha 氮。因此 LC2010 在 seed0 上也支持“低投入达到高产平台”的叙事，但此前 seed1 显示资源使用不稳定，不能直接作为跨 seed 稳定成功结论。",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lc_complete.to_string(index=False))
    print(f"Wrote {lc_path}")
    print(f"Updated {clean_path}")
    print(f"Wrote {DOC_PATH}")


if __name__ == "__main__":
    main()

