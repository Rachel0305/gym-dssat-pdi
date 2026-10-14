from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "036_02"
OUT = ROOT / "benchmark_results" / "036_02_correct_03601_baseline_metric_postprocess"
TABLES = OUT / "tables"
DOC = ROOT / "docs" / "036_02_correct_03601_baseline_metric_postprocess_record.md"
PROMPT = ROOT / "prompts" / "036_02_correct_03601_baseline_metric_postprocess.md"

PPO_03601 = (
    ROOT
    / "benchmark_results"
    / "036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun"
    / "evaluation"
    / "036_01_checkpoint_validation_summary.csv"
)
BASE_FOUR_SITE = (
    ROOT
    / "benchmark_results"
    / "031_36_missing_dssat_auto_completion_for_03134"
    / "evaluation"
    / "031_36_full_completed_template_aware_unified_baseline_summary.csv"
)
BASE_SY = (
    ROOT
    / "benchmark_results"
    / "031_29_sy_auto_and_recorded_template_completion"
    / "evaluation"
    / "031_29_sy_baseline_summary.csv"
)

CANONICAL_GROUPS = {
    "null": "null",
    "dssat_auto": "dssat_auto",
    "official_extension_expert": "official_extension_expert",
}


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


def normalize_baseline(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    work = df.copy()
    work["baseline_source_file"] = source_name
    work["station_code"] = work["station_code"].astype(str)
    work["site"] = work.get("site", "").astype(str)
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work["scenario"] = work["scenario"].astype(str)

    def group_name(s: str) -> str:
        if s in CANONICAL_GROUPS:
            return CANONICAL_GROUPS[s]
        if s == "recorded_farmer" or s.startswith("recorded_farmer_template"):
            return "recorded_farmer_or_template"
        return "other"

    work["baseline_group"] = work["scenario"].map(group_name)
    for col in [
        "grain_yield_kg_ha",
        "actual_irrigation_mm",
        "actual_nitrogen_kg_ha",
        "etcp_mm",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "max_water_stress",
        "max_nitrogen_stress",
    ]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    keep = [
        "station_code",
        "site",
        "year",
        "scenario",
        "baseline_group",
        "grain_yield_kg_ha",
        "actual_irrigation_mm",
        "actual_nitrogen_kg_ha",
        "etcp_mm",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "max_water_stress",
        "max_nitrogen_stress",
        "baseline_source_file",
    ]
    return work[keep].copy()


def load_baselines() -> pd.DataFrame:
    parts = [
        normalize_baseline(read_csv(BASE_FOUR_SITE), BASE_FOUR_SITE.relative_to(ROOT).as_posix()),
        normalize_baseline(read_csv(BASE_SY), BASE_SY.relative_to(ROOT).as_posix()),
    ]
    base = pd.concat(parts, ignore_index=True, sort=False)
    base = base[base["baseline_group"].isin(["null", "dssat_auto", "official_extension_expert", "recorded_farmer_or_template"])].copy()
    return base


def baseline_coverage(base: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (station, year), g in base.groupby(["station_code", "year"], dropna=False):
        groups = sorted(g["baseline_group"].dropna().astype(str).unique().tolist())
        scenarios = sorted(g["scenario"].dropna().astype(str).unique().tolist())
        rows.append(
            {
                "station_code": station,
                "year": int(year),
                "available_rows": int(len(g)),
                "available_groups": ",".join(groups),
                "available_scenarios": ",".join(scenarios),
                "has_null": "null" in groups,
                "has_dssat_auto": "dssat_auto" in groups,
                "has_official_extension_expert": "official_extension_expert" in groups,
                "has_recorded_farmer_or_template": "recorded_farmer_or_template" in groups,
                "canonical_group_count": int(len(groups)),
                "has_multiple_recorded_templates": int((g["baseline_group"] == "recorded_farmer_or_template").sum()) > 1,
                "max_yield_available": pd.to_numeric(g["grain_yield_kg_ha"], errors="coerce").max(),
                "max_wp_et_available": pd.to_numeric(g["WP_ET_kg_m3"], errors="coerce").max(),
                "max_pfp_n_available": pd.to_numeric(g["PFP_N_kg_kg"], errors="coerce").max(),
            }
        )
    return pd.DataFrame(rows).sort_values(["station_code", "year"]).reset_index(drop=True)


def add_corrected_flags(ppo: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    out = ppo.copy()
    out["station_code"] = out["station_code"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    if "WP_ET_kg_m3" not in out.columns:
        out["WP_ET_kg_m3"] = np.nan
        out["WP_ET_kg_m3_source"] = "missing_in_03601_daily_summary_requires_dssat_replay"
    else:
        out["WP_ET_kg_m3"] = pd.to_numeric(out["WP_ET_kg_m3"], errors="coerce")
        out["WP_ET_kg_m3_source"] = "03601_existing"
    out["final_grnwt"] = pd.to_numeric(out["final_grnwt"], errors="coerce")
    out["PFP_N"] = pd.to_numeric(out["PFP_N"], errors="coerce")

    merged = out.merge(
        coverage[
            [
                "station_code",
                "year",
                "available_rows",
                "available_groups",
                "available_scenarios",
                "canonical_group_count",
                "has_multiple_recorded_templates",
                "max_yield_available",
                "max_wp_et_available",
                "max_pfp_n_available",
            ]
        ],
        on=["station_code", "year"],
        how="left",
    )
    merged["baseline_available_rows_corrected"] = merged["available_rows"].fillna(0).astype(int)
    merged["baseline_groups_corrected"] = merged["available_groups"].fillna("")
    merged["baseline_scenarios_corrected"] = merged["available_scenarios"].fillna("")
    merged["gap_yield_vs_available_baseline_max"] = merged["final_grnwt"] - merged["max_yield_available"]
    merged["gap_wp_et_vs_available_baseline_max"] = merged["WP_ET_kg_m3"] - merged["max_wp_et_available"]
    merged["gap_pfp_n_vs_available_baseline_max"] = merged["PFP_N"] - merged["max_pfp_n_available"]
    merged["any_metric_win_available_baseline"] = (
        (merged["gap_yield_vs_available_baseline_max"] > 0)
        | (merged["gap_wp_et_vs_available_baseline_max"] > 0)
        | (merged["gap_pfp_n_vs_available_baseline_max"] > 0)
    ).astype(object)
    merged.loc[merged["baseline_available_rows_corrected"] == 0, "any_metric_win_available_baseline"] = np.nan
    return merged


def summarize_by_station_checkpoint(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["run_status"].astype(str).str.startswith("ok")].copy()
    return (
        ok.groupby(["station_code", "checkpoint_step"], as_index=False)
        .agg(
            validation_years=("year", "nunique"),
            mean_final_grnwt=("final_grnwt", "mean"),
            mean_total_irrigation=("total_irrigation", "mean"),
            mean_total_n=("total_n", "mean"),
            mean_PFP_N=("PFP_N", "mean"),
            mean_WP_ET_kg_m3=("WP_ET_kg_m3", "mean"),
            baseline_matched_years=("baseline_available_rows_corrected", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            any_metric_win_available_count=("any_metric_win_available_baseline", lambda s: int(pd.Series(s).fillna(False).sum())),
            yield_win_available_count=("gap_yield_vs_available_baseline_max", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            wp_et_win_available_count=("gap_wp_et_vs_available_baseline_max", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            pfp_n_win_available_count=("gap_pfp_n_vs_available_baseline_max", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            mean_gap_yield_vs_available_max=("gap_yield_vs_available_baseline_max", "mean"),
            mean_gap_wp_et_vs_available_max=("gap_wp_et_vs_available_baseline_max", "mean"),
            mean_gap_pfp_n_vs_available_max=("gap_pfp_n_vs_available_baseline_max", "mean"),
            max_swfac=("max_swfac", "max"),
            max_nstres=("max_nstres", "max"),
        )
        .sort_values(["station_code", "checkpoint_step"])
        .reset_index(drop=True)
    )


def anomaly_flags(corrected: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    zero = corrected[pd.to_numeric(corrected["final_grnwt"], errors="coerce").fillna(np.nan).eq(0)]
    for r in zero.itertuples(index=False):
        rows.append(
            {
                "severity": "high",
                "issue": "ppo_final_grnwt_zero",
                "station_code": r.station_code,
                "year": int(r.year),
                "checkpoint_step": int(r.checkpoint_step),
                "details": f"final_grnwt=0; daily_csv_path={getattr(r, 'daily_csv_path', '')}",
            }
        )
    missing_wp = corrected[corrected["WP_ET_kg_m3"].isna()]
    rows.append(
        {
            "severity": "medium",
            "issue": "ppo_wp_et_missing_all_rows",
            "station_code": "ALL",
            "year": "",
            "checkpoint_step": "",
            "details": f"{len(missing_wp)}/{len(corrected)} PPO rows lack WP_ET_kg_m3; requires checkpoint replay with DSSAT summary retention.",
        }
    )
    missing_base = corrected[corrected["baseline_available_rows_corrected"].eq(0)]
    if len(missing_base):
        rows.append(
            {
                "severity": "high",
                "issue": "baseline_unmatched_rows",
                "station_code": "MULTI",
                "year": "",
                "checkpoint_step": "",
                "details": f"{len(missing_base)} PPO rows still have no matched baseline after merging known sources.",
            }
        )
    multi_template = coverage[coverage["has_multiple_recorded_templates"].fillna(False)]
    if len(multi_template):
        rows.append(
            {
                "severity": "low",
                "issue": "multiple_recorded_family_rows",
                "station_code": "ALL",
                "year": "",
                "checkpoint_step": "",
                "details": f"{len(multi_template)} station-years have more than one recorded_farmer/template row; available-baseline max is conservative but not always a single canonical recorded_farmer.",
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def write_record(corrected: pd.DataFrame, by: pd.DataFrame, coverage: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    lines = [
        "# 036_02 修正 036_01 基线与指标后处理记录",
        "",
        "## 结论先说",
        "",
        "- 036_02 不训练、不改模型，只修正 036_01 的比较后处理。",
        "- SYA 已通过 `031_29_sy_baseline_summary.csv` 接入可用基线；但 SYA recorded farmer 是多个 template，不是单一原始 recorded farmer。",
        "- PPO 的 `WP_ET_kg_m3` 在 036_01 输出中缺失，因此本任务不伪造 WP_ET；若要比较 WP_ET，需要后续用 checkpoint 做只评估重放并保存 DSSAT Summary/ET。",
        "- FQA2018 的 PPO 产量为 0 已列入高优先级异常。",
        "- 快速抽查 FQA2018 日值表显示：`topwt` 最高约 4213–4221 kg/ha，`grnwt` 始终为 0，episode 约 83 天结束；因此它更像是籽粒形成/物候终止异常或真实失败，而不是 PPO 动作没有进入 DSSAT。",
        "",
        "## 汇总表",
        "",
        md_table(by, 80),
        "",
        "## 异常与边界",
        "",
        md_table(anomalies, 80),
        "",
        "## 基线覆盖概览",
        "",
        md_table(
            coverage.groupby("station_code", as_index=False).agg(
                years=("year", "nunique"),
                min_group_count=("canonical_group_count", "min"),
                max_group_count=("canonical_group_count", "max"),
                multi_recorded_template_years=("has_multiple_recorded_templates", "sum"),
            ),
            20,
        ),
        "",
        "## 输出文件",
        "",
        "- `benchmark_results/036_02_correct_03601_baseline_metric_postprocess/tables/036_02_corrected_checkpoint_validation_summary.csv`",
        "- `benchmark_results/036_02_correct_03601_baseline_metric_postprocess/tables/036_02_corrected_by_station_checkpoint.csv`",
        "- `benchmark_results/036_02_correct_03601_baseline_metric_postprocess/tables/036_02_baseline_coverage_by_station_year.csv`",
        "- `benchmark_results/036_02_correct_03601_baseline_metric_postprocess/tables/036_02_anomaly_flags.csv`",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    ppo = read_csv(PPO_03601)
    base = load_baselines()
    coverage = baseline_coverage(base)
    corrected = add_corrected_flags(ppo, coverage)
    by = summarize_by_station_checkpoint(corrected)
    anomalies = anomaly_flags(corrected, coverage)

    base.to_csv(TABLES / "036_02_combined_baseline_rows.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(TABLES / "036_02_baseline_coverage_by_station_year.csv", index=False, encoding="utf-8-sig")
    corrected.to_csv(TABLES / "036_02_corrected_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
    by.to_csv(TABLES / "036_02_corrected_by_station_checkpoint.csv", index=False, encoding="utf-8-sig")
    anomalies.to_csv(TABLES / "036_02_anomaly_flags.csv", index=False, encoding="utf-8-sig")
    write_record(corrected, by, coverage, anomalies)

    payload = {
        "task": TASK_ID,
        "ppo_rows": int(len(ppo)),
        "corrected_rows": int(len(corrected)),
        "stations": sorted(corrected["station_code"].astype(str).unique().tolist()),
        "wp_et_missing_rows": int(corrected["WP_ET_kg_m3"].isna().sum()),
        "baseline_unmatched_rows": int(corrected["baseline_available_rows_corrected"].eq(0).sum()),
        "zero_yield_rows": int(pd.to_numeric(corrected["final_grnwt"], errors="coerce").eq(0).sum()),
        "record_md": DOC.relative_to(ROOT).as_posix(),
        "corrected_summary": (TABLES / "036_02_corrected_checkpoint_validation_summary.csv").relative_to(ROOT).as_posix(),
        "by_station": (TABLES / "036_02_corrected_by_station_checkpoint.csv").relative_to(ROOT).as_posix(),
    }
    (OUT / "036_02_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(by.to_string(index=False))
    print(anomalies.to_string(index=False))


if __name__ == "__main__":
    main()
