from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "prompts" / "032_21_five_site_half_split_free_timing_ppo_readiness_audit.md"
OUT = ROOT / "benchmark_results" / "032_21_five_site_half_split_free_timing_ppo_readiness_audit"
TABLES = OUT / "tables"
DOC = ROOT / "docs" / "032_21_five_site_half_split_free_timing_ppo_readiness_audit_record.md"

POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"
UNIFIED_BASELINE = ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134" / "evaluation" / "031_36_full_completed_template_aware_unified_baseline_summary.csv"
SY_BASELINE = ROOT / "benchmark_results" / "031_29_sy_auto_and_recorded_template_completion" / "evaluation" / "031_29_sy_baseline_summary.csv"
LC_03220_SUMMARY = ROOT / "benchmark_results" / "032_20_lc_75k_ppo_consistent_snapshot_daily_rebuild" / "032_20_lc2005_2020_five_scenario_summary_snapshot_derived.csv"
FOUR_SITE_PPO_03134 = ROOT / "benchmark_results" / "031_34_four_site_all_year_frozen_maskableppo_transfer" / "evaluation" / "031_34_full_candidate_vs_baseline.csv"
SY_PPO_03130 = ROOT / "benchmark_results" / "031_30_sy_ppo_vs_completed_baselines" / "evaluation" / "031_30_sy_ppo_vs_completed_baselines.csv"

STATIONS = ["SYA", "HLA", "FQA", "LCA", "YCA"]
STATION_TO_SITE = {"SYA": "SY", "HLA": "HLA", "FQA": "FQ", "LCA": "LC", "YCA": "YC"}
SCENARIOS = ["null", "recorded_farmer", "dssat_auto", "official_extension_expert"]


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    (OUT / "configs").mkdir(parents=True, exist_ok=True)


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


def norm_scenario(value: Any) -> str:
    text = str(value).strip()
    mapping = {
        "recorded": "recorded_farmer",
        "recorded_farmer_template": "recorded_farmer",
        "recorded_farmer_template_02705": "recorded_farmer",
        "official_expert": "official_extension_expert",
        "extension_expert": "official_extension_expert",
        "extension_expert_fixed_dap": "official_extension_expert",
    }
    return mapping.get(text, text)


def load_available_years() -> pd.DataFrame:
    pool = pd.read_csv(POOL, keep_default_na=False)
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").astype("Int64")
    pool = pool[pool["station_code"].isin(STATIONS) & pool["year"].ge(2000)].copy()
    pool["site"] = pool["station_code"].map(STATION_TO_SITE)
    pool["weather_file_normalized"] = pool["weather_file"].astype(str).str.replace("\\", "/", regex=False)
    pool["weather_path"] = pool["weather_file_normalized"].map(lambda p: (ROOT / p).as_posix())
    pool["weather_file_exists"] = pool["weather_file_normalized"].map(lambda p: (ROOT / p).exists())
    return pool.sort_values(["station_code", "year"]).reset_index(drop=True)


def split_years(available: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for station, group in available[available["weather_file_exists"]].groupby("station_code", sort=False):
        years = sorted(group["year"].astype(int).unique().tolist())
        train_n = len(years) // 2
        train_years = set(years[:train_n])
        for year in years:
            rows.append(
                {
                    "station_code": station,
                    "site": STATION_TO_SITE[station],
                    "year": year,
                    "split": "train" if year in train_years else "validation",
                    "split_rule": "按年份排序；前半训练、后半验证；奇数年份数时后半验证多1年",
                    "n_available_years_for_station": len(years),
                }
            )
    return pd.DataFrame(rows).sort_values(["station_code", "year"]).reset_index(drop=True)


def load_baseline_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if UNIFIED_BASELINE.exists():
        df = pd.read_csv(UNIFIED_BASELINE, keep_default_na=False)
        frames.append(df)
    if SY_BASELINE.exists():
        df = pd.read_csv(SY_BASELINE, keep_default_na=False)
        if "station_code" not in df.columns:
            df["station_code"] = "SYA"
        frames.append(df)
    if LC_03220_SUMMARY.exists():
        df = pd.read_csv(LC_03220_SUMMARY, keep_default_na=False)
        df = df[df["scenario"].isin(SCENARIOS)].copy()
        df["station_code"] = "LCA"
        df["site"] = "LC"
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["station_code", "site", "year", "scenario"])
    base = pd.concat(frames, ignore_index=True, sort=False)
    if "station_code" not in base.columns:
        base["station_code"] = base["site"].map({v: k for k, v in STATION_TO_SITE.items()})
    if "site" not in base.columns:
        base["site"] = base["station_code"].map(STATION_TO_SITE)
    base["year"] = pd.to_numeric(base["year"], errors="coerce").astype("Int64")
    base["scenario_norm"] = base["scenario"].map(norm_scenario)
    base = base[base["station_code"].isin(STATIONS) & base["year"].ge(2000) & base["scenario_norm"].isin(SCENARIOS)].copy()
    return base


def baseline_coverage(split: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, item in split.iterrows():
        station = item["station_code"]
        year = int(item["year"])
        subset = baselines[baselines["station_code"].eq(station) & baselines["year"].astype(int).eq(year)]
        present = sorted(set(subset["scenario_norm"].astype(str).tolist()))
        missing = [scenario for scenario in SCENARIOS if scenario not in present]
        rows.append(
            {
                "station_code": station,
                "site": item["site"],
                "year": year,
                "split": item["split"],
                "baseline_present_count": len([s for s in SCENARIOS if s in present]),
                "baseline_complete_4": len(missing) == 0,
                "present_scenarios": ",".join(present),
                "missing_scenarios": ",".join(missing),
            }
        )
    return pd.DataFrame(rows).sort_values(["station_code", "year"]).reset_index(drop=True)


def existing_ppo_reference(split: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if FOUR_SITE_PPO_03134.exists():
        df = pd.read_csv(FOUR_SITE_PPO_03134, keep_default_na=False)
        rows.append(
            df.assign(source="031_34_four_site_single_source_year_frozen_transfer")[
                ["station_code", "site", "year", "source", "run_status", "final_grain_kg_ha", "total_irrigation", "total_n", "action_sequence"]
            ]
        )
    if SY_PPO_03130.exists():
        df = pd.read_csv(SY_PPO_03130, keep_default_na=False)
        if "station_code" not in df.columns:
            df["station_code"] = "SYA"
        if "site" not in df.columns:
            df["site"] = "SY"
        rows.append(
            df.assign(source="031_30_sy_frozen_transfer_reference")[
                [c for c in ["station_code", "site", "year", "source", "run_status", "final_grain_kg_ha", "total_irrigation", "total_n", "action_sequence"] if c in df.columns or c == "source"]
            ]
        )
    if LC_03220_SUMMARY.exists():
        df = pd.read_csv(LC_03220_SUMMARY, keep_default_na=False)
        df = df[df["scenario"].eq("rl_candidate")].copy()
        df["station_code"] = "LCA"
        df["site"] = "LC"
        df["run_status"] = "ok"
        df["source"] = "032_20_lc_multiyear_75k_snapshot_rebuild_reference"
        df["total_irrigation"] = df["irrigation_event_total_mm"]
        df["total_n"] = df["nitrogen_event_total_kg_ha"]
        rows.append(df[["station_code", "site", "year", "source", "run_status", "final_grain_kg_ha", "total_irrigation", "total_n"]])
    if not rows:
        return pd.DataFrame()
    ref = pd.concat(rows, ignore_index=True, sort=False)
    ref["year"] = pd.to_numeric(ref["year"], errors="coerce").astype("Int64")
    ref = ref[ref["station_code"].isin(STATIONS) & ref["year"].ge(2000)].copy()
    return split.merge(
        ref,
        on=["station_code", "site", "year"],
        how="left",
    ).assign(existing_ppo_reference_available=lambda d: d["source"].notna())


def summarize(split: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    merged = split.merge(
        coverage[["station_code", "year", "baseline_complete_4"]],
        on=["station_code", "year"],
        how="left",
    )
    rows = []
    for station, group in merged.groupby("station_code", sort=False):
        train = group[group["split"].eq("train")]
        val = group[group["split"].eq("validation")]
        rows.append(
            {
                "station_code": station,
                "site": STATION_TO_SITE[station],
                "available_years": len(group),
                "year_range": f"{int(group['year'].min())}-{int(group['year'].max())}",
                "train_years": ",".join(map(str, train["year"].astype(int).tolist())),
                "validation_years": ",".join(map(str, val["year"].astype(int).tolist())),
                "train_n": len(train),
                "validation_n": len(val),
                "baseline_complete_years": int(group["baseline_complete_4"].fillna(False).sum()),
                "baseline_incomplete_years": int((~group["baseline_complete_4"].fillna(False)).sum()),
                "ready_for_training_without_baseline_completion": bool(group["baseline_complete_4"].fillna(False).all()),
            }
        )
    return pd.DataFrame(rows)


def write_record(
    available: pd.DataFrame,
    split: pd.DataFrame,
    coverage: pd.DataFrame,
    site_summary: pd.DataFrame,
    ppo_reference: pd.DataFrame,
) -> None:
    gaps = coverage[~coverage["baseline_complete_4"]].copy()
    ppo_ref_summary = (
        ppo_reference.groupby(["station_code", "site", "source"], dropna=False)
        .agg(rows=("year", "count"))
        .reset_index()
        if not ppo_reference.empty
        else pd.DataFrame()
    )
    lines = [
        "# 032_21 五站点自由时序 PPO 前半训练/后半验证准备审计记录",
        "",
        "## 结论先说",
        "",
        "- 本轮没有训练、没有运行 DSSAT，只做数据准备审计。",
        "- 从 2000 年开始，只纳入 scenario pool 中存在且天气文件实际存在的年份。",
        "- 建议切分规则：每个站点按年份排序，前半训练、后半验证；奇数年份数时验证集多 1 年。",
        "- 下一步若直接训练新自由时序 PPO，可以先跑基线完整的站点；基线不完整年份不会阻止训练，但会影响训练后与四情景的完整比较。",
        "",
        "## 每站点建议切分",
        "",
        md_table(site_summary, max_rows=20),
        "",
        "## 逐年切分清单",
        "",
        md_table(split, max_rows=160),
        "",
        "## 四基线覆盖缺口",
        "",
        md_table(gaps[["station_code", "site", "year", "split", "baseline_present_count", "missing_scenarios"]], max_rows=160),
        "",
        "## 历史 PPO 参考结果库存",
        "",
        "这些结果只能作为历史参考，不能直接等同于下一轮“同一配置、前半训练、后半验证”的新主线结果。",
        "",
        md_table(ppo_ref_summary, max_rows=50),
        "",
        "## 下一步建议",
        "",
        "1. 若先复核 LC 以外站点是否也出现“多年输入不同但动作几乎固定”的现象，优先跑 HLA 或 FQ，因为气候/生育期与 LC 差异较大。",
        "2. 每个站点应独立训练一个自由时序 PPO 模型，不跨站点迁移；训练完成后迁移到同站点后半年份验证。",
        "3. 每个站点训练后必须输出：逐年指标表、节水节氮汇总图、抽样五情景日过程图、日值 QA。",
        "4. 若某站点基线缺口很多，可先训练，但正式对比图需要补齐对应年份四情景基线。",
        "",
        "## 解释边界",
        "",
        "- 本轮只回答“数据是否准备好”和“年份如何固定切分”，不评价 PPO 是否成功。",
        "- 旧阶段型 PPO、旧 DQN、031/032 历史自由 PPO 结果的动作空间、奖励、训练年份不完全一致，不能混成同一主线统计。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    available = load_available_years()
    split = split_years(available)
    baselines = load_baseline_rows()
    coverage = baseline_coverage(split, baselines)
    site_summary = summarize(split, coverage)
    ppo_reference = existing_ppo_reference(split)

    available.to_csv(TABLES / "032_21_available_years_weather_audit.csv", index=False, encoding="utf-8-sig")
    split.to_csv(TABLES / "032_21_half_split_years.csv", index=False, encoding="utf-8-sig")
    baselines.to_csv(TABLES / "032_21_baseline_source_rows_normalized.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(TABLES / "032_21_four_baseline_coverage_by_year.csv", index=False, encoding="utf-8-sig")
    site_summary.to_csv(TABLES / "032_21_site_split_readiness_summary.csv", index=False, encoding="utf-8-sig")
    ppo_reference.to_csv(TABLES / "032_21_existing_ppo_reference_inventory.csv", index=False, encoding="utf-8-sig")
    write_record(available, split, coverage, site_summary, ppo_reference)
    result = {
        "task": "032_21_five_site_half_split_free_timing_ppo_readiness_audit",
        "record_md": DOC.relative_to(ROOT).as_posix(),
        "site_summary_csv": (TABLES / "032_21_site_split_readiness_summary.csv").relative_to(ROOT).as_posix(),
        "baseline_gap_csv": (TABLES / "032_21_four_baseline_coverage_by_year.csv").relative_to(ROOT).as_posix(),
        "available_year_rows": int(len(available)),
        "split_year_rows": int(len(split)),
        "baseline_incomplete_years": int((~coverage["baseline_complete_4"].fillna(False)).sum()),
    }
    (OUT / "032_21_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
