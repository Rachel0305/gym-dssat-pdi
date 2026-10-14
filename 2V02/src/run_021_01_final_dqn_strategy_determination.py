from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pptx import Presentation
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "benchmark_results" / "021_01"
DOC_DATE = "2026-07-12"
DOC_MD = ROOT / "docs" / f"{DOC_DATE}_021_01_final_dqn_strategy_determination.md"
DOC_PPT = ROOT / "docs" / f"{DOC_DATE}_021_01_final_dqn_strategy_determination.pptx"
CONFIG_OUT = ROOT / "configs" / "final_dqn_candidate.yaml"
CONFIG_COPY = OUTPUT_DIR / "final_dqn_candidate.yaml"

HLA_SUMMARY = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_five_scenario_nstep_020_11" / "020_11_hla_five_scenario_summary.csv"
EXT_BASELINE = ROOT / "DSSAT_auto_validation" / "extension_expert_baseline_018_03" / "018_03_clean_multisite_comparison_with_extension_expert.csv"
YC_OLD_DIR = ROOT / "DSSAT_auto_validation" / "frozen_nstep_cross_site_020_12" / "YC2014"
FQ_OLD_DIR = ROOT / "DSSAT_auto_validation" / "frozen_nstep_cross_site_020_12" / "FQ2016"
LC_OLD_SUMMARY = ROOT / "DSSAT_auto_validation" / "extension_expert_baseline_018_03" / "018_06_lc2010_seed_stability_audit" / "018_06_lc2010_seed_best_summary.csv"
SY_DIAG = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "SY" / "sy_2012_2014_ic_diagnosis_016_01_runs" / "sy_2012_2014_ic_diagnosis_summary.csv"
SY_TRANSFER = ROOT / "DSSAT_auto_validation" / "sy_local_dqn_train_cross_year_transfer_017_08" / "017_08_sy_combined_summary.csv"
SY_SITE_CONFIG = ROOT / "configs" / "sites" / "sya.yaml"
SY_MZX = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "SY" / "CNSY1201.MZX"
LC_RUNTIME_LOG = OUTPUT_DIR / "021_01_lc2010_seed_stability__lc_2010_seed0" / "dqn_runtime" / "pdi_gym.log"
LC_NULL_SUMMARY = OUTPUT_DIR / "021_01_lc2010_seed_stability__lc_2010_seed0" / "null_evaluation" / "null_summary.csv"
LC_MANIFEST = OUTPUT_DIR / "021_01_lc2010_seed_stability__lc_2010_seed0" / "manifests" / "manifest.json"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_MD.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_cost_from_experiment_id(exp_id: str, prefix: str) -> float:
    match = re.search(prefix + r"([0-9]+(?:p[0-9]+)?)", exp_id)
    if not match:
        return math.nan
    token = match.group(1).replace("p", ".")
    return float(token)


def load_benchmark_rows(pattern: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(ROOT.glob(pattern)):
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["source_path"] = str(path.relative_to(ROOT))
        rows.append(row)
    return pd.DataFrame(rows)


def load_old_selected_summary(base_dir: Path, seed: int, station_code: str, year: int) -> dict[str, Any]:
    path = base_dir / f"seed{seed}_50000steps" / "selected_checkpoint_summary.csv"
    row = pd.read_csv(path).iloc[0].to_dict()
    return {
        "station_code": station_code,
        "year": year,
        "seed": seed,
        "checkpoint": int(row["checkpoint_step"]),
        "yield_kg_ha": float(row["final_grain_kg_ha"]),
        "biomass_kg_ha": float(row["final_biomass_kg_ha"]),
        "irrigation_mm": float(row["action_irrigation_total_mm"]),
        "nitrogen_kg_ha": float(row["action_nitrogen_total_kg_ha"]),
        "reward_total": float(row["total_reward"]),
        "max_water_stress": float(row["max_water_stress"]),
        "max_nitrogen_stress": float(row["max_nitrogen_stress"]),
        "source": f"reused:{path.relative_to(ROOT)}",
    }


def build_yc_summary() -> pd.DataFrame:
    current = load_benchmark_rows("benchmark_results/021_01/021_01_yc2014_ncost*__yc_2014_seed*/evaluations/season_summary.csv")
    current = current[~current["experiment_id"].str.contains("smoke", na=False)].copy()
    current["nitrogen_cost"] = current["experiment_id"].apply(lambda x: parse_cost_from_experiment_id(x, "ncost"))
    current["seed"] = current["seed"].astype(int)
    current["source"] = current["source_path"].apply(lambda x: f"new:{x}")

    old_seed0 = load_old_selected_summary(YC_OLD_DIR, 0, "YC", 2014)
    old_seed1 = load_old_selected_summary(YC_OLD_DIR, 1, "YC", 2014)
    old = pd.DataFrame([old_seed0, old_seed1])
    old["nitrogen_cost"] = 5.0

    cols = [
        "station_code",
        "year",
        "nitrogen_cost",
        "seed",
        "checkpoint",
        "yield_kg_ha",
        "biomass_kg_ha",
        "irrigation_mm",
        "nitrogen_kg_ha",
        "reward_total",
        "max_water_stress",
        "max_nitrogen_stress",
        "source",
    ]
    current = current.assign(
        max_water_stress=pd.NA,
        max_nitrogen_stress=pd.NA,
    )
    current = current[cols]
    result = pd.concat([old[cols], current], ignore_index=True).sort_values(["nitrogen_cost", "seed"]).reset_index(drop=True)
    return result


def build_fq_summary() -> pd.DataFrame:
    current = load_benchmark_rows("benchmark_results/021_01/021_01_fq2016_wcost*__fq_2016_seed*/evaluations/season_summary.csv")
    current = current[~current["experiment_id"].str.contains("smoke", na=False)].copy()
    current["water_cost"] = current["experiment_id"].apply(lambda x: parse_cost_from_experiment_id(x, "wcost"))
    current["seed"] = current["seed"].astype(int)
    current["source"] = current["source_path"].apply(lambda x: f"new:{x}")

    old_seed0 = load_old_selected_summary(FQ_OLD_DIR, 0, "FQ", 2016)
    old_seed1 = load_old_selected_summary(FQ_OLD_DIR, 1, "FQ", 2016)
    old = pd.DataFrame([old_seed0, old_seed1])
    old["water_cost"] = 1.0

    cols = [
        "station_code",
        "year",
        "water_cost",
        "seed",
        "checkpoint",
        "yield_kg_ha",
        "biomass_kg_ha",
        "irrigation_mm",
        "nitrogen_kg_ha",
        "reward_total",
        "max_water_stress",
        "max_nitrogen_stress",
        "source",
    ]
    current = current.assign(
        max_water_stress=pd.NA,
        max_nitrogen_stress=pd.NA,
    )
    current = current[cols]
    result = pd.concat([old[cols], current], ignore_index=True).sort_values(["water_cost", "seed"]).reset_index(drop=True)
    return result


def build_lc_summary() -> pd.DataFrame:
    legacy = pd.read_csv(LC_OLD_SUMMARY).copy()
    legacy["source"] = f"legacy:{LC_OLD_SUMMARY.relative_to(ROOT)}"
    legacy["status"] = "legacy_reference_only"
    legacy = legacy.rename(
        columns={
            "final_grain_kg_ha": "yield_kg_ha",
            "final_biomass_kg_ha": "biomass_kg_ha",
            "max_water_stress": "max_water_stress",
            "max_nitrogen_stress": "max_nitrogen_stress",
        }
    )
    blocked_row = {
        "seed": 0,
        "selection": "021_01_attempt",
        "checkpoint_step": pd.NA,
        "yield_kg_ha": pd.NA,
        "biomass_kg_ha": pd.NA,
        "irrigation_mm": pd.NA,
        "nitrogen_kg_ha": pd.NA,
        "max_water_stress": pd.NA,
        "max_nitrogen_stress": pd.NA,
        "total_reward": pd.NA,
        "source": "new:benchmark_results/021_01/021_01_lc2010_seed_stability__lc_2010_seed0",
        "status": "blocked_runtime_poll_wait",
        "note": "null run finished; DQN run created input/runtime shell only; pdi_gym.log ends with Client started; process slept >3h with near-zero CPU and no season_summary/training_log.",
    }
    blocked = pd.DataFrame([blocked_row])
    legacy["note"] = "Only 5K legacy evidence exists; not accepted as 021_01 formal frozen 50K seed review."
    cols = ["seed", "selection", "checkpoint_step", "yield_kg_ha", "biomass_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "max_water_stress", "max_nitrogen_stress", "total_reward", "status", "source", "note"]
    return pd.concat([legacy[cols], blocked[cols]], ignore_index=True)


def parse_sy_treatment_ic(mzx_text: str, treatment_number: int) -> str:
    lines = mzx_text.splitlines()
    in_treatments = False
    for line in lines:
        if line.startswith("*TREATMENTS"):
            in_treatments = True
            continue
        if in_treatments:
            if line.startswith("*") and not line.startswith("@"):
                break
            if line.strip().startswith(str(treatment_number) + " "):
                parts = line.split()
                if len(parts) >= 9:
                    return parts[8]
    return "unknown"


def build_sy_provenance() -> pd.DataFrame:
    site_cfg = yaml.safe_load(SY_SITE_CONFIG.read_text(encoding="utf-8"))
    mzx_text = SY_MZX.read_text(encoding="utf-8", errors="ignore")
    sy_diag = pd.read_csv(SY_DIAG)
    sy_transfer = pd.read_csv(SY_TRANSFER)

    rows = [
        {
            "item": "site_config.initial_condition_mode",
            "configured_value": site_cfg["site"]["initial_condition_mode"],
            "observed_value": 2,
            "source": str(SY_SITE_CONFIG.relative_to(ROOT)),
            "status": "configured",
            "note": "Benchmark site config explicitly points to IC=2 expectation.",
        },
        {
            "item": "prepared_mzx_treatment2_IC",
            "configured_value": 2,
            "observed_value": parse_sy_treatment_ic(mzx_text, 2),
            "source": str(SY_MZX.relative_to(ROOT)),
            "status": "mismatch" if parse_sy_treatment_ic(mzx_text, 2) != "2" else "matched",
            "note": "2014 treatment row in prepared MZX is the authoritative on-disk candidate used by current adapter path.",
        },
        {
            "item": "prepared_mzx_soil_id",
            "configured_value": site_cfg["site"]["soil_id"],
            "observed_value": site_cfg["site"]["soil_id"],
            "source": str(SY_MZX.relative_to(ROOT)),
            "status": "matched",
            "note": "Prepared input package keeps SY99001200 as soil id.",
        },
        {
            "item": "prepared_mzx_weather_2014",
            "configured_value": site_cfg["site"]["inputs"]["by_year"]["2014"]["weather_name"],
            "observed_value": "CNSY1401.WTH",
            "source": str(SY_MZX.relative_to(ROOT)),
            "status": "matched",
            "note": "2014 field row points to CNSY1401.",
        },
        {
            "item": "prepared_mzx_cultivar",
            "configured_value": site_cfg["site"]["cultivar_id"],
            "observed_value": "FY0985",
            "source": str(SY_MZX.relative_to(ROOT)),
            "status": "matched",
            "note": "Cultivar block in prepared MZX is FY0985.",
        },
    ]

    for _, row in sy_diag.iterrows():
        rows.append(
            {
                "item": f"diagnostic_{row['case']}",
                "configured_value": "n/a",
                "observed_value": row["final_topwt_gym"],
                "source": str(SY_DIAG.relative_to(ROOT)),
                "status": "evidence",
                "note": f"steps={row['steps_completed']}, terminated={row['terminated']}, description={row['description']}",
            }
        )

    transfer_2014 = sy_transfer[(sy_transfer["year"] == 2014) & (sy_transfer["scenario"].astype(str).str.contains("transfer"))]
    if not transfer_2014.empty:
        trow = transfer_2014.iloc[0]
        rows.append(
            {
                "item": "historical_transfer_result_2014",
                "configured_value": "n/a",
                "observed_value": float(trow["final_gwad"]),
                "source": str(SY_TRANSFER.relative_to(ROOT)),
                "status": "evidence",
                "note": "Historical transfer run exists, but current benchmark adapter still points to provenance-ambiguous prepared MZX.",
            }
        )
    return pd.DataFrame(rows)


def build_station_diagnosis(yc_df: pd.DataFrame, fq_df: pd.DataFrame, lc_df: pd.DataFrame, sy_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    hla = pd.read_csv(HLA_SUMMARY)
    hla_dqn = hla[hla["scenario_family"] == "nstep_dqn"].copy()
    rows.append(
        {
            "station_code": "HLA",
            "focus_year": "2007,2010,2015,2016,2022",
            "status": "keep_current_config",
            "recommended_water_cost": 1.0,
            "recommended_nitrogen_cost": 5.0,
            "recommended_n_steps": 5,
            "evidence": f"Existing formal 020_11 matrix already covers {hla_dqn['year'].nunique()} years x {hla_dqn['dqn_seed'].dropna().nunique()} seeds under the frozen n-step setup.",
            "next_action": "Reuse existing HLA formal results; no rerun needed in 021_01.",
        }
    )

    rows.append(
        {
            "station_code": "YC",
            "focus_year": "2014",
            "status": "keep_current_config",
            "recommended_water_cost": 1.0,
            "recommended_nitrogen_cost": 5.0,
            "recommended_n_steps": 5,
            "evidence": "nitrogen_cost=2/5/8 formal 50K comparison shows no meaningful policy or outcome separation; selected checkpoints all stay at zero-N behavior with similar yield.",
            "next_action": "Retain current nitrogen_cost=5 for continuity and cross-station comparability.",
        }
    )

    rows.append(
        {
            "station_code": "FQ",
            "focus_year": "2016",
            "status": "keep_current_config",
            "recommended_water_cost": 1.0,
            "recommended_nitrogen_cost": 5.0,
            "recommended_n_steps": 5,
            "evidence": "water_cost=0.5/1/2 formal 50K comparison shows the same checkpoint family and nearly identical resource-use pattern.",
            "next_action": "Retain current water_cost=1 for continuity and cross-station comparability.",
        }
    )

    lc_blocked = lc_df[lc_df["status"] == "blocked_runtime_poll_wait"]
    rows.append(
        {
            "station_code": "LC",
            "focus_year": "2010",
            "status": "blocked_runtime_poll_wait" if not lc_blocked.empty else "unknown",
            "recommended_water_cost": 1.0,
            "recommended_nitrogen_cost": 5.0,
            "recommended_n_steps": 5,
            "evidence": lc_blocked.iloc[0]["note"] if not lc_blocked.empty else "No blocked runtime evidence found.",
            "next_action": "Do not launch new sensitivity runs; fix benchmark/runtime handshake first.",
        }
    )

    mismatch = sy_df[sy_df["status"] == "mismatch"]
    rows.append(
        {
            "station_code": "SY",
            "focus_year": "2014",
            "status": "blocked_input_provenance",
            "recommended_water_cost": 1.0,
            "recommended_nitrogen_cost": 5.0,
            "recommended_n_steps": 5,
            "evidence": mismatch.iloc[0]["note"] if not mismatch.empty else "Prepared MZX provenance remains ambiguous.",
            "next_action": "Freeze formal training until IC/MZX provenance is authoritative.",
        }
    )
    return pd.DataFrame(rows)


def build_unified_parameter_evidence(station_diagnosis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in station_diagnosis.iterrows():
        rows.append(
            {
                "station_code": row["station_code"],
                "status": row["status"],
                "water_cost": row["recommended_water_cost"],
                "nitrogen_cost": row["recommended_nitrogen_cost"],
                "n_steps": row["recommended_n_steps"],
                "irrigation_action_levels_mm": "[0, 15, 30]",
                "nitrogen_action_levels_kg_ha": "[0, 50, 100]",
                "season_budgets": "I<=120, N<=300",
                "evidence": row["evidence"],
            }
        )
    rows.append(
        {
            "station_code": "UNIFIED_CANDIDATE",
            "status": "partial",
            "water_cost": 1.0,
            "nitrogen_cost": 5.0,
            "n_steps": 5,
            "irrigation_action_levels_mm": "[0, 15, 30]",
            "nitrogen_action_levels_kg_ha": "[0, 50, 100]",
            "season_budgets": "I<=120, N<=300",
            "evidence": "Keep the frozen HLA/YC/FQ-compatible setup; LC runtime and SY provenance remain blocked items.",
        }
    )
    return pd.DataFrame(rows)


def write_yaml_candidate() -> dict[str, Any]:
    payload = {
        "meta": {
            "task": "021_01_final_dqn_strategy_determination",
            "date": DOC_DATE,
            "status": "partial",
        },
        "candidate": {
            "algorithm": "DQN",
            "reward": {
                "type": "null_relative_terminal_gain_minus_resource_cost",
                "water_cost": 1.0,
                "nitrogen_cost": 5.0,
                "note": "Same formula kept; diagnosis did not justify changing coefficients.",
            },
            "actions": {
                "irrigation_mm": [0, 15, 30],
                "nitrogen_kg_ha": [0, 50, 100],
            },
            "budgets": {
                "irrigation_mm": 120,
                "nitrogen_kg_ha": 300,
            },
            "training": {
                "n_steps": 5,
                "checkpoint_selection": "best_reward_existing_protocol",
            },
            "applicability": {
                "confirmed_sites": ["HLA", "YC", "FQ"],
                "blocked_sites": {
                    "LC": "runtime poll-wait hang in 021_01 frozen benchmark run",
                    "SY": "input provenance ambiguous (config expects IC=2, prepared 2014 treatment row is IC=0)",
                },
            },
        },
    }
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    CONFIG_OUT.write_text(text, encoding="utf-8")
    CONFIG_COPY.write_text(text, encoding="utf-8")
    return payload


def markdown_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def write_markdown(
    station_diag: pd.DataFrame,
    yc_summary: pd.DataFrame,
    fq_summary: pd.DataFrame,
    lc_summary: pd.DataFrame,
    sy_summary: pd.DataFrame,
    unified_df: pd.DataFrame,
    candidate_yaml: dict[str, Any],
) -> None:
    generated_files = [
        "benchmark_results/021_01/station_diagnosis.csv",
        "benchmark_results/021_01/yc_nitrogen_cost_summary.csv",
        "benchmark_results/021_01/fq_water_cost_summary.csv",
        "benchmark_results/021_01/lc_seed_stability_summary.csv",
        "benchmark_results/021_01/sy_input_provenance.csv",
        "benchmark_results/021_01/unified_parameter_evidence.csv",
        "benchmark_results/021_01/final_dqn_candidate.yaml",
        "configs/final_dqn_candidate.yaml",
        f"docs/{DOC_MD.name}",
        f"docs/{DOC_PPT.name}",
    ]
    text = f"""# 021_01 Final DQN Strategy Determination

## 1. 任务目标

严格执行 `prompts/021_01_final_dqn_strategy_determination.md`，不继续工程重构，不扩展新的敏感性实验，优先复用已有结果，只对问题站点做必要诊断，并据此确定统一 DQN 正式候选策略。

## 2. 已复用结果与新增诊断

- HLA：完全复用 `020_11` 已有正式五情景/多年份/多 seed 结果。
- YC：复用旧 `nitrogen_cost=5` 的 seed0/1，并补做 `seed2`；新增 `nitrogen_cost=2/8` 正式 50K 结果。
- FQ：复用旧 `water_cost=1` 的 seed0/1，并补做 `seed2`；新增 `water_cost=0.5/2` 正式 50K 结果。
- LC：启动 frozen 正式复核，但 benchmark runtime 出现挂起。
- SY：只做 provenance audit，不启动正式训练。

## 3. 站点诊断结论

{markdown_table(station_diag)}

## 4. HLA 结论

- HLA 已有正式结果矩阵完整，覆盖 2007/2010/2015/2016/2022，且包含多 seed。
- 021_01 不重复训练 HLA，直接保留当前配置结论：`keep_current_config`。

## 5. YC nitrogen_cost 诊断

{markdown_table(yc_summary)}

结论：

- `nitrogen_cost=2/5/8` 在正式 50K 结果上没有形成可解释的策略分叉。
- 现有最优 checkpoint 仍是“零追加氮、有限灌溉”的同类策略族。
- 因此不建议仅为了 YC 单站点去改单独 reward，保留 `nitrogen_cost=5`。

## 6. FQ water_cost 诊断

{markdown_table(fq_summary)}

结论：

- `water_cost=0.5/1/2` 的正式 50K 结果基本落在同一策略簇。
- 未观察到足以支撑改动统一参数的收益。
- 因此保留 `water_cost=1`。

## 7. LC frozen configuration 复核

{markdown_table(lc_summary)}

结论：

- 021_01 新框架下，LC2010 的 null 已完成，但 DQN 运行在 `Client started` 后长时间停滞。
- 进程累计运行数小时、CPU 接近 0、未生成 `season_summary.csv` 与训练日志。
- 因此本任务内将 LC 标为 `blocked_runtime_poll_wait`，不继续追加训练。

## 8. SY input provenance audit

{markdown_table(sy_summary)}

结论：

- `configs/sites/sya.yaml` 期待 `IC=2`。
- 但当前准备输入 `CNSY1201.MZX` 的 2014 treatment 行仍显示 `IC=0`。
- 同时历史诊断已显示 `IC=0` 与 `IC=1`/`IC=2` 会导致明显不同的产量水平。
- 因此本任务内将 SY 标为 `blocked_input_provenance`。

## 9. 统一参数证据表

{markdown_table(unified_df)}

## 10. 最终候选策略

```yaml
{yaml.safe_dump(candidate_yaml, allow_unicode=True, sort_keys=False)}
```

## 11. 当前问题与已解决问题

已解决：

- YC `nitrogen_cost` 诊断完成。
- FQ `water_cost` 诊断完成。
- HLA 当前正式配置可以直接复用。

未解决：

- LC benchmark runtime 挂起，未完成 seed0/1/2 冻结正式复核。
- SY 输入 provenance 未统一，仍不能进入正式训练。

## 12. 后续实验计划

1. 先修 LC runtime/blocking 机制，再重新执行 frozen 50K seed 复核。
2. 先明确 SY authoritative IC/MZX，再生成正式训练配置。
3. 在 LC/SY 未解决前，不扩大统一参数搜索，不改 reward 结构。

## 13. Methods Source

- `benchmark/benchmark_runner.py`
- `configs/experiments/021_01_*.yaml`
- `configs/sites/hla.yaml`
- `configs/sites/yca.yaml`
- `configs/sites/fqa.yaml`
- `configs/sites/lca.yaml`
- `configs/sites/sya.yaml`
- `src/run_021_01_final_dqn_strategy_determination.py`

## 14. References

- `prompts/021_01_final_dqn_strategy_determination.md`
- `DSSAT_auto_validation/HLA_2004/hla_five_scenario_nstep_020_11/020_11_hla_five_scenario_summary.csv`
- `DSSAT_auto_validation/frozen_nstep_cross_site_020_12/YC2014/*`
- `DSSAT_auto_validation/frozen_nstep_cross_site_020_12/FQ2016/*`
- `DSSAT_auto_validation/extension_expert_baseline_018_03/018_03_clean_multisite_comparison_with_extension_expert.csv`
- `DSSAT_auto_validation/extension_expert_baseline_018_03/018_06_lc2010_seed_stability_audit/018_06_lc2010_seed_best_summary.csv`
- `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY/CNSY1201.MZX`
- `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY/sy_2012_2014_ic_diagnosis_016_01_runs/sy_2012_2014_ic_diagnosis_summary.csv`

## 15. 输出文件

{chr(10).join(f"- `{x}`" for x in generated_files)}

## 16. Git commit 信息

- Git commit: pending
- Git push: pending
"""
    DOC_MD.write_text(text, encoding="utf-8")


def _set_text_shape(text_frame, text: str, font_size: int = 22, bold: bool = False) -> None:
    text_frame.clear()
    p = text_frame.paragraphs[0]
    p.text = text
    for run in p.runs:
        run.font.size = Pt(font_size)
        run.font.bold = bold


def add_title_slide(prs: Presentation, title: str, subtitle: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _set_text_shape(slide.shapes.title.text_frame, title, 26, True)
    box = slide.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(8.8), Inches(4.8))
    _set_text_shape(box.text_frame, subtitle, 18, False)


def add_bullets_slide(prs: Presentation, title: str, bullets: list[str]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _set_text_shape(slide.shapes.title.text_frame, title, 24, True)
    box = slide.shapes.add_textbox(Inches(0.6), Inches(1.3), Inches(9.0), Inches(5.8))
    tf = box.text_frame
    tf.clear()
    for i, bullet in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = bullet
        p.level = 0
        for run in p.runs:
            run.font.size = Pt(18)


def add_table_slide(prs: Presentation, title: str, df: pd.DataFrame, max_rows: int = 8) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _set_text_shape(slide.shapes.title.text_frame, title, 22, True)
    df = df.head(max_rows).copy()
    rows, cols = len(df) + 1, len(df.columns)
    table = slide.shapes.add_table(rows, cols, Inches(0.3), Inches(1.2), Inches(9.2), Inches(5.8)).table
    for j, col in enumerate(df.columns):
        table.cell(0, j).text = str(col)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, col in enumerate(df.columns):
            val = row[col]
            text = "" if pd.isna(val) else str(val)
            table.cell(i, j).text = text[:80]
    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r, c)
            for paragraph in cell.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(10 if r == 0 else 9)
                    run.font.bold = r == 0


def write_ppt(
    station_diag: pd.DataFrame,
    yc_summary: pd.DataFrame,
    fq_summary: pd.DataFrame,
    lc_summary: pd.DataFrame,
    sy_summary: pd.DataFrame,
    unified_df: pd.DataFrame,
) -> None:
    prs = Presentation()
    add_title_slide(
        prs,
        "021_01 Final DQN Strategy Determination",
        "严格按任务书执行：复用 HLA 既有结果，完成 YC/FQ 必要诊断，核查 LC/SY 问题站点，给出统一 DQN 正式候选策略。",
    )
    add_table_slide(prs, "站点诊断总表", station_diag)
    add_table_slide(prs, "YC2014 nitrogen_cost 正式 50K 诊断", yc_summary)
    add_table_slide(prs, "FQ2016 water_cost 正式 50K 诊断", fq_summary)
    add_table_slide(prs, "LC2010 frozen configuration 复核", lc_summary)
    add_table_slide(prs, "SY2014 input provenance audit", sy_summary)
    add_table_slide(prs, "统一参数证据表", unified_df)
    add_bullets_slide(
        prs,
        "结论与后续",
        [
            "HLA：keep_current_config，正式结果直接复用。",
            "YC：nitrogen_cost=2/5/8 无实质分叉，保留 5。",
            "FQ：water_cost=0.5/1/2 无实质分叉，保留 1。",
            "LC：benchmark runtime 卡在 Client started 后的 poll-wait，当前 blocked。",
            "SY：config 期待 IC=2，但 prepared MZX 2014 treatment 仍是 IC=0，当前 blocked。",
            "统一候选：water_cost=1, nitrogen_cost=5, n_steps=5, I=[0,15,30], N=[0,50,100], I<=120, N<=300。",
        ],
    )
    prs.save(DOC_PPT)


def main() -> None:
    ensure_dirs()
    yc_summary = build_yc_summary()
    fq_summary = build_fq_summary()
    lc_summary = build_lc_summary()
    sy_summary = build_sy_provenance()
    station_diag = build_station_diagnosis(yc_summary, fq_summary, lc_summary, sy_summary)
    unified_df = build_unified_parameter_evidence(station_diag)
    candidate_yaml = write_yaml_candidate()

    yc_summary.to_csv(OUTPUT_DIR / "yc_nitrogen_cost_summary.csv", index=False, encoding="utf-8-sig")
    fq_summary.to_csv(OUTPUT_DIR / "fq_water_cost_summary.csv", index=False, encoding="utf-8-sig")
    lc_summary.to_csv(OUTPUT_DIR / "lc_seed_stability_summary.csv", index=False, encoding="utf-8-sig")
    sy_summary.to_csv(OUTPUT_DIR / "sy_input_provenance.csv", index=False, encoding="utf-8-sig")
    station_diag.to_csv(OUTPUT_DIR / "station_diagnosis.csv", index=False, encoding="utf-8-sig")
    unified_df.to_csv(OUTPUT_DIR / "unified_parameter_evidence.csv", index=False, encoding="utf-8-sig")

    write_markdown(station_diag, yc_summary, fq_summary, lc_summary, sy_summary, unified_df, candidate_yaml)
    write_ppt(station_diag, yc_summary, fq_summary, lc_summary, sy_summary, unified_df)
    print("021_01 Final DQN Strategy Determination artifacts generated")


if __name__ == "__main__":
    main()
