#!/usr/bin/env python3
"""Register reusable four-baseline snapshots without running DSSAT."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "028_06_screened_year_four_baseline_reuse_registry"
DOC = ROOT / "docs" / "2026-07-18_028_06_screened_year_four_baseline_reuse_registry.md"
REQUIRED = (
    "Summary.OUT",
    "PlantGro.OUT",
    "Weather.OUT",
    "SoilWat.OUT",
    "SoilNi.OUT",
    "MgmtEvent.OUT",
    "fileX.MZX",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hla_snapshot(year: int, scenario: str) -> Path:
    names = {
        "null": "null",
        "recorded": "recorded_farmer",
        "dssat_auto": "dssat_auto",
        "official_extension_expert": "extension_expert",
    }
    return ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_five_scenario_nstep_020_11" / "runs" / str(year) / names[scenario] / "pdi_tmp_snapshot_eval"


def yc_snapshot(year: int, scenario: str) -> Path | None:
    names = {"null": "null", "recorded": "recorded", "dssat_auto": "dssat_auto"}
    if scenario not in names:
        return None
    return ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_forward_screening_013_01" / "runs" / "YC" / f"{year}_{names[scenario]}" / "pdi_tmp_snapshot"


def fq_snapshot(year: int, scenario: str) -> Path | None:
    names = {"null": "null", "recorded": "recorded_shifted", "dssat_auto": "dssat_auto"}
    if scenario not in names:
        return None
    return ROOT / "DSSAT_auto_validation" / "fq_all_year_screen_and_dqn_transfer_014_01" / "runs" / str(year) / "seed0" / names[scenario] / "pdi_tmp_snapshot_eval"


def build_rows() -> list[dict[str, object]]:
    cases = [
        *(('HLA', year, '020_11_prepared_adapter') for year in (2007, 2015, 2016, 2022)),
        ('YC', 2008, '013_01_forward_screening'),
        *(('FQ', year, '014_01_weather_year_derived') for year in (2013, 2014, 2019, 2020, 2023)),
    ]
    scenarios = ("null", "recorded", "dssat_auto", "official_extension_expert")
    rows: list[dict[str, object]] = []
    for site, year, input_variant in cases:
        for scenario in scenarios:
            if site == "HLA":
                snapshot = hla_snapshot(year, scenario)
            elif site == "YC":
                snapshot = yc_snapshot(year, scenario)
            else:
                snapshot = fq_snapshot(year, scenario)
            missing = list(REQUIRED) if snapshot is None else [name for name in REQUIRED if not (snapshot / name).is_file()]
            complete = snapshot is not None and not missing
            rows.append(
                {
                    "site": site,
                    "year": year,
                    "scenario": scenario,
                    "status": "reusable_existing_snapshot" if complete else "missing_needs_run",
                    "input_variant": input_variant,
                    "scenario_variant": "recorded_shifted" if site == "FQ" and scenario == "recorded" else scenario,
                    "snapshot_path": "" if snapshot is None else str(snapshot.relative_to(ROOT)).replace("\\", "/"),
                    "missing_required_files": ";".join(missing),
                    "summary_sha256": sha256(snapshot / "Summary.OUT") if complete else "",
                    "filex_sha256": sha256(snapshot / "fileX.MZX") if complete else "",
                    "dssat_runs_this_task": 0,
                }
            )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    csv_path = OUT / "028_06_baseline_scenario_registry.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    reusable = [row for row in rows if row["status"] == "reusable_existing_snapshot"]
    missing = [row for row in rows if row["status"] == "missing_needs_run"]
    summary = {
        "status": "completed_audit_only",
        "scenario_units": len(rows),
        "reusable_existing_snapshots": len(reusable),
        "missing_scenarios": len(missing),
        "missing_case_keys": [f"{r['site']}{r['year']}:{r['scenario']}" for r in missing],
        "dssat_runs": 0,
        "training_calls": 0,
    }
    (OUT / "028_06_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 028_06 已筛选年份四基线复用与缺口登记",
        "",
        "## 结论",
        "",
        f"已逐项检查 {len(rows)} 个站点—年份—情景单元；{len(reusable)} 个已有快照通过必需文件检查，可直接复用；真正缺失 {len(missing)} 个。",
        "本任务运行 DSSAT 0 季、训练 0 次。",
        "",
        "## 真正缺失情景",
        "",
        "|站点|年份|情景|输入变体|",
        "|---|---:|---|---|",
    ]
    for row in missing:
        lines.append(f"|{row['site']}|{row['year']}|{row['scenario']}|{row['input_variant']}|")
    lines += [
        "",
        "## 来源边界",
        "",
        "- HLA 四个年份来自 020_11 prepared-adapter 链路；这不是未经转换的原始 treatment。",
        "- YC2008 来自 013_01 forward-screening 链路。",
        "- FQ 五个年份来自 014_01 weather-year 派生链路；其中 recorded 为 recorded_shifted。",
        "- official expert 不得由 recorded 或其他年份的结果替代。",
        "",
        "## 下一步硬边界",
        "",
        f"下一任务最多运行 {len(missing)} 季 official expert；其余 {len(reusable)} 个情景不得重复运行。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("028_06 baseline reuse registry completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
