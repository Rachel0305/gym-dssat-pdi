"""Read-only Phase A audit for the HLA NESDC 2000-2008 package.

This script stops at the rain-provenance gate. It does not generate fitting
weather, CLI files, WGEN realizations, DSSAT runs, or PPO training.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "external" / "nesdc_hla_2000_2008" / "raw"
ARCHIVE = RAW / "实体数据及关联说明文档_DP2011_HLA_QXSJ_DQJC.zip"
EXTRACT = OUT / "package_extract"
AUDIT_009 = (
    ROOT
    / "results"
    / "hl_fq_random_weather_8seed"
    / "source_audit"
    / "cern_d222_reaudit"
    / "hla_2004_rain_audit.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def decoded_zip_name(name: str) -> str:
    try:
        return name.encode("cp437").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def json_dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    raw_inventory = []
    for path in sorted(RAW.glob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        raw_inventory.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "size_bytes": stat.st_size,
                "mtime_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
                "sha256": sha256(path),
                "read_only_evidence": True,
            }
        )
    json_dump(OUT / "raw_inventory.json", raw_inventory)

    archive_rows = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for info in archive.infolist():
            archive_rows.append(
                {
                    "archive_path": decoded_zip_name(info.filename),
                    "is_directory": info.is_dir(),
                    "uncompressed_size": info.file_size,
                    "compressed_size": info.compress_size,
                    "crc32": f"{info.CRC:08X}",
                }
            )
    pd.DataFrame(archive_rows).to_csv(OUT / "archive_manifest.csv", index=False, encoding="utf-8-sig")

    workbook_inventory = []
    for path in sorted(EXTRACT.rglob("*.xls")):
        book = pd.ExcelFile(path)
        sheets = []
        for sheet in book.sheet_names:
            frame = pd.read_excel(path, sheet_name=sheet, header=None)
            sheets.append(
                {
                    "sheet": sheet,
                    "rows_including_title_and_header": int(frame.shape[0]),
                    "columns": int(frame.shape[1]),
                    "title": None if frame.empty else str(frame.iloc[0, 0]),
                    "raw_header": [None if pd.isna(value) else str(value) for value in frame.iloc[1].tolist()],
                }
            )
        workbook_inventory.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "sheets": sheets,
            }
        )
    json_dump(OUT / "workbook_inventory.json", workbook_inventory)

    precip_path = next(EXTRACT.rglob("海伦站自动观测降水、湿度.xls"))
    precip = pd.read_excel(precip_path, sheet_name="降水", header=1)
    precip.columns = [str(value).strip() for value in precip.columns]
    precip = precip.rename(columns={"年": "year", "月": "month", "合计": "raw_合计", "最高": "raw_最高"})
    for column in ["year", "month", "raw_合计", "raw_最高"]:
        precip[column] = pd.to_numeric(precip[column], errors="coerce")
    precip["source_file"] = precip_path.relative_to(ROOT).as_posix()
    precip["source_sheet"] = "降水"
    precip["time_granularity"] = "MONTH"
    precip["qc_note"] = "raw field names preserved; daily allocation unavailable"
    precip.to_csv(OUT / "nesdc_precip_monthly.csv", index=False, encoding="utf-8-sig")

    expected_months = pd.period_range("2001-01", "2008-12", freq="M")
    actual_months = pd.PeriodIndex(
        [f"{int(row.year):04d}-{int(row.month):02d}" for row in precip.itertuples()], freq="M"
    )
    missing_months = [str(value) for value in expected_months.difference(actual_months)]
    duplicate_month_rows = int(actual_months.duplicated().sum())

    audit_2004 = pd.read_csv(AUDIT_009, parse_dates=["date"])
    audit_2004["month"] = audit_2004["date"].dt.month
    external_monthly = (
        audit_2004.groupby("month")[["d222_20_20_mm", "legacy_rain_raw", "nasa_rain_mm", "ghcn_rain_mm"]]
        .sum(min_count=1)
        .reset_index()
    )
    nesdc_2004 = precip.loc[precip["year"] == 2004, ["month", "raw_合计", "raw_最高"]].copy()
    crosscheck = pd.DataFrame({"month": range(1, 13)}).merge(nesdc_2004, on="month", how="left")
    crosscheck = crosscheck.merge(external_monthly, on="month", how="left")
    crosscheck["nesdc_month_row_present"] = crosscheck["raw_合计"].notna() | crosscheck["raw_最高"].notna()
    crosscheck["daily_overlap_shift_audit"] = "NOT_RUN_MONTHLY_SOURCE_HAS_NO_DAY_FIELD"
    crosscheck.to_csv(OUT / "hla_2004_monthly_crosscheck.csv", index=False, encoding="utf-8-sig")

    raw_total_lt_raw_max = int((precip["raw_合计"] < precip["raw_最高"]).sum())
    sep_row = crosscheck.loc[crosscheck["month"] == 9].iloc[0]
    phase_a = {
        "task": "010 HLA Phase A rain provenance recovery",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "site": "HLA",
        "product": {
            "name": "2000-2008年黑龙江海伦站农田气象监测数据",
            "doi": "10.12199/nesdc.ecodb.mon.2020.dp2011.hla.005",
            "station_name": "黑龙江海伦站",
            "station_code_in_precip_workbook": None,
            "observation_site_codes_in_associated_document": ["HLAQX01", "HLAQX02"],
            "documented_product_range": "2000-01-01/2008-12-31",
            "documented_product_time_resolution": "MONTH",
            "precip_workbook_actual_range": "2001-01/2008-12",
            "precipitation_raw_fields": ["年", "月", "合计", "最高"],
            "precipitation_unit": "mm",
            "date_definition": "year and month only; no day field",
            "missing_value_code": "NOT_DOCUMENTED",
            "zero_vs_blank_semantics": "explicit numeric 0 exists; omitted month and blank semantics are not documented",
            "quality_control_code": "NOT_PRESENT",
        },
        "raw_inventory": raw_inventory,
        "precipitation_table": {
            "row_count": int(len(precip)),
            "monthly_rows_expected_2001_2008": int(len(expected_months)),
            "missing_months": missing_months,
            "duplicate_month_rows": duplicate_month_rows,
            "year_2004_rows": int((precip["year"] == 2004).sum()),
            "year_2004_months_present": [int(value) for value in precip.loc[precip["year"] == 2004, "month"]],
            "year_2004_daily_rows": 0,
            "year_2004_expected_daily_rows": 366,
            "jan_aug_2004_daily_precipitation_evidence": False,
            "raw_最高_greater_than_raw_合计_rows": raw_total_lt_raw_max,
            "field_semantics_warning": "The literal raw field names are physically ambiguous because raw_最高 exceeds raw_合计 on many rows.",
        },
        "overlap_audit": {
            "same_day_shift_minus1_0_plus1": "NOT_RUN_GRANULARITY_MISMATCH",
            "reason": "NESDC precipitation is monthly and has no day field.",
            "monthly_context_2004_september": {
                "nesdc_raw_合计": None if pd.isna(sep_row["raw_合计"]) else float(sep_row["raw_合计"]),
                "nesdc_raw_最高": None if pd.isna(sep_row["raw_最高"]) else float(sep_row["raw_最高"]),
                "d222_month_total_mm": float(sep_row["d222_20_20_mm"]),
                "legacy_month_total_mm": float(sep_row["legacy_rain_raw"]),
                "interpretation": "Unresolved conflict; the NESDC monthly row is zero while D222/legacy total 44.5 mm.",
            },
        },
        "HLA_RAIN_PROVENANCE": "FAIL",
        "READY_FOR_HLA_WGEN": False,
        "blockers": [
            "BLOCKED_BY_NESDC_MONTHLY_NOT_DAILY",
            "BLOCKED_BY_2004_DAILY_PRECIPITATION_PROVENANCE",
            "BLOCKED_BY_NESDC_D222_2004_09_CONFLICT",
            "BLOCKED_BY_NESDC_FIELD_SEMANTICS",
        ],
        "minimum_recovery_evidence": "Authoritative HLA daily precipitation for 2004-01-01 through 2004-08-31, with day-level dates, units, missing codes/QC, and enough overlap to resolve date boundary and units against D222.",
        "downstream": {
            "fitting_weather_qc": "NOT_RUN",
            "cli_generated": False,
            "climate_coverage": "NOT_RUN",
            "runtime_smoke": "NOT_RUN",
            "ppo_seeds_completed": 0,
            "historical_vs_random_comparison": "NOT_RUN",
            "fqa_touched": False,
        },
    }
    json_dump(OUT / "phase_a_gate.json", phase_a)

    findings = f"""# 010 HLA Phase A 降水 provenance 审计

`HLA_RAIN_PROVENANCE = FAIL`  
`READY_FOR_HLA_WGEN = FALSE`

## 决定性证据

- 产品说明明确写明时间分辨率为“月”；降水 Excel 只有 `年/月/合计/最高`，无日字段。
- 降水表实际覆盖 2001-01 至 2008-12，共 {len(precip)} 行；2004 年只有 1–9 月 9 行，缺 10–12 月。
- 2004 年不存在 366 条逐日记录，也不能为 2004-01-01 至 2004-08-31 的 244 天建立逐日 `date/rain_mm/source`。
- same-day 与 -1/0/+1 日 shift 审计因粒度不匹配无法执行。
- 2004-09 的 NESDC 两个原始降水字段均为 0，而 D222/项目旧表月累计均为 44.5 mm，冲突尚未解释。
- 原始 `最高` 大于 `合计` 的记录有 {raw_total_lt_raw_max} 行；按字面解释不满足“月最大值不超过月合计”，字段语义仍需数据字典确认。

## 停止决定

该月尺度产品能证明 HLA 2004 年前八个月并非可靠的“全零降水期”，但不能替代日尺度 WGEN fitting weather。按 010 的逐 gate 规则，本任务停止在 Phase A；未构建 fitting weather、CLI 或 WGEN pool，未运行 DSSAT/PPO，未读取或处理 FQA。
"""
    (OUT / "findings.md").write_text(findings, encoding="utf-8")
    print(json.dumps({"HLA_RAIN_PROVENANCE": "FAIL", "READY_FOR_HLA_WGEN": False}, indent=2))


if __name__ == "__main__":
    main()
