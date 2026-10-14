from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/yc_random_weather_ppo/004_23B_yc_external_weather_injection_smoke"
SITE_INPUT = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC"
SOURCE_WTH = SITE_INPUT / "CNYC0401.WTH"
SOURCE_FILEX = (
    ROOT
    / "results/yc_random_weather_ppo/004_03/training_verified/historical/ppo_seed_0/"
    / "rendered_inputs/YCA/2004/HISTORICAL_WEATHER_ppo_seed_0_2004/"
    / "YCA_2004_HISTORICAL_WEATHER_ppo_seed_0_2004.jinja2"
)
SOURCE_RENDERED_WTH = SOURCE_FILEX.parent / "CNYC0401.WTH"
WEATHER_FIELDS = ("SRAD", "TMAX", "TMIN", "RAIN")
TOLERANCE = 1e-5
RSS_STOP_MB = 1800.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_date_token(token: str, context_year: int) -> date | None:
    token = token.strip()
    try:
        if re.fullmatch(r"20\d{5}", token):
            return datetime.strptime(token, "%Y%j").date()
        if re.fullmatch(r"\d{5}", token):
            yy, doy = int(token[:2]), int(token[2:])
            year = context_year // 100 * 100 + yy
            if year - context_year > 50:
                year -= 100
            elif context_year - year > 50:
                year += 100
            return date(year, 1, 1) + timedelta(days=doy - 1)
    except ValueError:
        return None
    return None


def parse_wth(path: Path, context_year: int) -> list[dict[str, Any]]:
    header: list[str] | None = None
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="latin-1", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("@") and "DATE" in stripped.upper():
            header = [token.upper().lstrip("@") for token in stripped.split() if token != "@"]
            continue
        if not stripped or stripped.startswith(("*", "@")) or header is None:
            continue
        tokens = stripped.split()
        if len(tokens) < len(header):
            continue
        parsed = parse_date_token(tokens[0], context_year)
        if parsed is None:
            continue
        values = dict(zip(header, tokens))
        row: dict[str, Any] = {"DATE": parsed.isoformat(), "DOY": parsed.timetuple().tm_yday}
        for field in WEATHER_FIELDS:
            try:
                row[field] = float(values[field])
            except (KeyError, ValueError):
                row[field] = None
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No daily weather rows parsed from {path}")
    return rows


def first_section_row(text: str, section_header: str, row_pattern: str) -> str:
    lines = text.splitlines()
    in_section = False
    for line in lines:
        if line.lstrip().upper().startswith(section_header.upper()):
            in_section = True
            continue
        if in_section and line.startswith("*"):
            break
        if in_section and re.match(row_pattern, line):
            return line
    raise RuntimeError(f"Cannot find first row after {section_header}")


def parse_filex(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="latin-1", errors="replace")
    wsta_row = first_section_row(text, "@L ID_FIELD WSTA", r"\s*\d+\s+")
    pdate_row = first_section_row(text, "@P PDATE", r"\s*\d+\s+")
    general_row = first_section_row(text, "@N GENERAL", r"\s*\d+\s+GE\b")
    methods_row = first_section_row(text, "@N METHODS", r"\s*\d+\s+ME\b")
    wsta_tokens, pdate_tokens = wsta_row.split(), pdate_row.split()
    general_tokens, methods_tokens = general_row.split(), methods_row.split()
    if len(wsta_tokens) < 3 or len(pdate_tokens) < 3 or len(general_tokens) < 6 or len(methods_tokens) < 3:
        raise RuntimeError("FileX rows do not have expected DSSAT columns")
    pdate = pdate_tokens[1]
    sdate = general_tokens[5]
    calendar_year = 2000 + int(sdate[:2])
    return {
        "WSTA": wsta_tokens[2],
        "PDATE": pdate,
        "PDATE_DATE": (date(calendar_year, 1, 1) + timedelta(days=int(pdate[2:]) - 1)).isoformat(),
        "SDATE": sdate,
        "SDATE_DATE": (date(calendar_year, 1, 1) + timedelta(days=int(sdate[2:]) - 1)).isoformat(),
        "calendar_year": calendar_year,
        "WTHER": methods_tokens[2].upper(),
        "filex_row_wsta": wsta_row,
        "filex_row_pdate": pdate_row,
        "filex_row_sdate": general_row,
        "filex_row_methods": methods_row,
    }


def as_number(value: Any) -> float | None:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def state_value(state: dict[str, Any], *names: str) -> Any:
    lowered = {str(key).casefold(): value for key, value in state.items()}
    for name in names:
        if name.casefold() in lowered:
            return lowered[name.casefold()]
    return None


def state_date(state: dict[str, Any], context_year: int) -> tuple[str | None, str]:
    for name in ("date", "weather_date", "yrdoy", "doy"):
        value = state_value(state, name)
        if value is None:
            continue
        if isinstance(value, str):
            try:
                return date.fromisoformat(value.strip()).isoformat(), f"runtime_{name}_iso"
            except ValueError:
                pass
        if hasattr(value, "tolist"):
            try:
                value = value.tolist()
            except Exception:
                pass
        if isinstance(value, (list, tuple)):
            if len(value) == 366:
                doy = max(range(len(value)), key=lambda i: float(value[i])) + 1
                return (date(context_year, 1, 1) + timedelta(days=doy - 1)).isoformat(), "yrdoy_one_hot"
            continue
        parsed = parse_date_token(str(int(value)) if isinstance(value, float) and value.is_integer() else str(value), context_year)
        if parsed:
            return parsed.isoformat(), f"runtime_{name}"
        if name == "doy":
            try:
                doy = int(float(value))
                return (date(context_year, 1, 1) + timedelta(days=doy - 1)).isoformat(), "runtime_doy_plus_filex_year"
            except (TypeError, ValueError):
                pass
    return None, "unavailable"


def runtime_weather_rows(captures: list[dict[str, Any]], context_year: int) -> list[dict[str, Any]]:
    rows = []
    for index, capture in enumerate(captures):
        state = capture.get("state") or {}
        date_text, date_basis = state_date(state, context_year)
        row: dict[str, Any] = {
            "capture_index": index,
            "step": capture.get("step"),
            "DATE": date_text or "",
            "DOY": date.fromisoformat(date_text).timetuple().tm_yday if date_text else "",
            "YRDOY": as_number(state_value(state, "yrdoy")),
            "DAP": as_number(state_value(state, "dap")),
            "date_mapping_basis": date_basis,
            "terminated": bool(capture.get("done", False)),
            "truncated": False,
            "state_keys": ";".join(sorted(map(str, state.keys()))),
        }
        for field, aliases in {
            "SRAD": ("srad",), "TMAX": ("tmax",), "TMIN": ("tmin",), "RAIN": ("rain",)
        }.items():
            row[field] = as_number(state_value(state, *aliases))
        if all(row[field] is not None for field in WEATHER_FIELDS):
            rows.append(row)
    return rows


def current_rss_mb() -> float:
    try:
        import psutil
        process = psutil.Process()
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except Exception:
                pass
        return rss / (1024 * 1024)
    except Exception:
        return math.nan


def inventory_runtime_files(tmp_dir: Path, source_hash: str) -> list[dict[str, Any]]:
    records = []
    for path in sorted(p for p in tmp_dir.rglob("*") if p.is_file()):
        digest = sha256_file(path)
        records.append({
            "runtime_path": str(path),
            "relative_path": str(path.relative_to(tmp_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
            "is_wth": path.suffix.upper() == ".WTH",
            "matches_source_wth_sha256": digest == source_hash if path.suffix.upper() == ".WTH" else "",
            "classification": "NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT" if path.suffix.upper() not in {".WTH", ".MZX", ".YML"} else "runtime_input_or_control",
        })
    return records


def compare_weather(input_rows: list[dict[str, Any]], runtime_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_date: dict[str, dict[str, Any]] = {}
    for row in input_rows:
        by_date[str(row["DATE"])] = row
    comparisons: list[dict[str, Any]] = []
    missing_dates = 0
    max_diffs = {field: 0.0 for field in WEATHER_FIELDS}
    exact_cells = 0
    total_cells = 0
    unmapped_rows = 0
    seen_dates: set[str] = set()
    duplicate_runtime_dates = 0
    for row in runtime_rows:
        day = str(row.get("DATE") or "")
        if not day:
            unmapped_rows += 1
            continue
        if day in seen_dates:
            duplicate_runtime_dates += 1
        seen_dates.add(day)
        source = by_date.get(day)
        if source is None:
            missing_dates += 1
            comparisons.append({"DATE": day, "runtime_capture_index": row.get("capture_index"), "source_row_found": False})
            continue
        item: dict[str, Any] = {
            "DATE": day,
            "DOY": row.get("DOY"),
            "DAP": row.get("DAP"),
            "runtime_capture_index": row.get("capture_index"),
            "source_row_found": True,
            "terminated": row.get("terminated"),
            "truncated": row.get("truncated"),
        }
        row_match = True
        for field in WEATHER_FIELDS:
            a, b = float(source[field]), float(row[field])
            difference = abs(a - b)
            max_diffs[field] = max(max_diffs[field], difference)
            exact_cells += int(difference == 0.0)
            total_cells += 1
            item[f"input_{field}"] = a
            item[f"runtime_{field}"] = b
            item[f"abs_diff_{field}"] = difference
            row_match &= difference == 0.0
        item["exact_numeric_match"] = row_match
        item["within_tolerance"] = all(item[f"abs_diff_{field}"] <= TOLERANCE for field in WEATHER_FIELDS)
        comparisons.append(item)
    overlap_rows = sum(bool(row.get("source_row_found")) for row in comparisons)
    overlap_days = len({str(row.get("DATE")) for row in comparisons if row.get("source_row_found")})
    all_within = (
        len(runtime_rows) > 0
        and unmapped_rows == 0
        and missing_dates == 0
        and duplicate_runtime_dates == 0
        and overlap_rows == len(runtime_rows)
        and all(bool(row.get("within_tolerance")) for row in comparisons if row.get("source_row_found"))
    )
    exact = all_within and exact_cells == total_cells and total_cells > 0
    status = "EXACT_MATCH" if exact else (
        "NUMERIC_MATCH_WITH_TOLERANCE" if all_within else (
            "MISMATCH" if overlap_rows > 0 and (missing_dates > 0 or any(v > TOLERANCE for v in max_diffs.values())) else "INSUFFICIENT"
        )
    )
    return comparisons, {
        "n_runtime_weather_rows": len(runtime_rows),
        "n_overlap_days": overlap_days,
        "n_overlap_rows": overlap_rows,
        "unmapped_runtime_rows": unmapped_rows,
        "runtime_dates_not_in_input": missing_dates,
        "duplicate_runtime_dates": duplicate_runtime_dates,
        "max_abs_diff_SRAD": max_diffs["SRAD"],
        "max_abs_diff_TMAX": max_diffs["TMAX"],
        "max_abs_diff_TMIN": max_diffs["TMIN"],
        "max_abs_diff_RAIN": max_diffs["RAIN"],
        "exact_numeric_cells": exact_cells,
        "compared_numeric_cells": total_cells,
        "strict_tolerance": TOLERANCE,
        "daily_weather_match_status": status,
    }


def markdown_report(source: dict[str, Any], execution: dict[str, Any], decision: dict[str, Any]) -> str:
    cmp = execution.get("comparison", {})
    segments = execution.get("runtime_episode_segments", [])
    offset_diag = execution.get("date_offset_diagnostic", {})
    timing_audit = execution.get("state_timing_audit", {})
    lifecycle_note = execution.get("reset_lifecycle_diagnosis", "未发现额外 runtime 生命周期问题。")
    return f"""# 004_23B YC 外部实测天气注入 smoke

## 结论

- `external_weather_injection_status = {decision['external_weather_injection_status']}`
- `wth_copy_status = {decision['wth_copy_status']}`
- `daily_weather_match_status = {decision['daily_weather_match_status']}`
- `wgen_bypass_status = {decision['wgen_bypass_status']}`
- `next_step = {decision['next_step']}`
- 重复性：`{decision['repeatability']}`（按单次最小 smoke 设计，未做第二次 episode）
- 日期/DAP 映射：`{decision.get('date_dap_mapping_status', 'NOT_ASSESSED')}`。

## 输入与 provenance

- 样本：YC observed weather，calendar year `{source['calendar_year']}`；使用既有 004_03 historical-weather run 的输入上下文，未按天气极端性选择。
- 源 WTH：`{source['source_wth_path']}`，SHA256 `{source['source_wth_sha256']}`。
- 源 FileX：`{source['source_filex_path']}`，SHA256 `{source['source_filex_sha256']}`。
- FileX：WSTA `{source['WSTA']}`，PDATE `{source['PDATE']}` (`{source['PDATE_DATE']}`)，SDATE `{source['SDATE']}` (`{source['SDATE_DATE']}`)，WTHER `{source['WTHER']}`。
- 源 WTH 与 004_03 renderer 既有副本 byte-identical：`{source['source_wth_matches_renderer_copy']}`。

## Runtime 核验

- Gym-DSSAT 参数：`random_weather=False`；runtime FileX 的 WTHER=`{execution.get('runtime_WTHER', 'UNAVAILABLE')}`。
- runtime 实际 WTH：`{execution.get('runtime_wth_path', 'UNAVAILABLE')}`；启动后拷贝 SHA256 `{execution.get('runtime_wth_sha256_before', 'UNAVAILABLE')}`，episode 后 SHA256 `{execution.get('runtime_wth_sha256_after', 'UNAVAILABLE')}`。
- 原始 WTH 与 runtime copy：`{decision['wth_copy_status']}`。
- runtime 日天气行数 `{cmp.get('n_runtime_weather_rows', 0)}`；按日期键 join `{cmp.get('n_overlap_rows', 0)}` 行 / `{cmp.get('n_overlap_days', 0)}` 个唯一日期（join 不代表数值匹配）；未映射 `{cmp.get('unmapped_runtime_rows', 0)}` 行；输入外日期 `{cmp.get('runtime_dates_not_in_input', 0)}`；重复 runtime 日期 `{cmp.get('duplicate_runtime_dates', 0)}`。
- 最大绝对差：SRAD `{cmp.get('max_abs_diff_SRAD', 'NA')}`、TMAX `{cmp.get('max_abs_diff_TMAX', 'NA')}`、TMIN `{cmp.get('max_abs_diff_TMIN', 'NA')}`、RAIN `{cmp.get('max_abs_diff_RAIN', 'NA')}`；严格容差 `{cmp.get('strict_tolerance', TOLERANCE)}`。
- 日期键优先使用 PDI 的 YRDOY；具体日期映射见 `runtime_weather_daily.csv`。
- runtime 日期序列分段数：`{len(segments)}`；详情见 `execution_record.json`。固定偏移诊断中 `source_date = runtime_YRDOY - 1 day` 匹配 `{offset_diag.get('source_minus_one_day_within_tolerance_rows', 'NA')}` / `{offset_diag.get('source_minus_one_day_compared_rows', 'NA')}` 行，仅作诊断，不自动接受。
- 生命周期诊断：{lifecycle_note}
- PDI YRDOY 同日直接比较状态为 `{cmp.get('daily_weather_match_status', 'INSUFFICIENT')}`；只有源码/运行时序能解释固定偏移后，才会另行闭合日期映射并重判。
- Gym-DSSAT/PDI 时序审计：`{timing_audit.get('offset_semantics_status', 'NOT_REVIEWED')}`。wrapper 收发时序与仍未知的 DSSAT/PDI 引擎内部日历时序分别记录于 `runtime_state_timing_audit.md/json`。
- WGEN bypass 判断依据：`random_weather=False`、WTHER=M、runtime copied WTH 哈希和逐日比较；判定 `{decision['wgen_bypass_status']}`。

## 实验边界

本次没有训练 PPO、运行 WGEN 或生成 synthetic weather。Gym-DSSAT 构造后直接从 `raw._state` 捕获首状态，未调用 `raw.reset()`，随后只运行一个固定 no-op episode。所有 crop、yield、stress、Summary.OUT 等均为 `NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT`，不作农艺解释。runtime 临时 WTH 在 cleanup 前已复制至本任务结果目录。

## 产物

- 此报告针对 `{execution.get('output_dir', '本次结果目录')}` 中的 clean run；较早 attempt 的失败/重复轨迹保留在其独立目录，不并入本次统计。
- `source_input_manifest.json`
- `runtime_file_inventory.csv`
- `input_weather_daily.csv`
- `runtime_weather_daily.csv`
- `weather_loop_comparison.csv`
- `runtime_date_offset_diagnostic.csv`
- `weather_loop_comparison_minus1_day_diagnostic.csv`（诊断用途，不自动作为 match）
- `runtime_state_timing_audit.md/json`
- `execution_record.json`
- `decision.json`
"""


def audit_existing_output(root: Path, output: Path) -> int:
    manifest_path = output / "source_input_manifest.json"
    execution_path = output / "execution_record.json"
    decision_path = output / "decision.json"
    runtime_path = output / "runtime_weather_daily.csv"
    input_path = output / "input_weather_daily.csv"
    if not all(path.is_file() for path in (manifest_path, execution_path, decision_path, runtime_path, input_path)):
        raise FileNotFoundError("Existing-run audit requires manifest, execution, decision, and daily CSV artifacts")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    with runtime_path.open("r", encoding="utf-8-sig", newline="") as stream:
        runtime_rows = list(csv.DictReader(stream))
    with input_path.open("r", encoding="utf-8-sig", newline="") as stream:
        input_rows = list(csv.DictReader(stream))
    input_by_date = {row["DATE"]: row for row in input_rows}
    compare_rows_input: list[dict[str, Any]] = []
    for row in runtime_rows:
        normalized: dict[str, Any] = dict(row)
        for field in WEATHER_FIELDS:
            try:
                normalized[field] = float(row[field])
            except (ValueError, KeyError):
                normalized[field] = None
        try:
            normalized["capture_index"] = int(row["capture_index"])
        except (ValueError, KeyError):
            pass
        try:
            normalized["DAP"] = float(row["DAP"])
        except (ValueError, KeyError):
            pass
        normalized["terminated"] = str(row.get("terminated", "")).casefold() == "true"
        normalized["truncated"] = str(row.get("truncated", "")).casefold() == "true"
        compare_rows_input.append(normalized)
    refreshed_comparison, refreshed_summary = compare_weather(input_rows, compare_rows_input)
    write_csv(output / "weather_loop_comparison.csv", refreshed_comparison)
    execution["comparison"] = refreshed_summary

    segments: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    previous_date: date | None = None
    for row in runtime_rows:
        try:
            row_date = date.fromisoformat(row.get("DATE", ""))
        except ValueError:
            row_date = None
        if current and row_date is not None and previous_date is not None and row_date != previous_date + timedelta(days=1):
            segments.append(current)
            current = []
        current.append(row)
        previous_date = row_date
    if current:
        segments.append(current)

    diagnostic_rows: list[dict[str, Any]] = []
    minus_one_day_details: list[dict[str, Any]] = []
    offset_summary: dict[str, int] = {
        "source_minus_one_day_exact_rows": 0,
        "source_minus_one_day_within_tolerance_rows": 0,
        "source_minus_one_day_compared_rows": 0,
    }
    for segment_index, segment in enumerate(segments, start=1):
        dates = [row.get("DATE", "") for row in segment]
        for offset in (-1, 0, 1):
            joined = exact_rows = within_tolerance_rows = exact_cells = within_tolerance_cells = compared_cells = 0
            max_diffs = {field: 0.0 for field in WEATHER_FIELDS}
            for runtime_row in segment:
                try:
                    runtime_date = date.fromisoformat(runtime_row["DATE"])
                except (ValueError, KeyError):
                    continue
                source_date = (runtime_date + timedelta(days=offset)).isoformat()
                source_row = input_by_date.get(source_date)
                if source_row is None:
                    if offset == -1:
                        minus_one_day_details.append({
                            "runtime_sequence": segment_index,
                            "runtime_YRDOY_DATE": runtime_date.isoformat(),
                            "source_date_diagnostic_only": source_date,
                            "source_row_found": False,
                            "diagnostic_mapping_accepted": False,
                        })
                    continue
                joined += 1
                row_exact = row_within_tolerance = True
                detail: dict[str, Any] = {
                    "runtime_sequence": segment_index,
                    "runtime_YRDOY_DATE": runtime_date.isoformat(),
                    "source_date_diagnostic_only": source_date,
                    "source_row_found": True,
                    "diagnostic_mapping_accepted": False,
                }
                for field in WEATHER_FIELDS:
                    try:
                        difference = abs(float(source_row[field]) - float(runtime_row[field]))
                    except (ValueError, KeyError):
                        row_exact = False
                        continue
                    max_diffs[field] = max(max_diffs[field], difference)
                    compared_cells += 1
                    exact_cells += int(difference == 0.0)
                    row_exact &= difference == 0.0
                    within_tolerance_cells += int(difference <= TOLERANCE)
                    row_within_tolerance &= difference <= TOLERANCE
                    if offset == -1:
                        detail[f"source_{field}"] = float(source_row[field])
                        detail[f"runtime_{field}"] = float(runtime_row[field])
                        detail[f"abs_diff_{field}"] = difference
                exact_rows += int(row_exact)
                within_tolerance_rows += int(row_within_tolerance)
                if offset == -1:
                    detail["four_variable_exact_match"] = row_exact
                    detail["four_variable_within_tolerance"] = row_within_tolerance
                    detail["tolerance"] = TOLERANCE
                    minus_one_day_details.append(detail)
            diagnostic_rows.append({
                "runtime_sequence": segment_index,
                "runtime_rows": len(segment),
                "runtime_date_start": min(dates) if dates else "",
                "runtime_date_end": max(dates) if dates else "",
                "source_date_offset_days": offset,
                "offset_interpretation": "source_date = runtime_YRDOY + offset_days",
                "n_joined_days": joined,
                "n_four_variable_exact_rows": exact_rows,
                "n_four_variable_within_tolerance_rows": within_tolerance_rows,
                "n_exact_numeric_cells": exact_cells,
                "n_within_tolerance_numeric_cells": within_tolerance_cells,
                "n_compared_numeric_cells": compared_cells,
                "strict_tolerance": TOLERANCE,
                "max_abs_diff_SRAD": max_diffs["SRAD"],
                "max_abs_diff_TMAX": max_diffs["TMAX"],
                "max_abs_diff_TMIN": max_diffs["TMIN"],
                "max_abs_diff_RAIN": max_diffs["RAIN"],
                "diagnostic_only_not_accepted_as_match": True,
            })
            if offset == -1:
                offset_summary["source_minus_one_day_exact_rows"] += exact_rows
                offset_summary["source_minus_one_day_within_tolerance_rows"] += within_tolerance_rows
                offset_summary["source_minus_one_day_compared_rows"] += joined
    write_csv(output / "runtime_date_offset_diagnostic.csv", diagnostic_rows)
    write_csv(output / "weather_loop_comparison_minus1_day_diagnostic.csv", minus_one_day_details)

    segment_summaries = []
    for index, segment in enumerate(segments, start=1):
        dates = [row.get("DATE", "") for row in segment]
        dap_values = []
        for row in segment:
            try:
                dap_values.append(float(row["DAP"]))
            except (ValueError, KeyError):
                pass
        segment_summaries.append({
            "sequence": index,
            "rows": len(segment),
            "date_start": min(dates) if dates else "",
            "date_end": max(dates) if dates else "",
            "dap_min": min(dap_values) if dap_values else None,
            "dap_max": max(dap_values) if dap_values else None,
        })
    execution["runtime_episode_segments"] = segment_summaries
    execution["date_offset_diagnostic"] = offset_summary
    if execution.get("raw_reset_called"):
        execution["reset_lifecycle_diagnosis"] = (
            "本次 runner 记录 raw.reset() 已调用；Gym-DSSAT reset 生命周期可能先推进未完成 episode。"
            "需结合日期分段和调用栈解释，不将该轨迹当作 clean single-episode trace。"
        )
    else:
        execution["reset_lifecycle_diagnosis"] = (
            "本次直接以 Gym-DSSAT 构造后的 raw._state 作为首状态，未调用 raw.reset()；"
            "若仅有一个连续日期段，则重复轨迹不能再由 reset 双启动解释。"
        )
    execution["runner_lifecycle_fix"] = "Executed constructor-state path; raw.reset() was not called." if not execution.get("raw_reset_called") else "raw.reset() was called; clean lifecycle gate failed."
    execution["source_minus_one_day_offset_is_diagnostic_only"] = True
    timing_path = output / "runtime_state_timing_audit.json"
    if timing_path.is_file():
        execution["state_timing_audit"] = json.loads(timing_path.read_text(encoding="utf-8"))
    write_json(execution_path, execution)

    duplicate_dates = execution.get("comparison", {}).get("duplicate_runtime_dates", 0)
    one_segment = len(segment_summaries) == 1
    timing_explains_offset = bool(execution.get("state_timing_audit", {}).get("offset_semantics_confirmed"))
    no_reset = execution.get("raw_reset_called") is False
    clean_single = duplicate_dates == 0 and one_segment and no_reset
    decision["date_dap_mapping_status"] = (
        "CLOSED_FIXED_MINUS_ONE_DAY" if clean_single and timing_explains_offset
        else "UNRESOLVED_FIXED_MINUS_ONE_DAY" if clean_single and offset_summary["source_minus_one_day_within_tolerance_rows"] == offset_summary["source_minus_one_day_compared_rows"]
        else "NOT_CLOSED_DUPLICATE_OR_DISCONTINUOUS_RUNTIME_SEQUENCE"
    )
    decision["wth_copy_status"] = "BYTE_IDENTICAL" if execution.get("runtime_wth_initial_byte_identical") else decision.get("wth_copy_status", "NOT_VERIFIED")
    decision["daily_weather_match_status"] = execution.get("comparison", {}).get("daily_weather_match_status", "INSUFFICIENT")
    offset_is_stable = (
        clean_single
        and offset_summary["source_minus_one_day_compared_rows"] > 0
        and offset_summary["source_minus_one_day_within_tolerance_rows"] == offset_summary["source_minus_one_day_compared_rows"]
    )
    if decision["daily_weather_match_status"] == "MISMATCH" and offset_is_stable and decision["wth_copy_status"] == "BYTE_IDENTICAL" and execution.get("runtime_WTHER") == "M" and execution.get("runtime_random_weather") is False:
        decision["external_weather_injection_status"] = "PARTIALLY_VERIFIED"
    else:
        decision["external_weather_injection_status"] = "FAILED" if decision["daily_weather_match_status"] == "MISMATCH" else decision.get("external_weather_injection_status")
    decision["wgen_bypass_status"] = "SUPPORTED" if decision["wth_copy_status"] == "BYTE_IDENTICAL" and execution.get("runtime_WTHER") == "M" and execution.get("runtime_random_weather") is False else "UNVERIFIED"
    decision["next_step"] = "STOP_INSUFFICIENT" if offset_is_stable and not timing_explains_offset else ("FIX_INJECTION_PATH" if decision["daily_weather_match_status"] == "MISMATCH" else decision.get("next_step"))
    decision["date_offset_diagnostic_accepted_for_match"] = False
    write_json(decision_path, decision)
    report = root / "docs/yc_random_weather_004_23B_external_weather_injection_smoke.md"
    report.write_text(markdown_report(manifest, execution, decision), encoding="utf-8")
    print(json.dumps({"audited_existing_outputs": str(output), "decision": decision, "runtime_episode_segments": segment_summaries, "date_offset_diagnostic": offset_summary}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--weather-wth", type=Path, default=None, help="Measured WTH override for isolated task-local smoke inputs")
    parser.add_argument("--audit-existing", action="store_true", help="Audit existing outputs only; never launch Gym-DSSAT")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (args.output_dir or (root / "results/yc_random_weather_ppo/004_23B_yc_external_weather_injection_smoke")).resolve()
    if args.audit_existing:
        return audit_existing_output(root, output)
    allowed_existing = {
        "source_input_manifest.json",
        "input_weather_daily.csv",
        "runtime_file_inventory.csv",
        "runtime_weather_daily.csv",
        "weather_loop_comparison.csv",
        "execution_record.json",
        "decision.json",
        "inputs/CNYC0401_measured_2004.MZX",
        "runtime_capture/attempt_001_runtime_CNYC0401.WTH",
        "runtime_capture/attempt_001_runtime_fileX.MZX",
        "runtime_capture/attempt_001_startup_failure.json",
    }
    if output.exists():
        existing_files = [
            str(path.relative_to(output)).replace("\\", "/")
            for path in output.rglob("*")
            if path.is_file()
        ]
        unexpected = [
            rel for rel in existing_files
            if rel not in allowed_existing and not rel.startswith("attempt_001/")
        ]
        if unexpected:
            raise FileExistsError(f"Refusing to overwrite existing task evidence: {unexpected}")
    output.mkdir(parents=True, exist_ok=True)
    for rel in [
        "prompt_02/004_23B_yc_external_weather_injection_smoke.md",
        "src/run_004_23B_yc_external_weather_injection_smoke.py",
    ]:
        if not (root / rel).is_file():
            raise FileNotFoundError(root / rel)

    source_wth = (args.weather_wth or (root / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC/CNYC0401.WTH")).resolve()
    source_filex = (
        root / "results/yc_random_weather_ppo/004_03/training_verified/historical/ppo_seed_0/"
        / "rendered_inputs/YCA/2004/HISTORICAL_WEATHER_ppo_seed_0_2004/"
        / "YCA_2004_HISTORICAL_WEATHER_ppo_seed_0_2004.jinja2"
    )
    renderer_copy = source_filex.parent / "CNYC0401.WTH"
    site_input = source_wth.parent
    cultivar, soil = site_input / "MZCER048.CUL", site_input / "SOIL.SOL"
    required = [source_wth, source_filex, renderer_copy, cultivar, soil]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing provenance input(s): " + ", ".join(missing))

    filex_info = parse_filex(source_filex)
    source_hash = sha256_file(source_wth)
    manifest = {
        "task": "004_23B",
        "selection_rationale": "Use an existing 004_03 historical-weather YC 2004 episode context; year selected for traceable ordinary historical input provenance, not weather extremes.",
        "source_wth_path": str(source_wth),
        "source_wth_sha256": source_hash,
        "source_wth_size_bytes": source_wth.stat().st_size,
        "source_filex_path": str(source_filex),
        "source_filex_sha256": sha256_file(source_filex),
        "source_renderer_copy_path": str(renderer_copy),
        "source_renderer_copy_sha256": sha256_file(renderer_copy),
        "source_wth_matches_renderer_copy": source_hash == sha256_file(renderer_copy),
        "cultivar_path": str(cultivar),
        "cultivar_sha256": sha256_file(cultivar),
        "soil_path": str(soil),
        "soil_sha256": sha256_file(soil),
        **filex_info,
        "auxiliary_file_paths": [str(cultivar), str(source_wth), str(soil)],
        "random_weather": False,
        "expected_WTHER": "M",
        "runtime_date_mapping_priority": ["PDI YRDOY", "PDI DOY plus FileX year"],
    }
    manifest_path = output / "source_input_manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("source_wth_sha256") != source_hash or existing_manifest.get("source_filex_sha256") != manifest["source_filex_sha256"]:
            raise RuntimeError("Existing source manifest differs from current inputs; preserving it and stopping")
    else:
        write_json(manifest_path, manifest)
    input_rows = parse_wth(source_wth, int(manifest["calendar_year"]))
    input_csv = output / "input_weather_daily.csv"
    if not input_csv.exists():
        write_csv(input_csv, input_rows, ["DATE", "DOY", "SRAD", "TMAX", "TMIN", "RAIN"])

    copied_filex = output / "inputs" / "CNYC0401_measured_2004.MZX"
    copied_filex.parent.mkdir(parents=True, exist_ok=True)
    if copied_filex.exists():
        if sha256_file(copied_filex) != manifest["source_filex_sha256"]:
            raise RuntimeError("Existing task-local FileX differs; preserving it and stopping")
    else:
        shutil.copyfile(source_filex, copied_filex)
    if sha256_file(copied_filex) != manifest["source_filex_sha256"]:
        raise RuntimeError("Task-local FileX copy is not byte-identical to provenance source")

    prior_wth = output / "runtime_capture" / "attempt_001_runtime_CNYC0401.WTH"
    if prior_wth.is_file():
        prior_hash = sha256_file(prior_wth)
        if prior_hash != source_hash:
            raise RuntimeError("Preserved attempt_001 runtime WTH differs from source; do not continue")
        prior_failure = output / "runtime_capture" / "attempt_001_startup_failure.json"
        if not prior_failure.exists():
            write_json(prior_failure, {
                "attempt": 1,
                "status": "STARTUP_STALLED_NO_INITIAL_PDI_STATE",
                "runtime_tmp_path": "/tmp/tmpofuc29jv",
                "runtime_wth_archived_path": str(prior_wth),
                "runtime_wth_sha256": prior_hash,
                "source_wth_sha256": source_hash,
                "wth_byte_identical": True,
                "initial_pdi_state_received": False,
                "daily_steps_confirmed": 0,
                "observation": "PDI initialization remained blocked waiting for its first state reply; the agent interrupted only this task-owned Python process after about three minutes. No PPO was involved. Any possible crop-runtime startup output is non-scientific side effect.",
                "logs_directory_was_missing_at_first_start": True,
                "runtime_temp_preserved": True,
            })

    captures: list[dict[str, Any]] = []
    output_logs = output / "logs"
    output_logs.mkdir(parents=True, exist_ok=True)
    partial_instances: list[Any] = []
    execution: dict[str, Any] = {
        "started_at": datetime.now().astimezone().isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": os.getcwd(),
        "project_root": str(root),
        "output_dir": str(output),
        "task_local_filex_path": str(copied_filex),
        "task_local_filex_sha256": sha256_file(copied_filex),
        "runtime": {},
        "fixed_management_action": {"amir": 0.0, "anfer": 0.0},
        "crop_outputs_classification": "NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT",
        "repeatability": "NOT_TESTED",
        "raw_reset_called": False,
        "initial_state_source": "DssatPdi constructor raw._state",
        "runtime_attempt": output.name,
        "error": None,
    }
    prior_failure_path = output / "runtime_capture" / "attempt_001_startup_failure.json"
    if prior_failure_path.exists():
        execution["previous_startup_attempt"] = json.loads(prior_failure_path.read_text(encoding="utf-8"))
    tmp_dir: Path | None = None
    raw = None
    try:
        import gym
        import gym_dssat_pdi.envs.dssat_pdi as dssat_module
        import zmq

        try:
            package_version = importlib.metadata.version("gym-dssat-pdi")
        except importlib.metadata.PackageNotFoundError:
            package_version = "unknown"
        execution["runtime"] = {
            "gym_version": getattr(gym, "__version__", "unknown"),
            "gym_dssat_pdi_module": str(Path(dssat_module.__file__).resolve()),
            "gym_dssat_pdi_version": package_version,
            "run_dssat_location": "/opt/dssat_pdi/run_dssat",
            "pdi_initial_state_receive_timeout_seconds": 30,
        }
        env_class = dssat_module.DssatPdi
        original_get_state_class = env_class._get_state
        original_init_class = env_class.__init__

        def timed_get_state(instance: Any, *call_args: Any, **call_kwargs: Any):
            if getattr(instance, "_server", None) is not None:
                instance._server.setsockopt(zmq.RCVTIMEO, 30_000)
            try:
                return original_get_state_class(instance, *call_args, **call_kwargs)
            except zmq.Again as timeout_error:
                raise TimeoutError("PDI did not return a state within 30 seconds") from timeout_error

        def tracked_init(instance: Any, *init_args: Any, **init_kwargs: Any):
            partial_instances.append(instance)
            return original_init_class(instance, *init_args, **init_kwargs)

        env_class._get_state = timed_get_state
        env_class.__init__ = tracked_init
        try:
            env = gym.make(
                "gym_dssat_pdi:GymDssatPdi-v0",
                log_saving_path=str(output_logs / f"pdi_gym_{output.name}.log"),
                mode="all",
                seed=23,
                random_weather=False,
                evaluation=True,
                fileX_template_path=str(copied_filex),
                experiment_number=1,
                auxiliary_file_paths=[str(cultivar), str(source_wth), str(soil)],
                run_dssat_location="/opt/dssat_pdi/run_dssat",
            )
        finally:
            env_class._get_state = original_get_state_class
            env_class.__init__ = original_init_class
        raw = env.unwrapped
        execution["raw_reset_called"] = False
        execution["initial_state_source"] = "DssatPdi constructor raw._state"
        tmp_dir = Path(raw._tmp_folder).resolve()
        execution["runtime_tmp_dir"] = str(tmp_dir)
        execution["runtime_random_weather"] = bool(raw.random_weather)
        execution["runtime_wther_property"] = str(raw.wther)
        execution["runtime_rseed1_field"] = int(raw._rseed1)
        runtime_filex = tmp_dir / "fileX.MZX"
        execution["runtime_filex_path"] = str(runtime_filex)
        runtime_filex_text = runtime_filex.read_text(encoding="latin-1", errors="replace")
        runtime_filex_info = parse_filex(runtime_filex)
        execution["runtime_WTHER"] = runtime_filex_info["WTHER"]
        execution["runtime_WSTA"] = runtime_filex_info["WSTA"]
        execution["runtime_PDATE"] = runtime_filex_info["PDATE"]
        runtime_wth = tmp_dir / f"{manifest['WSTA']}.WTH"
        execution["runtime_wth_path"] = str(runtime_wth)
        if not runtime_wth.is_file():
            raise FileNotFoundError(f"Gym-DSSAT did not copy expected WTH into runtime temp: {runtime_wth}")
        execution["runtime_wth_sha256_before"] = sha256_file(runtime_wth)
        execution["runtime_wth_size_before"] = runtime_wth.stat().st_size
        archive_wth = output / "runtime_capture" / "runtime_copied_CNYC0401.WTH"
        archive_wth.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runtime_wth, archive_wth)
        execution["archived_runtime_wth_path"] = str(archive_wth)
        execution["archived_runtime_wth_sha256"] = sha256_file(archive_wth)
        execution["runtime_wth_initial_byte_identical"] = source_hash == execution["runtime_wth_sha256_before"]
        inventory_before = inventory_runtime_files(tmp_dir, source_hash)
        write_csv(output / "runtime_file_inventory.csv", inventory_before)

        original_get_state = raw._get_state

        def capture_get_state(*call_args: Any, **call_kwargs: Any):
            result = original_get_state(*call_args, **call_kwargs)
            state = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            done = bool(result[2]) if isinstance(result, tuple) and len(result) > 2 else False
            if isinstance(state, dict) and state:
                captures.append({"step": len(captures), "state": dict(state), "done": done})
            return result

        raw._get_state = capture_get_state
        initial_state = getattr(raw, "_state", None)
        if isinstance(initial_state, dict) and initial_state:
            captures.append({"step": 0, "state": dict(initial_state), "done": bool(raw.done)})
        max_rss = current_rss_mb()
        stop_reason = None
        for step_index in range(450):
            if raw.done:
                break
            result = raw.step({"amir": 0.0, "anfer": 0.0})
            execution["steps_attempted"] = step_index + 1
            if result is None:
                raise RuntimeError("DssatPdi.step returned None; inspect runtime log for failure")
            if not isinstance(result, tuple) or len(result) < 3:
                raise RuntimeError(f"Unexpected DssatPdi.step result structure: {type(result)}")
            if bool(result[2]):
                break
            if step_index % 10 == 0:
                max_rss = max(max_rss, current_rss_mb())
            if math.isfinite(max_rss) and max_rss > RSS_STOP_MB:
                stop_reason = f"RSS guard exceeded {RSS_STOP_MB} MiB"
                execution["resource_guard_triggered"] = stop_reason
                break
        else:
            stop_reason = "Reached 450-step hard limit"
        execution["max_process_tree_rss_mb"] = max_rss if math.isfinite(max_rss) else None
        execution["runtime_done"] = bool(raw.done)
        execution["terminated"] = bool(raw.done)
        execution["truncated"] = bool(stop_reason is not None)
        execution["stop_reason"] = stop_reason
        execution["runtime_state_capture_count"] = len(captures)
        execution["captured_state_keys"] = sorted({str(key) for item in captures for key in item["state"].keys()})
        if runtime_wth.exists():
            execution["runtime_wth_sha256_after"] = sha256_file(runtime_wth)
            execution["runtime_wth_size_after"] = runtime_wth.stat().st_size
        runtime_rows = runtime_weather_rows(captures, int(manifest["calendar_year"]))
        if stop_reason and runtime_rows:
            runtime_rows[-1]["truncated"] = True
        write_csv(
            output / "runtime_weather_daily.csv",
            runtime_rows,
            ["capture_index", "step", "DATE", "DOY", "YRDOY", "DAP", "date_mapping_basis", *WEATHER_FIELDS, "terminated", "truncated", "state_keys"],
        )
        comparisons, comparison_summary = compare_weather(input_rows, runtime_rows)
        write_csv(output / "weather_loop_comparison.csv", comparisons)
        execution["comparison"] = comparison_summary
        execution["source_wth_sha256"] = source_hash
        execution["runtime_wth_sha256"] = execution.get("runtime_wth_sha256_before")
        execution["runtime_filex_text_sha256"] = sha256_file(runtime_filex)
        execution["runtime_filex_text"] = runtime_filex_text
        execution["runtime_wther_matches_expected"] = execution["runtime_WTHER"] == "M"
    except Exception as exc:
        execution["error"] = f"{type(exc).__name__}: {exc}"
        execution["traceback"] = traceback.format_exc()
        if raw is None and partial_instances:
            partial = partial_instances[-1]
            partial_dir = Path(getattr(partial, "_tmp_folder", "")).resolve()
            execution["partial_runtime_tmp_dir"] = str(partial_dir)
            partial_wth = partial_dir / f"{manifest['WSTA']}.WTH"
            if partial_wth.is_file():
                execution["partial_runtime_wth_sha256"] = sha256_file(partial_wth)
                execution["partial_runtime_wth_path"] = str(partial_wth)
                archived = output / "runtime_capture" / f"{output.name}_partial_runtime_CNYC0401.WTH"
                archived.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(partial_wth, archived)
                execution["partial_runtime_wth_archive"] = str(archived)
                execution["partial_runtime_wth_archive_sha256"] = sha256_file(archived)
            try:
                partial._cleanup_process()
                if getattr(partial, "_server", None) is not None:
                    partial._close_server()
                if getattr(partial, "_zmq_context", None) is not None:
                    partial._zmq_context.term()
                if getattr(partial, "_f_out", None) is not None and not partial._f_out.closed:
                    partial._f_out.close()
            except Exception as cleanup_exc:
                execution["partial_runtime_cleanup_error"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
        if captures:
            runtime_rows = runtime_weather_rows(captures, int(manifest["calendar_year"]))
            write_csv(output / "runtime_weather_daily.csv", runtime_rows)
            comparisons, comparison_summary = compare_weather(input_rows, runtime_rows)
            write_csv(output / "weather_loop_comparison.csv", comparisons)
            execution["comparison"] = comparison_summary
    finally:
        if raw is not None:
            try:
                if tmp_dir and (tmp_dir / f"{manifest['WSTA']}.WTH").is_file():
                    execution["runtime_wth_sha256_after"] = sha256_file(tmp_dir / f"{manifest['WSTA']}.WTH")
                    write_csv(output / "runtime_file_inventory.csv", inventory_runtime_files(tmp_dir, source_hash))
                raw.close()
            except Exception as close_exc:
                execution["cleanup_error"] = f"{type(close_exc).__name__}: {close_exc}"
    execution["finished_at"] = datetime.now().astimezone().isoformat()
    execution["comparison"] = execution.get("comparison", {})
    execution["input_weather_rows"] = len(input_rows)
    execution["runtime_capture_rows"] = len(runtime_weather_rows(captures, int(manifest["calendar_year"])))
    if not (output / "runtime_weather_daily.csv").exists():
        write_csv(output / "runtime_weather_daily.csv", [])
    if not (output / "weather_loop_comparison.csv").exists():
        write_csv(output / "weather_loop_comparison.csv", [])
    if not (output / "runtime_file_inventory.csv").exists():
        write_csv(output / "runtime_file_inventory.csv", [])
    cmp = execution.get("comparison", {})
    wth_equal = bool(execution.get("runtime_wth_initial_byte_identical"))
    runtime_wther_m = execution.get("runtime_WTHER") == "M" and execution.get("runtime_random_weather") is False
    daily_status = cmp.get("daily_weather_match_status", "INSUFFICIENT")
    if wth_equal and runtime_wther_m and daily_status in {"EXACT_MATCH", "NUMERIC_MATCH_WITH_TOLERANCE"}:
        injection = "RUNTIME_VERIFIED"
    elif execution.get("error") or daily_status == "MISMATCH" or ("runtime_wth_initial_byte_identical" in execution and not wth_equal):
        injection = "FAILED"
    elif wth_equal:
        injection = "PARTIALLY_VERIFIED"
    else:
        injection = "FAILED"
    if runtime_wther_m and wth_equal and daily_status in {"EXACT_MATCH", "NUMERIC_MATCH_WITH_TOLERANCE"}:
        bypass = "VERIFIED"
    elif runtime_wther_m and wth_equal:
        bypass = "SUPPORTED"
    else:
        bypass = "UNVERIFIED"
    if wth_equal and daily_status in {"EXACT_MATCH", "NUMERIC_MATCH_WITH_TOLERANCE"}:
        next_step = "EXTERNAL_AUGMENTATION_PILOT"
    elif daily_status == "MISMATCH" or ("runtime_wth_initial_byte_identical" in execution and not wth_equal):
        next_step = "FIX_INJECTION_PATH"
    else:
        next_step = "STOP_INSUFFICIENT"
    decision = {
        "external_weather_injection_status": injection,
        "wth_copy_status": "BYTE_IDENTICAL" if wth_equal else ("MISMATCH" if "runtime_wth_initial_byte_identical" in execution else "NOT_VERIFIED"),
        "daily_weather_match_status": daily_status,
        "wgen_bypass_status": bypass,
        "next_step": next_step,
        "repeatability": "NOT_TESTED",
        "tolerance": TOLERANCE,
        "tolerance_rationale": "Only for small decimal-to-binary floating representation differences in PDI state; exact match is preferred.",
        "scientific_boundary": "NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT",
        "no_ppo": True,
        "no_wgen": True,
        "no_synthetic_weather": True,
        "no_commit": True,
        "no_push": True,
    }
    write_json(output / "execution_record.json", execution)
    write_json(output / "decision.json", decision)
    report_path = root / "docs/yc_random_weather_004_23B_external_weather_injection_smoke.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown_report(manifest, execution, decision), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "report": str(report_path), "decision": decision}, ensure_ascii=False, indent=2))
    return 0 if injection == "RUNTIME_VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
