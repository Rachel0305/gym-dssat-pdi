from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import run_004_23B_yc_external_weather_injection_smoke as smoke


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/yc_random_weather_ppo/004_23B_yc_external_weather_injection_smoke/attempt_005_marker"
SOURCE_WTH = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC/CNYC0401.WTH"
SOURCE_CUL = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC/MZCER048.CUL"
SOURCE_SOL = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC/SOIL.SOL"
SOURCE_FILEX = smoke.SOURCE_FILEX
BASE_RUNNER = ROOT / "src/run_004_23B_yc_external_weather_injection_smoke.py"
PREVIOUS_EXECUTION = ROOT / "results/yc_random_weather_ppo/004_23B_yc_external_weather_injection_smoke/attempt_003/execution_record.json"
REPORT_PATH = ROOT / "docs/yc_random_weather_004_23B_external_weather_injection_smoke.md"
TOLERANCE = smoke.TOLERANCE
MARKER_VALUES = [
    {"RAIN": 2.3, "SRAD": 13.7, "TMAX": 30.8, "TMIN": 23.6},
    {"RAIN": 0.0, "SRAD": 21.3, "TMAX": 34.2, "TMIN": 25.1},
    {"RAIN": 6.4, "SRAD": 8.3, "TMAX": 27.6, "TMIN": 21.9},
]
WEATHER_FIELDS = ("SRAD", "TMAX", "TMIN", "RAIN")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def within_tolerance(left: dict[str, Any] | None, right: dict[str, Any] | None) -> tuple[bool, dict[str, float | None]]:
    if left is None or right is None:
        return False, {field: None for field in WEATHER_FIELDS}
    diffs: dict[str, float | None] = {}
    for field in WEATHER_FIELDS:
        try:
            diffs[field] = abs(float(left[field]) - float(right[field]))
        except (KeyError, TypeError, ValueError):
            diffs[field] = None
    return all(value is not None and value <= TOLERANCE for value in diffs.values()), diffs


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def count_date_segments(rows: list[dict[str, Any]]) -> int:
    segments = 0
    previous: date | None = None
    for row in rows:
        try:
            current = date.fromisoformat(str(row.get("DATE", "")))
        except ValueError:
            continue
        if previous is None or current != previous + timedelta(days=1):
            segments += 1
        previous = current
    return segments


def marker_tuple(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(round(float(row[field]), 1) for field in ("RAIN", "SRAD", "TMAX", "TMIN"))


def create_marker_weather(original: Path, target: Path, marker_dates: list[date]) -> list[dict[str, Any]]:
    source_rows = {row["DATE"]: row for row in smoke.parse_wth(original, 2004)}
    existing_tuples = {marker_tuple(row) for row in source_rows.values()}
    changes: list[dict[str, Any]] = []
    marker_by_date: dict[str, dict[str, float]] = {}
    for marker_date, new_values in zip(marker_dates, MARKER_VALUES, strict=True):
        key = marker_date.isoformat()
        if key not in source_rows:
            raise ValueError(f"Marker date is absent from source weather: {key}")
        old_values = {field: float(source_rows[key][field]) for field in WEATHER_FIELDS}
        if marker_tuple(new_values) in existing_tuples:
            raise ValueError(f"Marker tuple already exists in source series: {key} {marker_tuple(new_values)}")
        if not (0 <= new_values["RAIN"] <= 40 and 0 <= new_values["SRAD"] <= 35):
            raise ValueError(f"Marker outside the configured physical QA envelope: {key}")
        if not (-15 <= new_values["TMIN"] <= 35 and -10 <= new_values["TMAX"] <= 50 and new_values["TMAX"] >= new_values["TMIN"]):
            raise ValueError(f"Invalid marker temperature tuple: {key}")
        marker_by_date[key] = new_values
        changes.append({
            "DATE": key,
            "DOY": marker_date.timetuple().tm_yday,
            "marker_id": f"M{len(changes) + 1}",
            "marker": True,
            **{f"original_{field}": old_values[field] for field in WEATHER_FIELDS},
            **{f"marker_{field}": new_values[field] for field in WEATHER_FIELDS},
            "tuple_order": "RAIN,SRAD,TMAX,TMIN",
        })

    original_bytes = original.read_bytes()
    original_lines = original_bytes.splitlines(keepends=True)
    output_lines: list[bytes] = []
    changed = 0
    for raw_line in original_lines:
        line_body = raw_line.rstrip(b"\r\n")
        ending = raw_line[len(line_body):]
        text = line_body.decode("latin-1")
        tokens = text.split()
        if len(tokens) >= 5 and re.fullmatch(r"\d{7}", tokens[0]):
            day = smoke.parse_date_token(tokens[0], 2004)
            if day and day.isoformat() in marker_by_date:
                values = marker_by_date[day.isoformat()]
                replacement = (
                    f"{tokens[0]}"
                    f"{values['SRAD']:6.1f}{values['TMAX']:6.1f}"
                    f"{values['TMIN']:6.1f}{values['RAIN']:6.1f}"
                ).encode("latin-1")
                if len(replacement) != len(line_body):
                    raise ValueError(f"Marker row width changed for {day.isoformat()}")
                output_lines.append(replacement + ending)
                changed += 1
                continue
        output_lines.append(raw_line)
    if changed != 3:
        raise ValueError(f"Expected exactly 3 changed daily rows, got {changed}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"".join(output_lines))
    if len(target.read_bytes().splitlines()) != len(original_bytes.splitlines()):
        raise ValueError("Marker copy changed WTH line count")
    if len([1 for a, b in zip(original_lines, output_lines, strict=True) if a != b]) != 3:
        raise ValueError("Marker copy changed rows other than the three preselected days")
    return changes


def analyze_mapping(marker_dates: list[date], changes: list[dict[str, Any]], marker_weather: Path, runtime_weather: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = {row["DATE"]: row for row in smoke.parse_wth(marker_weather, 2004)}
    runtime_rows = read_rows(runtime_weather)
    runtime_by_date: dict[str, dict[str, Any]] = {}
    duplicate_dates = 0
    for row in runtime_rows:
        key = row.get("DATE", "")
        if key in runtime_by_date:
            duplicate_dates += 1
        runtime_by_date[key] = row
    comparison: list[dict[str, Any]] = []
    marker_summaries: list[dict[str, Any]] = []
    all_marker_prev = True
    all_marker_unique = True
    for index, day in enumerate(marker_dates, start=1):
        source = source_rows[day.isoformat()]
        expected = {field: float(source[field]) for field in WEATHER_FIELDS}
        matches = []
        for row in runtime_rows:
            matched, diffs = within_tolerance(expected, row)
            if matched:
                matches.append((row, diffs))
        first = matches[0][0] if matches else None
        first_date = first.get("DATE", "") if first else ""
        if first_date == day.isoformat():
            mapping = "SAME_DAY"
        elif first_date == (day + timedelta(days=1)).isoformat():
            mapping = "PREVIOUS_SOURCE_DAY"
        else:
            mapping = "OTHER_OR_INCONSISTENT"
        all_marker_prev &= mapping == "PREVIOUS_SOURCE_DAY"
        all_marker_unique &= len(matches) == 1
        same_row = runtime_by_date.get(day.isoformat())
        next_row = runtime_by_date.get((day + timedelta(days=1)).isoformat())
        same_ok, same_diffs = within_tolerance(expected, same_row)
        next_ok, next_diffs = within_tolerance(expected, next_row)
        marker_summaries.append({
            "marker_id": f"M{index}",
            "source_date": day.isoformat(),
            "marker_tuple_RAIN_SRAD_TMAX_TMIN": [expected["RAIN"], expected["SRAD"], expected["TMAX"], expected["TMIN"]],
            "first_runtime_match_capture_index": first.get("capture_index") if first else None,
            "first_runtime_match_YRDOY": first.get("YRDOY") if first else None,
            "first_runtime_match_DATE": first_date,
            "within_tolerance_runtime_match_count": len(matches),
            "mapping_classification": mapping,
            "same_day_match": same_ok,
            "source_day_to_runtime_next_day_match": next_ok,
            "next_day_abs_diff": next_diffs,
            "exact_tuple_rows": sum(all(float(row[field]) == expected[field] for field in WEATHER_FIELDS) for row, _ in matches),
        })
        comparison.append({
            "record_type": "MARKER",
            "marker_id": f"M{index}",
            "source_date": day.isoformat(),
            "runtime_date_same_day": day.isoformat(),
            "runtime_date_under_previous_source_day_rule": (day + timedelta(days=1)).isoformat(),
            "first_tuple_match_runtime_date": first_date,
            "first_tuple_match_YRDOY": first.get("YRDOY") if first else None,
            "tuple_match_count": len(matches),
            "mapping_classification": mapping,
            "same_day_four_variable_match": same_ok,
            "previous_source_day_four_variable_match": next_ok,
            "same_day_max_abs_diff": max((v for v in same_diffs.values() if v is not None), default=None),
            "previous_source_day_max_abs_diff": max((v for v in next_diffs.values() if v is not None), default=None),
            "mapping_accepted": False,
        })

    neighbor_dates = [marker_dates[0] - timedelta(days=2), marker_dates[0] - timedelta(days=1), marker_dates[-1] + timedelta(days=1), marker_dates[-1] + timedelta(days=2)]
    all_neighbors_prev = True
    for index, day in enumerate(neighbor_dates, start=1):
        source = source_rows.get(day.isoformat())
        expected = {field: float(source[field]) for field in WEATHER_FIELDS} if source else None
        same_row = runtime_by_date.get(day.isoformat())
        next_date = (day + timedelta(days=1)).isoformat()
        next_row = runtime_by_date.get(next_date)
        same_ok, same_diffs = within_tolerance(expected, same_row)
        next_ok, next_diffs = within_tolerance(expected, next_row)
        all_neighbors_prev &= bool(next_ok)
        comparison.append({
            "record_type": "UNMODIFIED_NEIGHBOR",
            "marker_id": "",
            "source_date": day.isoformat(),
            "runtime_date_same_day": day.isoformat(),
            "runtime_date_under_previous_source_day_rule": next_date,
            "first_tuple_match_runtime_date": "",
            "first_tuple_match_YRDOY": "",
            "tuple_match_count": "",
            "mapping_classification": "PREVIOUS_SOURCE_DAY" if next_ok else "OTHER_OR_INCONSISTENT",
            "same_day_four_variable_match": same_ok,
            "previous_source_day_four_variable_match": next_ok,
            "same_day_max_abs_diff": max((v for v in same_diffs.values() if v is not None), default=None),
            "previous_source_day_max_abs_diff": max((v for v in next_diffs.values() if v is not None), default=None),
            "mapping_accepted": False,
            "neighbor_index": index,
        })

    summary = {
        "marker_summaries": marker_summaries,
        "marker_dates_all_classify_previous_source_day": all_marker_prev,
        "each_marker_tuple_has_one_runtime_match": all_marker_unique,
        "all_four_unmodified_neighbors_match_previous_source_day": all_neighbors_prev,
        "neighbor_dates": [day.isoformat() for day in neighbor_dates],
        "runtime_rows": len(runtime_rows),
        "runtime_unique_dates": len(runtime_by_date),
        "runtime_duplicate_dates": duplicate_dates,
        "runtime_date_segment_count": count_date_segments(runtime_rows),
        "all_seven_dates_support_previous_source_day": all_marker_prev and all_marker_unique and all_neighbors_prev,
    }
    return comparison, summary


def report_text(previous_report: str, manifest: dict[str, Any], changes: list[dict[str, Any]], execution: dict[str, Any], decision: dict[str, Any]) -> str:
    table = ["| Marker | Source date | 原值 RAIN/SRAD/TMAX/TMIN | Marker RAIN/SRAD/TMAX/TMIN | 首次 runtime YRDOY | 结论 |", "|---|---|---|---|---:|---|"]
    for item, observed in zip(changes, execution["marker_mapping"]["marker_summaries"], strict=True):
        old = ", ".join(str(item[f"original_{field}"]) for field in ("RAIN", "SRAD", "TMAX", "TMIN"))
        new = ", ".join(str(item[f"marker_{field}"]) for field in ("RAIN", "SRAD", "TMAX", "TMIN"))
        table.append(f"| {item['marker_id']} | {item['DATE']} | {old} | {new} | {observed['first_runtime_match_YRDOY']} | {observed['mapping_classification']} |")
    preserved = previous_report.split("\n", 1)[1] if previous_report.startswith("# ") else previous_report
    return f"""# 004_23B YC 外部实测天气注入 smoke

## 最新判定：独立 marker 日期语义闭合

- `date_dap_mapping_status = {decision['date_dap_mapping_status']}`
- `daily_weather_match_status = {decision['daily_weather_match_status']}`
- `external_weather_injection_status = {decision['external_weather_injection_status']}`
- `wgen_bypass_status = {decision['wgen_bypass_status']}`
- `next_step = {decision['next_step']}`。这里只判定后续允许进入 augmentation pilot；本任务未开始 augmentation。

本次为独立 black-box marker 实验，运行一个 measured-weather / no-op episode。所有 crop、yield、stress、Summary.OUT 等继续标记为 `NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT`，不作农业解释。

### 预选 marker 与 runtime 命中

输入源 WTH：`{manifest['original_wth_path']}`，SHA256 `{manifest['original_wth_sha256']}`。任务副本：`{manifest['marker_wth_path']}`，SHA256 `{manifest['marker_wth_sha256']}`。仅修改 3 条日天气记录，原始 WTH 未改。

WSTA `{manifest['WSTA']}`，SDATE `{manifest['SDATE']}` (`{manifest['SDATE_DATE']}`)，PDATE `{manifest['PDATE']}` (`{manifest['PDATE_DATE']}`)，WTHER `{manifest['WTHER']}`。Marker dates 为 SDATE+40..42，即 `{manifest['marker_dates'][0]}` 至 `{manifest['marker_dates'][-1]}`；到上一 clean run 终止日期 `{manifest['previous_clean_terminal_date']}` 尚有至少 `{manifest['minimum_days_to_previous_terminal']}` 天。

{chr(10).join(table)}

- 三个 marker 均唯一出现在 runtime state 的 `YRDOY = source date + 1`；每个 marker 的 tuple 首次匹配位置、DAP、精确/容差匹配记录在 `marker_mapping_comparison.csv`。
- marker 前后未修改的四个邻近日（前 2 日、后 2 日）也全部只在 `YRDOY = source date + 1` 下四变量匹配。七日连续证据通过：`{execution['marker_mapping']['all_seven_dates_support_previous_source_day']}`。
- Runtime trace `{execution['runtime_rows']}` 行、连续日期段 `{execution['runtime_date_segment_count']}`、重复日期 `{execution['runtime_duplicate_dates']}`；`raw.reset()` 调用 `{execution['raw_reset_called']}`。
- Marker WTH 与 runtime copy：`{execution['runtime_wth_byte_identical']}`；Gym-DSSAT `random_weather=False`、runtime `WTHER={execution['runtime_WTHER']}`。
- 底层 harness 返回码 `{execution['runtime_runner_returncode']}` 是因为它仍按 PDI YRDOY 同日口径报告预期的 MISMATCH；runtime episode 正常完成，marker 全局门控通过后才按实证日期映射给出本节最终判定。

### 解释边界

本次 black-box 实验直接在任务隔离 WTH 中改写三个已知日期的四变量组合，随后从构造态开始观测。marker 与两侧未修改邻日共同证明本 runtime 的经验映射为：`source weather date = runtime YRDOY - 1 day`。因此本任务按验证后的 runtime 日期映射判定 daily weather match；这不是依据旧数据中“看起来匹配更好”而自动选偏移。

attempt_003 的 clean smoke 原始结论与产物仍原样保留在独立目录；该次同日 MISMATCH / 日期语义未闭合只描述 attempt_003，不被改写。本节为新增的独立日期语义闭合实验。

## attempt_003 原始报告（保留）

以下为 marker 实验前报告正文快照；attempt_003 结果文件均未改写。

{preserved}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite marker experiment output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for path in (SOURCE_WTH, SOURCE_FILEX, PREVIOUS_EXECUTION, BASE_RUNNER, REPORT_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    previous_report = REPORT_PATH.read_text(encoding="utf-8")
    (output / "attempt_003_report_snapshot.md").write_text(previous_report, encoding="utf-8")
    filex = smoke.parse_filex(SOURCE_FILEX)
    start_date = date.fromisoformat(filex["SDATE_DATE"])
    marker_dates = [start_date + timedelta(days=40 + i) for i in range(3)]
    previous = json.loads(PREVIOUS_EXECUTION.read_text(encoding="utf-8"))
    terminal_date = date.fromisoformat(previous["runtime_episode_segments"][-1]["date_end"])
    margin = (terminal_date - (marker_dates[-1] + timedelta(days=1))).days
    if marker_dates[0] <= start_date or margin < 45:
        raise RuntimeError(f"Preselected marker dates are not sufficiently far from boundaries: margin={margin}")

    marker_wth = output / "inputs" / "CNYC0401.WTH"
    original_hash_before = sha256_file(SOURCE_WTH)
    changes = create_marker_weather(SOURCE_WTH, marker_wth, marker_dates)
    marker_hash = sha256_file(marker_wth)
    task_cul = marker_wth.parent / SOURCE_CUL.name
    task_sol = marker_wth.parent / SOURCE_SOL.name
    shutil.copyfile(SOURCE_CUL, task_cul)
    shutil.copyfile(SOURCE_SOL, task_sol)
    if sha256_file(task_cul) != sha256_file(SOURCE_CUL) or sha256_file(task_sol) != sha256_file(SOURCE_SOL):
        raise RuntimeError("Task-local cultivar/soil copy integrity check failed")
    if sha256_file(SOURCE_WTH) != original_hash_before:
        raise RuntimeError("Original observed WTH changed while creating marker copy")
    smoke.write_csv(output / "marker_weather_changes.csv", changes)
    manifest = {
        "task": "004_23B independent black-box runtime date-semantics closure",
        "created_at": datetime.now().astimezone().isoformat(),
        "original_wth_path": str(SOURCE_WTH),
        "original_wth_sha256": original_hash_before,
        "marker_wth_path": str(marker_wth),
        "marker_wth_sha256": marker_hash,
        "only_task_copy_modified": True,
        "changed_daily_rows": 3,
        "marker_dates": [d.isoformat() for d in marker_dates],
        "marker_doy": [d.timetuple().tm_yday for d in marker_dates],
        "marker_selection_rule": "SDATE + 40, +41, +42; preselected by episode distance, not weather outcome",
        "previous_clean_terminal_date": terminal_date.isoformat(),
        "minimum_days_to_previous_terminal": margin,
        "marker_values_order": "RAIN,SRAD,TMAX,TMIN",
        "marker_values": [marker_tuple(item) for item in MARKER_VALUES],
        "unique_against_original_source_series": True,
        "WSTA": filex["WSTA"],
        "SDATE": filex["SDATE"],
        "SDATE_DATE": filex["SDATE_DATE"],
        "PDATE": filex["PDATE"],
        "PDATE_DATE": filex["PDATE_DATE"],
        "calendar_year": filex["calendar_year"],
        "WTHER": filex["WTHER"],
        "source_filex_path": str(SOURCE_FILEX),
        "source_filex_sha256": smoke.sha256_file(SOURCE_FILEX),
        "task_cultivar_copy": str(task_cul),
        "source_cultivar_path": str(SOURCE_CUL),
        "source_cultivar_sha256": sha256_file(SOURCE_CUL),
        "task_cultivar_sha256": sha256_file(task_cul),
        "task_soil_copy": str(task_sol),
        "source_soil_path": str(SOURCE_SOL),
        "source_soil_sha256": sha256_file(SOURCE_SOL),
        "task_soil_sha256": sha256_file(task_sol),
        "random_weather": False,
        "management": {"amir": 0.0, "anfer": 0.0},
        "no_raw_reset": True,
        "no_training_wgen_or_synthetic_generation": True,
        "crop_outputs_classification": "NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT",
    }
    write_json(output / "marker_input_manifest.json", manifest)

    runtime_output = output / "runtime"
    command = [
        sys.executable,
        str(BASE_RUNNER),
        "--project-root", str(ROOT),
        "--output-dir", str(runtime_output),
        "--weather-wth", str(marker_wth),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (output / "runtime_runner_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "runtime_runner_stderr.log").write_text(completed.stderr, encoding="utf-8")
    child_execution_path = runtime_output / "execution_record.json"
    child_decision_path = runtime_output / "decision.json"
    if not child_execution_path.is_file() or not child_decision_path.is_file():
        execution = {
            "runtime_runner_returncode": completed.returncode,
            "command": command,
            "runtime_completed": False,
            "error": "Base clean-smoke runner did not produce its execution record and decision.",
            "crop_outputs_classification": "NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT",
        }
        decision = {
            "date_dap_mapping_status": "UNRESOLVED",
            "daily_weather_match_status": "INSUFFICIENT",
            "external_weather_injection_status": "PARTIALLY_VERIFIED",
            "wgen_bypass_status": "SUPPORTED",
            "next_step": "STOP_INSUFFICIENT",
            "no_augmentation_started": True,
        }
        write_json(output / "execution_record.json", execution)
        write_json(output / "decision.json", decision)
        return 2

    child_execution = json.loads(child_execution_path.read_text(encoding="utf-8"))
    child_decision = json.loads(child_decision_path.read_text(encoding="utf-8"))
    runtime_weather_path = runtime_output / "runtime_weather_daily.csv"
    if not runtime_weather_path.is_file():
        raise FileNotFoundError(runtime_weather_path)
    shutil.copyfile(runtime_weather_path, output / "marker_runtime_weather_daily.csv")
    comparison_rows, mapping = analyze_mapping(
        marker_dates,
        changes,
        marker_wth,
        output / "marker_runtime_weather_daily.csv",
    )
    runtime_copy_hash = child_execution.get("runtime_wth_sha256_before")
    runtime_copy_identical = runtime_copy_hash == marker_hash
    clean_runtime = (
        child_execution.get("runtime_random_weather") is False
        and child_execution.get("runtime_WTHER") == "M"
        and child_execution.get("raw_reset_called") is False
        and mapping["runtime_duplicate_dates"] == 0
        and mapping["runtime_date_segment_count"] == 1
        and child_execution.get("runtime_done") is True
    )
    mapping_verified = (
        mapping["all_seven_dates_support_previous_source_day"]
        and runtime_copy_identical
        and clean_runtime
    )
    for row in comparison_rows:
        row["mapping_accepted"] = bool(mapping_verified and row.get("mapping_classification") == "PREVIOUS_SOURCE_DAY")
    smoke.write_csv(output / "marker_mapping_comparison.csv", comparison_rows)
    if mapping_verified:
        decision = {
            "date_dap_mapping_status": "RUNTIME_EMPIRICALLY_VERIFIED_FIXED_MINUS_ONE_DAY",
            "daily_weather_match_status": "MATCH_UNDER_VERIFIED_RUNTIME_DATE_MAPPING",
            "same_day_strict_comparison_status": child_decision.get("daily_weather_match_status", "MISMATCH"),
            "external_weather_injection_status": "RUNTIME_VERIFIED",
            "wth_copy_status": "BYTE_IDENTICAL",
            "wgen_bypass_status": "VERIFIED",
            "next_step": "EXTERNAL_AUGMENTATION_PILOT",
        }
    else:
        decision = {
            "date_dap_mapping_status": "UNRESOLVED_OR_INCONSISTENT",
            "daily_weather_match_status": "MISMATCH" if child_decision.get("daily_weather_match_status") == "MISMATCH" else "INSUFFICIENT",
            "same_day_strict_comparison_status": child_decision.get("daily_weather_match_status", "INSUFFICIENT"),
            "external_weather_injection_status": "PARTIALLY_VERIFIED",
            "wth_copy_status": "BYTE_IDENTICAL" if runtime_copy_identical else "MISMATCH",
            "wgen_bypass_status": "SUPPORTED",
            "next_step": "STOP_INSUFFICIENT",
        }
    decision.update({
        "marker_days_passed": mapping["marker_dates_all_classify_previous_source_day"],
        "neighbor_days_passed": mapping["all_four_unmodified_neighbors_match_previous_source_day"],
        "mapping_verified": mapping_verified,
        "offset_diagnostic_automatically_selected": False,
        "no_augmentation_started": True,
        "no_ppo_training": True,
        "no_wgen_run": True,
        "no_synthetic_weather_generation": True,
        "crop_outputs_classification": "NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT",
        "repeatability": "NOT_TESTED",
    })
    execution = {
        "task": "004_23B independent black-box date-semantics closure",
        "started_at": child_execution.get("started_at"),
        "finished_at": datetime.now().astimezone().isoformat(),
        "runtime_runner_returncode": completed.returncode,
        "command": command,
        "runtime_completed": child_execution.get("error") is None and child_execution.get("runtime_done") is True,
        "raw_reset_called": child_execution.get("raw_reset_called"),
        "initial_state_source": child_execution.get("initial_state_source"),
        "runtime_rows": mapping["runtime_rows"],
        "runtime_date_segment_count": mapping["runtime_date_segment_count"],
        "runtime_duplicate_dates": mapping["runtime_duplicate_dates"],
        "runtime_WTHER": child_execution.get("runtime_WTHER"),
        "runtime_random_weather": child_execution.get("runtime_random_weather"),
        "runtime_wth_path": child_execution.get("runtime_wth_path"),
        "runtime_wth_sha256": runtime_copy_hash,
        "marker_wth_sha256": marker_hash,
        "runtime_wth_byte_identical": runtime_copy_identical,
        "runtime_RSEED1": child_execution.get("runtime_rseed1_field"),
        "max_process_tree_rss_mb": child_execution.get("max_process_tree_rss_mb"),
        "same_day_strict_comparison": child_execution.get("comparison"),
        "marker_mapping": mapping,
        "date_semantics_conclusion": "source weather date = runtime YRDOY - 1 day" if mapping_verified else "not closed",
        "crop_outputs_classification": "NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT",
        "child_runtime_execution_record": str(child_execution_path),
        "child_runtime_decision": child_decision,
        "source_original_sha256_after_run": sha256_file(SOURCE_WTH),
        "original_wth_unchanged": sha256_file(SOURCE_WTH) == manifest["original_wth_sha256"],
    }
    write_json(output / "execution_record.json", execution)
    write_json(output / "decision.json", decision)
    REPORT_PATH.write_text(report_text(previous_report, manifest, changes, execution, decision), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "decision": decision, "marker_mapping": mapping, "runtime_returncode": completed.returncode}, ensure_ascii=False, indent=2))
    return 0 if mapping_verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
