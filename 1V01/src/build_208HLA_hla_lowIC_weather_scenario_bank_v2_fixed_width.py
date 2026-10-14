"""Build the corrected HLA 10-year x 6-weather scenario bank.

Unlike 202HLA/v1, v2 writes DSSAT WTH data rows using the source-compatible
fixed-width contract: DATE width 7 followed by four fields of width 6.  It
also maps the 50 augmented scenarios to 1950--1999 to avoid future-century
ambiguity in YYDDD experiment dates.  Original inputs are copied, never edited.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "208HLA"
TASK_NAME = "hla_lowIC_weather_scenario_bank_v2_fixed_width"
CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
RECORD = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
DATA_RE = re.compile(
    r"^\s*(\d{7})\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*$"
)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weather_name(prefix: str, year: int) -> str:
    return f"{prefix}{int(year) % 100:02d}01.WTH"


def window_bounds(planting_doy: int, window: str) -> tuple[int | None, int | None]:
    if window == "none":
        return None, None
    mapping = {"dap_1_30": (1, 30), "dap_31_60": (31, 60), "dap_61_90": (61, 90)}
    start, end = mapping[window]
    return planting_doy + start - 1, planting_doy + end - 1


def fixed_width_weather(
    source: Path,
    target: Path,
    source_year: int,
    target_year: int,
    planting_doy: int,
    multiplier: float,
    window: str,
) -> dict[str, Any]:
    start_doy, end_doy = window_bounds(planting_doy, window)
    output: list[str] = []
    changed_days = 0
    total_before = total_after = window_before = window_after = 0.0
    data_widths: set[int] = set()
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        match = DATA_RE.match(line)
        if not match:
            output.append(line)
            continue
        date_text, srad_text, tmax_text, tmin_text, rain_text = match.groups()
        doy = int(date_text[-3:])
        srad, tmax, tmin, rain = map(float, [srad_text, tmax_text, tmin_text, rain_text])
        rain = max(0.0, rain)
        in_window = start_doy is not None and end_doy is not None and start_doy <= doy <= end_doy
        new_rain = rain * multiplier if in_window else rain
        total_before += rain
        total_after += new_rain
        if in_window:
            changed_days += 1
            window_before += rain
            window_after += new_rain
        rendered = f"{target_year:04d}{doy:03d}{srad:6.1f}{tmax:6.1f}{tmin:6.1f}{new_rain:6.1f}"
        data_widths.add(len(rendered))
        output.append(rendered)
    if data_widths != {31}:
        raise RuntimeError(f"DSSAT WTH width violation for {target}: {sorted(data_widths)}")
    target.write_text("\n".join(output) + "\n", encoding="utf-8")
    return {
        "source_year": source_year,
        "pseudo_year": target_year,
        "changed_days": changed_days,
        "total_rain_before_mm": total_before,
        "total_rain_after_mm": total_after,
        "window_rain_before_mm": window_before,
        "window_rain_after_mm": window_after,
        "window_rain_delta_mm": window_after - window_before,
        "data_line_width": 31,
        "source_sha256": sha256(source),
        "target_sha256": sha256(target),
    }


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_root = ROOT / cfg["source_input_root"]
    target_root = ROOT / cfg["target_input_root"]
    if target_root.exists() or OUT.exists():
        raise FileExistsError("208HLA refuses to overwrite an existing input or result root")
    source_dir = source_root / "HL"
    target_dir = target_root / "HL"
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    shutil.copytree(source_root, target_root)
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    shutil.copy2(CONFIG, OUT / "configs" / CONFIG.name)

    source_static = {
        path.relative_to(source_dir).as_posix(): sha256(path)
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.upper() != ".WTH"
    }
    rows: list[dict[str, Any]] = []
    for index, source_year in enumerate(map(int, cfg["train_years"])):
        source_weather = source_dir / weather_name(cfg["weather_prefix"], source_year)
        for variant in cfg["variants"]:
            name = str(variant["name"])
            target_year = source_year if name == "original" else int(variant["target_start_year"]) + index
            target_weather = target_dir / weather_name(cfg["weather_prefix"], target_year)
            if name == "original":
                stats = {
                    "source_year": source_year,
                    "pseudo_year": target_year,
                    "changed_days": 0,
                    "total_rain_before_mm": "",
                    "total_rain_after_mm": "",
                    "window_rain_before_mm": 0.0,
                    "window_rain_after_mm": 0.0,
                    "window_rain_delta_mm": 0.0,
                    "data_line_width": 31,
                    "source_sha256": sha256(source_weather),
                    "target_sha256": sha256(target_weather),
                }
            else:
                stats = fixed_width_weather(
                    source_weather,
                    target_weather,
                    source_year,
                    target_year,
                    int(cfg["planting_doy"]),
                    float(variant["rain_multiplier"]),
                    str(variant["window"]),
                )
            rows.append(
                {
                    "station_code": "HLA",
                    "site": "HL",
                    "source_year": source_year,
                    "pseudo_year": target_year,
                    "variant": name,
                    "split": "train",
                    "planting_doy": int(cfg["planting_doy"]),
                    "rain_multiplier": float(variant["rain_multiplier"]),
                    "window": str(variant["window"]),
                    "source_weather": rel(source_weather),
                    "target_weather": rel(target_weather),
                    **stats,
                }
            )
    for year in map(int, cfg["validation_years"]):
        source_weather = source_dir / weather_name(cfg["weather_prefix"], year)
        target_weather = target_dir / weather_name(cfg["weather_prefix"], year)
        rows.append(
            {
                "station_code": "HLA", "site": "HL", "source_year": year, "pseudo_year": year,
                "variant": "validation_original", "split": "validation", "planting_doy": int(cfg["planting_doy"]),
                "rain_multiplier": 1.0, "window": "none", "source_weather": rel(source_weather),
                "target_weather": rel(target_weather), "changed_days": 0, "total_rain_before_mm": "",
                "total_rain_after_mm": "", "window_rain_before_mm": 0.0, "window_rain_after_mm": 0.0,
                "window_rain_delta_mm": 0.0, "data_line_width": 31, "source_sha256": sha256(source_weather),
                "target_sha256": sha256(target_weather),
            }
        )

    manifest = OUT / f"{TASK_ID}_weather_scenario_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    target_static = {
        path.relative_to(target_dir).as_posix(): sha256(path)
        for path in target_dir.rglob("*")
        if path.is_file() and path.suffix.upper() != ".WTH"
    }
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    issues: list[str] = []
    if len(train) != 60 or len({row["pseudo_year"] for row in train}) != 60:
        issues.append("training scenario count/year uniqueness failed")
    if any(int(row["data_line_width"]) != 31 for row in train):
        issues.append("fixed-width WTH contract failed")
    if any(row["source_sha256"] != row["target_sha256"] for row in validation):
        issues.append("validation WTH changed")
    if source_static != target_static:
        issues.append("static input hashes changed")
    summary = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": "completed_generation_and_audit" if not issues else "failed_postflight",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "source_years": len({row["source_year"] for row in train}),
        "variants": len({row["variant"] for row in train}),
        "unique_training_years": len({row["pseudo_year"] for row in train}),
        "augmented_year_range": [min(row["pseudo_year"] for row in train if row["variant"] != "original"), max(row["pseudo_year"] for row in train if row["variant"] != "original")],
        "dssat_data_line_width": 31,
        "source_weather_modified": False,
        "validation_weather_hash_mismatch_rows": sum(row["source_sha256"] != row["target_sha256"] for row in validation),
        "static_hash_match": source_static == target_static,
        "issues": issues,
        "physical_gate_required_before_training": True,
        "manifest": rel(manifest),
    }
    (OUT / f"{TASK_ID}_generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    record = [
        "# 208HLA HLA lowIC 十年六天气 v2 固定宽度情景库",
        "",
        "## 修复",
        "",
        "- v1 将 WTH 的 SRAD/TMAX/TMIN/RAIN 误写为每列宽度 7；DSSAT 原输入契约为每列宽度 6。",
        "- v2 强制 `DATE(7)+4×weather(6)=31` 字符；只改变训练 RAIN。",
        "- 50 个增强情景映射到 1950--1999，10 个 original 保持 2004--2013；验证 2014--2023 原样复制。",
        "- 新输入根独立生成；原 WTH、MZX、SOL、CUL 均未修改。",
        "",
        "## 生成审计",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 下一门槛",
        "",
        "- 训练前必须用 original、无扰动克隆、dry、wet 做 DSSAT 物理响应审计；门槛通过后才允许 2K smoke。",
    ]
    RECORD.write_text("\n".join(record) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if issues:
        raise RuntimeError(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
