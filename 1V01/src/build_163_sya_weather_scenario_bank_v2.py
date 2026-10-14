"""Build the SYA 2005 weather-only scenario bank without pseudo-years.

Each scenario gets an independent DSSAT input root.  The rendered experiment
still uses simulation year 2005 and every scenario weather file is named
``CNSY0501.WTH`` with 2005 date rows.  Only RAIN is perturbed in the selected
DAP window; source weather and static input files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import date
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "163_sya_weather_scenario_bank_v2.json"
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def planting_doy(path: Path, station: str) -> int:
    values: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("station_code", "")).strip() != station:
                continue
            value = str(row.get("planting_date", "")).strip()
            if value:
                year, month, day = map(int, value[:10].split("-"))
                values.append(date(year, month, day).timetuple().tm_yday)
    if not values:
        raise ValueError(f"No planting dates found for {station}: {path}")
    return int(round(median(values)))


def weather_name(prefix: str, year: int) -> str:
    return f"{prefix}{int(year) % 100:02d}01.WTH"


def window_bounds(planting: int, window: str) -> tuple[int | None, int | None]:
    if window == "none":
        return None, None
    bounds = {"dap_1_30": (1, 30), "dap_31_60": (31, 60), "dap_61_90": (61, 90)}
    if window not in bounds:
        raise ValueError(f"Unknown window: {window}")
    start, end = bounds[window]
    return planting + start - 1, planting + end - 1


def copy_static_inputs(source_dir: Path, target_dir: Path) -> None:
    """Copy only non-weather inputs so a scenario cannot silently select another year."""
    target_dir.mkdir(parents=True, exist_ok=False)
    allowed_suffixes = {".MZX", ".MZA", ".MZT", ".CUL", ".SOL", ".CLI", ".PRM", ".WDB"}
    for source in sorted(source_dir.iterdir()):
        if source.is_dir() or source.suffix.upper() == ".WTH":
            continue
        if source.suffix.upper() not in allowed_suffixes:
            continue
        target = target_dir / source.name
        shutil.copy2(source, target)


def write_variant_weather(
    source: Path,
    target: Path,
    source_year: int,
    planting: int,
    multiplier: float,
    window: str,
) -> dict[str, Any]:
    start, end = window_bounds(planting, window)
    changed_rows = 0
    total_before = 0.0
    total_after = 0.0
    window_before = 0.0
    window_after = 0.0
    output: list[str] = []
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        match = DATA_RE.match(line)
        if not match:
            output.append(line)
            continue
        date_text, srad_text, tmax_text, tmin_text, rain_text = match.groups()
        doy = int(date_text[-3:])
        srad = float(srad_text)
        tmax = float(tmax_text)
        tmin = float(tmin_text)
        rain = max(0.0, float(rain_text))
        in_window = start is not None and end is not None and start <= doy <= end
        new_rain = max(0.0, rain * multiplier) if in_window else rain
        total_before += rain
        total_after += new_rain
        if in_window:
            changed_rows += 1
            window_before += rain
            window_after += new_rain
        # DSSAT's WTH daily records are fixed-width: DATE(7) plus four
        # meteorological fields of width 6.  Preserve this contract exactly;
        # reformatting to width 7 makes the file look readable but can shift
        # columns and cause DSSAT to ignore the weather record.
        output.append(f"{int(source_year):04d}{doy:03d}{srad:6.1f}{tmax:6.1f}{tmin:6.1f}{new_rain:6.1f}")
    target.write_text("\n".join(output) + "\n", encoding="utf-8")
    return {
        "source_year": int(source_year),
        "simulation_year": int(source_year),
        "weather_date_year": int(source_year),
        "changed_days": int(changed_rows),
        "total_rain_before_mm": float(total_before),
        "total_rain_after_mm": float(total_after),
        "window_rain_before_mm": float(window_before),
        "window_rain_after_mm": float(window_after),
        "window_rain_delta_mm": float(window_after - window_before),
        "source_sha256": sha256(source),
        "target_sha256": sha256(target),
    }


def build(config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    station = str(cfg["station_code"])
    short = str(cfg["site"])
    prefix = str(cfg["weather_prefix"])
    source_year = int(cfg["source_year"])
    source_root = ROOT / str(cfg["source_input_root"])
    target_root = ROOT / str(cfg["target_input_root"])
    manifest_root = ROOT / str(cfg["manifest_output_root"])
    source_dir = source_root / short
    source_weather = source_dir / weather_name(prefix, source_year)
    if not source_dir.exists() or not source_weather.exists():
        raise FileNotFoundError(f"Missing source inputs: {source_dir} / {source_weather}")
    if target_root.exists() or manifest_root.exists():
        raise FileExistsError("Refuse to overwrite an existing v2 scenario bank or manifest root")
    planting = planting_doy(ROOT / str(cfg["phenology_csv"]), station)
    target_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for variant in cfg["variants"]:
        name = str(variant["name"])
        scenario_root = target_root / name
        scenario_site_dir = scenario_root / short
        copy_static_inputs(source_dir, scenario_site_dir)
        target_weather = scenario_site_dir / weather_name(prefix, source_year)
        stats = write_variant_weather(
            source=source_weather,
            target=target_weather,
            source_year=source_year,
            planting=planting,
            multiplier=float(variant["rain_multiplier"]),
            window=str(variant["window"]),
        )
        rows.append(
            {
                "station_code": station,
                "site": short,
                "variant": name,
                "source_year": source_year,
                "simulation_year": source_year,
                "weather_date_year": source_year,
                "planting_doy": planting,
                "rain_multiplier": float(variant["rain_multiplier"]),
                "window": str(variant["window"]),
                "scenario_input_root": rel(scenario_root),
                "source_weather": rel(source_weather),
                "target_weather": rel(target_weather),
                **stats,
            }
        )
    manifest_root.mkdir(parents=True, exist_ok=False)
    manifest = manifest_root / "163_sya_weather_scenario_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "completed_generation_only",
        "config": rel(config_path),
        "station_code": station,
        "site": short,
        "source_year": source_year,
        "simulation_years": sorted({row["simulation_year"] for row in rows}),
        "weather_date_years": sorted({row["weather_date_year"] for row in rows}),
        "variants": [row["variant"] for row in rows],
        "target_input_root": rel(target_root),
        "manifest_csv": rel(manifest),
        "source_weather_modified": False,
        "static_inputs_modified": False,
        "only_rain_changed": True,
    }
    (manifest_root / "163_generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    print(json.dumps(build(config.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
