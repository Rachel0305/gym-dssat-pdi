"""Build a traceable multi-site weather scenario bank for PPO training.

This module creates a new DSSAT input root and never edits the canonical input
root. Version 1 intentionally changes rainfall only, in crop-development
windows defined from station-level observed phenology. Temperature and solar
radiation are deferred to a separate factor so the first PPO comparison stays
interpretable.
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
CONFIG_DEFAULT = ROOT / "configs" / "160_multisite_weather_scenario_bank_v1.json"
DATA_RE = re.compile(
    r"^\s*(\d{7})\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*$"
)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def station_phenology(path: Path) -> dict[str, int]:
    grouped: dict[str, list[int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            station = str(row.get("station_code", "")).strip()
            value = str(row.get("planting_date", "")).strip()
            if not station or not value:
                continue
            year, month, day = map(int, value[:10].split("-"))
            grouped.setdefault(station, []).append(date(year, month, day).timetuple().tm_yday)
    expected = {"HLA", "SYA", "LCA", "YCA", "FQA"}
    missing = sorted(expected - set(grouped))
    if missing:
        raise ValueError(f"Missing phenology for stations: {missing}")
    return {station: int(round(median(values))) for station, values in grouped.items()}


def weather_name(prefix: str, year: int) -> str:
    return f"{prefix}{int(year) % 100:02d}01.WTH"


def window_bounds(planting_doy: int, window: str) -> tuple[int | None, int | None]:
    if window == "none":
        return None, None
    bounds = {
        "dap_1_30": (1, 30),
        "dap_31_60": (31, 60),
        "dap_61_90": (61, 90),
    }
    if window not in bounds:
        raise ValueError(f"Unknown window: {window}")
    start, end = bounds[window]
    return planting_doy + start - 1, planting_doy + end - 1


def perturb_weather(
    source: Path,
    target: Path,
    source_year: int,
    target_year: int,
    planting_doy: int,
    multiplier: float,
    window: str,
) -> dict[str, float | int | str]:
    start_doy, end_doy = window_bounds(planting_doy, window)
    changed_days = 0
    total_rain_before = 0.0
    total_rain_after = 0.0
    window_rain_before = 0.0
    window_rain_after = 0.0
    output: list[str] = []
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        match = DATA_RE.match(line)
        if not match:
            output.append(line)
            continue
        _date_text, srad_text, tmax_text, tmin_text, rain_text = match.groups()
        doy = int(_date_text[-3:])
        srad = float(srad_text)
        tmax = float(tmax_text)
        tmin = float(tmin_text)
        rain = max(0.0, float(rain_text))
        in_window = start_doy is not None and end_doy is not None and start_doy <= doy <= end_doy
        new_rain = max(0.0, rain * multiplier) if in_window else rain
        total_rain_before += rain
        total_rain_after += new_rain
        if in_window:
            changed_days += 1
            window_rain_before += rain
            window_rain_after += new_rain
        output.append(f"{int(target_year):04d}{doy:03d}{srad:7.1f}{tmax:7.1f}{tmin:7.1f}{new_rain:7.1f}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(output) + "\n", encoding="utf-8")
    return {
        "source_year": int(source_year),
        "pseudo_year": int(target_year),
        "changed_days": int(changed_days),
        "total_rain_before_mm": float(total_rain_before),
        "total_rain_after_mm": float(total_rain_after),
        "window_rain_before_mm": float(window_rain_before),
        "window_rain_after_mm": float(window_rain_after),
        "window_rain_delta_mm": float(window_rain_after - window_rain_before),
        "source_sha256": sha256(source),
        "target_sha256": sha256(target),
    }


def scenario_pseudo_year(cfg: dict[str, Any], source_year: int, variant: dict[str, Any], source_index: int) -> int:
    if variant["name"] == "original":
        return int(source_year)
    start = int(cfg["pseudo_year"]["start"])
    stride = int(cfg["pseudo_year"]["source_year_stride"])
    group = int(variant["pseudo_year_group"])
    return start + (group - 1) * stride + int(source_index)


def copy_input_root(source_root: Path, target_root: Path, stations: list[str], cfg: dict[str, Any]) -> None:
    if target_root.exists():
        raise FileExistsError(f"Refuse to overwrite existing scenario root: {rel(target_root)}")
    target_root.mkdir(parents=True, exist_ok=False)
    for station in stations:
        short = cfg["stations"][station]["short"]
        source_dir = source_root / short
        target_dir = target_root / short
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)
        shutil.copytree(source_dir, target_dir)


def build(cfg_path: Path, stations: list[str] | None = None) -> dict[str, Any]:
    cfg = load_json(cfg_path)
    selected = stations or list(cfg["stations"])
    unknown = sorted(set(selected) - set(cfg["stations"]))
    if unknown:
        raise ValueError(f"Unknown stations: {unknown}")
    source_root = ROOT / cfg["source_input_root"]
    target_root = ROOT / cfg["target_input_root"]
    manifest_root = ROOT / cfg["manifest_output_root"]
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    phenology = station_phenology(ROOT / cfg["phenology_csv"])
    copy_input_root(source_root, target_root, selected, cfg)
    rows: list[dict[str, Any]] = []
    variants = list(cfg["variants"])
    validation_years = [int(x) for x in cfg["validation_years"]]
    for station in selected:
        info = cfg["stations"][station]
        short = str(info["short"])
        prefix = str(info["weather_prefix"])
        train_years = [int(x) for x in cfg["train_years"][station]]
        target_dir = target_root / short
        for source_index, source_year in enumerate(train_years):
            source_weather = source_root / short / weather_name(prefix, source_year)
            if not source_weather.exists():
                raise FileNotFoundError(source_weather)
            for variant in variants:
                name = str(variant["name"])
                pseudo_year = scenario_pseudo_year(cfg, source_year, variant, source_index)
                target_weather = target_dir / weather_name(prefix, pseudo_year)
                if name == "original":
                    stats = {
                        "source_year": source_year,
                        "pseudo_year": pseudo_year,
                        "changed_days": 0,
                        "total_rain_before_mm": None,
                        "total_rain_after_mm": None,
                        "window_rain_before_mm": 0.0,
                        "window_rain_after_mm": 0.0,
                        "window_rain_delta_mm": 0.0,
                        "source_sha256": sha256(source_weather),
                        "target_sha256": sha256(target_weather),
                    }
                else:
                    stats = perturb_weather(
                        source=source_weather,
                        target=target_weather,
                        source_year=source_year,
                        target_year=pseudo_year,
                        planting_doy=phenology[station],
                        multiplier=float(variant["rain_multiplier"]),
                        window=str(variant["window"]),
                    )
                rows.append(
                    {
                        "station_code": station,
                        "site": short,
                        "source_year": source_year,
                        "pseudo_year": pseudo_year,
                        "variant": name,
                        "split": "train",
                        "planting_doy": phenology[station],
                        "rain_multiplier": float(variant["rain_multiplier"]),
                        "window": str(variant["window"]),
                        "source_weather": rel(source_weather),
                        "target_weather": rel(target_weather),
                        **stats,
                    }
                )
        for year in validation_years:
            source_weather = source_root / short / weather_name(prefix, year)
            target_weather = target_root / short / weather_name(prefix, year)
            if not source_weather.exists() or not target_weather.exists():
                raise FileNotFoundError(f"Missing validation weather for {station} {year}")
            rows.append(
                {
                    "station_code": station,
                    "site": short,
                    "source_year": year,
                    "pseudo_year": year,
                    "variant": "validation_original",
                    "split": "validation",
                    "planting_doy": phenology[station],
                    "rain_multiplier": 1.0,
                    "window": "none",
                    "source_weather": rel(source_weather),
                    "target_weather": rel(target_weather),
                    "changed_days": 0,
                    "source_sha256": sha256(source_weather),
                    "target_sha256": sha256(target_weather),
                }
            )
    if manifest_root.exists():
        raise FileExistsError(f"Refuse to overwrite existing manifest root: {rel(manifest_root)}")
    manifest_root.mkdir(parents=True, exist_ok=False)
    manifest_path = manifest_root / "160_weather_scenario_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "completed_generation_only",
        "config": rel(cfg_path),
        "source_input_root": rel(source_root),
        "target_input_root": rel(target_root),
        "manifest_csv": rel(manifest_path),
        "stations": selected,
        "train_rows": sum(row["split"] == "train" for row in rows),
        "validation_rows": sum(row["split"] == "validation" for row in rows),
        "train_variants": sorted({row["variant"] for row in rows if row["split"] == "train"}),
        "validation_variants": sorted({row["variant"] for row in rows if row["split"] == "validation"}),
        "source_weather_modified": False,
        "cultivar_soil_mzx_modified": False,
        "temperature_perturbation": False,
    }
    (manifest_root / "160_generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--stations", default="", help="Comma-separated station codes; default is all five")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else (ROOT / args.config)
    stations = [x.strip() for x in args.stations.split(",") if x.strip()] or None
    print(json.dumps(build(cfg_path.resolve(), stations), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
