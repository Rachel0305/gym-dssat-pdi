"""Weather augmentation utilities for 061_00 LC smoke training.

The generator creates a separate input root and never edits the original DSSAT
input folder. It copies LC lowIC inputs, then adds pseudo-year WTH files for
weather variants. Pseudo-years are used because the existing gym-DSSAT rendering
pipeline resolves WTH files by station/year stem (e.g., CNLC3501.WTH).
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_all_year_direct_action_safe_ppo as direct_ppo


STATION = "LCA"
SITE_SHORT = "LC"
WEATHER_PREFIX = "CNLC"


@dataclass(frozen=True)
class WeatherVariant:
    source_year: int
    pseudo_year: int
    variant: str
    rain_multiplier: float
    window: str
    source_weather: Path
    target_weather: Path


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def target_weather_name(year: int) -> str:
    return f"{WEATHER_PREFIX}{int(year) % 100:02d}01.WTH"


def planting_doy_for_lc() -> int:
    reps = direct_ppo.representative_phenology()
    return int(reps[STATION]["planting_doy"])


def window_doy_bounds(year: int, window: str) -> tuple[int | None, int | None]:
    if window == "none":
        return None, None
    planting = pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(days=planting_doy_for_lc() - 1)
    if window == "dap_1_30":
        start = planting
        end = planting + pd.Timedelta(days=29)
    elif window == "dap_31_60":
        start = planting + pd.Timedelta(days=30)
        end = planting + pd.Timedelta(days=59)
    else:
        raise ValueError(f"Unknown weather perturbation window: {window}")
    return int(start.dayofyear), int(end.dayofyear)


def build_variant_rows(cfg: dict[str, Any]) -> pd.DataFrame:
    aug = cfg["weather_augmentation"]
    source_root = ROOT / aug["source_input_root"]
    target_root = ROOT / aug["augmented_input_root"]
    rows = []
    for source_year in map(int, aug["train_years"]):
        for variant in aug["variants"]:
            name = str(variant["name"])
            offset = int(variant["pseudo_year_offset"])
            pseudo_year = int(source_year + offset)
            source_weather = source_root / SITE_SHORT / target_weather_name(source_year)
            target_weather = target_root / SITE_SHORT / target_weather_name(pseudo_year)
            rows.append(
                {
                    "station_code": STATION,
                    "site": SITE_SHORT,
                    "source_year": source_year,
                    "pseudo_year": pseudo_year,
                    "variant": name,
                    "split": "train",
                    "rain_multiplier": float(variant["rain_multiplier"]),
                    "window": str(variant["window"]),
                    "source_weather": rel(source_weather),
                    "target_weather": rel(target_weather),
                }
            )
    for year in map(int, aug["validation_years_original_only"]):
        source_weather = source_root / SITE_SHORT / target_weather_name(year)
        target_weather = target_root / SITE_SHORT / target_weather_name(year)
        rows.append(
            {
                "station_code": STATION,
                "site": SITE_SHORT,
                "source_year": year,
                "pseudo_year": year,
                "variant": "validation_original",
                "split": "validation",
                "rain_multiplier": 1.0,
                "window": "none",
                "source_weather": rel(source_weather),
                "target_weather": rel(target_weather),
            }
        )
    return pd.DataFrame(rows)


def perturb_wth_text(text: str, source_year: int, pseudo_year: int, rain_multiplier: float, window: str) -> tuple[str, dict[str, float]]:
    start_doy, end_doy = window_doy_bounds(source_year, window)
    changed_days = 0
    rain_before = 0.0
    rain_after = 0.0
    out_lines = []
    data_re = re.compile(r"^\s*(\d{4})(\d{3})\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s*$")
    for line in text.splitlines():
        match = data_re.match(line)
        if not match:
            out_lines.append(line)
            continue
        _yyyy, doy_text, srad_text, tmax_text, tmin_text, rain_text = match.groups()
        doy = int(doy_text)
        srad = float(srad_text)
        tmax = float(tmax_text)
        tmin = float(tmin_text)
        rain = float(rain_text)
        new_rain = rain
        in_window = start_doy is not None and end_doy is not None and start_doy <= doy <= end_doy
        if in_window:
            new_rain = max(0.0, rain * float(rain_multiplier))
            changed_days += 1
            rain_before += rain
            rain_after += new_rain
        out_lines.append(f"{int(pseudo_year):04d}{doy:03d}{srad:6.1f}{tmax:6.1f}{tmin:6.1f}{new_rain:6.1f}")
    return "\n".join(out_lines) + "\n", {
        "changed_days": float(changed_days),
        "window_rain_before_mm": float(rain_before),
        "window_rain_after_mm": float(rain_after),
        "window_rain_delta_mm": float(rain_after - rain_before),
    }


def prepare_augmented_inputs(cfg_path: Path, reuse_existing: bool = True) -> dict[str, Any]:
    cfg = read_config(cfg_path)
    aug = cfg["weather_augmentation"]
    source_root = ROOT / aug["source_input_root"]
    target_root = ROOT / aug["augmented_input_root"]
    source_lc = source_root / SITE_SHORT
    target_lc = target_root / SITE_SHORT
    if not source_lc.exists():
        raise FileNotFoundError(source_lc)
    if target_root.exists() and not reuse_existing:
        raise FileExistsError(f"Augmented root already exists and will not be overwritten: {rel(target_root)}")
    if not target_lc.exists():
        target_lc.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_lc, target_lc)

    mapping = build_variant_rows(cfg)
    generation_rows = []
    for row in mapping.itertuples(index=False):
        source_weather = ROOT / str(row.source_weather)
        target_weather = ROOT / str(row.target_weather)
        if not source_weather.exists():
            raise FileNotFoundError(source_weather)
        if str(row.variant) in {"original", "validation_original"}:
            if not target_weather.exists():
                shutil.copy2(source_weather, target_weather)
            stats = {
                "changed_days": 0.0,
                "window_rain_before_mm": 0.0,
                "window_rain_after_mm": 0.0,
                "window_rain_delta_mm": 0.0,
            }
        else:
            text = source_weather.read_text(encoding="utf-8", errors="replace")
            new_text, stats = perturb_wth_text(
                text,
                int(row.source_year),
                int(row.pseudo_year),
                float(row.rain_multiplier),
                str(row.window),
            )
            target_weather.write_text(new_text, encoding="utf-8")
        generation_rows.append({**row._asdict(), **stats})

    task_id = str(cfg.get("task_id", "061_00"))
    task_name = str(cfg.get("task_name", "lca_lowIC_weather_augmentation_smoke_maskableppo"))
    out_dir = ROOT / "benchmark_results" / f"{task_id}_{task_name}" / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = out_dir / "061_00_weather_variant_mapping.csv"
    pd.DataFrame(generation_rows).to_csv(mapping_path, index=False, encoding="utf-8-sig")
    return {
        "augmented_input_root": rel(target_root),
        "mapping_csv": rel(mapping_path),
        "train_variant_count": int((mapping["split"] == "train").sum()),
        "validation_original_count": int((mapping["split"] == "validation").sum()),
        "variants": sorted(mapping["variant"].unique().tolist()),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "061_00_lca_lowIC_weather_augmentation_smoke_maskableppo.json")
    parser.add_argument("--no-reuse-existing", action="store_true")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    print(json.dumps(prepare_augmented_inputs(cfg_path, reuse_existing=not args.no_reuse_existing), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
