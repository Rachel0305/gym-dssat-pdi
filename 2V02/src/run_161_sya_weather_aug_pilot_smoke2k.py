"""161: SYA 2005 six-scenario weather-augmentation PPO pilot.

This runner consumes the already-generated 160 scenario bank. It does not
generate or overwrite weather inputs. The PPO core, reward, safety rules, and
action grid are inherited from the established SYA 046_02/046_10 framework;
only the training weather cases and isolated input root are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_all_year_direct_action_safe_ppo as direct_ppo


TASK_ID = "161"
TASK_NAME = "sya_weather_aug_pilot_smoke2k"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
PROMPT = ROOT / "prompts" / "2026-08-18_sya_weather_aug_pilot_smoke2k.md"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
STATION = "SYA"
SITE = "SY"
EXPECTED_VARIANTS = ["original", "early_dry", "mid_dry", "late_dry", "early_wet", "mid_wet"]
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


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("station_code") != STATION or cfg.get("site") != SITE:
        raise ValueError("161 pilot is fixed to SYA/SY")
    if int(cfg.get("source_year")) != 2005:
        raise ValueError("161 pilot is fixed to source year 2005")
    return cfg


def representative_planting_doy() -> int:
    return int(direct_ppo.representative_phenology()[STATION]["planting_doy"])


def target_root(cfg: dict[str, Any]) -> Path:
    return ROOT / str(cfg["input_root"])


def source_root(cfg: dict[str, Any]) -> Path:
    return ROOT / str(cfg["source_input_root"])


def weather_name(year: int) -> str:
    return f"CNSY{int(year) % 100:02d}01.WTH"


def read_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / str(cfg["source_manifest"])
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = pd.read_csv(path, keep_default_na=False)
    manifest["source_year"] = pd.to_numeric(manifest["source_year"], errors="coerce").astype(int)
    manifest["pseudo_year"] = pd.to_numeric(manifest["pseudo_year"], errors="coerce").astype(int)
    selected_train = manifest[
        manifest["station_code"].eq(STATION)
        & manifest["split"].eq("train")
        & manifest["source_year"].eq(int(cfg["source_year"]))
    ].copy()
    selected_validation = manifest[
        manifest["station_code"].eq(STATION)
        & manifest["split"].eq("validation")
        & manifest["source_year"].isin(list(map(int, cfg["validation_years"])))
    ].copy()
    selected = pd.concat([selected_train, selected_validation], ignore_index=True)
    if selected_train["variant"].tolist() != EXPECTED_VARIANTS:
        selected_train = selected_train.set_index("variant").loc[EXPECTED_VARIANTS].reset_index()
        selected = pd.concat([selected_train, selected_validation], ignore_index=True)
    return selected.reset_index(drop=True)


def parse_weather(path: Path) -> dict[int, tuple[float, float, float, float]]:
    rows: dict[int, tuple[float, float, float, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = DATA_RE.match(line)
        if not match:
            continue
        date_text, srad, tmax, tmin, rain = match.groups()
        rows[int(date_text[-3:])] = (float(srad), float(tmax), float(tmin), float(rain))
    if len(rows) < 300:
        raise ValueError(f"Weather file has too few parsed daily rows: {path} ({len(rows)})")
    return rows


def expected_window(planting_doy: int, window: str) -> tuple[int | None, int | None]:
    if window == "none":
        return None, None
    bounds = {"dap_1_30": (1, 30), "dap_31_60": (31, 60), "dap_61_90": (61, 90)}
    if window not in bounds:
        raise ValueError(f"Unknown perturbation window: {window}")
    start, end = bounds[window]
    return planting_doy + start - 1, planting_doy + end - 1


def compare_weather(cfg: dict[str, Any], row: pd.Series) -> dict[str, Any]:
    src = source_root(cfg) / "SY" / weather_name(int(row["source_year"]))
    dst = target_root(cfg) / "SY" / weather_name(int(row["pseudo_year"]))
    if not src.exists() or not dst.exists():
        raise FileNotFoundError(f"Missing weather source/target: {src} / {dst}")
    source_data = parse_weather(src)
    target_data = parse_weather(dst)
    all_doys = sorted(set(source_data) & set(target_data))
    if len(all_doys) < 300:
        raise ValueError(f"Insufficient shared DOYs for {row['variant']}: {len(all_doys)}")
    start, end = expected_window(representative_planting_doy(), str(row["window"]))
    changed_rain = 0
    unchanged_weather_rows = 0
    ratio_errors = 0
    nonwindow_rain_changes = 0
    factor = float(row["rain_multiplier"])
    for doy in all_doys:
        srad0, tmax0, tmin0, rain0 = source_data[doy]
        srad1, tmax1, tmin1, rain1 = target_data[doy]
        if not (np.isclose(srad0, srad1) and np.isclose(tmax0, tmax1) and np.isclose(tmin0, tmin1)):
            unchanged_weather_rows += 1
        in_window = start is not None and end is not None and start <= doy <= end
        if in_window:
            if not np.isclose(rain1, max(0.0, rain0 * factor), atol=0.11):
                ratio_errors += 1
            if not np.isclose(rain0, rain1):
                changed_rain += 1
        elif not np.isclose(rain0, rain1):
            nonwindow_rain_changes += 1
    return {
        "variant": str(row["variant"]),
        "source_weather": rel(src),
        "target_weather": rel(dst),
        "source_sha256": sha256(src),
        "target_sha256": sha256(dst),
        "changed_rain_rows": changed_rain,
        "nonwindow_rain_changes": nonwindow_rain_changes,
        "temperature_or_radiation_mismatch_rows": unchanged_weather_rows,
        "rain_ratio_errors": ratio_errors,
        "source_total_rain_mm": float(sum(v[3] for v in source_data.values())),
        "target_total_rain_mm": float(sum(v[3] for v in target_data.values())),
        "passed": bool(unchanged_weather_rows == 0 and nonwindow_rain_changes == 0 and ratio_errors == 0),
    }


def static_input_audit(cfg: dict[str, Any]) -> dict[str, Any]:
    source_dir = source_root(cfg) / "SY"
    target_dir = target_root(cfg) / "SY"
    if not source_dir.exists() or not target_dir.exists():
        raise FileNotFoundError(f"Missing static input directory: {source_dir} / {target_dir}")
    source_files = {p.relative_to(source_dir).as_posix(): p for p in source_dir.rglob("*") if p.is_file() and p.suffix.upper() != ".WTH"}
    target_files = {p.relative_to(target_dir).as_posix(): p for p in target_dir.rglob("*") if p.is_file() and p.suffix.upper() != ".WTH"}
    missing = sorted(set(source_files) - set(target_files))
    extra = sorted(set(target_files) - set(source_files))
    mismatched = sorted(name for name in set(source_files) & set(target_files) if sha256(source_files[name]) != sha256(target_files[name]))
    return {
        "source_static_file_count": len(source_files),
        "target_static_file_count": len(target_files),
        "missing_static_files": missing,
        "extra_static_files": extra,
        "mismatched_static_files": mismatched,
        "passed": not missing and not extra and not mismatched,
    }


def preflight(cfg: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest = read_manifest(cfg)
    train = manifest[manifest["split"].eq("train")].copy()
    validation = manifest[manifest["split"].eq("validation")].copy()
    issues: list[str] = []
    if train["variant"].tolist() != EXPECTED_VARIANTS:
        issues.append(f"training variants mismatch: {train['variant'].tolist()}")
    if len(validation) != len(cfg["validation_years"]):
        issues.append(f"validation row count mismatch: {len(validation)}")
    weather_audit = pd.DataFrame([compare_weather(cfg, row) for _, row in manifest.iterrows()])
    if not weather_audit["passed"].all():
        issues.append("weather-only perturbation audit failed")
    static_audit = static_input_audit(cfg)
    if not static_audit["passed"]:
        issues.append("static input audit failed")
    summary = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "station_code": STATION,
        "source_year": int(cfg["source_year"]),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_variants": train["variant"].tolist(),
        "validation_years": validation["source_year"].astype(int).tolist(),
        "weather_audit": weather_audit.to_dict(orient="records"),
        "static_input_audit": static_audit,
        "issues": issues,
        "next_step_allowed": not issues,
    }
    return summary, manifest, weather_audit


def make_split(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        pseudo_year = int(row.pseudo_year)
        rows.append(
            {
                "station_code": STATION,
                "site": SITE,
                "station_name": SITE,
                "source_year": int(row.source_year),
                "year": pseudo_year,
                "split": str(row.split),
                "weather_variant": str(row.variant),
                "scenario_type": f"161_{row.variant}",
                "weather_file": weather_name(pseudo_year),
                "target_weather": str(row.target_weather),
                "selected_for_train": str(row.split) == "train",
                "selected_for_eval": str(row.split) == "validation",
                "selection_reason": "161_sya_2005_weather_aug_pilot",
                "rain_multiplier": float(row.rain_multiplier),
                "perturbation_window": str(row.window),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "year"]).reset_index(drop=True)


def write_record(cfg: dict[str, Any], preflight_result: dict[str, Any], status: str, error: str = "") -> None:
    lines = [
        f"# {TASK_ID} SYA weather augmentation pilot smoke",
        "",
        f"- Status: `{status}`.",
        f"- Training: SYA 2005 with six existing weather scenarios; timesteps={cfg['training']['total_timesteps']}.",
        f"- Validation: original SYA {cfg['validation_years']} only; not formal generalization validation.",
        "- Controlled change: weather input root and training scenario mapping only.",
        "- Unchanged: reward, action safety, observation contract, cultivar, soil, MZX, and PPO action grid.",
        "",
        "## Preflight",
        "",
        "```json",
        json.dumps(preflight_result, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    if error:
        lines.extend(["## Error", "", "```text", error, "```", ""])
    DOC.write_text("\n".join(lines), encoding="utf-8")


def run(cfg_path: Path, dry_run: bool = False) -> dict[str, Any]:
    cfg = load_config(cfg_path)
    preflight_result, manifest, weather_audit = preflight(cfg)
    if dry_run:
        return {**preflight_result, "mode": "dry_run"}
    if not preflight_result["next_step_allowed"]:
        raise RuntimeError("161 preflight failed: " + "; ".join(preflight_result["issues"]))
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing 161 output: {rel(OUT)}")
    for sub in ["configs", "evaluation", "logs", "models", "daily_outputs", "audits"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, OUT / "configs" / cfg_path.name)
    shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)
    (OUT / "configs" / "161_preflight.json").write_text(json.dumps(preflight_result, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest.to_csv(OUT / "configs" / "161_selected_manifest.csv", index=False, encoding="utf-8-sig")
    weather_audit.to_csv(OUT / "audits" / "161_weather_input_audit.csv", index=False, encoding="utf-8-sig")
    split = make_split(manifest)
    split.to_csv(OUT / "configs" / "161_training_validation_split.csv", index=False, encoding="utf-8-sig")

    old_engine = {
        "TASK_ID": engine.TASK_ID,
        "TASK_NAME": engine.TASK_NAME,
        "BASE_OUT": engine.BASE_OUT,
        "BASE_DOC": engine.BASE_DOC,
        "PROMPT": engine.PROMPT,
        "LOWIC_INPUT_ROOT": engine.LOWIC_INPUT_ROOT,
        "STATION": engine.STATION,
        "SITES": list(engine.SITES),
        "BINARY_IRRIGATION_LEVELS": list(engine.BINARY_IRRIGATION_LEVELS),
        "BINARY_NITROGEN_LEVELS": list(engine.BINARY_NITROGEN_LEVELS),
    }
    old_base = {
        "SITES": list(base03222.SITES),
        "SITE_NAMES": dict(base03222.SITE_NAMES),
        "OUT": base03222.OUT,
        "DOC": base03222.DOC,
        "PROMPT": base03222.PROMPT,
        "TOTAL_TIMESTEPS": base03222.TOTAL_TIMESTEPS,
        "CHECKPOINT_STEPS": list(base03222.CHECKPOINT_STEPS),
        "load_split": base03222.load_split,
        "build_selection": base03222.build_selection,
    }
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    status = "completed"
    error = ""
    try:
        engine.TASK_ID = TASK_ID
        engine.TASK_NAME = TASK_NAME
        engine.BASE_OUT = OUT
        engine.BASE_DOC = DOC
        engine.PROMPT = PROMPT
        engine.LOWIC_INPUT_ROOT = target_root(cfg)
        engine.STATION = STATION
        engine.SITES = [STATION]
        engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        base03222.SITES = [STATION]
        base03222.SITE_NAMES.clear()
        base03222.SITE_NAMES.update({STATION: SITE})
        base03222.load_split = lambda: split.copy()
        base03222.build_selection = lambda _split: split.copy()
        engine.patch_base_module(OUT, DOC, int(cfg["training"]["total_timesteps"]), list(map(int, cfg["training"]["checkpoint_steps"])))
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = target_root(cfg)
        engine.run_training(int(cfg["training"]["total_timesteps"]), list(map(int, cfg["training"]["checkpoint_steps"])), suffix="")
    except Exception:
        status = "failed"
        error = traceback.format_exc()
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
        for key, value in old_engine.items():
            setattr(engine, key, value)
        for key, value in old_base.items():
            if key == "SITE_NAMES":
                base03222.SITE_NAMES.clear()
                base03222.SITE_NAMES.update(value)
            else:
                setattr(base03222, key, value)
    write_record(cfg, preflight_result, status, error)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": status,
        "output_root": rel(OUT),
        "record_md": rel(DOC),
        "preflight": rel(OUT / "configs" / "161_preflight.json"),
        "weather_audit": rel(OUT / "audits" / "161_weather_input_audit.csv"),
        "split": rel(OUT / "configs" / "161_training_validation_split.csv"),
        "error": error[-5000:] if error else "",
    }
    (OUT / "161_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if status != "completed":
        raise RuntimeError(error)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else (ROOT / args.config).resolve()
    print(json.dumps(run(cfg_path, dry_run=args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
