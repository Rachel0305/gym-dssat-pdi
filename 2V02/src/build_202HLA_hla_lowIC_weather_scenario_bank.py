"""Build the isolated HLA lowIC 10-year x 6-weather scenario bank.

This is a thin, auditable wrapper around ``weather_scenario_bank_160.py``.
It adds HLA/lowIC-specific collision and provenance checks, while leaving the
shared weather-generation implementation unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import weather_scenario_bank_160 as shared


TASK_ID = "202HLA"
TASK_NAME = "hla_lowIC_weather_scenario_bank_v1"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
RECORD = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
EXPECTED_VARIANTS = ["original", "early_dry", "mid_dry", "late_dry", "early_wet", "mid_wet"]


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
    if cfg.get("task_id") != TASK_ID or cfg.get("task_name") != TASK_NAME:
        raise ValueError("202HLA task identity mismatch")
    if list(cfg.get("stations", {})) != ["HLA"]:
        raise ValueError("202HLA is an HLA-only scenario-bank task")
    if cfg.get("source_input_root") != "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual":
        raise ValueError("202HLA must reuse the exact 054_00 lowIC input root")
    variants = [str(item.get("name")) for item in cfg.get("variants", [])]
    if variants != EXPECTED_VARIANTS:
        raise ValueError(f"weather variants mismatch: {variants}")
    return cfg


def planned_rows(cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    years = list(map(int, cfg["train_years"]["HLA"]))
    for source_index, source_year in enumerate(years):
        for variant in cfg["variants"]:
            pseudo_year = shared.scenario_pseudo_year(cfg, source_year, variant, source_index)
            rows.append(
                {
                    "source_year": source_year,
                    "pseudo_year": int(pseudo_year),
                    "weather_file": shared.weather_name("CNHL", int(pseudo_year)),
                    "variant": str(variant["name"]),
                }
            )
    return pd.DataFrame(rows)


def preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    source_root = ROOT / cfg["source_input_root"]
    target_root = ROOT / cfg["target_input_root"]
    manifest_root = ROOT / cfg["manifest_output_root"]
    plan = planned_rows(cfg)
    train_years = list(map(int, cfg["train_years"]["HLA"]))
    validation_years = list(map(int, cfg["validation_years"]))
    expected_files = [shared.weather_name("CNHL", year) for year in train_years + validation_years]
    source_weather = [source_root / "HL" / name for name in expected_files]
    duplicate_pseudo_years = sorted(plan.loc[plan["pseudo_year"].duplicated(keep=False), "pseudo_year"].unique().tolist())
    duplicate_target_names = sorted(plan.loc[plan["weather_file"].duplicated(keep=False), "weather_file"].unique().tolist())
    validation_names = {shared.weather_name("CNHL", year) for year in validation_years}
    augmented_validation_collisions = sorted(set(plan["weather_file"]) & validation_names)
    issues: list[str] = []
    if not source_root.exists():
        issues.append(f"source input root missing: {rel(source_root)}")
    missing_weather = [rel(path) for path in source_weather if not path.exists()]
    if missing_weather:
        issues.append(f"missing source weather files: {missing_weather}")
    if len(plan) != 60:
        issues.append(f"planned training rows must be 60, got {len(plan)}")
    if duplicate_pseudo_years:
        issues.append(f"duplicate pseudo years: {duplicate_pseudo_years}")
    if duplicate_target_names:
        issues.append(f"duplicate target WTH filenames: {duplicate_target_names}")
    if augmented_validation_collisions:
        issues.append(f"augmented/validation WTH filename collisions: {augmented_validation_collisions}")
    if target_root.exists():
        issues.append(f"target input root already exists: {rel(target_root)}")
    if manifest_root.exists():
        issues.append(f"manifest output root already exists: {rel(manifest_root)}")
    return {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "source_input_root": rel(source_root),
        "target_input_root": rel(target_root),
        "manifest_output_root": rel(manifest_root),
        "train_source_years": train_years,
        "validation_original_years": validation_years,
        "planned_training_rows": int(len(plan)),
        "planned_variants": EXPECTED_VARIANTS,
        "unique_pseudo_years": int(plan["pseudo_year"].nunique()),
        "unique_target_weather_files": int(plan["weather_file"].nunique()),
        "duplicate_pseudo_years": duplicate_pseudo_years,
        "duplicate_target_weather_files": duplicate_target_names,
        "augmented_validation_filename_collisions": augmented_validation_collisions,
        "source_weather_hashes": {rel(path): sha256(path) for path in source_weather if path.exists()},
        "issues": issues,
        "next_step_allowed": not issues,
    }


def postflight(cfg: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    manifest_root = ROOT / cfg["manifest_output_root"]
    generated = manifest_root / "160_weather_scenario_manifest.csv"
    manifest = pd.read_csv(generated, keep_default_na=False)
    train = manifest[manifest["split"].eq("train")].copy()
    validation = manifest[manifest["split"].eq("validation")].copy()
    source_root = ROOT / cfg["source_input_root"]
    target_root = ROOT / cfg["target_input_root"]
    source_static = {
        path.relative_to(source_root / "HL").as_posix(): sha256(path)
        for path in (source_root / "HL").rglob("*")
        if path.is_file() and path.suffix.upper() != ".WTH"
    }
    target_static = {
        path.relative_to(target_root / "HL").as_posix(): sha256(path)
        for path in (target_root / "HL").rglob("*")
        if path.is_file() and path.suffix.upper() != ".WTH"
    }
    source_hash_after = {
        rel(ROOT / path): sha256(ROOT / path)
        for path in before["source_weather_hashes"]
        if (ROOT / path).exists()
    }
    static_mismatch = sorted(name for name in set(source_static) & set(target_static) if source_static[name] != target_static[name])
    source_hash_changed = sorted(
        name for name, value in before["source_weather_hashes"].items() if source_hash_after.get(name) != value
    )
    validation_hash_mismatch = validation[validation["source_sha256"].astype(str) != validation["target_sha256"].astype(str)]
    target_name_duplicates = train["target_weather"].astype(str).duplicated(keep=False)
    issues: list[str] = []
    if len(train) != 60 or len(validation) != 10:
        issues.append(f"generated row counts mismatch: train={len(train)}, validation={len(validation)}")
    if set(train["variant"]) != set(EXPECTED_VARIANTS):
        issues.append(f"generated variants mismatch: {sorted(set(train['variant']))}")
    if target_name_duplicates.any():
        issues.append("generated target weather paths are not unique")
    if source_hash_changed:
        issues.append(f"source WTH hashes changed: {source_hash_changed}")
    if not validation_hash_mismatch.empty:
        issues.append("validation weather is not byte-identical to source")
    if set(source_static) != set(target_static) or static_mismatch:
        issues.append("static input file set/hash mismatch")
    alias_manifest = manifest_root / f"{TASK_ID}_weather_scenario_manifest.csv"
    alias_summary = manifest_root / f"{TASK_ID}_generation_summary.json"
    shutil.copy2(generated, alias_manifest)
    summary = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": "completed_generation_and_audit" if not issues else "failed_postflight",
        "source_input_root": cfg["source_input_root"],
        "target_input_root": cfg["target_input_root"],
        "manifest_csv": rel(alias_manifest),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "source_year_count": int(train["source_year"].nunique()),
        "variant_count": int(train["variant"].nunique()),
        "unique_target_weather_paths": int(train["target_weather"].nunique()),
        "source_weather_modified": bool(source_hash_changed),
        "validation_weather_hash_mismatch_rows": int(len(validation_hash_mismatch)),
        "static_source_file_count": len(source_static),
        "static_target_file_count": len(target_static),
        "static_hash_mismatch_files": static_mismatch,
        "issues": issues,
        "next_step_allowed": not issues,
    }
    alias_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_record(cfg: dict[str, Any], before: dict[str, Any], after: dict[str, Any] | None) -> None:
    lines = [
        f"# {TASK_ID} HLA lowIC 十年六天气情景库记录",
        "",
        "## 实验定位",
        "",
        "- 新工作线；不覆盖 160、054 或其他既有输入和结果。",
        "- 复用生成器：`src/weather_scenario_bank_160.py`。",
        f"- 原始输入：`{cfg['source_input_root']}`，与 054_00 lowIC 一致。",
        f"- 新输入根：`{cfg['target_input_root']}`。",
        "- 只扰动训练天气 RAIN；验证 2014--2023 保持原样。",
        "",
        "## 生成前审计",
        "",
        "```json",
        json.dumps(before, ensure_ascii=False, indent=2),
        "```",
    ]
    if after is not None:
        lines.extend(["", "## 生成后审计", "", "```json", json.dumps(after, ensure_ascii=False, indent=2), "```"])
    RECORD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else (ROOT / args.config).resolve()
    cfg = load_config(cfg_path)
    before = preflight(cfg)
    if args.dry_run:
        write_record(cfg, before, None)
        print(json.dumps({**before, "mode": "dry_run"}, ensure_ascii=False, indent=2))
        return
    if not before["next_step_allowed"]:
        raise RuntimeError(json.dumps(before, ensure_ascii=False, indent=2))
    shared.build(cfg_path, ["HLA"])
    after = postflight(cfg, before)
    write_record(cfg, before, after)
    print(json.dumps(after, ensure_ascii=False, indent=2))
    if not after["next_step_allowed"]:
        raise RuntimeError(json.dumps(after, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
