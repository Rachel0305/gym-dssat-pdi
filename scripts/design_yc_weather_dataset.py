from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "yc_weather_dataset_design"

CONFIG_055 = ROOT / "configs" / "055_00_yca_lowIC_expanded_action_maskableppo.json"
SNAPSHOT_001 = ROOT / "results" / "yc_weather_audit" / "yc_weather_config_snapshot.json"
DIAG_001 = ROOT / "results" / "yc_weather_audit" / "yc_weather_reset_diagnostic.json"
BANK_MANIFEST = (
    ROOT
    / "benchmark_results"
    / "217YCA_yca_lowIC_weather_scenario_bank_v1_fixed_width"
    / "217YCA_weather_scenario_manifest.csv"
)
PHYSICAL_RESULT = (
    ROOT
    / "benchmark_results"
    / "217YCA_yca_lowIC_weather_physical_gate"
    / "217YCA_result.json"
)
PHYSICAL_DETAILS = (
    ROOT
    / "benchmark_results"
    / "217YCA_yca_lowIC_weather_physical_gate"
    / "evaluation"
    / "217YCA_scenario_physical_gate_details.csv"
)


def rel(path: Path | str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_wth(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("*") or stripped.startswith("@"):
            continue
        parts = stripped.split()
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        yyddd = parts[0]
        yy = int(yyddd[:-3])
        year = 2000 + yy if yy < 50 else 1900 + yy
        doy = int(yyddd[-3:])
        try:
            date = datetime.strptime(f"{year} {doy:03d}", "%Y %j")
        except ValueError:
            continue
        rows.append(
            {
                "date_code": int(yyddd),
                "year": year,
                "doy": doy,
                "month": date.month,
                "srad": float(parts[1]),
                "tmax": float(parts[2]),
                "tmin": float(parts[3]),
                "rain": float(parts[4]),
            }
        )
    return rows


def max_dry_spell(rows: list[dict[str, float | int]]) -> int:
    longest = 0
    current = 0
    for row in rows:
        if float(row["rain"]) <= 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def wth_quality(path: Path) -> dict[str, object]:
    rows = parse_wth(path)
    rain = [float(r["rain"]) for r in rows]
    srad = [float(r["srad"]) for r in rows]
    tmax = [float(r["tmax"]) for r in rows]
    tmin = [float(r["tmin"]) for r in rows]
    wet_days = sum(1 for x in rain if x > 0.0)
    monthly = defaultdict(float)
    for row in rows:
        monthly[int(row["month"])] += float(row["rain"])
    monthly_payload = {f"{m:02d}": round(monthly[m], 3) for m in range(1, 13)}
    checks = {
        "rows": len(rows),
        "annual_rain_mm": round(sum(rain), 3),
        "wet_day_count": wet_days,
        "wet_day_ratio": round(wet_days / len(rows), 4) if rows else math.nan,
        "mean_wet_day_rain_mm": round(sum(rain) / wet_days, 3) if wet_days else 0.0,
        "max_dry_spell_days": max_dry_spell(rows),
        "rain_min_mm": min(rain) if rain else math.nan,
        "srad_min": min(srad) if srad else math.nan,
        "tmax_lt_tmin_count": sum(1 for hi, lo in zip(tmax, tmin) if hi < lo),
        "monthly_rain_hash": sha256_text(json.dumps(monthly_payload, sort_keys=True)),
    }
    checks["physical_status"] = (
        "pass"
        if rows
        and checks["rain_min_mm"] >= 0
        and checks["srad_min"] >= 0
        and checks["tmax_lt_tmin_count"] == 0
        and checks["rows"] in {365, 366}
        else "review"
    )
    return checks


def git_output(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace").strip()


def inventory_path(path: Path) -> dict[str, object]:
    exists = path.exists()
    tracked = False
    if exists:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel(path)],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        ).returncode == 0
    file_count = 0
    byte_count = 0
    if exists:
        for child in path.rglob("*") if path.is_dir() else [path]:
            if child.is_file():
                file_count += 1
                byte_count += child.stat().st_size
    return {
        "path": rel(path),
        "exists": exists,
        "tracked": tracked,
        "file_count": file_count,
        "bytes": byte_count,
        "recommendation": "retain_as_audit_evidence_or_build_cache; not deleted by 002",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_json(CONFIG_055)
    snapshot = load_json(SNAPSHOT_001)
    diag = load_json(DIAG_001)
    physical_result = load_json(PHYSICAL_RESULT) if PHYSICAL_RESULT.exists() else {}
    bank_rows = read_csv(BANK_MANIFEST)
    physical_rows = read_csv(PHYSICAL_DETAILS) if PHYSICAL_DETAILS.exists() else []

    selected_variants = ["original", "early_dry", "mid_dry", "late_dry", "early_wet"]
    selected = [
        row
        for row in bank_rows
        if row["split"] == "train" and row["source_year"] == "2005" and row["variant"] in selected_variants
    ]
    selected.sort(key=lambda r: selected_variants.index(r["variant"]))

    physical_by_variant = {row["weather_variant"]: row for row in physical_rows if row.get("source_year") == "2005"}
    sample_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []

    for row in selected:
        weather_path = ROOT / row["target_weather"]
        params = {
            "source_year": int(row["source_year"]),
            "pseudo_year": int(row["pseudo_year"]),
            "variant": row["variant"],
            "rain_multiplier": row["rain_multiplier"],
            "window": row["window"],
            "changed_days": row["changed_days"],
        }
        scenario_id = f"YC_217YCA_{row['source_year']}_{row['variant']}"
        quality = wth_quality(weather_path)
        gate = physical_by_variant.get(row["variant"], {})
        gate_ok = gate.get("run_status") == "ok" and gate.get("mat", "") not in {"", "0"}
        sample_rows.append(
            {
                "scenario_id": scenario_id,
                "site": row["site"],
                "station_code": row["station_code"],
                "split": row["split"],
                "source_years": row["source_year"],
                "generator_name": "existing_217YCA_fixed_width_rain_window_multiplier",
                "generator_version": "217YCA_v1_fixed_width",
                "weather_selection_seed": "not_random_deterministic_variant_order",
                "weather_generation_seed": "not_random_deterministic_multiplier",
                "ppo_seed": "not_applicable_no_training",
                "weather_file": rel(weather_path),
                "file_sha256": sha256_file(weather_path),
                "climate_parameter_file": "",
                "parameters_hash": sha256_text(json.dumps(params, sort_keys=True)),
                "generation_config": json.dumps(params, sort_keys=True),
                "validation_status": "pass_existing_physical_gate" if gate_ok and quality["physical_status"] == "pass" else "review",
                "leakage_status": "train_source_year_only; no validation_year_used",
                "dssat_smoke_source": rel(PHYSICAL_RESULT),
            }
        )
        quality_rows.append(
            {
                "scenario_id": scenario_id,
                "weather_file": rel(weather_path),
                "physical_status": quality["physical_status"],
                "rows": quality["rows"],
                "annual_rain_mm": quality["annual_rain_mm"],
                "wet_day_count": quality["wet_day_count"],
                "wet_day_ratio": quality["wet_day_ratio"],
                "mean_wet_day_rain_mm": quality["mean_wet_day_rain_mm"],
                "max_dry_spell_days": quality["max_dry_spell_days"],
                "rain_min_mm": quality["rain_min_mm"],
                "srad_min": quality["srad_min"],
                "tmax_lt_tmin_count": quality["tmax_lt_tmin_count"],
                "monthly_rain_hash": quality["monthly_rain_hash"],
                "existing_gate_run_status": gate.get("run_status", ""),
                "existing_gate_mat": gate.get("mat", ""),
                "existing_gate_rain_log_minus_wth_mm": gate.get("rain_log_minus_wth_mm", ""),
            }
        )

    manifest_schema = [
        {"field": "scenario_id", "required": "yes", "description": "stable unique scenario key"},
        {"field": "site", "required": "yes", "description": "short site code such as YC"},
        {"field": "station_code", "required": "yes", "description": "DSSAT station code such as YCA"},
        {"field": "split", "required": "yes", "description": "train, validation, test, or diagnostic"},
        {"field": "source_years", "required": "yes", "description": "historical years used to fit or resample this weather"},
        {"field": "generator_name", "required": "yes", "description": "implementation name; do not conflate with PPO seed"},
        {"field": "generator_version", "required": "yes", "description": "version or commit of generator parameters"},
        {"field": "weather_selection_seed", "required": "conditional", "description": "seed selecting source years/scenarios"},
        {"field": "weather_generation_seed", "required": "conditional", "description": "seed for stochastic generation after parameters are fixed"},
        {"field": "ppo_seed", "required": "no", "description": "PPO training seed; must stay separate from weather seeds"},
        {"field": "weather_file", "required": "yes", "description": "generated or selected DSSAT .WTH path"},
        {"field": "file_sha256", "required": "yes", "description": "hash of exact WTH bytes"},
        {"field": "climate_parameter_file", "required": "conditional", "description": ".CLI or external parameter file if used"},
        {"field": "parameters_hash", "required": "yes", "description": "hash of generator configuration and fitted parameters"},
        {"field": "generation_config", "required": "yes", "description": "machine-readable compact JSON config"},
        {"field": "validation_status", "required": "yes", "description": "quality gate status"},
        {"field": "leakage_status", "required": "yes", "description": "evidence no validation/test years fit the generator"},
        {"field": "dssat_smoke_source", "required": "conditional", "description": "DSSAT reset/single-season smoke evidence path"},
    ]

    route_rows = [
        {
            "route": "A_gym_dssat_random_weather_wgen_cli",
            "status": "blocked_for_current_YC_lowIC",
            "evidence": "gym_dssat supports random_weather arg, but 055_00 env_args set random_weather=false and YC lowIC input has no .CLI",
            "main_risk": "must create/verify YC .CLI from train years only; default package climate must not leak into YC",
            "recommended_next": "do not enable until .CLI parameter estimation, hash, and DSSAT smoke are documented",
        },
        {
            "route": "B_external_literature_or_official_python_generator",
            "status": "design_only_pending_verified_implementation",
            "evidence": "requires fit parameters from train years and code-level reproducibility; no such YC generator is currently frozen in repo",
            "main_risk": "ad hoc Gaussian perturbations would break weather physics and validation independence",
            "recommended_next": "select one generator family, implement tiny 3-5 WTH pilot, then fixed/different seed reproducibility gates",
        },
        {
            "route": "C_existing_217YCA_fixed_width_rain_scenario_bank",
            "status": "feasible_as_limited_scenario_bank_evidence",
            "evidence": "existing 217YCA manifest and physical gate cover six deterministic source-year 2005 variants",
            "main_risk": "not a full statistical weather generator; only rainfall-window stress scenarios",
            "recommended_next": "usable for 003 smoke design only if claims stay limited to scenario-bank diversity",
        },
    ]

    cleanup_rows = [
        inventory_path(ROOT / ".codex-yc-weather-pptx-build"),
        inventory_path(ROOT / "results" / "yc_weather_audit" / "rendered_probe"),
        inventory_path(ROOT / "results" / "yc_weather_audit" / "gym_reset_probe"),
    ]

    snapshot_out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_branch": git_output(["branch", "--show-current"]),
        "git_head": git_output(["rev-parse", "--short", "HEAD"]),
        "task": "002_yc_weather_dataset_design",
        "scope": "YC/YCA only; design and lightweight evidence; no PPO training",
        "config_055": rel(CONFIG_055),
        "training_entry": snapshot.get("training_entry"),
        "random_weather_current": snapshot.get("random_weather"),
        "current_weather_file_example": snapshot.get("weather_file"),
        "episode_weather_changes": snapshot.get("episode_weather_changes"),
        "weather_reproducible": snapshot.get("weather_reproducible"),
        "gym_supports_random_weather_arg": diag.get("package_info", {}).get("supports_random_weather_arg"),
        "train_years": config["scope"]["train_years"],
        "validation_years": config["scope"]["validation_years"],
        "test_years": [],
        "small_sample_policy": "reuse existing 217YCA deterministic rain-window scenarios for feasibility evidence; do not claim statistical generation",
        "small_sample_count": len(sample_rows),
        "existing_physical_gate": physical_result.get("gate", {}),
        "outputs": {
            "manifest_schema": rel(OUT / "yc_weather_manifest_schema.csv"),
            "small_sample_manifest": rel(OUT / "yc_weather_small_sample_manifest.csv"),
            "quality_summary": rel(OUT / "yc_weather_small_sample_quality_summary.csv"),
            "route_comparison": rel(OUT / "yc_weather_route_comparison.csv"),
            "cleanup_inventory": rel(OUT / "yc_weather_cleanup_inventory.csv"),
        },
    }

    write_csv(OUT / "yc_weather_manifest_schema.csv", manifest_schema, ["field", "required", "description"])
    write_csv(
        OUT / "yc_weather_small_sample_manifest.csv",
        sample_rows,
        [
            "scenario_id",
            "site",
            "station_code",
            "split",
            "source_years",
            "generator_name",
            "generator_version",
            "weather_selection_seed",
            "weather_generation_seed",
            "ppo_seed",
            "weather_file",
            "file_sha256",
            "climate_parameter_file",
            "parameters_hash",
            "generation_config",
            "validation_status",
            "leakage_status",
            "dssat_smoke_source",
        ],
    )
    write_csv(
        OUT / "yc_weather_small_sample_quality_summary.csv",
        quality_rows,
        [
            "scenario_id",
            "weather_file",
            "physical_status",
            "rows",
            "annual_rain_mm",
            "wet_day_count",
            "wet_day_ratio",
            "mean_wet_day_rain_mm",
            "max_dry_spell_days",
            "rain_min_mm",
            "srad_min",
            "tmax_lt_tmin_count",
            "monthly_rain_hash",
            "existing_gate_run_status",
            "existing_gate_mat",
            "existing_gate_rain_log_minus_wth_mm",
        ],
    )
    write_csv(OUT / "yc_weather_route_comparison.csv", route_rows, ["route", "status", "evidence", "main_risk", "recommended_next"])
    write_csv(OUT / "yc_weather_cleanup_inventory.csv", cleanup_rows, ["path", "exists", "tracked", "file_count", "bytes", "recommendation"])
    (OUT / "yc_weather_dataset_design_snapshot.json").write_text(
        json.dumps(snapshot_out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
