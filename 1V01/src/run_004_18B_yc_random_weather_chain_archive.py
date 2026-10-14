"""Build the small, hash-addressed YC random-weather archive index."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/archive/yc_random_weather"
WEATHER = ROOT / "results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery"
SUMMARY = ROOT / "results/yc_random_weather_ppo/004_18A_yc_weather_summary_for_reporting"


def long_path(path: Path) -> str:
    value = str(path.absolute())
    return "\\\\?\\" + value if os.name == "nt" else value


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(long_path(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


ASSETS = [
    ("archival_contract", "metadata", "FORMAL", ".gitattributes", "Path-scoped Git byte preservation for frozen weather assets"),
    ("motivation", "prompt", "DIAGNOSTIC", "prompt_02/004_yc_ppo_weather_augmentation_experiment.md", "Initial weather-augmentation design; later 004_05 is the formal experiment"),
    ("motivation", "prompt", "FORMAL", "prompt_02/004_05_yc_weather_augmentation_multi_seed_archetype_experiment.md", "Final controlled paired-seed question"),
    ("fitting_cli", "input", "FORMAL", "results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv", "Frozen 2004-2013 fitting weather"),
    ("fitting_cli", "input", "FORMAL", "results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI", "Frozen YC WGEN CLI"),
    ("fitting_cli", "source", "FORMAL", "scripts/build_dssat_cli.py", "CLI generation code"),
    ("fitting_cli", "source", "DIAGNOSTIC", "scripts/audit_yc_wgen_parameters_004_01.py", "168-value parameter audit"),
    ("fitting_cli", "metadata", "FORMAL", "results/yc_wgen_cli_pilot/003_06_05_02/final/final_cli_metadata.json", "CLI generation provenance"),
    ("fitting_cli", "metadata", "DIAGNOSTIC", "results/yc_wgen_cli_pilot/004_01/parameter_audit/parameter_method_summary.json", "168 monthly values independently checked"),
    ("fitting_cli", "report", "DIAGNOSTIC", "docs/yc_wgen_parameter_audit_and_synthetic_weather_validation.md", "Parameter method valid; full-year generation blocked"),
    ("wgen_qc", "prompt", "DIAGNOSTIC", "prompt_02/003_06_06_yc_wgen_weather_qc_and_dssat_smoke.md", "Earlier WGEN/QC pilot"),
    ("wgen_qc", "prompt", "FORMAL", "prompt_02/004_02_yc_random_weather_episode_ensemble_validation.md", "100-episode validation"),
    ("wgen_qc", "source", "FORMAL", "results/yc_random_weather_episode_validation/004_02/run_episode_ensemble.py", "Episode-level weather generator"),
    ("wgen_qc", "source", "FORMAL", "results/yc_random_weather_episode_validation/004_02/analyze_episode_ensemble.py", "Ensemble QC analysis"),
    ("wgen_qc", "metadata", "FORMAL", "results/yc_random_weather_episode_validation/004_02/summary.json", "100/100 generation pass; five repeats match"),
    ("wgen_qc", "metadata", "FORMAL", "results/yc_random_weather_episode_validation/004_02/weather_seed_reproducibility.csv", "Five repeated seeds"),
    ("wgen_qc", "metadata", "FORMAL", "results/yc_random_weather_episode_validation/004_02/weather_seed_diversity.csv", "100 unique sequences"),
    ("wgen_qc", "metadata", "DIAGNOSTIC", "results/yc_wgen_cli_pilot/003_06_06/weather_qc/weather_qc_summary.json", "Early weather QC"),
    ("wgen_qc", "report", "FORMAL", "docs/yc_random_weather_episode_ensemble_validation.md", "100 episode validation report"),
    ("ppo_004_05", "prompt", "FORMAL", "prompt_02/004_03_yc_random_weather_ppo_controlled_pilot.md", "Controlled pilot"),
    ("ppo_004_05", "source", "FORMAL", "results/yc_random_weather_ppo/004_03/run_controlled_pilot.py", "WGEN FileX/runtime setup"),
    ("ppo_004_05", "source", "FORMAL", "results/yc_random_weather_ppo/004_05/run_yc_weather_augmentation_multi_seed_archetype_004_05.py", "8-seed experiment runner"),
    ("ppo_004_05", "metadata", "FORMAL", "results/yc_random_weather_ppo/004_05/config/training_weather_schedule_random.csv", "100000 selections, each weather seed 1250 times"),
    ("ppo_004_05", "metadata", "FORMAL", "results/yc_random_weather_ppo/004_05/multi_seed_decision.json", "NO_POSITIVE_SIGNAL; paired success 0/8"),
    ("ppo_004_05", "metadata", "FORMAL", "results/yc_random_weather_ppo/004_05/all_evaluation_episode_level_0_7.csv", "Observed and held-out episode summaries"),
    ("ppo_004_05", "metadata", "FORMAL", "results/yc_random_weather_ppo/004_05/paired_seed_performance_comparison.csv", "Paired PPO results"),
    ("ppo_004_05", "metadata", "FORMAL", "results/yc_random_weather_ppo/004_05/new_training_run_manifest.csv", "Training run manifest"),
    ("ppo_004_05", "report", "FORMAL", "docs/yc_random_weather_multi_seed_archetype_experiment.md", "Formal eight-seed conclusion"),
    ("weather_audit", "source", "DIAGNOSTIC", "src/audit_yc_wgen_weather_domain_coverage.py", "First weather archive inventory"),
    ("weather_audit", "report", "DIAGNOSTIC", "docs/yc_random_weather_weather_domain_coverage_audit.md", "Weather pool expansion DEFER"),
    ("weather_audit", "prompt", "DIAGNOSTIC", "prompt_02/004_16_yc_wgen_weather_reconstruction_and_coverage.md", "Provenance gate specification"),
    ("weather_audit", "source", "DIAGNOSTIC", "src/run_004_16_yc_wgen_weather_reconstruction_and_coverage.py", "Read-only gate runner"),
    ("weather_audit", "metadata", "DIAGNOSTIC", "results/yc_random_weather_ppo/004_16_yc_wgen_weather_reconstruction/decision.json", "FAILED means no materialization; not WGEN failure"),
    ("weather_audit", "report", "DIAGNOSTIC", "docs/yc_random_weather_004_16_wgen_weather_reconstruction_and_coverage.md", "No formal coverage conclusion"),
    ("recovery_004_17", "prompt", "DIAGNOSTIC", "prompt_02/004_17_yc_runtime_wgen_capture_and_weather_recovery.md", "Runtime recovery gate"),
    ("recovery_004_17", "source", "DIAGNOSTIC", "src/run_004_17_yc_runtime_wgen_capture_and_weather_recovery.py", "Runtime capture and verification"),
    ("recovery_004_17", "source", "FORMAL", "src/archive_runtime_weather.py", "Future episode archive helper"),
    ("recovery_004_17", "metadata", "FORMAL", "results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery/weather_archive_manifest_verified.csv", "Authoritative 100-row mapping; two provisional"),
    ("recovery_004_17", "metadata", "DIAGNOSTIC", "results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery/hash_verification_context_replay_reconciled.csv", "Historical matching ledger"),
    ("recovery_004_17", "metadata", "FORMAL", "results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery/provenance/final_archive_integrity.csv", "100/100 file hashes"),
    ("recovery_004_17", "metadata", "FORMAL", "results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery/provenance/final_archive_integrity.json", "Independent integrity conclusion"),
    ("recovery_004_17", "metadata", "DIAGNOSTIC", "results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery/decision.json", "PARTIALLY_VERIFIED; coverage insufficient"),
    ("recovery_004_17", "report", "DIAGNOSTIC", "results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery/provenance/runtime_weather_generation_chain.md", "Runtime hash is PDI state-derived, no raw WTH"),
    ("recovery_004_17", "report", "DIAGNOSTIC", "docs/yc_random_weather_004_17_runtime_weather_recovery_and_coverage.md", "98 exact + 2 provisional"),
    ("reporting_004_18A", "prompt", "DIAGNOSTIC", "prompt_02/004_18A_yc_weather_summary_for_reporting.md", "Descriptive reporting scope"),
    ("reporting_004_18A", "source", "DIAGNOSTIC", "src/run_004_18A_yc_weather_summary_for_reporting.py", "Reproducible summary and figures"),
    ("reporting_004_18A", "report", "DIAGNOSTIC", "docs/yc_random_weather_004_18A_weather_summary_for_reporting.md", "Sensitivity changes identified"),
    ("reporting_004_18A", "report", "DIAGNOSTIC", "docs/yc_random_weather_weather_summary_for_presentation.md", "Chinese group-meeting summary"),
    ("archival_contract", "report", "FORMAL", "docs/random_weather_archival_contract.md", "Future YC/HL/FQ/LC/SY weather archive contract"),
    ("archival_contract", "source", "DIAGNOSTIC", "src/run_004_18B_yc_random_weather_chain_archive.py", "Rebuilds this index"),
]


def main() -> None:
    if json.loads((SUMMARY / "summary_status.json").read_text(encoding="utf-8"))["artifact_status"] != "COMPLETE":
        raise ValueError("004_18A gate not complete")
    manifest_path = WEATHER / "weather_archive_manifest_verified.csv"
    with open(long_path(manifest_path), encoding="utf-8-sig", newline="") as stream:
        weather_rows = list(csv.DictReader(stream))
    if len(weather_rows) != 100 or len({int(r["weather_seed"]) for r in weather_rows}) != 100:
        raise ValueError("Weather archive manifest does not have 100 distinct seeds")
    assets = list(ASSETS)
    for row in weather_rows:
        seed = int(row["weather_seed"])
        status = "FORMAL" if row["verification_level"] == "EXACT_MATCH" else "PROVISIONAL"
        path = str((WEATHER / row["canonical_csv_path"]).relative_to(ROOT)).replace("\\", "/")
        assets.append(("recovery_004_17", "canonical_weather", status, path,
                       f"seed={seed}; set={row['set']}; context={row['crop_year']}; historical={row['verification_level']}"))
    for path in sorted(SUMMARY.glob("*.csv")):
        assets.append(("reporting_004_18A", "table", "DIAGNOSTIC", path.relative_to(ROOT).as_posix(), "Descriptive only"))
    assets.append(("reporting_004_18A", "metadata", "DIAGNOSTIC", (SUMMARY / "summary_status.json").relative_to(ROOT).as_posix(), "Coverage verdict pending"))
    for path in sorted((SUMMARY / "figures").glob("*.png")):
        assets.append(("reporting_004_18A", "figure", "DIAGNOSTIC", path.relative_to(ROOT).as_posix(), "Group-meeting figure"))
    rows = []
    for stage, kind, status, rel, note in assets:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = digest(path)
        if kind == "canonical_weather":
            weather = next(r for r in weather_rows if int(r["weather_seed"]) == int(note.split(";")[0].split("=")[1]))
            if actual != weather["canonical_series_sha256"].upper():
                raise ValueError(f"Canonical weather hash mismatch: {path}")
        rows.append({"stage": stage, "artifact_type": kind, "status": status, "path": rel,
                     "sha256": actual, "size_bytes": path.stat().st_size, "note": note})
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "yc_random_weather_experiment_chain_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = OUT / "yc_random_weather_experiment_chain_manifest.json"
    json_path.write_text(json.dumps({"schema": "yc_random_weather_archive_v1", "files": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"indexed_files": len(rows), "canonical_weather": sum(r["artifact_type"] == "canonical_weather" for r in rows),
                      "provisional_weather": sum(r["artifact_type"] == "canonical_weather" and r["status"] == "PROVISIONAL" for r in rows),
                      "indexed_bytes": sum(r["size_bytes"] for r in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
