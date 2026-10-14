"""Read-only reread of 044 attempt_05 evidence; writes only its new gate report."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent / "attempt_05"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    manifest = list(csv.DictReader((RUN / "episode_manifest.csv").open(encoding="utf-8", newline="")))
    result = json.loads((RUN / "run_result.json").read_text(encoding="utf-8"))
    checks = {}
    checks["run_pass"] = result["status"] == "PASS_SMOKE_ONLY"
    checks["episode_index_unique_sequential"] = [int(r["episode_index"]) for r in manifest] == list(range(1, len(manifest) + 1))
    checks["episode_status"] = sum(r["status"] == "completed" for r in manifest) == 19 and sum(r["status"] == "partial_at_stop" for r in manifest) == 1
    checks["step_archive_closure"] = sum(int(r["days_used"]) for r in manifest) == result["actual_steps"] == 2016
    checks["weather_seed_train_pool"] = all(1001 <= int(r["weather_seed"]) <= 1080 for r in manifest)
    checks["pdi_runtime_seed"] = all(r["weather_seed"] == r["pdi_rseed1"] for r in manifest)
    checks["capture_one_per_step"] = all(r["days_used"] == r["weather_rows"] == r["capture_calls"] for r in manifest)
    checks["physical_screen"] = all(r["physical_status"] == "PASS" for r in manifest)
    weather_ok = True
    runtime_ok = True
    for row in manifest:
        weather = ROOT / row["weather_path"]
        if sha(weather) != row["weather_sha256"]:
            weather_ok = False
        with weather.open(encoding="utf-8", newline="") as f:
            daily = list(csv.DictReader(f))
        if len(daily) != int(row["weather_rows"]) or list(daily[0]) != ["DATE", "DOY", "RAIN", "SRAD", "TMAX", "TMIN"]:
            weather_ok = False
        evidence = json.loads((ROOT / row["runtime_evidence_path"]).read_text(encoding="utf-8"))
        if not (evidence["filex_wther"] == "W" and evidence["pdi_runtime_episode_seed_confirmed"] and evidence["pdi_yaml_bootstrap_seed_confirmed"] and evidence["cli_source_match"] and evidence["filex_wsta_confirmed"]):
            runtime_ok = False
    checks["all_weather_files_reread"] = weather_ok
    checks["all_runtime_evidence"] = runtime_ok
    resource = list(csv.DictReader((RUN / "resource_usage.csv").open(encoding="utf-8", newline="")))
    peak = max(float(r["process_tree_rss_mb"]) for r in resource)
    checks["rss_below_1536_mb"] = peak < 1536
    model = ROOT / result["model_path"]
    checks["model_hash_reread"] = model.is_file() and sha(model) == result["model_sha256"]
    output = {"status": "PASS_SMOKE_ONLY" if all(checks.values()) else "FAIL", "checks": checks, "actual_steps": result["actual_steps"], "episodes": len(manifest), "completed_episodes": sum(r["status"] == "completed" for r in manifest), "partial_episodes": sum(r["status"] == "partial_at_stop" for r in manifest), "unique_weather_hashes": len({r["weather_sha256"] for r in manifest}), "peak_process_tree_rss_mb": round(peak, 2), "failed_attempts_preserved": ["attempt_01", "attempt_02", "attempt_04"], "scope": "single FQA 2007 PPO seed0 WGEN 2K smoke; no formal 8-seed training", "limitations": ["fixed-year smoke does not validate 2005-2013 year scheduling", "weather climate distribution and heldout pool not assessed", "shared native FIELD coordinate warning remains unresolved"]}
    gate = RUN / "final_gate.json"
    if gate.exists():
        raise FileExistsError(gate)
    gate.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))
    if output["status"] != "PASS_SMOKE_ONLY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
