"""Reread 048 paired rollouts and create an evidence-limited diagnostic gate."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MODELS = {
    "2k": ROOT / "results/fqa_wgen_ppo_smoke_044/attempt_05/models/fqa_ppo_seed0_2k.zip",
    "10k": ROOT / "results/fqa_multiyear_wgen_ppo_047/attempt_01/models/fqa_multiyear_wgen_ppo_seed0_10k.zip",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    checks = {}
    metrics = {}
    manifests = {}
    runs = {}
    weather_ok = True
    runtime_ok = True
    endpoint_ok = True
    resource_ok = True
    for label, model in MODELS.items():
        directory = OUT / label
        metrics[label] = csv_rows(directory / "endpoint_metrics.csv")
        manifests[label] = csv_rows(directory / "episode_manifest.csv")
        runs[label] = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        preflight = json.loads((directory / "preflight.json").read_text(encoding="utf-8"))
        checks[f"{label}_frozen_model"] = model.is_file() and sha(model) == preflight["model_sha256"] == runs[label]["model_sha256"]
        checks[f"{label}_run_pass"] = runs[label]["status"] == "PASS_DIAGNOSTIC_INPUT" and runs[label]["episodes"] == 2 and runs[label]["weather_days"] == sum(int(row["days"]) for row in manifests[label])
        schedule = json.loads((directory / "schedule.json").read_text(encoding="utf-8"))
        checks[f"{label}_heldout_schedule"] = [(int(row["episode_index"]), int(row["year"]), int(row["weather_seed"])) for row in metrics[label]] == [(1, 2007, 1081), (2, 2007, 1100)] and [(row["episode_index"], row["historical_year"], row["weather_seed"]) for row in schedule] == [(1, 2007, 1081), (2, 2007, 1100)]
        checks[f"{label}_source_hash"] = preflight["prompt_sha256"] == sha(ROOT / "prompt_02/048_fqa_wgen_10k_heldout_pair_diagnostic.md") and preflight["source_045_sha256"] == sha(ROOT / "results/fqa_wgen_multiyear_heldout_gate_045/run_gate.py")
        resources = csv_rows(directory / "resource_usage.csv")
        resource_ok &= len(resources) == 2 and max(float(row["process_tree_rss_mb"]) for row in resources) < 1536 and float(runs[label]["elapsed_seconds"]) < 300
        for endpoint, archive in zip(metrics[label], manifests[label]):
            path = ROOT / archive["weather_path"]
            daily = csv_rows(path)
            dates = [date.fromisoformat(row["DATE"]) for row in daily]
            weather_ok &= sha(path) == archive["weather_sha256"] == endpoint["weather_sha256"] and len(daily) == int(archive["days"]) == int(endpoint["days"]) and len(daily) > 0 and all(dates[i] == dates[i-1] + timedelta(days=1) for i in range(1, len(dates))) and list(daily[0]) == ["DATE", "DOY", "RAIN", "SRAD", "TMAX", "TMIN"] and archive["physical_status"] == "PASS"
            weather_ok &= all(float(row["RAIN"]) >= 0 and float(row["SRAD"]) > 0 and float(row["TMAX"]) >= float(row["TMIN"]) for row in daily)
            proof = json.loads((ROOT / archive["runtime_evidence_path"]).read_text(encoding="utf-8"))
            runtime_ok &= proof["wther"] == "W" and proof["wsta_confirmed"] and proof["yaml_bootstrap_confirmed"] and proof["runtime_cli_sha256"] == proof["source_cli_sha256"] and int(proof["runtime_rseed1"]) == int(endpoint["weather_seed"]) == int(archive["pdi_rseed1"])
            y, n, irr = (float(endpoint[key]) for key in ("yield_kg_ha", "nitrogen_kg_ha", "irrigation_mm"))
            endpoint_ok &= y > 0 and n >= 0 and irr >= 0 and (abs(float(endpoint["pfp_n_kg_kg"]) - y/n) < 1e-9 if n > 0 else endpoint["pfp_n_kg_kg"] == "")
    checks["weather_hash_days_dates_physical"] = bool(weather_ok)
    checks["runtime_wgen_seed_cli"] = bool(runtime_ok)
    checks["endpoint_pfp_n"] = bool(endpoint_ok)
    checks["resources_below_limit"] = bool(resource_ok)
    paired = []
    for a, b in zip(metrics["2k"], metrics["10k"]):
        if (a["year"], a["weather_seed"], a["weather_sha256"]) != (b["year"], b["weather_seed"], b["weather_sha256"]):
            checks["same_realized_weather_pairs"] = False
            break
        row = {"year": int(a["year"]), "weather_seed": int(a["weather_seed"]), "weather_sha256": a["weather_sha256"]}
        for key in ("yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "pfp_n_kg_kg"):
            row[f"2k_{key}"] = round(float(a[key]), 6)
            row[f"10k_{key}"] = round(float(b[key]), 6)
            row[f"delta_10k_minus_2k_{key}"] = round(float(b[key]) - float(a[key]), 6)
        paired.append(row)
    else:
        checks["same_realized_weather_pairs"] = len(paired) == 2
    if all(checks.values()):
        path = OUT / "paired_endpoints.csv"
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    report = {"status": "PASS_PAIRED_DIAGNOSTIC_LIMITED" if all(checks.values()) else "FAIL", "checks": checks, "paired": paired, "interpretation": "Two 2007 heldout WGEN realizations only; no policy gain, generalization, WP_ET, or NUE claim; no authorization for 100K expansion."}
    with (OUT / "final_gate.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
