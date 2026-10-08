"""Audit the paired 5K/10K checkpoint smoke against saved 048 weather evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
MODEL_PATHS = {
    "5k": ROOT / "results/fqa_multiyear_wgen_ppo_047/attempt_01/models/checkpoint_5000.zip",
    "10k": ROOT / "results/fqa_multiyear_wgen_ppo_047/attempt_01/models/checkpoint_10000.zip",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    checks = {}
    data = {}
    manifests = {}
    results = {}
    for label, model in MODEL_PATHS.items():
        out = BASE / label
        data[label] = read_csv(out / "endpoint_and_action_summary.csv")
        manifests[label] = read_csv(out / "episode_manifest.csv")
        results[label] = json.loads((out / "result.json").read_text(encoding="utf-8"))
        preflight = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
        checks[f"{label}_checkpoint_hash"] = model.is_file() and sha(model) == preflight["checkpoint_sha256"] == results[label]["checkpoint_sha256"]
        checks[f"{label}_run_complete"] = results[label]["status"] == "PASS_CHECKPOINT_DIAGNOSTIC_INPUT" and results[label]["episodes"] == 2 and results[label]["trace_rows"] == sum(int(row["days"]) for row in data[label])
        checks[f"{label}_heldout_seeds"] = [int(row["weather_seed"]) for row in data[label]] == [1081, 1100]
        checks[f"{label}_rss_wall"] = float(results[label]["peak_rss_mb"]) < 1536 and float(results[label]["elapsed_seconds"]) < 300
        trace = read_csv(out / "daily_action_state_trace.csv")
        checks[f"{label}_trace_closure"] = len(trace) == results[label]["trace_rows"] and all(int(row["policy_action_index"]) in range(16) for row in trace)
        checks[f"{label}_source_hashes"] = preflight["prompt_sha256"] == sha(ROOT / "prompt_02/049_fqa_047_checkpoint_learning_signal_smoke.md") and preflight["source_048_sha256"] == sha(ROOT / "results/fqa_wgen_10k_heldout_diagnostic_048/run_eval.py")

    paired = []
    weather_ok = True
    runtime_ok = True
    for a, b in zip(data["5k"], data["10k"]):
        seed = int(a["weather_seed"])
        match = int(a["year"]) == int(b["year"]) == 2007 and int(b["weather_seed"]) == seed and a["weather_sha256"] == b["weather_sha256"]
        old_048 = next(row for row in read_csv(ROOT / "results/fqa_wgen_10k_heldout_diagnostic_048/10k/episode_manifest.csv") if int(row["weather_seed"]) == seed)
        match = match and a["weather_sha256"] == old_048["weather_sha256"]
        weather_ok &= match
        for label in ("5k", "10k"):
            manifest = next(row for row in manifests[label] if int(row["weather_seed"]) == seed)
            wpath = ROOT / manifest["weather_path"]
            proof = json.loads((ROOT / manifest["runtime_evidence_path"]).read_text(encoding="utf-8"))
            weather_ok &= sha(wpath) == manifest["weather_sha256"] == a["weather_sha256"] and len(read_csv(wpath)) == int(manifest["days"])
            runtime_ok &= proof["wther"] == "W" and proof["wsta_confirmed"] and proof["yaml_bootstrap_confirmed"] and proof["runtime_cli_sha256"] == proof["source_cli_sha256"] and int(proof["runtime_rseed1"]) == seed
        item = {"year": 2007, "weather_seed": seed, "weather_sha256": a["weather_sha256"]}
        for key in ("yield_kg_ha", "irrigation_mm_wrapper", "nitrogen_kg_ha_wrapper", "pfp_n_wrapper_basis", "episode_return", "distinct_policy_action_indices", "positive_irrigation_days", "positive_nitrogen_days"):
            va, vb = float(a[key]), float(b[key])
            item[f"5k_{key}"] = va
            item[f"10k_{key}"] = vb
            item[f"delta_10k_minus_5k_{key}"] = vb - va
        item["5k_action_counts"] = a["action_index_counts_json"]
        item["10k_action_counts"] = b["action_index_counts_json"]
        paired.append(item)
    checks["identical_heldout_weather_realizations"] = bool(weather_ok)
    checks["runtime_wgen_provenance"] = bool(runtime_ok)
    checks["paired_schedule"] = len(paired) == 2 and [row["weather_seed"] for row in paired] == [1081, 1100]
    status = "PASS_CHECKPOINT_DIAGNOSTIC_ONLY" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "checks": checks,
        "paired": paired,
        "interpretation": "Policy action distribution and endpoints differ between same-run 5K and 10K checkpoints in these two heldout realizations; one yield decreases and one increases, while wrapper cumulative water/N stay fixed. This is evidence of some policy change, not proof of useful learning, convergence, or generalization.",
        "limitations": ["two 2007 WGEN heldout realizations only", "wrapper cumulative water/N not verified against Summary.OUT", "PFP_N uses wrapper cumulative N", "no ETCP replay; WP_ET and NUE unavailable", "2K comparator from 048 is a separate smoke model and is not a within-run learning checkpoint"],
    }
    if status == "PASS_CHECKPOINT_DIAGNOSTIC_ONLY":
        with (BASE / "paired_checkpoint_diagnostic.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    with (BASE / "final_gate.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False))
    if status != "PASS_CHECKPOINT_DIAGNOSTIC_ONLY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
