"""Close the paired 5K/10K heldout evaluation and summarize all 20 seeds."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SEEDS = list(range(1081, 1101))
MODEL_PATHS = {
    "5k": ROOT / "results/fqa_multiyear_wgen_ppo_047/attempt_01/models/checkpoint_5000.zip",
    "10k": ROOT / "results/fqa_multiyear_wgen_ppo_047/attempt_01/models/checkpoint_10000.zip",
}
METRICS = (
    "yield_kg_ha",
    "irrigation_mm_wrapper",
    "nitrogen_kg_ha_wrapper",
    "pfp_n_wrapper_basis",
    "episode_return",
    "distinct_policy_action_indices",
    "positive_irrigation_days",
    "positive_nitrogen_days",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, obj) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inside_root(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if path != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise ValueError(f"Path escapes project root: {relative}")
    return path


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "population_sd": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    for name in ("paired_checkpoint_full20.csv", "final_gate.json"):
        if (BASE / name).exists():
            raise FileExistsError(BASE / name)

    checks: dict[str, bool] = {}
    datasets: dict[str, dict] = {}
    for label in ("5k", "10k"):
        out = BASE / label
        endpoint = read_csv(out / "endpoint_and_action_summary.csv")
        manifest = read_csv(out / "episode_manifest.csv")
        trace = read_csv(out / "daily_action_state_trace.csv")
        resources = read_csv(out / "resource_usage.csv")
        schedule = json.loads((out / "schedule.json").read_text(encoding="utf-8"))
        result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        preflight = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
        model_path = MODEL_PATHS[label]

        checks[f"{label}_checkpoint_hash"] = (
            model_path.is_file()
            and sha(model_path) == preflight["checkpoint_sha256"] == result["checkpoint_sha256"]
        )
        checks[f"{label}_runner_closure"] = (
            result["status"] == "PASS_CHECKPOINT_FULL20_INPUT"
            and int(result["episodes"]) == 20
            and int(result["trace_rows"]) == int(result["weather_days"]) == len(trace)
            and int(result["trace_rows"]) == sum(int(row["days"]) for row in endpoint)
        )
        expected = [(i, 2007, seed) for i, seed in enumerate(SEEDS, 1)]
        checks[f"{label}_schedule"] = (
            [(int(r["episode_index"]), int(r["historical_year"]), int(r["weather_seed"])) for r in schedule] == expected
            and [(int(r["episode_index"]), int(r["year"]), int(r["weather_seed"])) for r in endpoint] == expected
            and [(int(r["episode_index"]), int(r["year"]), int(r["weather_seed"])) for r in manifest] == expected
        )
        checks[f"{label}_resource_limits"] = (
            len(resources) == 20
            and max(float(r["process_tree_rss_mb"]) for r in resources) < 1536
            and float(result["peak_rss_mb"]) < 1536
            and float(result["elapsed_seconds"]) < 600
        )
        prompt = ROOT / "prompt_02/050_fqa_047_checkpoint_full20_heldout_evaluation.md"
        source048 = ROOT / "results/fqa_wgen_10k_heldout_diagnostic_048/run_eval.py"
        source049 = ROOT / "results/fqa_047_checkpoint_learning_signal_049/run_checkpoint_eval.py"
        checks[f"{label}_source_hashes"] = (
            preflight["prompt_sha256"] == sha(prompt)
            and preflight["source_048_sha256"] == sha(source048)
            and preflight["source_049_sha256"] == sha(source049)
        )

        trace_by_seed: dict[int, list[dict]] = {seed: [] for seed in SEEDS}
        trace_ok = True
        for row in trace:
            seed = int(row["weather_seed"])
            trace_ok &= seed in trace_by_seed and int(row["historical_year"]) == 2007
            trace_by_seed[seed].append(row)
            trace_ok &= int(row["policy_action_index"]) in range(16)
        archive_ok = runtime_ok = endpoint_ok = trace_ok
        episode_index = {seed: i for i, seed in enumerate(SEEDS, 1)}
        for ep, arc in zip(endpoint, manifest):
            seed = int(ep["weather_seed"])
            dailies = trace_by_seed[seed]
            archive_path = inside_root(arc["weather_path"])
            archive = read_csv(archive_path)
            dates = [date.fromisoformat(r["DATE"]) for r in archive]
            archive_ok &= (
                sha(archive_path) == ep["weather_sha256"] == arc["weather_sha256"]
                and len(archive) == int(ep["days"]) == int(arc["days"]) == int(arc["weather_rows"])
                and len(dates) > 0
                and all(dates[i] == dates[i - 1] + timedelta(days=1) for i in range(1, len(dates)))
                and list(archive[0]) == ["DATE", "DOY", "RAIN", "SRAD", "TMAX", "TMIN"]
                and arc["physical_status"] == "PASS"
                and all(float(r["RAIN"]) >= 0 and float(r["SRAD"]) > 0 and float(r["TMAX"]) >= float(r["TMIN"]) for r in archive)
            )
            archive_ok &= len(dailies) == int(ep["days"])
            archive_ok &= [int(r["step_dap"]) for r in dailies] == list(range(1, len(dailies) + 1))
            archive_ok &= all(int(r["episode_index"]) == episode_index[seed] for r in dailies)

            runtime_path = inside_root(arc["runtime_evidence_path"])
            proof = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime_ok &= (
                proof["wther"] == "W"
                and proof["wsta_confirmed"]
                and proof["yaml_bootstrap_confirmed"]
                and proof["runtime_cli_sha256"] == proof["source_cli_sha256"]
                and int(proof["yaml_bootstrap_seed"]) == SEEDS[0]
                and int(proof["runtime_rseed1"]) == int(proof["scheduled_seed"]) == seed
                and int(arc["pdi_rseed1"]) == seed
            )

            y = float(ep["yield_kg_ha"])
            n = float(ep["nitrogen_kg_ha_wrapper"])
            irr = float(ep["irrigation_mm_wrapper"])
            pfp = float(ep["pfp_n_wrapper_basis"])
            actions = json.loads(ep["action_index_counts_json"])
            endpoint_ok &= y > 0 and n >= 0 and irr >= 0 and n > 0 and abs(pfp - y / n) < 1e-9
            endpoint_ok &= sum(int(v) for v in actions.values()) == len(dailies)
            endpoint_ok &= len(actions) == int(ep["distinct_policy_action_indices"])
            endpoint_ok &= sum(float(r["safe_action_amir_mm"] or 0) > 0 for r in dailies) == int(ep["positive_irrigation_days"])
            endpoint_ok &= sum(float(r["safe_action_anfer_kg_ha"] or 0) > 0 for r in dailies) == int(ep["positive_nitrogen_days"])
            endpoint_ok &= {str(k): int(v) for k, v in actions.items()} == {
                str(k): sum(int(r["policy_action_index"]) == int(k) for r in dailies) for k in actions
            }

        checks[f"{label}_weather_hash_dates_physical_trace"] = bool(archive_ok)
        checks[f"{label}_runtime_wgen_seed_cli"] = bool(runtime_ok)
        checks[f"{label}_endpoint_and_action_closure"] = bool(endpoint_ok)
        datasets[label] = {"endpoint": endpoint, "manifest": manifest, "trace": trace, "resources": resources, "result": result}

    by_label_seed = {
        label: {int(row["weather_seed"]): row for row in datasets[label]["endpoint"]}
        for label in ("5k", "10k")
    }
    manifests_by_label_seed = {
        label: {int(row["weather_seed"]): row for row in datasets[label]["manifest"]}
        for label in ("5k", "10k")
    }

    prior048 = {int(r["weather_seed"]): r for r in read_csv(ROOT / "results/fqa_wgen_10k_heldout_diagnostic_048/10k/episode_manifest.csv")}
    prior049 = {
        label: {int(r["weather_seed"]): r for r in read_csv(ROOT / f"results/fqa_047_checkpoint_learning_signal_049/{label}/episode_manifest.csv")}
        for label in ("5k", "10k")
    }
    same_weather = True
    prior_weather = True
    paired: list[dict] = []
    for seed in SEEDS:
        a, b = by_label_seed["5k"][seed], by_label_seed["10k"][seed]
        weather_hash = a["weather_sha256"]
        same_weather &= weather_hash == b["weather_sha256"]
        if seed in (1081, 1100):
            prior_weather &= weather_hash == prior048[seed]["weather_sha256"]
            for label in ("5k", "10k"):
                prior_weather &= weather_hash == prior049[label][seed]["weather_sha256"]
        item = {"year": 2007, "weather_seed": seed, "weather_sha256": weather_hash}
        for metric in METRICS:
            va, vb = float(a[metric]), float(b[metric])
            item[f"5k_{metric}"] = va
            item[f"10k_{metric}"] = vb
            item[f"delta_10k_minus_5k_{metric}"] = vb - va
        item["5k_action_counts_json"] = a["action_index_counts_json"]
        item["10k_action_counts_json"] = b["action_index_counts_json"]
        paired.append(item)

    checks["same_seed_same_weather_hash_5k_10k"] = bool(same_weather)
    checks["seed_1081_1100_match_048_049_weather"] = bool(prior_weather)
    checks["paired_schedule_exact_20"] = [int(row["weather_seed"]) for row in paired] == SEEDS

    summary: dict[str, dict] = {}
    for label in ("5k", "10k"):
        summary[label] = {
            metric: summarize([float(row[metric]) for row in datasets[label]["endpoint"]])
            for metric in METRICS
        }
        total_actions: dict[str, int] = {}
        episodes_with_action: dict[str, int] = {}
        for row in datasets[label]["endpoint"]:
            for action, count in json.loads(row["action_index_counts_json"]).items():
                total_actions[action] = total_actions.get(action, 0) + int(count)
                episodes_with_action[action] = episodes_with_action.get(action, 0) + 1
        summary[label]["total_policy_action_counts"] = dict(sorted(total_actions.items(), key=lambda x: int(x[0])))
        summary[label]["episodes_containing_action"] = dict(sorted(episodes_with_action.items(), key=lambda x: int(x[0])))

    delta_summary = {
        metric: summarize([float(row[f"delta_10k_minus_5k_{metric}"]) for row in paired])
        for metric in METRICS
    }
    yield_deltas = [float(row["delta_10k_minus_5k_yield_kg_ha"]) for row in paired]
    delta_directions = {
        "yield_10k_higher": sum(x > 0 for x in yield_deltas),
        "yield_10k_lower": sum(x < 0 for x in yield_deltas),
        "yield_equal": sum(x == 0 for x in yield_deltas),
    }
    checks["resources_all_below_limits"] = all(checks[f"{label}_resource_limits"] for label in ("5k", "10k"))
    status = "PASS_CHECKPOINT_FULL20_PAIRED_ARCHIVE_ONLY" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "checks": checks,
        "paired_seed_count": len(paired),
        "checkpoint_summaries": summary,
        "paired_delta_summaries_10k_minus_5k": delta_summary,
        "yield_direction_counts": delta_directions,
        "paired_results_csv": "results/fqa_047_checkpoint_full20_heldout_050/paired_checkpoint_full20.csv",
        "interpretation": "Deterministic policies differ between the same-run 5K and 10K checkpoints, but the 10K yield direction across 20 heldout 2007 WGEN realizations is descriptive and is not evidence of convergence or generalization. A full 20-seed run does not by itself prove that 10K is too short or justify starting 100K.",
        "limitations": [
            "one historical year (2007) and 20 synthetic WGEN heldout realizations",
            "no paired classical management control in this diagnostic",
            "wrapper cumulative irrigation and nitrogen were not reconciled against Summary.OUT",
            "PFP_N uses wrapper cumulative nitrogen",
            "no ETCP replay; WP_ET and NUE are unavailable",
            "descriptive paired summaries only; no significance or convergence claim",
        ],
        "resource_usage": {
            label: {
                "peak_rss_mb": datasets[label]["result"]["peak_rss_mb"],
                "elapsed_seconds": datasets[label]["result"]["elapsed_seconds"],
                "episodes": datasets[label]["result"]["episodes"],
                "trace_rows": datasets[label]["result"]["trace_rows"],
                "weather_days": datasets[label]["result"]["weather_days"],
            }
            for label in ("5k", "10k")
        },
    }
    if status == "PASS_CHECKPOINT_FULL20_PAIRED_ARCHIVE_ONLY":
        write_csv(BASE / "paired_checkpoint_full20.csv", paired)
    write_json(BASE / "final_gate.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status != "PASS_CHECKPOINT_FULL20_PAIRED_ARCHIVE_ONLY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
