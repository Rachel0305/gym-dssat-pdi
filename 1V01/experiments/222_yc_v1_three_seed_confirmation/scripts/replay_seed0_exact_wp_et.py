"""Read-only exact Summary.OUT/ETCP replay for YC V1.

This wrapper reuses the project's frozen YCA replay engine.  It never edits
055_00 or 221YCA outputs and writes each method/checkpoint into a separate
directory under the new confirmation experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
YCA_SRC = SRC / "055_yca_lowIC_site_transfer"
for entry in (SRC, YCA_SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_055_03_yca_lowIC_five_scenario_figures as yca_replay


EXPERIMENT = ROOT / "experiments" / "222_yc_v1_three_seed_confirmation"
RESULTS = EXPERIMENT / "results"
LOGS = EXPERIMENT / "logs"
CHECKPOINTS = [25_000, 50_000, 75_000, 100_000]
YEARS = list(range(2014, 2024))

POLICIES: dict[str, dict[str, Any]] = {
    "Original": {
        "config": ROOT / "configs" / "055_00_yca_lowIC_expanded_action_maskableppo.json",
        "ppo_root": ROOT / "benchmark_results" / "055_00_yca_lowIC_expanded_action_maskableppo",
        "input_root": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual",
        "source_label": "055_00_frozen",
    },
    "Augmented": {
        "config": ROOT / "experiments" / "221_yc_ppo_domain_augmentation" / "configs" / "221YCA_yc_ppo_domain_augmentation.json",
        "ppo_root": ROOT / "benchmark_results" / "221YCA_yc_ppo_domain_augmentation",
        "input_root": ROOT / "DSSAT_auto_validation" / "yca_lowIC_weather_scenario_bank_217_v1_fixed_width",
        "source_label": "221YCA_frozen",
    },
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def inventory_row(cfg: dict[str, Any], ppo_root: Path, checkpoint: int, year: int) -> pd.Series:
    inventory_path = yca_replay.resolve_training_inventory(ppo_root, checkpoint, preferred_prefix=str(cfg["task_id"]))
    validation_path = yca_replay.resolve_validation_summary(
        ppo_root, checkpoint, list(map(int, cfg["scope"]["validation_years"])), preferred_prefix=str(cfg["task_id"])
    )
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    validation = pd.read_csv(validation_path, keep_default_na=False)
    models = inventory[pd.to_numeric(inventory["checkpoint_step"], errors="coerce").eq(checkpoint)]
    rows = validation[
        pd.to_numeric(validation["checkpoint_step"], errors="coerce").eq(checkpoint)
        & pd.to_numeric(validation["year"], errors="coerce").eq(year)
    ]
    if len(models) == 0 or len(rows) != 1:
        raise RuntimeError(f"missing source row for {cfg['task_id']} checkpoint={checkpoint} year={year}")
    return rows.iloc[0]


def replay_one(method: str, meta: dict[str, Any], cfg: dict[str, Any], checkpoint: int, year: int) -> dict[str, Any]:
    # The existing YCA replay module uses this map to resolve the lowIC input root.
    yca_replay.INPUT_PROFILES["lowIC"] = Path(meta["input_root"])
    source = inventory_row(cfg, Path(meta["ppo_root"]), checkpoint, year)
    out = EXPERIMENT / "snapshots" / method / f"seed{int(cfg.get('seed', 0))}" / f"ckpt{checkpoint}"
    out.mkdir(parents=True, exist_ok=True)
    snapshot = yca_replay.replay_ppo_snapshot(cfg, checkpoint, year, out, Path(meta["ppo_root"]))
    daily = yca_replay.daily_from_snapshot(snapshot, year, "rl_candidate")
    final_yield = float(pd.to_numeric(daily["grain_yield_kg_ha"], errors="coerce").iloc[-1])
    expected_i = float(pd.to_numeric(source.get("total_irrigation"), errors="coerce"))
    expected_n = float(pd.to_numeric(source.get("total_n"), errors="coerce"))
    metrics = yca_replay.baseline.metrics_from_snapshot(snapshot, final_yield)
    row = {
        "method": method,
        "source_run": meta["source_label"],
        "task_id": cfg["task_id"],
        "seed": int(cfg.get("seed", 0)),
        "checkpoint": checkpoint,
        "evaluation_year": year,
        "yield": final_yield,
        "ETCP": float(metrics["etcp_mm"]),
        "WP_ET": float(metrics["WP_ET_kg_m3"]),
        "total_irrigation": float(metrics["actual_irrigation_mm"]),
        "total_nitrogen": float(metrics["actual_nitrogen_kg_ha"]),
        "PFP_N": float(metrics["PFP_N_kg_kg"]) if pd.notna(metrics["PFP_N_kg_kg"]) else np.nan,
        "reward": float(pd.to_numeric(source.get("reward_stress_aware_sum"), errors="coerce")),
        "source_yield": float(pd.to_numeric(source.get("final_grnwt"), errors="coerce")),
        "source_total_irrigation": expected_i,
        "source_total_nitrogen": expected_n,
        "yield_abs_diff_vs_source": final_yield - float(pd.to_numeric(source.get("final_grnwt"), errors="coerce")),
        "irrigation_abs_diff_vs_source": float(metrics["actual_irrigation_mm"]) - expected_i,
        "nitrogen_abs_diff_vs_source": float(metrics["actual_nitrogen_kg_ha"]) - expected_n,
        "summary_match_score": float(metrics["summary_match_score"]),
        "summary_row_index": int(metrics["summary_row_index"]),
        "snapshot_path": rel(snapshot),
        "status": "ok",
    }
    return row


def selected_policies(all_seeds: bool) -> list[tuple[str, dict[str, Any]]]:
    """Return frozen seed-0 policies or the paired three-seed set."""
    if not all_seeds:
        return list(POLICIES.items())
    selected: list[tuple[str, dict[str, Any]]] = list(POLICIES.items())
    for seed in (1, 2):
        for method, base in POLICIES.items():
            task_id = f"222YCA_{'O' if method == 'Original' else 'A'}{seed}"
            task_name = f"yc_v1_{'original' if method == 'Original' else 'augmented'}_seed{seed}"
            meta = dict(base)
            meta.update(
                {
                    "config": EXPERIMENT / "configs" / f"{task_id}_{task_name}.json",
                    "ppo_root": ROOT / "benchmark_results" / f"{task_id}_{task_name}",
                    "source_label": f"{task_id}_formal",
                }
            )
            selected.append((method, meta))
    return selected


def run(smoke: bool, all_seeds: bool) -> dict[str, Any]:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    checkpoints = [25_000] if smoke else CHECKPOINTS
    years = [2014] if smoke else YEARS
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    policies = selected_policies(all_seeds)
    for method, meta in policies:
        cfg = read_json(Path(meta["config"]))
        for checkpoint in checkpoints:
            for year in years:
                try:
                    row = replay_one(method, meta, cfg, checkpoint, year)
                    rows.append(row)
                    print(json.dumps({"method": method, "checkpoint": checkpoint, "year": year, "WP_ET": row["WP_ET"], "status": "ok"}, ensure_ascii=False), flush=True)
                except Exception:
                    failure = {
                        "method": method,
                        "checkpoint": checkpoint,
                        "year": year,
                        "status": "failed",
                        "error": traceback.format_exc(),
                    }
                    failures.append(failure)
                    print(json.dumps(failure, ensure_ascii=False), flush=True)
    by_year = pd.DataFrame(rows)
    if smoke:
        out_csv = RESULTS / "yc_v1_exact_efficiency_replay_smoke.csv"
    elif all_seeds:
        out_csv = RESULTS / "yc_v1_exact_efficiency_replay_all_seeds.csv"
    else:
        out_csv = RESULTS / "yc_v1_exact_efficiency_replay.csv"
    by_year.to_csv(out_csv, index=False, encoding="utf-8-sig")
    payload = {
        "task": "yc_v1_exact_efficiency_replay",
        "phase": "smoke" if smoke else ("full_three_seeds" if all_seeds else "full_seed0"),
        "policies": [f"{method}_seed{int(read_json(Path(meta['config'])).get('seed', 0))}" for method, meta in policies],
        "checkpoints": checkpoints,
        "years": years,
        "rows_completed": len(by_year),
        "rows_expected": len(policies) * len(checkpoints) * len(years),
        "failures": failures,
        "output_csv": rel(out_csv),
        "metric_definition": "WP_ET = YPEM*0.1 when valid, otherwise HWAM/(ETCP*10); ETCP is Summary.OUT ETCP in mm; PFP_N = YPNAM when NICM > 0.",
        "source_engine": "src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py",
        "read_only_replay": True,
    }
    if smoke:
        manifest = LOGS / "yc_v1_exact_efficiency_replay_smoke.json"
    elif all_seeds:
        manifest = LOGS / "yc_v1_exact_efficiency_replay_all_seeds.json"
    else:
        manifest = LOGS / "yc_v1_exact_efficiency_replay_seed0.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError(f"exact replay had {len(failures)} failure(s)")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Replay one year at 25K per method")
    parser.add_argument("--all-seeds", action="store_true", help="Replay seed 0, 1, and 2 for both methods")
    args = parser.parse_args()
    run(smoke=args.smoke, all_seeds=args.all_seeds)
