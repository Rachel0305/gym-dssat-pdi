#!/usr/bin/env python3
"""Persist deterministic snapshots for existing 027_07 selected PPO models."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as old


SOURCE = ROOT / "benchmark_results" / "027_07_site_specific_stage_maskable_ppo_attempt2"
OUT = ROOT / "benchmark_results" / "028_04_existing_yc_fq_lc_ppo_frozen_daily"
SELECTED = SOURCE / "027_07_three_site_selected_seed_summary.csv"
CASES = {("YC", 0), ("YC", 2), ("LC", 0), ("FQ", 0)}


def close(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    selected = pd.read_csv(SELECTED)
    selected = selected[selected.apply(lambda r: (str(r["site"]), int(r["seed"])) in CASES, axis=1)]
    if len(selected) != len(CASES):
        raise RuntimeError(f"Expected {len(CASES)} selected rows, found {len(selected)}")

    results = []
    action_frames = []
    for item in selected.sort_values(["site", "seed"]).to_dict("records"):
        site = str(item["site"])
        seed = int(item["seed"])
        checkpoint = int(item["selected_checkpoint"])
        spec = old.SPECS[site]
        site_source = SOURCE / site
        readiness, scaler, baselines = old.load_readiness(spec, site_source)
        model_path = site_source / f"seed{seed}" / f"checkpoint_{checkpoint:06d}.zip"
        expected_hash = str(item["model_sha256"])
        actual_hash = old.sha256(model_path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"{site}/seed{seed}: model hash mismatch")

        case_out = OUT / site / f"seed{seed}_checkpoint{checkpoint}"
        runtime = case_out / "runtime_eval"
        env = old.make_stage_env(
            spec,
            runtime,
            scaler,
            readiness["reward_config"],
            seed=28040 + seed,
            phase="frozen_daily_evidence",
        )
        try:
            model = MaskablePPO.load(model_path, device="cpu")
            before = int(model.num_timesteps)
            result, stages = old.evaluate(model, env, checkpoint, baselines)
            after = int(model.num_timesteps)
            if before != after:
                raise RuntimeError(f"{site}/seed{seed}: num_timesteps changed during frozen evaluation")
            source_snapshot = old.snapshot_from_env(env.raw_env)
            snapshot = case_out / "snapshot"
            shutil.copytree(source_snapshot, snapshot)
        finally:
            env.close()

        checks = {
            "model_hash_matches": actual_hash == expected_hash,
            "learn_calls_zero": before == after,
            "yield_matches_027_07": close(result["final_yield"], item["yield"], 1e-6),
            "irrigation_matches_027_07": close(result["irrigation_total"], item["irrigation"], 1e-9),
            "nitrogen_matches_027_07": close(result["nitrogen_total"], item["nitrogen"], 1e-9),
            "wp_matches_027_07": close(result["WP_ET_kg_m3"], item["WP_ET"], 1e-9),
            "snapshot_summary_exists": (snapshot / "Summary.OUT").exists(),
            "snapshot_plantgro_exists": (snapshot / "PlantGro.OUT").exists(),
            "snapshot_weather_exists": (snapshot / "Weather.OUT").exists(),
            "snapshot_soilwat_exists": (snapshot / "SoilWat.OUT").exists(),
        }
        expected_pfp = item.get("PFP_N")
        if pd.notna(expected_pfp):
            checks["pfp_matches_027_07"] = close(result["PFP_N_kg_kg"], expected_pfp, 1e-9)
        if not all(checks.values()):
            raise RuntimeError(f"{site}/seed{seed}: frozen evidence checks failed: {checks}")

        case_out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(stages).to_csv(case_out / "stage_actions.csv", index=False, encoding="utf-8-sig")
        payload = {
            "site": site,
            "year": spec.year,
            "seed": seed,
            "checkpoint": checkpoint,
            "model_path": str(model_path.relative_to(ROOT)),
            "model_sha256": actual_hash,
            "snapshot_path": str(snapshot.relative_to(ROOT)),
            "learn_calls": 0,
            "checks": checks,
            "metrics": result,
        }
        (case_out / "frozen_evidence.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
        )
        results.append(payload)
        action_frames.append(pd.DataFrame([{"site": site, "year": spec.year, "seed": seed, **r} for r in stages]))

    summary_rows = []
    for payload in results:
        m = payload["metrics"]
        summary_rows.append({
            "site": payload["site"], "year": payload["year"], "seed": payload["seed"],
            "checkpoint": payload["checkpoint"], "model_sha256": payload["model_sha256"],
            "yield": m["final_yield"], "biomass": m["final_biomass"],
            "irrigation": m["irrigation_total"], "nitrogen": m["nitrogen_total"],
            "WP_ET": m["WP_ET_kg_m3"], "PFP_N": m["PFP_N_kg_kg"],
            "reward": m["episode_total_reward"],
            "advisor_any_metric_strict_winner": m["advisor_any_metric_strict_winner"],
            "snapshot_path": payload["snapshot_path"], "learn_calls": 0,
        })
    pd.DataFrame(summary_rows).to_csv(OUT / "028_04_frozen_ppo_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(action_frames, ignore_index=True).to_csv(OUT / "028_04_frozen_ppo_stage_actions.csv", index=False, encoding="utf-8-sig")
    result_doc = {
        "status": "completed",
        "cases": len(results),
        "training_calls": 0,
        "dssat_seasons": len(results),
        "all_checks_passed": True,
    }
    (OUT / "028_04_result.json").write_text(json.dumps(result_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result_doc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
