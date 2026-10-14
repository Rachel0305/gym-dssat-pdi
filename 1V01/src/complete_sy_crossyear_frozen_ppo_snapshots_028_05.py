#!/usr/bin/env python3
"""Persist SY2012/2015 frozen PPO snapshots without training."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_sy_all_authoritative_years_frozen_stage_ppo_026_06 as base
import run_sy_icdat_aligned_all_years_frozen_stage_ppo_026_07 as aligned


SOURCE = ROOT / "benchmark_results" / "026_07_attempt2"
OUT = ROOT / "benchmark_results" / "028_05_sy_crossyear_frozen_ppo_daily"
YEARS = (2012, 2015)


def close(a: float, b: float, tol: float) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return abs(float(a) - float(b)) <= tol


def evaluate(year: int, seed: int, model_path: Path, targets: dict[str, float]) -> tuple[dict, list[dict], Path]:
    scenario = f"frozen_ppo_seed{seed}"
    case_out = OUT / str(year) / f"seed{seed}"
    run_dir, env_args = aligned.prepare_linked_run_aligned(year, scenario, seed, case_out / "runtime")
    env = base.TransferStageEnv(base.sy.make_raw_env(env_args), base.SCALER)
    model_hash = base.sha256(model_path)
    actions = []
    total_reward = 0.0
    try:
        model = base.MaskablePPO.load(model_path, device="cpu")
        before = int(model.num_timesteps)
        obs, info = env.reset()
        done = False
        while not done:
            mask = base.get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            stage_index = env.stage_index
            dap = int(info.get("dap", env._dap()))
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            actions.append({
                "year": year, "seed": seed, "stage_index": stage_index, "dap": dap,
                "action_index": int(action), "mask_valid": bool(mask[int(action)]), **env.stage_rows[-1],
            })
            done = bool(terminated or truncated)
        after = int(model.num_timesteps)
        if before != after:
            raise RuntimeError(f"SY{year}/seed{seed}: num_timesteps changed")
        if env.last_result is None:
            raise RuntimeError("Frozen season ended without result")
        tmp = Path(getattr(env.raw_env.unwrapped, "_tmp_folder"))
        snapshot = case_out / "snapshot"
        shutil.copytree(tmp, snapshot)
        metrics = base.metrics_from_snapshot(
            snapshot, env.last_result["final_yield"], env.last_result["irrigation_total"], env.last_result["nitrogen_total"]
        )
        result = {
            "year": year, "scenario": scenario, "seed": seed,
            "source_model": str(model_path.relative_to(ROOT)), "model_sha256": model_hash,
            "action_sequence": ",".join(str(r["action_index"]) for r in actions),
            "final_gwad": env.last_result["final_yield"], "final_cwad": env.last_result["final_biomass"],
            "irrigation_total": env.last_result["irrigation_total"], "fertilizer_total": env.last_result["nitrogen_total"],
            "frozen_training_objective_return": total_reward, "invalid_action_attempts": env.invalid_attempts,
            "learn_calls": 0, "num_timesteps_before": before, "num_timesteps_after": after,
            "snapshot_path": str(snapshot.relative_to(ROOT)), **metrics,
        }
        result["yield_pass"] = result["final_gwad"] >= targets["yield_min"]
        result["wp_et_pass"] = result["WP_ET_kg_m3"] >= targets["wp_et_min"]
        result["pfp_n_pass"] = result["PFP_N_kg_kg"] >= targets["pfp_n_min"]
        result["local_primary_pass"] = bool(result["yield_pass"] and result["wp_et_pass"] and result["pfp_n_pass"])
        return result, actions, snapshot
    finally:
        env.close()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    aligned.install_runtime_patch(OUT)
    published = pd.read_csv(SOURCE / "026_07_sy_all_years_frozen_ppo_summary.csv")
    rows, action_rows, checks = [], [], []
    for year in YEARS:
        baselines = pd.read_csv(SOURCE / f"026_07_sy{year}_four_baselines.csv")
        targets = base.local_targets(baselines)
        for seed, model_path in sorted(base.MODELS.items()):
            expected = published[(published["year"].eq(year)) & (published["seed"].eq(seed))].iloc[0]
            result, actions, snapshot = evaluate(year, seed, model_path, targets)
            case_checks = {
                "model_hash_matches": result["model_sha256"] == str(expected["model_sha256"]),
                "learn_calls_zero": result["learn_calls"] == 0 and result["num_timesteps_before"] == result["num_timesteps_after"],
                "action_sequence_matches": result["action_sequence"] == str(expected["action_sequence"]),
                "yield_matches": close(result["final_gwad"], expected["final_gwad"], 1e-6),
                "irrigation_matches": close(result["irrigation_total"], expected["irrigation_total"], 1e-9),
                "nitrogen_matches": close(result["fertilizer_total"], expected["fertilizer_total"], 1e-9),
                "wp_matches": close(result["WP_ET_kg_m3"], expected["WP_ET_kg_m3"], 1e-9),
                "pfp_matches": close(result["PFP_N_kg_kg"], expected["PFP_N_kg_kg"], 1e-9),
                "original_mzx_unchanged": base.sha256(base.MZX) == aligned.ORIGINAL_MZX_SHA256,
                "snapshot_complete": all((snapshot / n).exists() for n in ("Summary.OUT", "PlantGro.OUT", "Weather.OUT", "SoilWat.OUT")),
            }
            if not all(case_checks.values()):
                raise RuntimeError(f"SY{year}/seed{seed}: checks failed {case_checks}")
            case_dir = OUT / str(year) / f"seed{seed}"
            pd.DataFrame(actions).to_csv(case_dir / "stage_actions.csv", index=False, encoding="utf-8-sig")
            (case_dir / "frozen_evidence.json").write_text(json.dumps({"metrics": result, "checks": case_checks}, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(result)
            action_rows.extend(actions)
            checks.append({"year": year, "seed": seed, **case_checks})
    pd.DataFrame(rows).to_csv(OUT / "028_05_sy_crossyear_frozen_ppo_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(action_rows).to_csv(OUT / "028_05_sy_crossyear_stage_actions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(checks).to_csv(OUT / "028_05_reproduction_checks.csv", index=False, encoding="utf-8-sig")
    result = {"status": "completed", "cases": len(rows), "training_calls": 0, "dssat_seasons": len(rows), "all_checks_passed": True}
    (OUT / "028_05_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
