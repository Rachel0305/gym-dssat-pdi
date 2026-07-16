from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calculate_five_site_wue_nue_from_summary_019_10 import num, parse_summary_out, select_matching_row
from run_sy2014_stage_mc_dqn_seed1_short_022_02 import FixedObservationScaler
from run_fq_yc_new_cultivar_forward_screening_013_01 import (
    set_management_for_treatment,
    set_treatment_pointers,
    zero_target_reported_rows,
)
from stage_decision_env_ppo_026 import StageDecisionEnv026
import run_extension_expert_baseline_018_03 as expert
import run_sy_local_dqn_train_cross_year_transfer_017_08 as sy


OUT = ROOT / "benchmark_results" / "026_05"
INPUT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "SY"
SCALER = ROOT / "benchmark_results" / "021_24" / "021_24_observation_scaler.csv"
MODELS = {
    0: ROOT / "benchmark_results" / "026_03" / "checkpoint_000120.zip",
    1: ROOT / "benchmark_results" / "026_02" / "checkpoint_000060.zip",
    2: ROOT / "benchmark_results" / "026_04" / "checkpoint_000240.zip",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_from_snapshot(snapshot: Path, final_yield: float, irrigation: float, nitrogen: float) -> dict[str, Any]:
    rows = parse_summary_out(snapshot / "Summary.OUT")
    scored = []
    for index, candidate in enumerate(rows):
        hwam, ircm = num(candidate, "HWAM"), num(candidate, "IRCM")
        if hwam is None or ircm is None:
            continue
        score = abs(hwam - final_yield) + abs(ircm - irrigation)
        scored.append((score, -index, candidate, index))
    if not scored:
        raise ValueError("No Summary.OUT row contains numeric HWAM/IRCM")
    score, _, row, row_index = min(scored, key=lambda item: (item[0], item[1]))
    if abs(float(num(row, "HWAM")) - final_yield) > 2.0:
        raise ValueError(f"Yield mismatch: expected {final_yield}, got {num(row, 'HWAM')}")
    if abs(float(num(row, "IRCM")) - irrigation) > 2.0:
        raise ValueError(f"Irrigation mismatch: expected {irrigation}, got {num(row, 'IRCM')}")
    ircm, nicm = num(row, "IRCM"), num(row, "NICM")
    etcp, ypem, ypnam = num(row, "ETCP"), num(row, "YPEM"), num(row, "YPNAM")
    wp = ypem * 0.1 if ypem is not None and ypem >= 0 else final_yield / etcp / 10.0
    pfp = ypnam if nicm is not None and nicm > 0 and ypnam is not None and ypnam >= 0 else math.nan
    return {
        "summary_irrigation_total": ircm,
        "summary_nitrogen_total": nicm,
        "etcp_mm": etcp,
        "WP_ET_kg_m3": wp,
        "PFP_N_kg_kg": pfp,
        "summary_match_score": score,
        "summary_row_index": row_index,
        "management_or_action_n_total": nitrogen,
        "n_accounting_difference_vs_summary": nitrogen - float(nicm) if nicm is not None else math.nan,
    }


class TransferStageEnv(StageDecisionEnv026):
    def __init__(self, raw_env: Any, scaler_path: Path) -> None:
        gym.Env.__init__(self)
        self.raw_env = raw_env
        self.scaler = FixedObservationScaler(scaler_path)
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float32)
        self.obs = None
        self.state: dict[str, Any] = {}
        self.stage_index = 0
        self.used_i = 0.0
        self.used_n = 0.0
        self.executed = []
        self.stage_rows: list[dict[str, Any]] = []
        self.last_result: dict[str, Any] | None = None
        self.invalid_attempts = 0

    def step(self, action_index: int):
        mask = self.action_masks()
        index = int(action_index)
        if index < 0 or index >= len(mask) or not bool(mask[index]):
            self.invalid_attempts += 1
        return super().step(index)


def prepare_sy2012_linked_run(scenario: str, seed: int, root: Path) -> tuple[Path, dict[str, Any]]:
    preparation_scenario = scenario if scenario.startswith(("dqn", "transfer")) else f"transfer_{scenario}"
    run_dir = sy.prepare_run_dir(2012, preparation_scenario, seed=seed, root=root)
    input_dir = run_dir / "input"
    filex = input_dir / "CNSY1201.MZX"
    source = (INPUT / "CNSY1201.MZX").read_text(encoding="latin-1", errors="ignore")
    text = set_treatment_pointers(source, 1, "1", "1")
    text = set_management_for_treatment(text, 1, "L", "L")
    text = zero_target_reported_rows(text, 1)
    filex.write_text(text, encoding="latin-1", errors="ignore")
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env_args["fileX_template_path"] = str(filex)
    (run_dir / "env_args.json").write_text(json.dumps(env_args, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir, env_args


def run_baselines() -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    sy.OUT_DIR = OUT / "baseline_source"
    sy.configure_globals()
    rows: list[dict[str, Any]] = []
    for scenario in ("null", "recorded", "dssat_auto"):
        _, _, summary = sy.run_zero_action(2012, scenario)
        snapshot = ROOT / summary["run_dir"] / "pdi_tmp_snapshot_eval"
        metrics = metrics_from_snapshot(snapshot, summary["final_gwad"], summary["irrigation_total"], summary["fertilizer_total"])
        rows.append({"scenario": scenario, **summary, **metrics})

    case = {"site": "SY", "station": "Shenyang", "year": 2012, "region": "northeast_greatwall_spring_maize"}
    schedule = expert.build_region_schedule()
    schedule = schedule[schedule["region"].eq(case["region"])].copy()
    expert_run = OUT / "expert_source" / "runs" / "SY2012"
    expert_run, env_args = prepare_sy2012_linked_run("official_extension_expert", 0, OUT / "expert_source" / "runs")
    _, _, summary = expert.run_fixed_schedule(case, env_args, schedule, expert_run)
    snapshot = expert_run / "pdi_tmp_snapshot_eval"
    irrigation = float(summary["action_irrigation_total"])
    nitrogen = float(summary["action_fertilizer_total"])
    metrics = metrics_from_snapshot(snapshot, summary["final_gwad"], irrigation, nitrogen)
    rows.append({
        "scenario": "official_extension_expert",
        "final_gwad": summary["final_gwad"],
        "final_cwad": summary["final_cwad"],
        "irrigation_total": irrigation,
        "fertilizer_total": nitrogen,
        "run_dir": str(expert_run.relative_to(ROOT)),
        **metrics,
    })

    frame = pd.DataFrame(rows)
    current_mzx = (INPUT / "CNSY1201.MZX").read_text(encoding="latin-1", errors="ignore")
    provenance = {
        "mzx_sha256": sha256(INPUT / "CNSY1201.MZX"),
        "weather_sha256": sha256(INPUT / "CNSY1201.WTH"),
        "soil_sha256": sha256(INPUT / "SOIL.SOL"),
        "cultivar_sha256": sha256(INPUT / "MZCER048.CUL"),
        "treatment_2012_present": "1 1 1 0 Sim2012" in current_mzx,
        "treatment_2012_ic1_pattern": " 1  1  0  1  1" in current_mzx,
    }
    required = {"null", "recorded", "dssat_auto", "official_extension_expert"}
    finite_core = bool(np.isfinite(frame[["final_gwad", "irrigation_total", "fertilizer_total", "WP_ET_kg_m3"]].to_numpy(dtype=float)).all())
    expert_positive_n = float(frame.loc[frame["scenario"].eq("official_extension_expert"), "summary_nitrogen_total"].iloc[0]) > 0
    passed = set(frame["scenario"]) == required and finite_core and expert_positive_n and all(provenance[key] for key in ("treatment_2012_present", "treatment_2012_ic1_pattern"))
    return rows, passed, provenance


def local_targets(baselines: pd.DataFrame) -> dict[str, float]:
    main = baselines[baselines["scenario"].isin(["dssat_auto", "official_extension_expert"])].copy()
    positive_n = main[pd.to_numeric(main["summary_nitrogen_total"], errors="coerce") > 0]
    return {
        "yield_min": float(main["final_gwad"].max()),
        "wp_et_min": float(main["WP_ET_kg_m3"].max()),
        "pfp_n_min": float(positive_n["PFP_N_kg_kg"].max()),
    }


def run_frozen_model(seed: int, model_path: Path, targets: dict[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scenario = f"frozen_ppo_seed{seed}"
    run_dir, env_args = prepare_sy2012_linked_run(scenario, seed, OUT / "ppo_runs")
    raw_env = sy.make_raw_env(env_args)
    env = TransferStageEnv(raw_env, SCALER)
    model_hash_before = sha256(model_path)
    actions: list[dict[str, Any]] = []
    total_reward = 0.0
    try:
        model = MaskablePPO.load(model_path, device="cpu")
        obs, info = env.reset()
        done = False
        while not done:
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            stage_index = env.stage_index
            dap = int(info.get("dap", env._dap()))
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            executed = env.stage_rows[-1]
            actions.append({"seed": seed, "stage_index": stage_index, "dap": dap, "action_index": int(action), "mask_valid": bool(mask[int(action)]), **executed})
            done = bool(terminated or truncated)
        if env.last_result is None:
            raise RuntimeError("Frozen PPO season ended without result")
        tmp = Path(getattr(env.raw_env.unwrapped, "_tmp_folder"))
        metrics = metrics_from_snapshot(tmp, env.last_result["final_yield"], env.last_result["irrigation_total"], env.last_result["nitrogen_total"])
        result = {
            "scenario": scenario,
            "seed": seed,
            "source_model": str(model_path.relative_to(ROOT)),
            "model_sha256": model_hash_before,
            "action_sequence": ",".join(str(row["action_index"]) for row in actions),
            "final_gwad": env.last_result["final_yield"],
            "final_cwad": env.last_result["final_biomass"],
            "irrigation_total": env.last_result["irrigation_total"],
            "fertilizer_total": env.last_result["nitrogen_total"],
            "frozen_sy2014_reward": total_reward,
            "invalid_action_attempts": env.invalid_attempts,
            **metrics,
        }
        result["yield_pass"] = result["final_gwad"] >= targets["yield_min"]
        result["wp_et_pass"] = result["WP_ET_kg_m3"] >= targets["wp_et_min"]
        result["pfp_n_pass"] = result["PFP_N_kg_kg"] >= targets["pfp_n_min"]
        result["local_primary_pass"] = bool(result["yield_pass"] and result["wp_et_pass"] and result["pfp_n_pass"])
    finally:
        env.close()
    if sha256(model_path) != model_hash_before:
        raise RuntimeError(f"Frozen model hash changed: {model_path}")
    return result, actions


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    missing = [str(path) for path in MODELS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen models: {missing}")
    model_hashes = {str(seed): sha256(path) for seed, path in MODELS.items()}
    baseline_rows, baselines_passed, provenance = run_baselines()
    baselines = pd.DataFrame(baseline_rows)
    baselines.to_csv(OUT / "026_05_sy2012_four_baselines.csv", index=False)
    if not baselines_passed:
        payload = {"status": "completed", "branch": "C_baseline_or_execution_blocked", "provenance": provenance, "model_hashes": model_hashes, "training_steps": 0, "next_step_allowed": False}
        (OUT / "026_05_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    targets = local_targets(baselines)
    results: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for seed, model_path in MODELS.items():
        result, actions = run_frozen_model(seed, model_path, targets)
        results.append(result)
        action_rows.extend(actions)
    result_frame = pd.DataFrame(results)
    pass_count = int(result_frame["local_primary_pass"].sum())
    engineering_checks = {
        "four_baselines_complete": baselines_passed,
        "three_models_evaluated": len(results) == 3,
        "zero_training_steps": True,
        "all_model_hashes_unchanged": all(sha256(MODELS[int(seed)]) == digest for seed, digest in model_hashes.items()),
        "zero_invalid_actions": int(result_frame["invalid_action_attempts"].sum()) == 0,
        "all_metrics_finite": bool(np.isfinite(result_frame[["final_gwad", "irrigation_total", "fertilizer_total", "WP_ET_kg_m3", "PFP_N_kg_kg"]].to_numpy(dtype=float)).all()),
    }
    engineering_pass = all(engineering_checks.values())
    transfer_pass = engineering_pass and pass_count >= 2
    branch = "A_fixed_models_transfer" if transfer_pass else ("B_models_do_not_transfer" if engineering_pass else "C_baseline_or_execution_blocked")
    result_frame.to_csv(OUT / "026_05_frozen_ppo_transfer_summary.csv", index=False)
    pd.DataFrame(action_rows).to_csv(OUT / "026_05_frozen_ppo_stage_actions.csv", index=False)
    comparison = pd.concat([baselines.assign(group="baseline"), result_frame.assign(group="frozen_ppo")], ignore_index=True, sort=False)
    comparison.to_csv(OUT / "026_05_all_scenarios_summary.csv", index=False)
    payload = {
        "status": "completed",
        "branch": branch,
        "provenance": provenance,
        "model_hashes": model_hashes,
        "local_targets": targets,
        "engineering_checks": engineering_checks,
        "local_primary_pass_count": pass_count,
        "frozen_model_results": results,
        "training_steps": 0,
        "scientific_success_claimed": False,
        "next_step_allowed": bool(transfer_pass),
    }
    (OUT / "026_05_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
