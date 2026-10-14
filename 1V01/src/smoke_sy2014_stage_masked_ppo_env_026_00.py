from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_sy2014_stage_mc_dqn_seed1_short_022_02 import summary_metrics
from stage_based_dqn_core_022 import terminal_complete_returns
from stage_decision_env_ppo_026 import StageDecisionEnv026


OUT = ROOT / "benchmark_results" / "026_00"
SCALER = ROOT / "benchmark_results" / "021_24" / "021_24_observation_scaler.csv"
THRESHOLDS = ROOT / "benchmark_results" / "022_01" / "022_01_thresholds_and_provenance.json"
ACTIONS = (3, 4, 7, 1, 1, 0)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    env = StageDecisionEnv026(OUT / "runtime", SCALER, seed=1)
    rewards = []
    mask_rows = []
    try:
        obs, info = env.reset(seed=1)
        for index, action in enumerate(ACTIONS):
            mask = env.action_masks()
            mask_rows.append({"stage_index": index, "dap": info["dap"], "selected_action": action, "selected_is_valid": bool(mask[action]), "valid_actions": ",".join(map(str, mask.nonzero()[0].tolist()))})
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
        result = env.last_result
        if result is None:
            raise RuntimeError("Missing terminal result")
        thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
        metrics = summary_metrics(env.raw_env, result, thresholds)
    finally:
        env.close()
    expected_g0 = terminal_complete_returns(result["executed_actions"], result["final_yield"])[0] / 1000.0
    total_reward = sum(rewards)
    checks = {
        "six_stage_decisions": len(result["stage_rows"]) == 6,
        "all_selected_actions_mask_valid": all(row["selected_is_valid"] for row in mask_rows),
        "yield_11202_pm2": abs(result["final_yield"] - 11202.00439453125) <= 2.0,
        "irrigation_total_60": abs(result["irrigation_total"] - 60.0) <= 1e-6,
        "nitrogen_total_200": abs(result["nitrogen_total"] - 200.0) <= 1e-6,
        "reward_closure": abs(total_reward - expected_g0) <= 1e-9,
        "reward_6p354_pm0p002": abs(total_reward - 6.35400439453125) <= 0.002,
        "wp_et_2p31": abs(metrics["WP_ET_kg_m3"] - 2.31) <= 0.01,
        "pfp_n_56": abs(metrics["PFP_N_kg_kg"] - 56.0) <= 0.1,
        "summary_resource_match": metrics["summary_match_score"] <= 2.0,
        "finite_rewards": all(math.isfinite(v) for v in rewards),
    }
    pd.DataFrame(mask_rows).to_csv(OUT / "026_00_stage_action_masks.csv", index=False)
    pd.DataFrame(result["stage_rows"]).to_csv(OUT / "026_00_stage_actions.csv", index=False)
    sb3_contrib_available = importlib.util.find_spec("sb3_contrib") is not None
    passed = all(checks.values())
    branch = "A_environment_ready" if passed and sb3_contrib_available else ("A_environment_ready_dependency_missing" if passed else "C_environment_failed")
    payload = {
        "status": "completed",
        "branch": branch,
        "checks": checks,
        "sb3_version": "2.8.0",
        "sb3_contrib_available": sb3_contrib_available,
        "ppo_training_steps": 0,
        "dssat_seasons": 1,
        "total_reward": total_reward,
        "expected_g0_scaled": expected_g0,
        "final_yield": result["final_yield"],
        "irrigation_total": result["irrigation_total"],
        "nitrogen_total": result["nitrogen_total"],
        "next_step_allowed": passed and sb3_contrib_available,
        "next_requirement": None if sb3_contrib_available else "explicit_user_approval_to_install_matching_sb3_contrib_in_container_venv",
    }
    (OUT / "026_00_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
