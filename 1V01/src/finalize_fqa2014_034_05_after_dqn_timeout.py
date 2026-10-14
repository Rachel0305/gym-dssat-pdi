from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

import run_fqa2014_linked_free_timing_ppo_dqn_50k_comparison_034_05 as task


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    start = time.time()
    task.ensure_dirs()
    config, env_config = task.load_config_and_env()
    ppo_model = task.OUT / "models" / "FQA" / "maskableppo_stress_aware_seed0.zip"
    train = pd.DataFrame(
        [
            {
                "algorithm": "MaskablePPO",
                "station_code": "FQA",
                "train_years": "2014",
                "seed": 0,
                "total_timesteps": 50000,
                "run_status": "ok",
                "model_path": str(ppo_model.relative_to(ROOT)).replace("\\", "/") if ppo_model.exists() else "",
                "notes": "PPO model completed before DQN runtime stop.",
                "task": task.TASK_ID,
                "linked_management_expected": True,
            },
            {
                "algorithm": "DQN",
                "station_code": "FQA",
                "train_years": "2014",
                "seed": 0,
                "total_timesteps": 50000,
                "run_status": "runtime_stopped",
                "model_path": "",
                "notes": "DQN 50K did not finish within the smoke waiting budget and was stopped to save compute; no result is reported.",
                "task": task.TASK_ID,
                "linked_management_expected": True,
            },
        ]
    )
    eval_rows = []
    failures = [
        {
            "phase": "train",
            "algorithm": "DQN",
            "status": "runtime_stopped",
            "traceback": "No exception. Process was manually stopped after PPO completed and DQN 50K produced no model within the smoke waiting budget.",
        }
    ]
    if ppo_model.exists():
        eval_rows.append(task.smoke03404.evaluate_with_snapshot(config, env_config, train.iloc[0]))
    else:
        failures.append({"phase": "eval", "algorithm": "MaskablePPO", "status": "missing_model", "traceback": str(ppo_model)})
    eval_df = pd.concat(eval_rows, ignore_index=True, sort=False) if eval_rows else pd.DataFrame()
    comp = task.scenario_comparison(eval_df) if not eval_df.empty else pd.DataFrame()
    failures_df = pd.DataFrame(failures)
    train.to_csv(task.OUT / "evaluation" / "034_05_training_summary.csv", index=False, encoding="utf-8-sig")
    eval_df.to_csv(task.OUT / "evaluation" / "034_05_eval_summary.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(task.OUT / "evaluation" / "034_05_scenario_comparison.csv", index=False, encoding="utf-8-sig")
    failures_df.to_csv(task.OUT / "evaluation" / "034_05_failures.csv", index=False, encoding="utf-8-sig")
    task.write_record(config, train, eval_df, comp, failures_df, time.time() - start)
    result = {
        "task": task.TASK_ID,
        "record_md": str(task.DOC.relative_to(ROOT)).replace("\\", "/"),
        "ppo_evaluated": bool(len(eval_df)),
        "dqn_status": "runtime_stopped",
        "comparison_csv": str((task.OUT / "evaluation" / "034_05_scenario_comparison.csv").relative_to(ROOT)).replace("\\", "/"),
    }
    (task.OUT / "034_05_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not comp.empty:
        print(comp[[
            "model_type",
            "scenario",
            "grain_yield_kg_ha",
            "actual_irrigation_mm",
            "actual_nitrogen_kg_ha",
            "WP_ET_kg_m3",
            "PFP_N_kg_kg",
            "simple_profit",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
