from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from custom_double_dqn import CustomDoubleDQN
from ppo_evaluate import latest_observation_dict, scalar
from run_hla2010_dqn_discrete_action_probe_012_01 import ACTION_TABLE, IRRIGATION_WINDOWS, NITROGEN_WINDOWS
from run_hla2010_dqn_economic_reward_probe_012_03 import (
    harvest_yields_from_mgmt,
    make_env,
)
from run_hla_official_reward_restart_smoke import install_official_reward_module, parse_events, prepare_case_at


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2015_double_dqn_economic_012_13"


def train_and_eval(year: int, timesteps: int, seed: int, water_cost: float, nitrogen_cost: float, label: str) -> Path:
    install_official_reward_module()
    run_name = f"{label}_seed{seed}_{timesteps}steps"
    case_dir = OUT_DIR / str(year) / run_name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    prepare_case_at(year, case_dir)
    env_args = json.loads((case_dir / "env_args.json").read_text(encoding="utf-8"))
    snapshot = case_dir / "pdi_tmp_snapshot_eval"
    debug_log = case_dir / "double_dqn_economic_debug.log"

    def log(msg: str) -> None:
        line = f"{pd.Timestamp.now().isoformat()} {msg}"
        print(line, flush=True)
        with debug_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log("train:start")
    env = make_env(env_args, water_cost, nitrogen_cost)
    model_dir = case_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = CustomDoubleDQN(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            learning_rate=1e-4,
            buffer_size=10000,
            learning_starts=min(100, max(10, timesteps // 10)),
            batch_size=32,
            train_freq=1,
            gradient_steps=1,
            gamma=0.99,
            exploration_fraction=0.35,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
        )
        log("learn:start")
        model.learn(total_timesteps=int(timesteps), progress_bar=False)
        log("learn:done")
        model.save(str(model_dir / "double_dqn_economic_reward_probe"))
    finally:
        env.close()
        log("train_env_close:done")

    rows = []
    eval_env = make_env(env_args, water_cost, nitrogen_cost)
    try:
        obs, info = eval_env.reset()
        for step in range(260):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(eval_env, obs, info)
            rows.append(
                {
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "action_index": eval_env.last_action_index,
                    "raw_amir": eval_env.last_raw_real_action.get("amir", np.nan),
                    "raw_anfer": eval_env.last_raw_real_action.get("anfer", np.nan),
                    "safe_amir": eval_env.last_safe_real_action.get("amir", np.nan),
                    "safe_anfer": eval_env.last_safe_real_action.get("anfer", np.nan),
                    "used_irrigation": eval_env.used_irrigation,
                    "used_nitrogen": eval_env.used_nitrogen,
                    "reward": reward,
                    "delta_grnwt": eval_env.last_reward_components.get("delta_grnwt", np.nan),
                    "water_cost_term": eval_env.last_reward_components.get("water_cost_term", np.nan),
                    "nitrogen_cost_term": eval_env.last_reward_components.get("nitrogen_cost_term", np.nan),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "done": done,
                }
            )
            if done:
                break
    finally:
        tmp = getattr(eval_env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        eval_env.close()
        log("eval_env_close:done")

    daily = pd.DataFrame(rows)
    daily.to_csv(case_dir / "double_dqn_economic_eval_daily.csv", index=False, encoding="utf-8-sig")
    events = parse_events(snapshot / "MgmtEvent.OUT")
    hvals = harvest_yields_from_mgmt(snapshot / "MgmtEvent.OUT")
    summary = {
        "algorithm": "CustomDoubleDQN",
        "reward_type": "economic_delta_grnwt_minus_input_cost",
        "water_cost": water_cost,
        "nitrogen_cost": nitrogen_cost,
        "action_table": ACTION_TABLE,
        "irrigation_windows": IRRIGATION_WINDOWS,
        "nitrogen_windows": NITROGEN_WINDOWS,
        "null_harvest_yield_kg_ha": hvals[0] if hvals else None,
        "double_dqn_harvest_yield_kg_ha": hvals[-1] if hvals else None,
        "daily_last_grnwt": float(daily["grnwt"].dropna().iloc[-1]) if not daily.empty else None,
        "daily_last_topwt": float(daily["topwt"].dropna().iloc[-1]) if not daily.empty else None,
        "irrigation_total": float(daily["safe_amir"].sum()) if not daily.empty else None,
        "nitrogen_total": float(daily["safe_anfer"].sum()) if not daily.empty else None,
        "economic_reward_total_eval": float(daily["reward"].sum()) if not daily.empty else None,
        "input_cost_total_eval": float(daily["water_cost_term"].sum() + daily["nitrogen_cost_term"].sum()) if not daily.empty else None,
        "no_op_count_eval": int((daily["action_index"] == 0).sum()) if not daily.empty else None,
        "max_swfac_eval": float(daily["swfac"].max()) if not daily.empty else None,
        "max_nstres_eval": float(daily["nstres"].max()) if not daily.empty else None,
        **events,
    }
    (case_dir / "event_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log("train_and_eval:done")
    return case_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--water-cost", type=float, default=1.0)
    parser.add_argument("--nitrogen-cost", type=float, default=5.0)
    parser.add_argument("--label", default="double_dqn_medium_N_cost")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    case_dir = train_and_eval(args.year, args.timesteps, args.seed, args.water_cost, args.nitrogen_cost, args.label)
    print(json.dumps({"case_dir": str(case_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
