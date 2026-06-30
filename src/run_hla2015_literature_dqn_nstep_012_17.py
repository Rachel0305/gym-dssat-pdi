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

from ppo_evaluate import latest_observation_dict, scalar
from run_hla_official_reward_restart_smoke import (
    install_official_reward_module,
    parse_events,
    prepare_case_at,
)
from run_hla2015_literature_aligned_dqn_012_15 import (
    LITERATURE_ACTION_TABLE,
    harvest_yields_from_mgmt,
    make_env,
)


OUT_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "hla2015_literature_dqn_nstep_012_17"
)


def train_and_eval(
    year: int,
    timesteps: int,
    seed: int,
    n_steps: int,
    w1: float,
    w2: float,
    w3: float,
    label: str,
) -> Path:
    install_official_reward_module()
    from stable_baselines3 import DQN

    run_name = f"{label}_nstep{n_steps}_seed{seed}_{timesteps}steps"
    case_dir = OUT_DIR / str(year) / run_name
    prepare_case_at(year, case_dir)
    env_args = json.loads((case_dir / "env_args.json").read_text(encoding="utf-8"))
    snapshot = case_dir / "pdi_tmp_snapshot_eval"
    if snapshot.exists():
        shutil.rmtree(snapshot)

    debug_log = case_dir / "dqn_literature_nstep_debug.log"

    def log(msg: str) -> None:
        line = f"{pd.Timestamp.now().isoformat()} {msg}"
        print(line, flush=True)
        with debug_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log("train:start")
    env = make_env(env_args, w1=w1, w2=w2, w3=w3)
    model_dir = case_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = DQN(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            learning_rate=1e-5,
            buffer_size=50000,
            learning_starts=min(1024, max(50, timesteps // 5)),
            batch_size=1024 if timesteps >= 1500 else 64,
            train_freq=1,
            gradient_steps=1,
            gamma=0.99,
            n_steps=int(n_steps),
            exploration_fraction=0.35,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            policy_kwargs={"net_arch": [256, 256, 256]},
        )
        log("learn:start")
        model.learn(total_timesteps=int(timesteps), progress_bar=False)
        log("learn:done")
        model.save(str(model_dir / "dqn_literature_nstep_probe"))
    finally:
        env.close()
        log("train_env_close:done")

    rows = []
    eval_env = make_env(env_args, w1=w1, w2=w2, w3=w3)
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
                    "terminal_yield_reward": eval_env.last_reward_components.get("terminal_yield_reward", np.nan),
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
    daily.to_csv(case_dir / "dqn_literature_nstep_eval_daily.csv", index=False, encoding="utf-8-sig")
    events = parse_events(snapshot / "MgmtEvent.OUT")
    hvals = harvest_yields_from_mgmt(snapshot / "MgmtEvent.OUT")
    summary = {
        "algorithm": "DQN",
        "variant": f"{n_steps}_step_return",
        "reward_type": "literature_terminal_yield_minus_input_cost",
        "w1_terminal_yield": w1,
        "w2_nitrogen": w2,
        "w3_water": w3,
        "n_steps": int(n_steps),
        "action_table": LITERATURE_ACTION_TABLE,
        "null_harvest_yield_kg_ha": hvals[0] if hvals else None,
        "dqn_harvest_yield_kg_ha": hvals[-1] if hvals else None,
        "daily_last_grnwt": float(daily["grnwt"].dropna().iloc[-1]) if not daily.empty else None,
        "daily_last_topwt": float(daily["topwt"].dropna().iloc[-1]) if not daily.empty else None,
        "irrigation_total": float(daily["safe_amir"].sum()) if not daily.empty else None,
        "nitrogen_total": float(daily["safe_anfer"].sum()) if not daily.empty else None,
        "literature_reward_total_eval": float(daily["reward"].sum()) if not daily.empty else None,
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
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--w1", type=float, default=0.158)
    parser.add_argument("--w2", type=float, default=0.79)
    parser.add_argument("--w3", type=float, default=1.10)
    parser.add_argument("--label", default="literature_aligned")
    args = parser.parse_args()
    case_dir = train_and_eval(
        year=args.year,
        timesteps=args.timesteps,
        seed=args.seed,
        n_steps=args.n_steps,
        w1=args.w1,
        w2=args.w2,
        w3=args.w3,
        label=args.label,
    )
    print(case_dir)


if __name__ == "__main__":
    main()
