from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


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
    / "hla2015_literature_dqn_warmstart_012_19"
)


def action_index_for(irrigation: float, nitrogen: float) -> int:
    for idx, values in LITERATURE_ACTION_TABLE.items():
        if abs(values["amir"] - irrigation) < 1e-9 and abs(values["anfer"] - nitrogen) < 1e-9:
            return int(idx)
    raise ValueError((irrigation, nitrogen))


NOOP_ACTION = action_index_for(0.0, 0.0)
I12_N0_ACTION = action_index_for(12.0, 0.0)


def heuristic_warmstart_action(dap: float) -> int:
    """A simple I60/N0-like target policy.

    Because one action is applied to the next DSSAT day in our wrapper logs,
    these pre-action DAPs are chosen to create about five 12 mm irrigation
    events in agronomic windows and no fertilization.
    """

    d = int(round(float(dap)))
    if d in {45, 52, 73, 80, 87}:
        return I12_N0_ACTION
    return NOOP_ACTION


def collect_warmstart_dataset(env_args: dict, w1: float, w2: float, w3: float, repeats: int = 8):
    obs_rows = []
    act_rows = []
    log_rows = []
    env = make_env(env_args, w1=w1, w2=w2, w3=w3)
    try:
        for episode in range(repeats):
            obs, info = env.reset()
            for step in range(260):
                latest_before = latest_observation_dict(env, obs, info)
                dap_before = scalar(latest_before.get("dap", np.nan))
                action = heuristic_warmstart_action(dap_before)
                obs_rows.append(np.asarray(obs, dtype=np.float32).copy())
                act_rows.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                latest_after = latest_observation_dict(env, obs, info)
                log_rows.append(
                    {
                        "episode": episode,
                        "step": step,
                        "dap_before": dap_before,
                        "target_action": action,
                        "target_action_label": f"I{LITERATURE_ACTION_TABLE[action]['amir']:.0f}_N{LITERATURE_ACTION_TABLE[action]['anfer']:.0f}",
                        "safe_amir": env.last_safe_real_action.get("amir", np.nan),
                        "safe_anfer": env.last_safe_real_action.get("anfer", np.nan),
                        "used_irrigation": getattr(env, "used_irrigation", np.nan),
                        "used_nitrogen": getattr(env, "used_nitrogen", np.nan),
                        "reward": float(reward),
                        "grnwt_after": scalar(latest_after.get("grnwt")),
                        "topwt_after": scalar(latest_after.get("topwt")),
                        "done": bool(terminated or truncated),
                    }
                )
                if bool(terminated or truncated):
                    break
    finally:
        env.close()
    return np.asarray(obs_rows, dtype=np.float32), np.asarray(act_rows, dtype=np.int64), pd.DataFrame(log_rows)


def warmstart_q_network(model, observations: np.ndarray, actions: np.ndarray, epochs: int, batch_size: int, lr: float):
    device = model.device
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.policy.q_net.parameters(), lr=lr)
    losses = []
    n = len(actions)
    for epoch in range(int(epochs)):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            q = model.policy.q_net(obs_tensor[idx])
            loss = F.cross_entropy(q, actions_tensor[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.q_net.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    model.policy.q_net_target.load_state_dict(model.policy.q_net.state_dict())
    return losses


def train_and_eval(
    year: int,
    timesteps: int,
    seed: int,
    n_steps: int,
    w1: float,
    w2: float,
    w3: float,
    warmstart_epochs: int,
    label: str,
) -> Path:
    install_official_reward_module()
    from stable_baselines3 import DQN

    run_name = f"{label}_nstep{n_steps}_ws{warmstart_epochs}_seed{seed}_{timesteps}steps"
    case_dir = OUT_DIR / str(year) / run_name
    prepare_case_at(year, case_dir)
    env_args = json.loads((case_dir / "env_args.json").read_text(encoding="utf-8"))
    snapshot = case_dir / "pdi_tmp_snapshot_eval"
    if snapshot.exists():
        shutil.rmtree(snapshot)

    debug_log = case_dir / "dqn_literature_warmstart_debug.log"

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
        log("warmstart_dataset:start")
        ws_obs, ws_actions, ws_log = collect_warmstart_dataset(env_args, w1=w1, w2=w2, w3=w3, repeats=8)
        ws_log.to_csv(case_dir / "warmstart_heuristic_dataset_log.csv", index=False, encoding="utf-8-sig")
        np.save(case_dir / "warmstart_observations.npy", ws_obs)
        np.save(case_dir / "warmstart_actions.npy", ws_actions)
        log(f"warmstart_dataset:done n={len(ws_actions)}")
        losses = warmstart_q_network(model, ws_obs, ws_actions, epochs=warmstart_epochs, batch_size=256, lr=1e-4)
        pd.DataFrame({"loss": losses}).to_csv(case_dir / "warmstart_supervised_losses.csv", index=False, encoding="utf-8-sig")
        log(f"warmstart_training:done final_loss={losses[-1] if losses else None}")
        model.save(str(model_dir / "dqn_after_warmstart_before_rl"))
        log("learn:start")
        model.learn(total_timesteps=int(timesteps), progress_bar=False)
        log("learn:done")
        model.save(str(model_dir / "dqn_literature_warmstart_probe"))
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
    daily.to_csv(case_dir / "dqn_literature_warmstart_eval_daily.csv", index=False, encoding="utf-8-sig")
    events = parse_events(snapshot / "MgmtEvent.OUT")
    hvals = harvest_yields_from_mgmt(snapshot / "MgmtEvent.OUT")
    summary = {
        "algorithm": "DQN",
        "variant": f"{n_steps}_step_return_plus_warmstart",
        "reward_type": "literature_terminal_yield_minus_input_cost",
        "w1_terminal_yield": w1,
        "w2_nitrogen": w2,
        "w3_water": w3,
        "n_steps": int(n_steps),
        "warmstart_epochs": int(warmstart_epochs),
        "warmstart_dataset_size": int(len(ws_actions)),
        "warmstart_target_irrigation_total_last_episode": float(
            ws_log[ws_log["episode"].eq(ws_log["episode"].max())]["safe_amir"].sum()
        ),
        "warmstart_target_nitrogen_total_last_episode": float(
            ws_log[ws_log["episode"].eq(ws_log["episode"].max())]["safe_anfer"].sum()
        ),
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
    parser.add_argument("--warmstart-epochs", type=int, default=8)
    parser.add_argument("--w1", type=float, default=0.158)
    parser.add_argument("--w2", type=float, default=0.79)
    parser.add_argument("--w3", type=float, default=1.10)
    parser.add_argument("--label", default="literature_warmstart")
    args = parser.parse_args()
    case_dir = train_and_eval(
        year=args.year,
        timesteps=args.timesteps,
        seed=args.seed,
        n_steps=args.n_steps,
        w1=args.w1,
        w2=args.w2,
        w3=args.w3,
        warmstart_epochs=args.warmstart_epochs,
        label=args.label,
    )
    print(case_dir)


if __name__ == "__main__":
    main()
