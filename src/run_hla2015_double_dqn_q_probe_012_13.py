from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from custom_double_dqn import CustomDoubleDQN
from ppo_evaluate import latest_observation_dict, scalar
from run_hla2010_dqn_discrete_action_probe_012_01 import IRRIGATION_WINDOWS
from run_hla2010_dqn_economic_reward_probe_012_03 import make_env


ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004"
CASE_DIR = ROOT / "hla2015_double_dqn_economic_012_13" / "2015" / "double_dqn_medium_N_cost_seed0_5000steps"
OUT_DIR = ROOT / "hla2015_double_dqn_economic_012_13" / "q_probe"


def in_irrigation_window(dap: float) -> bool:
    d = int(round(float(dap)))
    return any(left <= d <= right for left, right in IRRIGATION_WINDOWS)


def q_values(model, obs) -> np.ndarray:
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        q = model.policy.q_net(obs_tensor)
    return q.detach().cpu().numpy().reshape(-1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env_args = json.loads((CASE_DIR / "env_args.json").read_text(encoding="utf-8"))
    model = CustomDoubleDQN.load(str(CASE_DIR / "models" / "double_dqn_economic_reward_probe.zip"), device="cpu")
    env = make_env(env_args, water_cost=1.0, nitrogen_cost=5.0)
    rows = []
    try:
        obs, info = env.reset()
        for step in range(260):
            latest_before = latest_observation_dict(env, obs, info)
            dap = scalar(latest_before.get("dap", np.nan))
            q = q_values(model, obs)
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest_after = latest_observation_dict(env, obs, info)
            rows.append(
                {
                    "step": step,
                    "dap_before": dap,
                    "in_irrigation_window": in_irrigation_window(dap) if pd.notna(dap) else False,
                    "q_action0_noop": float(q[0]),
                    "q_action1_irrigation": float(q[1]),
                    "q_action2_nitrogen": float(q[2]),
                    "q_action3_both": float(q[3]),
                    "best_q_action": int(np.argmax(q)),
                    "predicted_action": action,
                    "q_irrigation_minus_noop": float(q[1] - q[0]),
                    "q_both_minus_noop": float(q[3] - q[0]),
                    "safe_amir": env.last_safe_real_action.get("amir", np.nan),
                    "safe_anfer": env.last_safe_real_action.get("anfer", np.nan),
                    "reward": float(reward),
                    "grnwt_after": scalar(latest_after.get("grnwt")),
                    "swfac_after": scalar(latest_after.get("swfac")),
                    "nstres_after": scalar(latest_after.get("nstres")),
                    "done": done,
                }
            )
            if done:
                break
    finally:
        env.close()
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "double_dqn_seed0_5k_q_values_daily.csv", index=False, encoding="utf-8-sig")
    iw = df[df["in_irrigation_window"].astype(bool)]
    summary = {
        "run": "double_dqn_seed0_5k",
        "steps": int(len(df)),
        "irrigation_window_steps": int(len(iw)),
        "iw_best_irrigation_or_both_count": int(iw["best_q_action"].isin([1, 3]).sum()),
        "iw_best_irrigation_or_both_rate": float(iw["best_q_action"].isin([1, 3]).mean()) if len(iw) else 0.0,
        "iw_mean_q_irrigation_minus_noop": float(iw["q_irrigation_minus_noop"].mean()) if len(iw) else np.nan,
        "safe_irrigation_total": float(df["safe_amir"].sum()),
        "safe_nitrogen_total": float(df["safe_anfer"].sum()),
        "final_grnwt": float(df["grnwt_after"].dropna().iloc[-1]),
        "total_reward": float(df["reward"].sum()),
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "double_dqn_seed0_5k_q_summary.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
