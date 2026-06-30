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

from ppo_evaluate import latest_observation_dict, scalar
from run_hla2010_dqn_discrete_action_probe_012_01 import IRRIGATION_WINDOWS, NITROGEN_WINDOWS
from run_hla2010_dqn_economic_reward_probe_012_03 import make_env


ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004"
DQN_2015 = ROOT / "hla2010_dqn_economic_reward_probe_012_03" / "2015"
OUT_DIR = ROOT / "hla2015_dqn_q_value_diagnostic_012_12"

RUNS = {
    "seed0_5k": DQN_2015 / "medium_N_cost_seed0_5000steps",
    "seed1_5k": DQN_2015 / "medium_N_cost_seed1_5000steps",
    "seed2_5k": DQN_2015 / "medium_N_cost_seed2_5000steps",
    "seed0_20k": DQN_2015 / "medium_N_cost_seed0_20000steps",
}


def in_windows(dap: float, windows: list[tuple[int, int]]) -> bool:
    d = int(round(float(dap)))
    return any(left <= d <= right for left, right in windows)


def q_values(model, obs) -> np.ndarray:
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        q = model.policy.q_net(obs_tensor)
    return q.detach().cpu().numpy().reshape(-1)


def run_diagnostic(label: str, case_dir: Path) -> pd.DataFrame:
    from stable_baselines3 import DQN

    model_path = case_dir / "models" / "dqn_economic_reward_probe.zip"
    env_args = json.loads((case_dir / "env_args.json").read_text(encoding="utf-8"))
    model = DQN.load(str(model_path), device="cpu")
    env = make_env(env_args, water_cost=1.0, nitrogen_cost=5.0)
    rows = []
    try:
        obs, info = env.reset()
        for step in range(260):
            latest_before = latest_observation_dict(env, obs, info)
            dap_before = scalar(latest_before.get("dap", np.nan))
            q = q_values(model, obs)
            best_action = int(np.argmax(q))
            predicted_action, _ = model.predict(obs, deterministic=True)
            predicted_action = int(np.asarray(predicted_action).item())
            obs, reward, terminated, truncated, info = env.step(predicted_action)
            done = bool(terminated or truncated)
            latest_after = latest_observation_dict(env, obs, info)
            rows.append(
                {
                    "run": label,
                    "step": step,
                    "dap_before": dap_before,
                    "in_irrigation_window": in_windows(dap_before, IRRIGATION_WINDOWS) if pd.notna(dap_before) else False,
                    "in_nitrogen_window": in_windows(dap_before, NITROGEN_WINDOWS) if pd.notna(dap_before) else False,
                    "q_action0_noop": float(q[0]),
                    "q_action1_irrigation": float(q[1]),
                    "q_action2_nitrogen": float(q[2]),
                    "q_action3_both": float(q[3]),
                    "best_q_action": best_action,
                    "predicted_action": predicted_action,
                    "q_best_minus_noop": float(np.max(q) - q[0]),
                    "q_irrigation_minus_noop": float(q[1] - q[0]),
                    "q_both_minus_noop": float(q[3] - q[0]),
                    "safe_amir": env.last_safe_real_action.get("amir", np.nan),
                    "safe_anfer": env.last_safe_real_action.get("anfer", np.nan),
                    "reward": float(reward),
                    "dap_after": scalar(latest_after.get("dap")),
                    "grnwt_after": scalar(latest_after.get("grnwt")),
                    "topwt_after": scalar(latest_after.get("topwt")),
                    "swfac_after": scalar(latest_after.get("swfac")),
                    "nstres_after": scalar(latest_after.get("nstres")),
                    "done": done,
                }
            )
            if done:
                break
    finally:
        env.close()
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, sub in df.groupby("run"):
        iw = sub[sub["in_irrigation_window"].astype(bool)]
        nw = sub[sub["in_nitrogen_window"].astype(bool)]
        rows.append(
            {
                "run": run,
                "steps": int(len(sub)),
                "irrigation_window_steps": int(len(iw)),
                "iw_best_noop_count": int((iw["best_q_action"] == 0).sum()),
                "iw_best_irrigation_count": int((iw["best_q_action"] == 1).sum()),
                "iw_best_nitrogen_count": int((iw["best_q_action"] == 2).sum()),
                "iw_best_both_count": int((iw["best_q_action"] == 3).sum()),
                "iw_pred_irrigation_or_both_count": int(iw["predicted_action"].isin([1, 3]).sum()),
                "iw_mean_q_irrigation_minus_noop": float(iw["q_irrigation_minus_noop"].mean()) if len(iw) else np.nan,
                "iw_mean_q_both_minus_noop": float(iw["q_both_minus_noop"].mean()) if len(iw) else np.nan,
                "nitrogen_window_steps": int(len(nw)),
                "nw_best_nitrogen_or_both_count": int(nw["best_q_action"].isin([2, 3]).sum()),
                "safe_irrigation_total": float(sub["safe_amir"].sum()),
                "safe_nitrogen_total": float(sub["safe_anfer"].sum()),
                "final_grnwt": float(sub["grnwt_after"].dropna().iloc[-1]),
                "total_reward": float(sub["reward"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["iw_best_irrigation_or_both_rate"] = (
        (out["iw_best_irrigation_count"] + out["iw_best_both_count"]) / out["irrigation_window_steps"]
    )
    out["iw_pred_irrigation_or_both_rate"] = out["iw_pred_irrigation_or_both_count"] / out["irrigation_window_steps"]
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for label, case_dir in RUNS.items():
        print(f"diagnosing {label}", flush=True)
        qdf = run_diagnostic(label, case_dir)
        qdf.to_csv(OUT_DIR / f"{label}_q_values_daily.csv", index=False, encoding="utf-8-sig")
        frames.append(qdf)
    all_q = pd.concat(frames, ignore_index=True, sort=False)
    all_q.to_csv(OUT_DIR / "hla2015_dqn_q_values_all_runs_daily.csv", index=False, encoding="utf-8-sig")
    summary = summarize(all_q)
    summary.to_csv(OUT_DIR / "hla2015_dqn_q_value_summary.csv", index=False, encoding="utf-8-sig")
    key = all_q[all_q["dap_before"].round().isin([27, 30, 35, 46, 53, 60, 67, 78, 92])]
    key.to_csv(OUT_DIR / "hla2015_dqn_q_values_key_daps.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
