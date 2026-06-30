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
from run_hla2015_literature_aligned_dqn_012_15 import LITERATURE_ACTION_TABLE, make_env


ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004"
CASE_DIR = (
    ROOT
    / "hla2015_literature_dqn_warmstart_012_19"
    / "2015"
    / "literature_warmstart_nstep5_ws8_seed1_5000steps"
)
OUT_DIR = ROOT / "hla2015_warmstart_seed1_q_value_diagnostic_012_21"


def in_windows(dap: float, windows: list[tuple[int, int]]) -> bool:
    d = int(round(float(dap)))
    return any(left <= d <= right for left, right in windows)


def q_values(model, obs) -> np.ndarray:
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        q = model.policy.q_net(obs_tensor)
    return q.detach().cpu().numpy().reshape(-1)


def action_label(action_index: int) -> str:
    action = LITERATURE_ACTION_TABLE[int(action_index)]
    return f"I{action['amir']:.0f}_N{action['anfer']:.0f}"


def run_q_diagnostic() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from stable_baselines3 import DQN

    model_path = CASE_DIR / "models" / "dqn_literature_warmstart_probe.zip"
    env_args = json.loads((CASE_DIR / "env_args.json").read_text(encoding="utf-8"))
    model = DQN.load(str(model_path), device="cpu")
    env = make_env(env_args, w1=0.158, w2=0.79, w3=1.10)
    rows = []
    top_rows = []
    try:
        obs, info = env.reset()
        for step in range(260):
            latest_before = latest_observation_dict(env, obs, info)
            dap_before = scalar(latest_before.get("dap", np.nan))
            q = q_values(model, obs)
            order = np.argsort(q)[::-1]
            best_action = int(order[0])
            predicted_action, _ = model.predict(obs, deterministic=True)
            predicted_action = int(np.asarray(predicted_action).item())
            q_record = {
                "step": step,
                "dap_before": dap_before,
                "in_irrigation_window": in_windows(dap_before, IRRIGATION_WINDOWS) if pd.notna(dap_before) else False,
                "in_nitrogen_window": in_windows(dap_before, NITROGEN_WINDOWS) if pd.notna(dap_before) else False,
                "best_q_action": best_action,
                "best_q_label": action_label(best_action),
                "predicted_action": predicted_action,
                "predicted_label": action_label(predicted_action),
                "q_best": float(q[best_action]),
                "q_noop": float(q[0]),
                "q_best_minus_noop": float(q[best_action] - q[0]),
            }
            for i in range(len(q)):
                q_record[f"q_{i:02d}_{action_label(i)}"] = float(q[i])
            rows.append(q_record)
            for rank, action_idx in enumerate(order[:8], start=1):
                action = LITERATURE_ACTION_TABLE[int(action_idx)]
                top_rows.append(
                    {
                        "step": step,
                        "dap_before": dap_before,
                        "rank": rank,
                        "action_index": int(action_idx),
                        "action_label": action_label(int(action_idx)),
                        "raw_irrigation_mm": action["amir"],
                        "raw_nitrogen_kg_ha": action["anfer"],
                        "q_value": float(q[action_idx]),
                        "q_minus_noop": float(q[action_idx] - q[0]),
                    }
                )
            obs, reward, terminated, truncated, info = env.step(predicted_action)
            latest_after = latest_observation_dict(env, obs, info)
            rows[-1].update(
                {
                    "safe_amir": env.last_safe_real_action.get("amir", np.nan),
                    "safe_anfer": env.last_safe_real_action.get("anfer", np.nan),
                    "used_irrigation": getattr(env, "used_irrigation", np.nan),
                    "used_nitrogen": getattr(env, "used_nitrogen", np.nan),
                    "reward": float(reward),
                    "grnwt_after": scalar(latest_after.get("grnwt")),
                    "topwt_after": scalar(latest_after.get("topwt")),
                    "swfac_after": scalar(latest_after.get("swfac")),
                    "nstres_after": scalar(latest_after.get("nstres")),
                    "done": bool(terminated or truncated),
                }
            )
            if bool(terminated or truncated):
                break
    finally:
        env.close()
    qdf = pd.DataFrame(rows)
    topdf = pd.DataFrame(top_rows)
    summary = summarize(qdf)
    return qdf, topdf, summary


def summarize(qdf: pd.DataFrame) -> pd.DataFrame:
    iw = qdf[qdf["in_irrigation_window"].astype(bool)]
    nw = qdf[qdf["in_nitrogen_window"].astype(bool)]

    def has_n(action_idx: int) -> bool:
        return LITERATURE_ACTION_TABLE[int(action_idx)]["anfer"] > 0

    def has_i(action_idx: int) -> bool:
        return LITERATURE_ACTION_TABLE[int(action_idx)]["amir"] > 0

    summary = {
        "run": "012_20_warmstart_seed1_5k",
        "steps": int(len(qdf)),
        "irrigation_window_steps": int(len(iw)),
        "nitrogen_window_steps": int(len(nw)),
        "all_best_contains_irrigation_rate": float(qdf["best_q_action"].map(has_i).mean()),
        "all_best_contains_nitrogen_rate": float(qdf["best_q_action"].map(has_n).mean()),
        "iw_best_contains_irrigation_rate": float(iw["best_q_action"].map(has_i).mean()) if len(iw) else np.nan,
        "iw_best_contains_nitrogen_rate": float(iw["best_q_action"].map(has_n).mean()) if len(iw) else np.nan,
        "nw_best_contains_nitrogen_rate": float(nw["best_q_action"].map(has_n).mean()) if len(nw) else np.nan,
        "predicted_irrigation_total": float(qdf["safe_amir"].sum()),
        "predicted_nitrogen_total": float(qdf["safe_anfer"].sum()),
        "final_grnwt": float(qdf["grnwt_after"].dropna().iloc[-1]),
        "final_topwt": float(qdf["topwt_after"].dropna().iloc[-1]),
        "total_reward": float(qdf["reward"].sum()),
        "max_swfac": float(qdf["swfac_after"].max()),
        "max_nstres": float(qdf["nstres_after"].max()),
    }
    for label, sub in [("iw", iw), ("nw", nw), ("all", qdf)]:
        counts = sub["best_q_label"].value_counts().head(5)
        for rank, (action_label_name, count) in enumerate(counts.items(), start=1):
            summary[f"{label}_top{rank}_best_label"] = action_label_name
            summary[f"{label}_top{rank}_best_count"] = int(count)
    return pd.DataFrame([summary])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qdf, topdf, summary = run_q_diagnostic()
    qdf.to_csv(OUT_DIR / "012_20_seed1_q_values_daily.csv", index=False, encoding="utf-8-sig")
    topdf.to_csv(OUT_DIR / "012_20_seed1_top8_actions_daily.csv", index=False, encoding="utf-8-sig")
    key = topdf[topdf["dap_before"].round().isin([21, 28, 35, 46, 53, 56, 60, 63, 67, 74, 85, 92])]
    key.to_csv(OUT_DIR / "012_20_seed1_top8_actions_key_daps.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "012_20_seed1_q_value_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "012_20_seed1_q_value_summary.json").write_text(
        json.dumps(summary.iloc[0].to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(OUT_DIR)


if __name__ == "__main__":
    main()
