from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from run_hla_official_reward_restart_smoke import (
    OUT_DIR,
    BudgetedDailyActionWrapper,
    LazyScalarGymDssatWrapper,
    install_official_reward_module,
    latest_observation_dict,
    parse_events,
    prepare_case_at,
    scalar,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate_old_irrigation_model(year: int, model_path: Path, label: str) -> Path:
    """Evaluate a historical 1-D irrigation PPO model in the current corrected all-mode HLA env.

    The old irrigation model has a 24-D observation space and a 1-D action space.
    The current corrected all-mode environment has the same 24-D observations but
    a 2-D action space: [amir, anfer].  This adapter sends the old model action
    to amir and fixes anfer=-1.0, i.e. zero fertilizer after denormalization.
    """
    import gym

    install_official_reward_module()
    case_dir = OUT_DIR / "old_model_current_env_eval" / str(year) / label
    prepare_case_at(year, case_dir)
    env_args = json.loads((case_dir / "env_args.json").read_text(encoding="utf-8"))

    snapshot = case_dir / "pdi_tmp_snapshot_eval"
    if snapshot.exists():
        shutil.rmtree(snapshot)

    model = PPO.load(str(model_path), device="cpu")
    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    env = BudgetedDailyActionWrapper(
        LazyScalarGymDssatWrapper(raw),
        allow_irrigation=True,
        allow_nitrogen=False,
    )

    meta = {
        "year": year,
        "label": label,
        "model_path": str(model_path),
        "model_observation_space": repr(model.observation_space),
        "model_action_space": repr(model.action_space),
        "env_observation_space": repr(env.observation_space),
        "env_action_space": repr(env.action_space),
        "env_action_names": list(env.formator.action_names),
        "adapter": "old 1-D action -> current amir; current anfer fixed to -1.0/zero N",
    }

    rows = []
    try:
        obs, info = env.reset()
        done = False
        for step in range(220):
            old_action, _ = model.predict(obs, deterministic=True)
            old_action = np.asarray(old_action, dtype=np.float32).reshape(-1)
            full_action = np.full(env.action_space.shape, -1.0, dtype=np.float32)
            full_action[0] = float(old_action[0])  # action_names[0] == amir in current env
            obs, reward, terminated, truncated, info = env.step(full_action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            rows.append(
                {
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "old_raw_action_0": float(old_action[0]),
                    "full_norm_amir": float(full_action[0]),
                    "full_norm_anfer": float(full_action[1]),
                    "safe_amir": env.last_safe_real_action.get("amir", np.nan),
                    "safe_anfer": env.last_safe_real_action.get("anfer", np.nan),
                    "used_irrigation": getattr(env, "used_irrigation", np.nan),
                    "used_nitrogen": getattr(env, "used_nitrogen", np.nan),
                    "reward": reward,
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
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        env.close()

    daily = pd.DataFrame(rows)
    daily.to_csv(case_dir / "old_model_eval_daily.csv", index=False, encoding="utf-8-sig")
    event_summary = parse_events(snapshot / "MgmtEvent.OUT")
    event_summary.update(
        {
            "label": label,
            "model_path": str(model_path),
            "csv_safe_i": float(daily["safe_amir"].sum()) if not daily.empty else 0.0,
            "csv_safe_n": float(daily["safe_anfer"].sum()) if not daily.empty else 0.0,
        }
    )
    (case_dir / "event_summary.json").write_text(json.dumps(event_summary, indent=2), encoding="utf-8")
    (case_dir / "eval_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return case_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2010)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default="old_irrigation_model")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    out = evaluate_old_irrigation_model(args.year, model_path, args.label)
    print(json.dumps({"out_dir": str(out)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
