from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import gymnasium as gymnasium_base
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_evaluate import latest_observation_dict, scalar
from run_hla2010_dqn_discrete_action_probe_012_01 import (
    IRRIGATION_WINDOWS,
    NITROGEN_WINDOWS,
    DiscreteBudgetedDailyActionWrapper,
)
from run_hla2010_dqn_economic_reward_probe_012_03 import EconomicRewardWrapper
from run_hla_official_reward_restart_smoke import LazyScalarGymDssatWrapper, install_official_reward_module, prepare_case_at


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2015_dqn_action_coverage_012_11"
YEAR = 2015
WATER_COST = 1.0
NITROGEN_COST = 5.0


def in_windows(dap: float, windows: list[tuple[int, int]]) -> bool:
    d = int(round(float(dap)))
    return any(left <= d <= right for left, right in windows)


class TrainingActionLogger(gymnasium_base.Env):
    """Pass-through wrapper that records every training transition."""

    def __init__(self, env, seed: int):
        super().__init__()
        self.env = env
        self.seed = int(seed)
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.rows: list[dict] = []
        self.global_step = 0
        self.episode = -1
        self.episode_step = 0

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        self.episode += 1
        self.episode_step = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        latest = latest_observation_dict(self.env, obs, info)
        dap = scalar(latest.get("dap", np.nan))
        raw = getattr(self.env, "last_raw_real_action", {"amir": np.nan, "anfer": np.nan})
        safe = getattr(self.env, "last_safe_real_action", {"amir": np.nan, "anfer": np.nan})
        action_index = int(getattr(self.env, "last_action_index", int(np.asarray(action).item())))
        self.rows.append(
            {
                "seed": self.seed,
                "global_step": self.global_step,
                "episode": self.episode,
                "episode_step": self.episode_step,
                "dap": dap,
                "action_index": action_index,
                "raw_amir": float(raw.get("amir", np.nan)),
                "raw_anfer": float(raw.get("anfer", np.nan)),
                "safe_amir": float(safe.get("amir", np.nan)),
                "safe_anfer": float(safe.get("anfer", np.nan)),
                "in_irrigation_window": in_windows(dap, IRRIGATION_WINDOWS) if pd.notna(dap) else False,
                "in_nitrogen_window": in_windows(dap, NITROGEN_WINDOWS) if pd.notna(dap) else False,
                "used_irrigation": float(getattr(self.env, "used_irrigation", np.nan)),
                "used_nitrogen": float(getattr(self.env, "used_nitrogen", np.nan)),
                "reward": float(reward),
                "delta_grnwt": float(getattr(self.env, "last_reward_components", {}).get("delta_grnwt", np.nan)),
                "grnwt": scalar(latest.get("grnwt")),
                "topwt": scalar(latest.get("topwt")),
                "swfac": scalar(latest.get("swfac")),
                "nstres": scalar(latest.get("nstres")),
                "done": done,
            }
        )
        self.global_step += 1
        self.episode_step += 1
        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    def render(self):
        return None


def make_logged_env(env_args: dict, seed: int):
    import gym

    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    base = DiscreteBudgetedDailyActionWrapper(
        LazyScalarGymDssatWrapper(raw),
        irrigation_windows=IRRIGATION_WINDOWS,
        nitrogen_windows=NITROGEN_WINDOWS,
    )
    economic = EconomicRewardWrapper(base, water_cost=WATER_COST, nitrogen_cost=NITROGEN_COST)
    return TrainingActionLogger(economic, seed=seed)


def summarize(log: pd.DataFrame, seed: int) -> dict:
    wants_irrigation = log["action_index"].isin([1, 3])
    wants_nitrogen = log["action_index"].isin([2, 3])
    in_i = log["in_irrigation_window"].astype(bool)
    in_n = log["in_nitrogen_window"].astype(bool)
    episodes = log.groupby("episode").agg(
        has_safe_irrigation=("safe_amir", lambda s: bool((s > 0).any())),
        has_safe_nitrogen=("safe_anfer", lambda s: bool((s > 0).any())),
        total_safe_irrigation=("safe_amir", "sum"),
        total_safe_nitrogen=("safe_anfer", "sum"),
        final_grnwt=("grnwt", "last"),
        final_reward=("reward", "sum"),
    )
    summary = {
        "seed": seed,
        "training_steps_logged": int(len(log)),
        "episodes": int(log["episode"].nunique()),
        "action0_count": int((log["action_index"] == 0).sum()),
        "action1_irrigation_count": int((log["action_index"] == 1).sum()),
        "action2_nitrogen_count": int((log["action_index"] == 2).sum()),
        "action3_both_count": int((log["action_index"] == 3).sum()),
        "irrigation_window_steps": int(in_i.sum()),
        "nitrogen_window_steps": int(in_n.sum()),
        "irrigation_window_wants_irrigation": int((in_i & wants_irrigation).sum()),
        "nitrogen_window_wants_nitrogen": int((in_n & wants_nitrogen).sum()),
        "safe_irrigation_events": int((log["safe_amir"] > 0).sum()),
        "safe_nitrogen_events": int((log["safe_anfer"] > 0).sum()),
        "raw_irrigation_blocked_count": int((wants_irrigation & (log["safe_amir"] <= 0)).sum()),
        "raw_nitrogen_blocked_count": int((wants_nitrogen & (log["safe_anfer"] <= 0)).sum()),
        "episodes_with_safe_irrigation": int(episodes["has_safe_irrigation"].sum()),
        "episodes_with_safe_nitrogen": int(episodes["has_safe_nitrogen"].sum()),
        "mean_episode_safe_irrigation": float(episodes["total_safe_irrigation"].mean()),
        "mean_episode_safe_nitrogen": float(episodes["total_safe_nitrogen"].mean()),
        "mean_episode_final_grnwt": float(episodes["final_grnwt"].mean()),
        "mean_episode_reward": float(episodes["final_reward"].mean()),
    }
    summary["irrigation_window_wants_irrigation_rate"] = (
        summary["irrigation_window_wants_irrigation"] / summary["irrigation_window_steps"]
        if summary["irrigation_window_steps"]
        else 0.0
    )
    summary["nitrogen_window_wants_nitrogen_rate"] = (
        summary["nitrogen_window_wants_nitrogen"] / summary["nitrogen_window_steps"]
        if summary["nitrogen_window_steps"]
        else 0.0
    )
    summary["episode_safe_irrigation_rate"] = summary["episodes_with_safe_irrigation"] / summary["episodes"] if summary["episodes"] else 0.0
    summary["episode_safe_nitrogen_rate"] = summary["episodes_with_safe_nitrogen"] / summary["episodes"] if summary["episodes"] else 0.0
    return summary


def run_seed(seed: int, timesteps: int) -> Path:
    install_official_reward_module()
    from stable_baselines3 import DQN

    case_dir = OUT_DIR / f"seed{seed}_{timesteps}steps"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    input_dir = case_dir / "input_case"
    prepare_case_at(YEAR, input_dir)
    env_args = json.loads((input_dir / "env_args.json").read_text(encoding="utf-8"))
    env = make_logged_env(env_args, seed=seed)
    try:
        model = DQN(
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
        model.learn(total_timesteps=int(timesteps), progress_bar=False)
        log = pd.DataFrame(env.rows)
    finally:
        env.close()
    log.to_csv(case_dir / "training_action_log.csv", index=False, encoding="utf-8-sig")
    summary = summarize(log, seed)
    (case_dir / "action_coverage_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(case_dir / "action_coverage_summary.csv", index=False, encoding="utf-8-sig")
    return case_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--timesteps", type=int, default=5000)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in args.seeds:
        print(f"running coverage seed={seed}", flush=True)
        case_dir = run_seed(seed, args.timesteps)
        summaries.append(pd.read_csv(case_dir / "action_coverage_summary.csv"))
    merged = pd.concat(summaries, ignore_index=True)
    merged.to_csv(OUT_DIR / f"hla2015_dqn_action_coverage_{args.timesteps}steps_summary.csv", index=False, encoding="utf-8-sig")
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
