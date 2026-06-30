from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import gymnasium as gymnasium_base
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_evaluate import latest_observation_dict, scalar
from run_hla_official_reward_restart_smoke import (
    LazyScalarGymDssatWrapper,
    install_official_reward_module,
    parse_events,
    prepare_case_at,
)


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_dqn_discrete_action_probe_012_01"

IRRIGATION_WINDOWS = [(20, 35), (45, 65), (70, 95)]
NITROGEN_WINDOWS = [(25, 40), (55, 70)]

ACTION_TABLE: dict[int, dict[str, float]] = {
    0: {"amir": 0.0, "anfer": 0.0},
    1: {"amir": 30.0, "anfer": 0.0},
    2: {"amir": 0.0, "anfer": 50.0},
    3: {"amir": 30.0, "anfer": 50.0},
}


class DiscreteBudgetedDailyActionWrapper(gymnasium_base.Env):
    """Map DQN discrete actions to gym-DSSAT linked water/nitrogen actions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env,
        irrigation_budget: float = 120.0,
        nitrogen_budget: float = 150.0,
        daily_irrigation_cap: float = 30.0,
        daily_nitrogen_cap: float = 50.0,
        min_interval_days: int = 7,
        irrigation_windows: list[tuple[int, int]] | None = None,
        nitrogen_windows: list[tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.env = env
        self.irrigation_budget = float(irrigation_budget)
        self.nitrogen_budget = float(nitrogen_budget)
        self.daily_irrigation_cap = float(daily_irrigation_cap)
        self.daily_nitrogen_cap = float(daily_nitrogen_cap)
        self.min_interval_days = int(min_interval_days)
        self.irrigation_windows = irrigation_windows
        self.nitrogen_windows = nitrogen_windows
        self.action_space = gymnasium_base.spaces.Discrete(len(ACTION_TABLE))
        self.observation_space = env.observation_space
        self.formator = env.formator
        self.reset_budget_state()
        self.last_action_index = 0
        self.last_raw_real_action = {"amir": 0.0, "anfer": 0.0}
        self.last_safe_real_action = {"amir": 0.0, "anfer": 0.0}

    def reset_budget_state(self) -> None:
        self.used_irrigation = 0.0
        self.used_nitrogen = 0.0
        self.last_operation_dap: int | None = None

    def reset(self, *args, **kwargs):
        self.reset_budget_state()
        self.last_action_index = 0
        self.last_raw_real_action = {"amir": 0.0, "anfer": 0.0}
        self.last_safe_real_action = {"amir": 0.0, "anfer": 0.0}
        return self.env.reset(*args, **kwargs)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)

    def close(self):
        return self.env.close()

    def render(self):
        return None

    def _current_dap(self) -> int:
        raw = getattr(self.env.unwrapped, "observation", None)
        if isinstance(raw, dict) and "dap" in raw:
            return int(round(scalar(raw.get("dap", 0))))
        return 0

    def _normalize_real_action(self, real_action: dict[str, float]) -> np.ndarray:
        values = []
        spaces = getattr(self.formator.action_space_dict, "spaces", self.formator.action_space_dict)
        for name in self.formator.action_names:
            space = spaces[name]
            low = float(np.asarray(space.low).flatten()[0])
            high = float(np.asarray(space.high).flatten()[0])
            real = float(real_action.get(name, 0.0))
            values.append(2.0 * ((real - low) / (high - low)) - 1.0)
        return np.asarray(values, dtype=np.float32)

    def _safe_real_action(self, raw_real: dict[str, float], dap: int) -> dict[str, float]:
        if dap < 1:
            return {"amir": 0.0, "anfer": 0.0}
        can_operate = self.last_operation_dap is None or (dap - self.last_operation_dap) >= self.min_interval_days
        if not can_operate:
            return {"amir": 0.0, "anfer": 0.0}
        in_irrigation_window = (
            True
            if self.irrigation_windows is None
            else any(left <= dap <= right for left, right in self.irrigation_windows)
        )
        in_nitrogen_window = (
            True
            if self.nitrogen_windows is None
            else any(left <= dap <= right for left, right in self.nitrogen_windows)
        )
        remaining_i = max(0.0, self.irrigation_budget - self.used_irrigation)
        remaining_n = max(0.0, self.nitrogen_budget - self.used_nitrogen)
        safe_i = (
            min(max(0.0, float(raw_real.get("amir", 0.0))), self.daily_irrigation_cap, remaining_i)
            if in_irrigation_window
            else 0.0
        )
        safe_n = (
            min(max(0.0, float(raw_real.get("anfer", 0.0))), self.daily_nitrogen_cap, remaining_n)
            if in_nitrogen_window
            else 0.0
        )
        return {"amir": safe_i, "anfer": safe_n}

    def step(self, action):
        action_index = int(np.asarray(action).item())
        raw_real = dict(ACTION_TABLE[action_index])
        dap = self._current_dap()
        safe_real = self._safe_real_action(raw_real, dap)
        safe_norm = self._normalize_real_action(safe_real)
        obs, reward, terminated, truncated, info = self.env.step(safe_norm)
        if safe_real["amir"] > 0 or safe_real["anfer"] > 0:
            self.last_operation_dap = dap
        self.used_irrigation += safe_real["amir"]
        self.used_nitrogen += safe_real["anfer"]
        self.last_action_index = action_index
        self.last_raw_real_action = raw_real
        self.last_safe_real_action = safe_real
        info = info if isinstance(info, dict) else {}
        info.update(
            {
                "dqn_action_index": action_index,
                "raw_real_action_amir": raw_real.get("amir", 0.0),
                "raw_real_action_anfer": raw_real.get("anfer", 0.0),
                "safe_real_action_amir": safe_real.get("amir", 0.0),
                "safe_real_action_anfer": safe_real.get("anfer", 0.0),
                "used_irrigation": self.used_irrigation,
                "used_nitrogen": self.used_nitrogen,
                "budget_dap": dap,
            }
        )
        return obs, reward, terminated, truncated, info


def harvest_yields_from_mgmt(path: Path) -> list[float]:
    values = []
    if not path.exists():
        return values
    for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
        if "Harvest Yield" in line:
            m = re.search(r"Harvest Yield\s+([0-9.]+)", line)
            if m:
                values.append(float(m.group(1)))
    return values


def train_and_eval(year: int, timesteps: int, seed: int) -> Path:
    install_official_reward_module()
    import gym
    from stable_baselines3 import DQN

    run_name = f"dqn_discrete_seed{seed}_{timesteps}steps"
    case_dir = OUT_DIR / str(year) / run_name
    prepare_case_at(year, case_dir)
    env_args = json.loads((case_dir / "env_args.json").read_text(encoding="utf-8"))
    snapshot = case_dir / "pdi_tmp_snapshot_eval"
    if snapshot.exists():
        shutil.rmtree(snapshot)

    debug_log = case_dir / "dqn_debug.log"

    def log(msg: str) -> None:
        line = f"{pd.Timestamp.now().isoformat()} {msg}"
        print(line, flush=True)
        with debug_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log("train:start")
    raw_env = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    env = DiscreteBudgetedDailyActionWrapper(
        LazyScalarGymDssatWrapper(raw_env),
        irrigation_windows=IRRIGATION_WINDOWS,
        nitrogen_windows=NITROGEN_WINDOWS,
    )
    model_dir = case_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        log("dqn_init:start")
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
        log("dqn_init:done")
        log("learn:start")
        model.learn(total_timesteps=int(timesteps), progress_bar=False)
        log("learn:done")
        model.save(str(model_dir / "dqn_discrete_probe"))
        log("model_save:done")
    finally:
        env.close()
        log("train_env_close:done")

    rows = []
    log("eval:start")
    eval_env = DiscreteBudgetedDailyActionWrapper(
        LazyScalarGymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped),
        irrigation_windows=IRRIGATION_WINDOWS,
        nitrogen_windows=NITROGEN_WINDOWS,
    )
    try:
        obs, info = eval_env.reset()
        for step in range(240):
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
    daily.to_csv(case_dir / "dqn_eval_daily.csv", index=False, encoding="utf-8-sig")
    event_summary = parse_events(snapshot / "MgmtEvent.OUT")
    hvals = harvest_yields_from_mgmt(snapshot / "MgmtEvent.OUT")
    event_summary.update(
        {
            "algorithm": "DQN",
            "action_table": ACTION_TABLE,
            "irrigation_windows": IRRIGATION_WINDOWS,
            "nitrogen_windows": NITROGEN_WINDOWS,
            "null_harvest_yield_kg_ha": hvals[0] if hvals else None,
            "dqn_harvest_yield_kg_ha": hvals[-1] if hvals else None,
            "daily_last_grnwt": float(daily["grnwt"].dropna().iloc[-1]) if not daily.empty else None,
            "daily_last_topwt": float(daily["topwt"].dropna().iloc[-1]) if not daily.empty else None,
            "no_op_count_eval": int((daily["action_index"] == 0).sum()) if not daily.empty else None,
            "nonzero_safe_action_count_eval": int(((daily["safe_amir"].abs() > 1e-6) | (daily["safe_anfer"].abs() > 1e-6)).sum()) if not daily.empty else None,
            "max_swfac_eval": float(daily["swfac"].max()) if not daily.empty else None,
            "max_nstres_eval": float(daily["nstres"].max()) if not daily.empty else None,
        }
    )
    (case_dir / "event_summary.json").write_text(json.dumps(event_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log("train_and_eval:done")
    return case_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2010)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    case_dir = train_and_eval(args.year, args.timesteps, args.seed)
    print(json.dumps({"case_dir": str(case_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
