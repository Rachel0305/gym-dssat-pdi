from __future__ import annotations

import argparse
import json
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
from run_hla_official_reward_restart_smoke import install_official_reward_module, parse_events
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc_dqn


YEAR = 2015
INPUT_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "hla2015_literature_aligned_dqn_012_15"
    / "2015"
    / "literature_aligned_seed0_5000steps"
    / "input"
)
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2015_success_strategy_validation_014_10"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_014_10_hla2015_success_strategy_validation_record.md"

WATER_COST = 1.0
NITROGEN_COST = 5.0

ACTION_TABLE_9 = {
    0: {"amir": 0.0, "anfer": 0.0},
    1: {"amir": 15.0, "anfer": 0.0},
    2: {"amir": 30.0, "anfer": 0.0},
    3: {"amir": 0.0, "anfer": 50.0},
    4: {"amir": 15.0, "anfer": 50.0},
    5: {"amir": 30.0, "anfer": 50.0},
    6: {"amir": 0.0, "anfer": 100.0},
    7: {"amir": 15.0, "anfer": 100.0},
    8: {"amir": 30.0, "anfer": 100.0},
}


class BaselineRelativeRewardWrapper(gymnasium_base.Env):
    metadata = {"render_modes": []}

    def __init__(self, env, null_baseline_yield: float):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.null_baseline_yield = float(null_baseline_yield)
        self.last_reward_components = {
            "yield_gain": 0.0,
            "water_cost_term": 0.0,
            "nitrogen_cost_term": 0.0,
            "baseline_relative_reward": 0.0,
        }

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.last_reward_components = {
            "yield_gain": 0.0,
            "water_cost_term": 0.0,
            "nitrogen_cost_term": 0.0,
            "baseline_relative_reward": 0.0,
        }
        return obs, info

    def step(self, action):
        obs, _old_reward, terminated, truncated, info = self.env.step(action)
        latest = latest_observation_dict(self.env, obs, info)
        grnwt = float(scalar(latest.get("grnwt", 0.0)) or 0.0)
        irrigation = float(getattr(self.env, "last_safe_real_action", {}).get("amir", 0.0))
        nitrogen = float(getattr(self.env, "last_safe_real_action", {}).get("anfer", 0.0))
        water_cost_term = WATER_COST * irrigation
        nitrogen_cost_term = NITROGEN_COST * nitrogen
        is_terminal = bool(terminated or truncated)
        yield_gain = max(0.0, grnwt - self.null_baseline_yield) if is_terminal else 0.0
        reward = float(yield_gain - water_cost_term - nitrogen_cost_term)
        self.last_reward_components = {
            "yield_gain": yield_gain,
            "water_cost_term": water_cost_term,
            "nitrogen_cost_term": nitrogen_cost_term,
            "baseline_relative_reward": reward,
        }
        info = info if isinstance(info, dict) else {}
        info.update(self.last_reward_components)
        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    def render(self):
        return None

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)


class DiscreteBudgetedActionWrapper(gymnasium_base.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env,
        action_table: dict[int, dict[str, float]],
        irrigation_budget: float = 120.0,
        nitrogen_budget: float = 300.0,
        daily_irrigation_cap: float = 30.0,
        daily_nitrogen_cap: float = 100.0,
        min_interval_days: int = 7,
    ):
        super().__init__()
        self.env = env
        self.action_table = action_table
        self.irrigation_budget = float(irrigation_budget)
        self.nitrogen_budget = float(nitrogen_budget)
        self.daily_irrigation_cap = float(daily_irrigation_cap)
        self.daily_nitrogen_cap = float(daily_nitrogen_cap)
        self.min_interval_days = int(min_interval_days)
        self.action_space = gymnasium_base.spaces.Discrete(len(action_table))
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
        remaining_i = max(0.0, self.irrigation_budget - self.used_irrigation)
        remaining_n = max(0.0, self.nitrogen_budget - self.used_nitrogen)
        safe_i = min(max(0.0, float(raw_real.get("amir", 0.0))), self.daily_irrigation_cap, remaining_i)
        safe_n = min(max(0.0, float(raw_real.get("anfer", 0.0))), self.daily_nitrogen_cap, remaining_n)
        return {"amir": safe_i, "anfer": safe_n}

    def step(self, action):
        action_index = int(np.asarray(action).item())
        raw_real = dict(self.action_table[action_index])
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
        info.update({"used_irrigation": self.used_irrigation, "used_nitrogen": self.used_nitrogen, "budget_dap": dap})
        return obs, reward, terminated, truncated, info


def prepare_case(case_dir: Path) -> Path:
    if case_dir.exists():
        shutil.rmtree(case_dir)
    input_dir = case_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for src in INPUT_DIR.iterdir():
        if src.is_file():
            shutil.copyfile(src, input_dir / src.name)
    filex = input_dir / "CNHL15N1.MZX"
    aux = [str(p) for p in sorted(input_dir.iterdir()) if p.is_file() and p.name != filex.name]
    env_args = {
        "log_saving_path": str(case_dir / f"{case_dir.name}.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": 1,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return case_dir


def make_env(env_args: dict[str, Any], null_baseline_yield: float):
    import gym
    from sb3_wrapper import GymDssatWrapper

    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    linked = DiscreteBudgetedActionWrapper(
        yc_dqn.LazyScalarGymDssatWrapper(GymDssatWrapper(raw)),
        action_table=ACTION_TABLE_9,
    )
    return BaselineRelativeRewardWrapper(linked, null_baseline_yield)


def harvest_yields_from_mgmt(path: Path) -> list[float]:
    import re

    values = []
    if not path.exists():
        return values
    for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
        if "Harvest Yield" in line:
            match = re.search(r"Harvest Yield\s+([0-9.]+)", line)
            if match:
                values.append(float(match.group(1)))
    return values


def log_line(path: Path, message: str) -> None:
    line = f"{pd.Timestamp.now().isoformat()} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def train_and_eval(timesteps: int, seed: int) -> Path:
    install_official_reward_module()
    from stable_baselines3 import DQN

    summary_path = (
        PROJECT_ROOT
        / "DSSAT_auto_validation"
        / "HLA_2004"
        / "hla_2010_2015_four_scenario_with_ppo"
        / "hla_2010_2015_four_scenario_summary.csv"
    )
    df_summary = pd.read_csv(summary_path)
    df_summary["requested_year"] = pd.to_numeric(df_summary["requested_year"], errors="coerce")
    df_summary["final_gwad"] = pd.to_numeric(df_summary["final_gwad"], errors="coerce")
    year_rows = df_summary[df_summary["requested_year"].eq(YEAR)].copy()
    year_rows = year_rows.sort_values("final_gwad", ascending=True)
    row = year_rows.iloc[0]
    null_baseline_yield = float(row["final_gwad"])
    run_name = f"baseline_relative_seed{seed}_{timesteps}steps"
    run_dir = OUT_DIR / str(YEAR) / run_name
    prepare_case(run_dir)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    debug_log = run_dir / "014_10_debug.log"
    snapshot = run_dir / "pdi_tmp_snapshot_eval"

    log_line(debug_log, f"train:start timesteps={timesteps} seed={seed} null_baseline={null_baseline_yield}")
    env = make_env(env_args, null_baseline_yield)
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
        log_line(debug_log, "learn:start")
        model.learn(total_timesteps=int(timesteps), progress_bar=False)
        log_line(debug_log, "learn:done")
        model_dir = run_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(model_dir / "dqn_hla2015_baseline_relative"))
    finally:
        env.close()
        log_line(debug_log, "train_env:closed")

    rows: list[dict[str, Any]] = []
    eval_env = make_env(env_args, null_baseline_yield)
    try:
        obs, info = eval_env.reset()
        for step in range(260):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            latest = latest_observation_dict(eval_env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "year": YEAR,
                    "seed": seed,
                    "timesteps": timesteps,
                    "step": step,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "dap": scalar(latest.get("dap")),
                    "action_index": int(np.asarray(action).item()),
                    "safe_amir": float(eval_env.env.last_safe_real_action.get("amir", 0.0)),
                    "safe_anfer": float(eval_env.env.last_safe_real_action.get("anfer", 0.0)),
                    "used_irrigation": float(eval_env.env.used_irrigation),
                    "used_nitrogen": float(eval_env.env.used_nitrogen),
                    "reward": float(reward),
                    "yield_gain": eval_env.last_reward_components.get("yield_gain", np.nan),
                    "water_cost_term": eval_env.last_reward_components.get("water_cost_term", np.nan),
                    "nitrogen_cost_term": eval_env.last_reward_components.get("nitrogen_cost_term", np.nan),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "done": bool(terminated or truncated),
                }
            )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(eval_env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        eval_env.close()
        log_line(debug_log, "eval_env:closed")

    daily = pd.DataFrame(rows)
    daily.to_csv(run_dir / "dqn_baseline_rel_eval_daily.csv", index=False, encoding="utf-8-sig")
    event_summary = parse_events(snapshot / "MgmtEvent.OUT")
    harvest_values = harvest_yields_from_mgmt(snapshot / "MgmtEvent.OUT")
    summary = {
        "year": YEAR,
        "seed": seed,
        "timesteps": timesteps,
        "null_baseline_yield": null_baseline_yield,
        "final_grnwt": float(daily["grnwt"].dropna().iloc[-1]) if not daily.empty else None,
        "final_topwt": float(daily["topwt"].dropna().iloc[-1]) if not daily.empty else None,
        "action_irrigation_total": float(daily["safe_amir"].sum()) if not daily.empty else None,
        "action_nitrogen_total": float(daily["safe_anfer"].sum()) if not daily.empty else None,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else None,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else None,
        "eval_reward_total": float(daily["reward"].sum()) if not daily.empty else None,
        "harvest_yield_values_from_mgmtevent": harvest_values,
        **event_summary,
        "run_dir": str(run_dir),
    }
    (run_dir / "event_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


def collect_summary() -> pd.DataFrame:
    rows = []
    for path in sorted((OUT_DIR / str(YEAR)).glob("*/*event_summary.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(OUT_DIR / "014_10_hla2015_baseline_relative_reward_summary.csv", index=False, encoding="utf-8-sig")
    return df


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无结果。_"
    keep = [
        "timesteps",
        "seed",
        "null_baseline_yield",
        "final_grnwt",
        "action_irrigation_total",
        "action_nitrogen_total",
        "max_water_stress",
        "max_nitrogen_stress",
        "eval_reward_total",
    ]
    show = df[[c for c in keep if c in df.columns]].copy()
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            show[col] = show[col].astype(str)
    lines = [
        "| " + " | ".join(show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for row in show.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_record(summary: pd.DataFrame) -> None:
    lines = [
        "# 014_10 HLA2015 成功策略复现记录",
        "",
        "## 目的",
        "",
        "把 HLA2010 的成功策略迁移到 HLA2015，验证能否稳定优于 null 并输出有意义的水氮操作。",
        "",
        "## 设置",
        "",
        "- 9 动作：灌溉 0/15/30 mm，施氮 0/50/100 kg/ha",
        "- baseline-relative reward",
        "- 输入包：`DSSAT_auto_validation/HLA_2004/hla2015_literature_aligned_dqn_012_15/2015/literature_aligned_seed0_5000steps/input`",
        "",
        "## 结果",
        "",
        df_to_md(summary),
        "",
        "## 文件",
        "",
        f"- 汇总：`{(OUT_DIR / '014_10_hla2015_baseline_relative_reward_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- 输出目录：`{OUT_DIR.relative_to(PROJECT_ROOT)}`",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.collect_only:
        train_and_eval(args.timesteps, args.seed)
    summary = collect_summary()
    write_record(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
