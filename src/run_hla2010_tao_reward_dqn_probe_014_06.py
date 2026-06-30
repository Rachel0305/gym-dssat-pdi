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
from run_hla_official_reward_restart_smoke import install_official_reward_module, parse_events, prepare_case_at
from run_hla2010_dqn_unified_recheck_014_03 import WINDOWS
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc_dqn


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_tao_reward_dqn_probe_014_06"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_014_06_hla2010_tao_reward_dqn_probe_record.md"

YEAR = 2010

# Tao et al. 2023 RF1 economic profit weights, simplified without nitrate leaching.
YIELD_WEIGHT = 0.158
NITROGEN_COST = 0.79
WATER_COST = 1.1


class TaoRF1RewardWrapper(gymnasium_base.Env):
    """Tao et al. IJCAI 2023 RF1-style reward, simplified without N leaching.

    Nonterminal: -0.79 * N_t - 1.1 * W_t
    Terminal:    0.158 * final_grnwt - 0.79 * N_t - 1.1 * W_t
    """

    metadata = {"render_modes": []}

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.last_reward_components = {
            "yield_reward": 0.0,
            "water_cost_term": 0.0,
            "nitrogen_cost_term": 0.0,
            "tao_reward": 0.0,
        }

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.last_reward_components = {
            "yield_reward": 0.0,
            "water_cost_term": 0.0,
            "nitrogen_cost_term": 0.0,
            "tao_reward": 0.0,
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
        yield_reward = YIELD_WEIGHT * grnwt if is_terminal else 0.0
        reward = float(yield_reward - water_cost_term - nitrogen_cost_term)
        self.last_reward_components = {
            "yield_reward": yield_reward,
            "water_cost_term": water_cost_term,
            "nitrogen_cost_term": nitrogen_cost_term,
            "tao_reward": reward,
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


def make_env(env_args: dict[str, Any]):
    import gym
    from sb3_wrapper import GymDssatWrapper

    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    linked = yc_dqn.YCDiscreteBudgetedWrapper(
        yc_dqn.LazyScalarGymDssatWrapper(GymDssatWrapper(raw)),
        irrigation_windows=WINDOWS["free_daily"]["irrigation"],
        nitrogen_windows=WINDOWS["free_daily"]["nitrogen"],
    )
    return TaoRF1RewardWrapper(linked)


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

    run_name = f"tao_rf1_simplified_seed{seed}_{timesteps}steps"
    run_dir = OUT_DIR / str(YEAR) / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    prepare_case_at(YEAR, run_dir)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    debug_log = run_dir / "014_06_debug.log"
    snapshot = run_dir / "pdi_tmp_snapshot_eval"

    log_line(debug_log, f"train:start timesteps={timesteps} seed={seed}")
    env = make_env(env_args)
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
        model.save(str(model_dir / "dqn_hla2010_tao_rf1_simplified"))
    finally:
        env.close()
        log_line(debug_log, "train_env:closed")

    rows: list[dict[str, Any]] = []
    eval_env = make_env(env_args)
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
                    "yield_reward": eval_env.last_reward_components.get("yield_reward", np.nan),
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
    daily.to_csv(run_dir / "dqn_tao_eval_daily.csv", index=False, encoding="utf-8-sig")
    event_summary = parse_events(snapshot / "MgmtEvent.OUT")
    harvest_values = harvest_yields_from_mgmt(snapshot / "MgmtEvent.OUT")
    summary = {
        "year": YEAR,
        "algorithm": "DQN",
        "reward_type": "Tao_RF1_simplified_no_leaching",
        "yield_weight": YIELD_WEIGHT,
        "water_cost": WATER_COST,
        "nitrogen_cost": NITROGEN_COST,
        "seed": seed,
        "timesteps": timesteps,
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
        df.to_csv(OUT_DIR / "014_06_hla2010_tao_reward_summary.csv", index=False, encoding="utf-8-sig")
    return df


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无结果。_"
    keep = [
        "timesteps",
        "seed",
        "final_grnwt",
        "final_topwt",
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
        "# 014_06 HLA2010 Tao et al. 奖励函数 DQN 探针记录",
        "",
        "## 目的",
        "",
        "测试 Tao et al. 2023 RF1 economic profit 奖励函数的简化版，是否能缓解 HLA2010 当前 economic DQN 的 no-op 退化。",
        "",
        "## 奖励函数",
        "",
        "```text",
        "非终止日: r_t = -0.79 * N_t - 1.1 * W_t",
        "终止日:   r_t = 0.158 * GRNWT_final - 0.79 * N_t - 1.1 * W_t",
        "```",
        "",
        "暂不包含硝态氮淋失项，因此是 Tao RF1 的简化版。",
        "",
        "## 当前结果",
        "",
        df_to_md(summary),
        "",
        "## 文件",
        "",
        f"- 汇总：`{(OUT_DIR / '014_06_hla2010_tao_reward_summary.csv').relative_to(PROJECT_ROOT)}`",
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
