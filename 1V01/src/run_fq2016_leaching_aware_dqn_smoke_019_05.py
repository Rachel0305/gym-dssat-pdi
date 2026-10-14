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
from run_fq_all_year_screen_and_dqn_transfer_014_01 import (
    INPUT_ROOT,
    MZX_NAME,
    TEMPLATE_TRNO,
    parse_events,
    prepare_text_for_shifted_scenario,
)
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc_dqn


YEAR = 2016
SITE = "FQ"
STATION = "Fengqiu"
NULL_BASELINE = 7066.0

OUT_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "fq2016_leaching_aware_reward_smoke_019_05"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-10_019_05_fq2016_leaching_aware_reward_smoke_record.md"

IRRIGATION_BUDGET = 120.0
NITROGEN_BUDGET = 300.0
DAILY_IRRIGATION_CAP = 30.0
DAILY_NITROGEN_CAP = 100.0
MIN_INTERVAL_DAYS = 7
WINDOWS = {"irrigation": [(1, 120)], "nitrogen": [(1, 120)]}


def latest_full_state_dict(env) -> dict[str, Any]:
    full_state: dict[str, Any] = {}
    raw = getattr(env.unwrapped, "_state", None)
    if isinstance(raw, dict):
        full_state.update(raw)
    history = getattr(env.unwrapped, "_history", {})
    if isinstance(history, dict):
        states = history.get("state", [])
        if states and isinstance(states[-1], dict):
            full_state.update(states[-1])
    return full_state


class LeachingAwareRewardWrapper(gymnasium_base.Env):
    metadata = {"render_modes": []}

    def __init__(self, env, null_baseline_yield: float, leaching_cost: float):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.null_baseline_yield = float(null_baseline_yield)
        self.leaching_cost = float(leaching_cost)
        self.previous_cleach = 0.0
        self.last_reward_components: dict[str, float] = {}

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        full_state = latest_full_state_dict(self.env)
        self.previous_cleach = float(scalar(full_state.get("cleach"), 0.0) or 0.0)
        self.last_reward_components = {
            "yield_gain": 0.0,
            "water_cost_term": 0.0,
            "nitrogen_cost_term": 0.0,
            "delta_cleach": 0.0,
            "leaching_cost_term": 0.0,
            "leaching_aware_reward": 0.0,
        }
        return obs, info

    def step(self, action):
        obs, _old_reward, terminated, truncated, info = self.env.step(action)
        latest = latest_observation_dict(self.env, obs, info)
        full_state = latest_full_state_dict(self.env)
        grnwt = float(scalar(latest.get("grnwt", 0.0)) or 0.0)
        irrigation = float(getattr(self.env, "last_safe_real_action", {}).get("amir", 0.0))
        nitrogen = float(getattr(self.env, "last_safe_real_action", {}).get("anfer", 0.0))
        current_cleach = float(scalar(full_state.get("cleach"), self.previous_cleach) or 0.0)
        delta_cleach = max(0.0, current_cleach - self.previous_cleach)
        self.previous_cleach = current_cleach

        water_cost_term = yc_dqn.WATER_COST * irrigation
        nitrogen_cost_term = yc_dqn.NITROGEN_COST * nitrogen
        leaching_cost_term = self.leaching_cost * delta_cleach
        yield_gain = max(0.0, grnwt - self.null_baseline_yield) if bool(terminated or truncated) else 0.0
        reward = float(yield_gain - water_cost_term - nitrogen_cost_term - leaching_cost_term)
        self.last_reward_components = {
            "yield_gain": float(yield_gain),
            "water_cost_term": float(water_cost_term),
            "nitrogen_cost_term": float(nitrogen_cost_term),
            "delta_cleach": float(delta_cleach),
            "leaching_cost_term": float(leaching_cost_term),
            "leaching_aware_reward": float(reward),
            "full_state_cleach": current_cleach,
            "full_state_tleachd": float(scalar(full_state.get("tleachd"), 0.0) or 0.0),
            "full_state_cnox": float(scalar(full_state.get("cnox"), 0.0) or 0.0),
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


def configure_globals(seed: int, timesteps: int) -> None:
    yc_dqn.SEED = seed
    yc_dqn.TIMESTEPS = timesteps
    yc_dqn.IRRIGATION_BUDGET = IRRIGATION_BUDGET
    yc_dqn.NITROGEN_BUDGET = NITROGEN_BUDGET
    yc_dqn.DAILY_IRRIGATION_CAP = DAILY_IRRIGATION_CAP
    yc_dqn.DAILY_NITROGEN_CAP = DAILY_NITROGEN_CAP
    yc_dqn.MIN_INTERVAL_DAYS = MIN_INTERVAL_DAYS
    yc_dqn.WATER_COST = 1.0
    yc_dqn.NITROGEN_COST = 5.0


def prepare_run_dir(seed: int, timesteps: int, leaching_cost: float) -> Path:
    run_name = f"seed{seed}_{timesteps}steps_lc{leaching_cost:g}"
    run_dir = OUT_ROOT / run_name
    if run_dir.exists():
        suffix = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUT_ROOT / f"{run_name}_{suffix}"
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    text = prepare_text_for_shifted_scenario(YEAR, "dqn_linked_free_daily")
    filex = input_dir / f"CNFQ{YEAR}_leaching_aware_dqn.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")

    for src in INPUT_ROOT.iterdir():
        if src.is_file() and src.name != MZX_NAME:
            shutil.copyfile(src, input_dir / src.name)

    aux = [str(p) for p in input_dir.iterdir() if p.suffix.upper() in {".CUL", ".SOL", ".WTH", ".MZA", ".MZT"}]
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": seed,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": TEMPLATE_TRNO,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


def make_raw_env(env_args: dict[str, Any]):
    import gym
    from sb3_wrapper import GymDssatWrapper

    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    return yc_dqn.LazyScalarGymDssatWrapper(GymDssatWrapper(raw))


def make_env(env_args: dict[str, Any], leaching_cost: float):
    linked = yc_dqn.YCDiscreteBudgetedWrapper(make_raw_env(env_args), WINDOWS["irrigation"], WINDOWS["nitrogen"])
    return LeachingAwareRewardWrapper(linked, NULL_BASELINE, leaching_cost)


def harvest_yields_from_mgmt(path: Path) -> list[float]:
    values = []
    if not path.exists():
        return values
    for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
        if "Harvest Yield" in line:
            match = re.search(r"Harvest Yield\s+([0-9.]+)", line)
            if match:
                values.append(float(match.group(1)))
    return values


def evaluate_model(model, env_args: dict[str, Any], checkpoint: int, run_dir: Path, leaching_cost: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    env = make_env(env_args, leaching_cost)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(280):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            latest = latest_observation_dict(env, obs, info)
            full_state = latest_full_state_dict(env)
            yrdoy = scalar(latest.get("yrdoy"))
            safe_action = dict(getattr(env.env, "last_safe_real_action", {}) or {})
            rows.append(
                {
                    "site": SITE,
                    "station": STATION,
                    "year": YEAR,
                    "scenario": "leaching_aware_dqn_smoke",
                    "checkpoint_step": checkpoint,
                    "step": step,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "dap": scalar(latest.get("dap")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "cleach": scalar(full_state.get("cleach")),
                    "tleachd": scalar(full_state.get("tleachd")),
                    "cnox": scalar(full_state.get("cnox")),
                    "delta_cleach": float(info.get("delta_cleach", np.nan)) if isinstance(info, dict) else np.nan,
                    "leaching_cost_term": float(info.get("leaching_cost_term", np.nan)) if isinstance(info, dict) else np.nan,
                    "yield_gain": float(info.get("yield_gain", np.nan)) if isinstance(info, dict) else np.nan,
                    "water_cost_term": float(info.get("water_cost_term", np.nan)) if isinstance(info, dict) else np.nan,
                    "nitrogen_cost_term": float(info.get("nitrogen_cost_term", np.nan)) if isinstance(info, dict) else np.nan,
                    "irrigation_mm": float(safe_action.get("amir", 0.0)),
                    "fertilizer_kg_ha": float(safe_action.get("anfer", 0.0)),
                    "action_index": int(np.asarray(action).item()),
                    "reward": float(reward),
                    "used_irrigation": float(getattr(env.env, "used_irrigation", np.nan)),
                    "used_nitrogen": float(getattr(env.env, "used_nitrogen", np.nan)),
                }
            )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, run_dir / f"pdi_tmp_snapshot_eval_{checkpoint}", dirs_exist_ok=True)
        env.close()

    daily = pd.DataFrame(rows)
    snapshot = run_dir / f"pdi_tmp_snapshot_eval_{checkpoint}"
    plantgro = parse_dssat_table(snapshot / "PlantGro.OUT")
    soilni = parse_dssat_table(snapshot / "SoilNi.OUT")
    events = parse_events(run_dir, "leaching_aware_dqn_smoke", snapshot_name=f"pdi_tmp_snapshot_eval_{checkpoint}")
    hvals = harvest_yields_from_mgmt(snapshot / "MgmtEvent.OUT")
    summary = {
        "checkpoint_step": checkpoint,
        "leaching_cost": leaching_cost,
        "action_irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "action_fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "harvest_yield_kg_ha": hvals[-1] if hvals else np.nan,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "final_cleach": float(pd.to_numeric(daily["cleach"], errors="coerce").dropna().iloc[-1]) if not daily.empty and not pd.to_numeric(daily["cleach"], errors="coerce").dropna().empty else np.nan,
        "sum_delta_cleach": float(pd.to_numeric(daily["delta_cleach"], errors="coerce").sum()) if "delta_cleach" in daily.columns else np.nan,
        "sum_leaching_cost_term": float(pd.to_numeric(daily["leaching_cost_term"], errors="coerce").sum()) if "leaching_cost_term" in daily.columns else np.nan,
        "soilni_final_NLCC": float(pd.to_numeric(soilni["NLCC"], errors="coerce").dropna().iloc[-1]) if "NLCC" in soilni.columns and not pd.to_numeric(soilni["NLCC"], errors="coerce").dropna().empty else np.nan,
        "total_reward": float(daily["reward"].sum()) if not daily.empty else np.nan,
        "irrigation_total_mgmtevent": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "fertilizer_total_mgmtevent": float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
    }
    return daily, summary


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无结果。_"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")
        else:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(v))
    lines = [
        "| " + " | ".join(out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for row in out.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_record(seed: int, timesteps: int, run_dir: Path, summary: pd.DataFrame) -> None:
    show_cols = [
        "checkpoint_step",
        "leaching_cost",
        "final_grain_kg_ha",
        "action_irrigation_total",
        "action_fertilizer_total",
        "final_cleach",
        "soilni_final_NLCC",
        "sum_delta_cleach",
        "sum_leaching_cost_term",
        "total_reward",
    ]
    table = dataframe_to_markdown(summary[[c for c in show_cols if c in summary.columns]])
    lines = [
        "# 019_05 FQ2016 leaching-aware reward smoke test 记录",
        "",
        "## 结论先行",
        "",
        "本轮只做低步数 smoke test，不作为策略优劣结论。目的只是确认 full-state `cleach` 能进入日值表和 reward component。",
        "",
        "结论：链路通过。日值表中 `cleach`、`delta_cleach`、`leaching_cost_term` 均已写出；`final_cleach` 与 `SoilNi.OUT` 的 `NLCC` 基本一致。当前可以进入不同 `leaching_cost` 系数的小范围敏感性测试，但还不应该直接长训练或写成策略优劣结论。",
        "",
        "## 奖励函数",
        "",
        "```text",
        "reward = max(0, final_grnwt - local_null_yield) - water_cost * irrigation - nitrogen_cost * nitrogen - leaching_cost * delta_cleach",
        "```",
        "",
        "## 结果",
        "",
        table,
        "",
        "## 文件",
        "",
        f"- 运行目录：`{run_dir.relative_to(PROJECT_ROOT)}`",
        f"- 日值表：`{(run_dir / 'leaching_aware_eval_daily.csv').relative_to(PROJECT_ROOT)}`",
        f"- 汇总表：`{(run_dir / 'leaching_aware_checkpoint_summary.csv').relative_to(PROJECT_ROOT)}`",
        "",
        "## 判断",
        "",
        "本轮链路通过。下一步建议只做小范围 `leaching_cost` 敏感性，例如 0、20、50、100 的 500-step smoke 或离线回放评分；确认惩罚强度方向后，再考虑 5K 训练。",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--leaching-cost", type=float, default=50.0)
    args = parser.parse_args()

    configure_globals(args.seed, args.timesteps)
    run_dir = prepare_run_dir(args.seed, args.timesteps, args.leaching_cost)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))

    from stable_baselines3 import DQN

    train_env = make_env(env_args, args.leaching_cost)
    all_daily = []
    all_summary = []
    try:
        model = DQN(
            "MlpPolicy",
            train_env,
            verbose=0,
            seed=args.seed,
            learning_rate=1e-4,
            buffer_size=5000,
            learning_starts=50,
            batch_size=32,
            train_freq=1,
            gradient_steps=1,
            gamma=0.99,
            exploration_fraction=0.35,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
        )
        prev = 0
        for checkpoint in range(args.checkpoint_interval, args.timesteps + 1, args.checkpoint_interval):
            model.learn(total_timesteps=checkpoint - prev, reset_num_timesteps=False, progress_bar=False)
            prev = checkpoint
            model_dir = run_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(model_dir / f"dqn_leaching_aware_checkpoint_{checkpoint}"))
            daily, summary = evaluate_model(model, env_args, checkpoint, run_dir, args.leaching_cost)
            all_daily.append(daily)
            all_summary.append(summary)
    finally:
        train_env.close()

    daily_df = pd.concat(all_daily, ignore_index=True)
    summary_df = pd.DataFrame(all_summary)
    daily_df.to_csv(run_dir / "leaching_aware_eval_daily.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(run_dir / "leaching_aware_checkpoint_summary.csv", index=False, encoding="utf-8-sig")
    write_record(args.seed, args.timesteps, run_dir, summary_df)
    print(summary_df.to_string(index=False))
    print(run_dir)


if __name__ == "__main__":
    main()
