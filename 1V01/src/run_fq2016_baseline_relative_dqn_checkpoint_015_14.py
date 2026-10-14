from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import gymnasium as gymnasium_base
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_all_year_screen_and_dqn_transfer_014_01 import (
    INPUT_ROOT,
    MZX_NAME,
    TEMPLATE_TRNO,
    parse_events,
    parse_weather,
    prepare_text_for_shifted_scenario,
)
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc_dqn


YEAR = 2016
SITE = "FQ"
STATION = "Fengqiu"
NULL_BASELINE = 7066.0
OUT_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "fq2016_baseline_relative_dqn_checkpoint_015_14"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-01_015_14_fq2016_baseline_relative_dqn_checkpoint_record.md"

IRRIGATION_BUDGET = 120.0
NITROGEN_BUDGET = 300.0
DAILY_IRRIGATION_CAP = 30.0
DAILY_NITROGEN_CAP = 100.0
MIN_INTERVAL_DAYS = 7
WINDOWS = {"irrigation": [(1, 120)], "nitrogen": [(1, 120)]}


class BaselineRelativeRewardWrapper(gymnasium_base.Env):
    metadata = {"render_modes": []}

    def __init__(self, env, null_baseline_yield: float):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.null_baseline_yield = float(null_baseline_yield)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, _old_reward, terminated, truncated, info = self.env.step(action)
        latest = latest_observation_dict(self.env, obs, info)
        grnwt = float(scalar(latest.get("grnwt", 0.0)) or 0.0)
        irrigation = float(getattr(self.env, "last_safe_real_action", {}).get("amir", 0.0))
        nitrogen = float(getattr(self.env, "last_safe_real_action", {}).get("anfer", 0.0))
        water_cost_term = yc_dqn.WATER_COST * irrigation
        nitrogen_cost_term = yc_dqn.NITROGEN_COST * nitrogen
        yield_gain = max(0.0, grnwt - self.null_baseline_yield) if bool(terminated or truncated) else 0.0
        reward = float(yield_gain - water_cost_term - nitrogen_cost_term)
        info = info if isinstance(info, dict) else {}
        info.update(
            {
                "yield_gain": yield_gain,
                "water_cost_term": water_cost_term,
                "nitrogen_cost_term": nitrogen_cost_term,
                "baseline_relative_reward": reward,
            }
        )
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


def prepare_run_dir(seed: int, timesteps: int) -> Path:
    run_dir = OUT_ROOT / f"seed{seed}_{timesteps}steps"
    if run_dir.exists():
        raise RuntimeError(f"Output directory already exists, refusing to overwrite: {run_dir}")
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    text = prepare_text_for_shifted_scenario(YEAR, "dqn_linked_free_daily")
    filex = input_dir / f"CNFQ{YEAR}_baseline_relative_dqn.MZX"
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


def make_env(env_args: dict[str, Any]):
    linked = yc_dqn.YCDiscreteBudgetedWrapper(make_raw_env(env_args), WINDOWS["irrigation"], WINDOWS["nitrogen"])
    return BaselineRelativeRewardWrapper(linked, NULL_BASELINE)


def evaluate_model(model, env_args: dict[str, Any], checkpoint: int, run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    env = make_env(env_args)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(280):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            safe_action = dict(getattr(env.env, "last_safe_real_action", {}) or {})
            rows.append(
                {
                    "site": SITE,
                    "station": STATION,
                    "year": YEAR,
                    "scenario": "baseline_relative_dqn",
                    "checkpoint_step": checkpoint,
                    "step": step,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "dap": scalar(latest.get("dap")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "irrigation_mm": float(safe_action.get("amir", 0.0)),
                    "fertilizer_kg_ha": float(safe_action.get("anfer", 0.0)),
                    "action_index": int(np.asarray(action).item()),
                    "reward": float(reward),
                    "used_irrigation": float(info.get("used_irrigation", np.nan)) if isinstance(info, dict) else np.nan,
                    "used_nitrogen": float(info.get("used_nitrogen", np.nan)) if isinstance(info, dict) else np.nan,
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
    events = parse_events(run_dir, "baseline_relative_dqn", snapshot_name=f"pdi_tmp_snapshot_eval_{checkpoint}")
    summary = {
        "checkpoint_step": checkpoint,
        "action_irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "action_fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "total_reward": float(daily["reward"].sum()) if not daily.empty else np.nan,
        "irrigation_total_mgmtevent": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "fertilizer_total_mgmtevent": float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
        "irrigation_events_mgmtevent": int(events.loc[events["unit"].eq("mm")].shape[0]) if not events.empty else 0,
        "fertilizer_events_mgmtevent": int(events.loc[events["unit"].str.contains("kg", na=False)].shape[0]) if not events.empty else 0,
    }
    return daily, summary


def plot_checkpoint_summary(summary: pd.DataFrame, run_dir: Path, seed: int, timesteps: int) -> None:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    x = summary["checkpoint_step"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(x, summary["final_grain_kg_ha"], marker="o", color="#1B4E8A", label="DQN")
    axes[0].axhline(7066, color="#555555", linestyle="-", linewidth=1.2, label="null")
    axes[0].axhline(7933, color="#C44E52", linestyle="--", linewidth=1.2, label="recorded")
    axes[0].axhline(8012, color="#DDAA00", linestyle="-", linewidth=1.2, label="auto")
    axes[0].set_ylabel("GWAD kg/ha")
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    axes[1].plot(x, summary["action_irrigation_total"], marker="o", color="#2E7D32", label="I")
    axes[1].plot(x, summary["action_fertilizer_total"], marker="^", color="#7B1FA2", label="N")
    axes[1].set_ylabel("Resource total")
    axes[1].legend(frameon=False)
    axes[2].plot(x, summary["total_reward"], marker="o", color="#222222")
    axes[2].axhline(0, color="#888888", linewidth=1)
    axes[2].set_ylabel("Reward")
    axes[2].set_xlabel("Checkpoint step")
    for ax in axes:
        ax.grid(True, color="#E8E8E8", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"FQ2016 baseline-relative DQN seed{seed} checkpoint summary")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fq2016_baseline_relative_seed{seed}_{timesteps}steps_checkpoint_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_record(seed: int, timesteps: int, checkpoint_interval: int, run_dir: Path, summary: pd.DataFrame) -> None:
    best_reward = summary.loc[summary["total_reward"].idxmax()]
    best_yield = summary.loc[summary["final_grain_kg_ha"].idxmax()]
    headers = list(summary.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in summary.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.3f}" if not float(val).is_integer() else f"{int(val)}")
            else:
                vals.append(str(val))
        table_lines.append("| " + " | ".join(vals) + " |")
    lines = [
        "# 015_14 FQ2016 baseline-relative DQN checkpoint record",
        "",
        "## 固定设置",
        "",
        f"- year: FQ2016",
        f"- seed: {seed}",
        f"- timesteps: {timesteps}",
        f"- checkpoint interval: {checkpoint_interval}",
        f"- null baseline: {NULL_BASELINE:.1f} kg/ha",
        "- reward: terminal max(0, GWAD_final - GWAD_null) - water_cost*I - nitrogen_cost*N",
        "- costs: water=1.0, nitrogen=5.0",
        "- action space: I in {0,15,30}, N in {0,50,100}",
        "- budget: I<=120, N<=300, min interval=7 days",
        "",
        "## 基准",
        "",
        "| scenario | GWAD kg/ha | I mm | N kg/ha |",
        "|---|---:|---:|---:|",
        "| null | 7066 | 0 | 0 |",
        "| recorded_shifted | 7933 | 75 | 144 |",
        "| DSSAT auto | 8012 | 59.9 | 0 |",
        "",
        "## 输出",
        "",
        f"- run dir: `{run_dir.relative_to(PROJECT_ROOT)}`",
        f"- summary: `{(run_dir / 'checkpoint_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- daily: `{(run_dir / 'dqn_eval_daily.csv').relative_to(PROJECT_ROOT)}`",
        "",
        "## Checkpoint results",
        "",
        "\n".join(table_lines),
        "",
        "## 判断",
        "",
        f"- Best reward checkpoint: {int(best_reward['checkpoint_step'])}, GWAD={best_reward['final_grain_kg_ha']:.1f}, I={best_reward['action_irrigation_total']:.1f}, N={best_reward['action_fertilizer_total']:.1f}, reward={best_reward['total_reward']:.1f}.",
        f"- Best yield checkpoint: {int(best_yield['checkpoint_step'])}, GWAD={best_yield['final_grain_kg_ha']:.1f}, I={best_yield['action_irrigation_total']:.1f}, N={best_yield['action_fertilizer_total']:.1f}, reward={best_yield['total_reward']:.1f}.",
    ]
    record_path = PROJECT_ROOT / "docs" / f"2026-07-05_017_01_fq2016_baseline_relative_dqn_seed{seed}_checkpoint_record.md"
    record_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    args = parser.parse_args()

    configure_globals(args.seed, args.timesteps)
    run_dir = prepare_run_dir(args.seed, args.timesteps)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))

    from stable_baselines3 import DQN

    train_env = make_env(env_args)
    all_daily = []
    all_summary = []
    try:
        model = DQN(
            "MlpPolicy",
            train_env,
            verbose=0,
            seed=args.seed,
            learning_rate=1e-4,
            buffer_size=10000,
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
            model.save(str(model_dir / f"dqn_baseline_relative_checkpoint_{checkpoint}"))
            daily, summary = evaluate_model(model, env_args, checkpoint, run_dir)
            all_daily.append(daily)
            all_summary.append(summary)
    finally:
        train_env.close()

    daily_df = pd.concat(all_daily, ignore_index=True)
    summary_df = pd.DataFrame(all_summary)
    daily_df.to_csv(run_dir / "dqn_eval_daily.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(run_dir / "checkpoint_summary.csv", index=False, encoding="utf-8-sig")
    plot_checkpoint_summary(summary_df, run_dir, args.seed, args.timesteps)
    write_record(args.seed, args.timesteps, args.checkpoint_interval, run_dir, summary_df)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
