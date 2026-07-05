from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import gymnasium as gymnasium_base
import numpy as np
import pandas as pd

import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_baseline_relative_dqn_015_10"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-01_015_10_yc2014_baseline_relative_dqn_record.md"

ACTION_TABLE_9: dict[int, dict[str, float]] = {
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
        water_cost_term = yc.WATER_COST * irrigation
        nitrogen_cost_term = yc.NITROGEN_COST * nitrogen
        yield_gain = max(0.0, grnwt - self.null_baseline_yield) if bool(terminated or truncated) else 0.0
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


def configure_yc_module(seed: int, timesteps: int) -> None:
    yc.OUT_DIR = OUT_DIR
    yc.DOC_PATH = DOC_PATH
    yc.SEED = seed
    yc.TIMESTEPS = timesteps
    yc.WATER_COST = 1.0
    yc.NITROGEN_COST = 5.0
    yc.IRRIGATION_BUDGET = 120.0
    yc.NITROGEN_BUDGET = 300.0
    yc.DAILY_IRRIGATION_CAP = 30.0
    yc.DAILY_NITROGEN_CAP = 100.0
    yc.MIN_INTERVAL_DAYS = 7
    yc.ACTION_TABLE = ACTION_TABLE_9


def make_train_env(env_args: dict[str, Any], null_baseline_yield: float):
    linked = yc.YCDiscreteBudgetedWrapper(
        yc.make_raw_env(env_args),
        yc.FREE_DAILY_WINDOWS["irrigation"],
        yc.FREE_DAILY_WINDOWS["nitrogen"],
    )
    return BaselineRelativeRewardWrapper(linked, null_baseline_yield)


def get_null_baseline_yield() -> float:
    null_daily, null_summary = yc.run_zero_action_scenario("null")
    null_dir = OUT_DIR / f"seed{yc.SEED}" / "null"
    null_dir.mkdir(parents=True, exist_ok=True)
    null_daily.to_csv(null_dir / "015_10_yc2014_null_daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([null_summary]).to_csv(null_dir / "015_10_yc2014_null_summary.csv", index=False, encoding="utf-8-sig")
    return float(null_summary["final_grain_kg_ha"])


def train_and_eval(timesteps: int, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    from stable_baselines3 import DQN

    null_baseline_yield = get_null_baseline_yield()
    scenario = "dqn_baseline_relative_9action"
    run_dir = yc.prepare_case_for_scenario(scenario)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))

    train_env = make_train_env(env_args, null_baseline_yield)
    try:
        model = DQN(
            "MlpPolicy",
            train_env,
            verbose=0,
            seed=seed,
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
        model.learn(total_timesteps=int(timesteps), progress_bar=False)
        model.save(str(run_dir / "dqn_baseline_relative_model"))
    finally:
        train_env.close()

    eval_env = make_train_env(env_args, null_baseline_yield)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = eval_env.reset()
        for step in range(260):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            latest = latest_observation_dict(eval_env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "scenario": scenario,
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "irrigation_mm": eval_env.last_safe_real_action.get("amir", 0.0),
                    "fertilizer_kg_ha": eval_env.last_safe_real_action.get("anfer", 0.0),
                    "action_index": eval_env.last_action_index,
                    "reward": float(reward),
                    "yield_gain": eval_env.last_reward_components.get("yield_gain", np.nan),
                    "water_cost_term": eval_env.last_reward_components.get("water_cost_term", np.nan),
                    "nitrogen_cost_term": eval_env.last_reward_components.get("nitrogen_cost_term", np.nan),
                }
            )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(eval_env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, run_dir / "pdi_tmp_snapshot_eval", dirs_exist_ok=True)
        eval_env.close()

    daily = pd.DataFrame(rows)
    if not daily.empty:
        daily = yc.build_plot_df(daily)
    events = yc.parse_events_eval(run_dir, scenario)
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot_eval" / "PlantGro.OUT")
    summary = {
        "scenario": scenario,
        "seed": seed,
        "timesteps": timesteps,
        "reward_type": "baseline_relative_site_year_null",
        "null_baseline_yield": null_baseline_yield,
        "action_irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "action_fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "mgmt_event_irrigation_total": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "mgmt_event_fertilizer_total": float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns else np.nan,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "total_reward": float(daily["reward"].sum()) if not daily.empty else np.nan,
        "run_dir": str(run_dir),
    }
    daily.to_csv(run_dir / "015_10_yc2014_baseline_relative_eval_daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(run_dir / "015_10_yc2014_baseline_relative_summary.csv", index=False, encoding="utf-8-sig")
    (run_dir / "event_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return daily, summary


def evaluate_model(model, env_args: dict[str, Any], null_baseline_yield: float, checkpoint_step: int, run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    scenario = f"checkpoint_{checkpoint_step}"
    eval_env = make_train_env(env_args, null_baseline_yield)
    rows: list[dict[str, Any]] = []
    snapshot_dir = run_dir / scenario / "pdi_tmp_snapshot_eval"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    try:
        obs, info = eval_env.reset()
        for step in range(260):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            latest = latest_observation_dict(eval_env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "checkpoint_step": checkpoint_step,
                    "scenario": scenario,
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "irrigation_mm": eval_env.last_safe_real_action.get("amir", 0.0),
                    "fertilizer_kg_ha": eval_env.last_safe_real_action.get("anfer", 0.0),
                    "action_index": eval_env.last_action_index,
                    "reward": float(reward),
                    "yield_gain": eval_env.last_reward_components.get("yield_gain", np.nan),
                    "water_cost_term": eval_env.last_reward_components.get("water_cost_term", np.nan),
                    "nitrogen_cost_term": eval_env.last_reward_components.get("nitrogen_cost_term", np.nan),
                }
            )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(eval_env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(tmp, snapshot_dir, dirs_exist_ok=True)
        eval_env.close()

    daily = pd.DataFrame(rows)
    if not daily.empty:
        daily = yc.build_plot_df(daily)
    events = yc.parse_events_eval(snapshot_dir.parent, scenario)
    plantgro = parse_dssat_table(snapshot_dir / "PlantGro.OUT")
    summary = {
        "checkpoint_step": checkpoint_step,
        "null_baseline_yield": null_baseline_yield,
        "action_irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "action_fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "mgmt_event_irrigation_total": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "mgmt_event_fertilizer_total": float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns else np.nan,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "total_reward": float(daily["reward"].sum()) if not daily.empty else np.nan,
    }
    return daily, summary


def run_checkpoint_train(timesteps: int, seed: int, checkpoint_interval: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from stable_baselines3 import DQN

    null_baseline_yield = get_null_baseline_yield()
    scenario = "dqn_baseline_relative_checkpoint"
    run_dir = yc.prepare_case_for_scenario(scenario)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    train_env = make_train_env(env_args, null_baseline_yield)
    all_daily: list[pd.DataFrame] = []
    all_summary: list[dict[str, Any]] = []
    try:
        model = DQN(
            "MlpPolicy",
            train_env,
            verbose=0,
            seed=seed,
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
        checkpoints = list(range(checkpoint_interval, timesteps + 1, checkpoint_interval))
        prev = 0
        for checkpoint in checkpoints:
            model.learn(total_timesteps=checkpoint - prev, reset_num_timesteps=False, progress_bar=False)
            prev = checkpoint
            model_dir = run_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(model_dir / f"dqn_baseline_relative_checkpoint_{checkpoint}"))
            daily, summary = evaluate_model(model, env_args, null_baseline_yield, checkpoint, run_dir)
            all_daily.append(daily)
            all_summary.append(summary)
    finally:
        train_env.close()

    all_daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    summary_df = pd.DataFrame(all_summary)
    all_daily_df.to_csv(run_dir / "015_10_yc2014_baseline_relative_checkpoint_daily.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(run_dir / "015_10_yc2014_baseline_relative_checkpoint_summary.csv", index=False, encoding="utf-8-sig")
    return all_daily_df, summary_df


def plot_checkpoint_summary(summary: pd.DataFrame, out_path: Path) -> None:
    if summary.empty:
        return
    summary = summary.sort_values("checkpoint_step")
    x = np.arange(len(summary))
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.bar(x - 0.22, summary["action_irrigation_total"], width=0.3, color="#74A9CF", label="I total")
    ax1.bar(x + 0.10, summary["action_fertilizer_total"], width=0.3, color="#A1D99B", label="N total")
    ax1.set_ylabel("Input amount")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(v // 1000)}K" for v in summary["checkpoint_step"]], rotation=0)
    ax2 = ax1.twinx()
    ax2.plot(x, summary["final_grain_kg_ha"], color="#CB181D", marker="o", lw=1.8, label="Grain")
    ax2.plot(x, summary["total_reward"], color="#54278F", marker="s", lw=1.5, label="Reward")
    ax2.set_ylabel("Yield / reward")
    ax1.grid(True, color="#E6E8F0", linewidth=0.8, alpha=0.9)
    ax1.set_title("YC2014 baseline-relative DQN checkpoint summary", loc="left")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    lines = [
        "# 015_10 YC2014 baseline-relative reward DQN recheck",
        "",
        "## 目的",
        "",
        "用和 HLA baseline-relative 成功线一致的奖励公式复核 YC2014，确认跨站点正式 reward 是否可以统一。",
        "",
        "## 奖励",
        "",
        "`reward_t = - 1.0 * I_t - 5.0 * N_t`",
        "",
        "`reward_T += max(0, GWAD_final - GWAD_null_site_year)`",
        "",
        "每个站点年份使用自己的 null baseline；本轮 YC2014 的 null baseline 来自同一输入包的 null 情景。",
        "",
        "## 结果",
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| null_baseline_yield | {summary['null_baseline_yield']:.3f} |",
        f"| final_grain_kg_ha | {summary['final_grain_kg_ha']:.3f} |",
        f"| final_biomass_kg_ha | {summary['final_biomass_kg_ha']:.3f} |",
        f"| action_irrigation_total | {summary['action_irrigation_total']:.3f} |",
        f"| action_fertilizer_total | {summary['action_fertilizer_total']:.3f} |",
        f"| max_water_stress | {summary['max_water_stress']:.3f} |",
        f"| max_nitrogen_stress | {summary['max_nitrogen_stress']:.3f} |",
        f"| total_reward | {summary['total_reward']:.3f} |",
        "",
        "## 判断",
        "",
        "- 如果该结果仍接近专家产量且水氮投入较低，说明 YC2014 在 baseline-relative 统一奖励下仍可保留为成功案例。",
        "- 如果该结果明显退化，则说明旧 YC 成功依赖 delta_grnwt 奖励，需要单独说明。",
    ]
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    configure_yc_module(args.seed, args.timesteps)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        _daily, summary_df = run_checkpoint_train(args.timesteps, args.seed, args.checkpoint_interval)
        fig_path = OUT_DIR / f"seed{args.seed}" / "dqn_baseline_relative_checkpoint" / "figures" / "yc2014_baseline_relative_checkpoint_summary.png"
        plot_checkpoint_summary(summary_df, fig_path)
        best = summary_df.sort_values(["total_reward", "final_grain_kg_ha"], ascending=False).iloc[0].to_dict()
        write_record(best)
        print(summary_df.to_string(index=False))
    else:
        _daily, summary = train_and_eval(args.timesteps, args.seed)
        write_record(summary)
        print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
