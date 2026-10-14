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

import run_hla_unified_dqn_long_train_015_09 as base
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc
from ppo_evaluate import latest_observation_dict, scalar
from run_hla_official_reward_restart_smoke import install_official_reward_module, prepare_case_at


OUT_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_baseline_relative_dqn_checkpoint_015_12"
SUMMARY_BASELINE = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "hla_2010_2015_four_scenario_with_ppo"
    / "hla_2010_2015_four_scenario_summary.csv"
)

# 奖励函数wrapper
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
    # 奖励计算
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


def get_null_baseline_yield(year: int) -> float:
    df = pd.read_csv(SUMMARY_BASELINE)
    scenario = df["scenario"].fillna("").astype(str).str.strip().str.lower()
    mask = df["requested_year"].astype(int).eq(int(year)) & scenario.eq("null")
    subset = df.loc[mask].copy()
    if subset.empty:
        year_df = df.loc[df["requested_year"].astype(int).eq(int(year))].copy()
        subset = year_df.loc[
            pd.to_numeric(year_df["irrigation_total"], errors="coerce").fillna(-1).eq(0)
            & pd.to_numeric(year_df["fertilizer_total"], errors="coerce").fillna(-1).eq(0)
        ].copy()
    if subset.empty:
        raise RuntimeError(f"Cannot find HLA{year} null baseline in {SUMMARY_BASELINE}")
    return float(subset.iloc[0]["final_gwad"])

# 环境构建和动作约束
def make_env(env_args: dict[str, Any], null_baseline_yield: float):
    linked = yc.YCDiscreteBudgetedWrapper(
        yc.make_raw_env(env_args),
        base.FREE_DAILY_WINDOWS["irrigation"],
        base.FREE_DAILY_WINDOWS["nitrogen"],
    )
    return BaselineRelativeRewardWrapper(linked, null_baseline_yield)


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.3f}" if not float(val).is_integer() else f"{int(val)}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_record(year: int, seed: int, timesteps: int, checkpoint_interval: int, null_baseline: float, run_dir: Path, summary: pd.DataFrame) -> None:
    doc_path = PROJECT_ROOT / "docs" / f"2026-07-01_015_12_hla{year}_baseline_relative_dqn_checkpoint_record.md"
    view_cols = [
        "checkpoint_step",
        "action_irrigation_total",
        "action_fertilizer_total",
        "final_grain_kg_ha",
        "final_biomass_kg_ha",
        "max_water_stress",
        "max_nitrogen_stress",
        "total_reward",
    ]
    view = summary[view_cols].sort_values("checkpoint_step").reset_index(drop=True)
    best_yield = view.loc[view["final_grain_kg_ha"].idxmax()]
    best_reward = view.loc[view["total_reward"].idxmax()]
    final = view.loc[view["checkpoint_step"].idxmax()]
    lines = [
        f"# 015_12 HLA{year} baseline-relative DQN checkpoint record",
        "",
        "## Settings",
        "",
        f"- Year: HLA{year}",
        f"- Seed: {seed}",
        f"- Timesteps: {timesteps}",
        f"- Checkpoint interval: {checkpoint_interval}",
        f"- Null baseline yield: {null_baseline:.1f} kg/ha",
        "- Reward: terminal max(0, GWAD_final - GWAD_null_site_year) minus water/nitrogen costs",
        "- Costs: water=1.0, nitrogen=5.0",
        "- Action space: 9-action, I in {0,15,30}, N in {0,50,100}",
        "- Budget: I<=120, N<=300, minimum interval=7 days",
        "",
        "## Outputs",
        "",
        f"- Run dir: `{run_dir.relative_to(PROJECT_ROOT)}`",
        f"- Summary: `{(run_dir / 'checkpoint_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- Daily: `{(run_dir / 'dqn_eval_daily.csv').relative_to(PROJECT_ROOT)}`",
        f"- Figure: `{(run_dir / 'figures' / f'hla{year}_baseline_relative_seed{seed}_{timesteps}steps_checkpoint_diagnostic.png').relative_to(PROJECT_ROOT)}`",
        "",
        "## Checkpoint Results",
        "",
        markdown_table(view),
        "",
        "## Key Judgement",
        "",
        f"- Best yield checkpoint: {int(best_yield['checkpoint_step'])}, yield={best_yield['final_grain_kg_ha']:.1f}, I={best_yield['action_irrigation_total']:.1f}, N={best_yield['action_fertilizer_total']:.1f}, reward={best_yield['total_reward']:.1f}.",
        f"- Best reward checkpoint: {int(best_reward['checkpoint_step'])}, yield={best_reward['final_grain_kg_ha']:.1f}, I={best_reward['action_irrigation_total']:.1f}, N={best_reward['action_fertilizer_total']:.1f}, reward={best_reward['total_reward']:.1f}.",
        f"- Final checkpoint: {int(final['checkpoint_step'])}, yield={final['final_grain_kg_ha']:.1f}, I={final['action_irrigation_total']:.1f}, N={final['action_fertilizer_total']:.1f}, reward={final['total_reward']:.1f}.",
    ]
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    args = parser.parse_args()

    base.configure_shared_settings()
    install_official_reward_module()
    null_baseline = get_null_baseline_yield(args.year)
    original_base_make_env = base.make_env
    base.make_env = lambda env_args: make_env(env_args, null_baseline)

    run_dir = OUT_ROOT / str(args.year) / f"baseline_relative_seed{args.seed}_{args.timesteps}steps"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    prepare_case_at(args.year, run_dir)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))

    from stable_baselines3 import DQN

    train_env = make_env(env_args, null_baseline)
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
            daily, summary = base.evaluate_model(model, env_args, checkpoint, run_dir)
            all_daily.append(daily)
            all_summary.append(summary)
    finally:
        base.make_env = original_base_make_env
        train_env.close()

    all_daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    summary_df = pd.DataFrame(all_summary)
    daily_path = run_dir / "dqn_eval_daily.csv"
    summary_path = run_dir / "checkpoint_summary.csv"
    fig_path = run_dir / "figures" / f"hla{args.year}_baseline_relative_seed{args.seed}_{args.timesteps}steps_checkpoint_diagnostic.png"
    all_daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    base.plot_checkpoints(all_daily_df, summary_df, fig_path, args.year, args.seed, args.timesteps)
    write_record(args.year, args.seed, args.timesteps, args.checkpoint_interval, null_baseline, run_dir, summary_df)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
