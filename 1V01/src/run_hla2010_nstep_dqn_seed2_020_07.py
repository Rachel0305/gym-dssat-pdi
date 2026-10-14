from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from stable_baselines3 import DQN

import run_hla_baseline_relative_dqn_checkpoint_015_12 as hla
import run_hla_unified_dqn_long_train_015_09 as base
from run_hla_official_reward_restart_smoke import install_official_reward_module, prepare_case_at

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_dqn_seed2_020_07"
DOC = ROOT / "docs" / "2026-07-10_020_07_hla2010_nstep_dqn_seed2_record.md"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False, name=None):
        vals = []
        for value in row:
            if pd.isna(value): vals.append("")
            elif isinstance(value, float): vals.append(f"{value:.3f}".rstrip("0").rstrip("."))
            else: vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--checkpoint-interval", type=int, required=True)
    args = parser.parse_args()
    if args.timesteps <= 0 or args.checkpoint_interval <= 0 or args.timesteps % args.checkpoint_interval:
        raise ValueError("timesteps must be positive and divisible by checkpoint interval")

    base.configure_shared_settings()
    install_official_reward_module()
    null_baseline = hla.get_null_baseline_yield(2010)
    run_dir = OUT_ROOT / f"nstep5_seed2_{args.timesteps}steps"
    if run_dir.exists(): shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    prepare_case_at(2010, run_dir)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    original_make_env = base.make_env
    base.make_env = lambda current_args: hla.make_env(current_args, null_baseline)
    train_env = hla.make_env(env_args, null_baseline)
    daily_frames, summary_rows = [], []
    try:
        model = DQN(
            "MlpPolicy", train_env, verbose=0, seed=2,
            learning_rate=1e-4, buffer_size=10000, learning_starts=50,
            batch_size=32, train_freq=1, gradient_steps=1, gamma=0.99,
            n_steps=5, exploration_fraction=0.35,
            exploration_initial_eps=1.0, exploration_final_eps=0.05,
        )
        previous = 0
        for checkpoint in range(args.checkpoint_interval, args.timesteps + 1, args.checkpoint_interval):
            model.learn(total_timesteps=checkpoint - previous, reset_num_timesteps=False, progress_bar=False)
            previous = checkpoint
            model_dir = run_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(model_dir / f"nstep5_checkpoint_{checkpoint}"))
            daily, summary = base.evaluate_model(model, env_args, checkpoint, run_dir)
            daily_frames.append(daily); summary_rows.append(summary)
    finally:
        base.make_env = original_make_env
        train_env.close()
    daily = pd.concat(daily_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    daily.to_csv(run_dir / "nstep5_eval_daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(run_dir / "checkpoint_summary.csv", index=False, encoding="utf-8-sig")
    base.plot_checkpoints(daily, summary, run_dir / "figures" / f"hla2010_nstep5_seed2_{args.timesteps}steps.png", 2010, 2, args.timesteps)
    chosen = summary.sort_values(["total_reward", "checkpoint_step"], ascending=[False, True]).iloc[0]
    lines = [
        "# 020_07 HLA2010 n-step DQN seed2 记录", "",
        "- 唯一改动：标准DQN n_steps从1改为5。", "- Seed: 2。",
        f"- Timesteps: {args.timesteps}；checkpoint interval: {args.checkpoint_interval}。",
        "- checkpoint选择：total_reward最大；并列取最早。", "", "## Checkpoint结果", "",
        md_table(summary[["checkpoint_step", "action_irrigation_total", "action_fertilizer_total", "final_grain_kg_ha", "final_biomass_kg_ha", "max_water_stress", "max_nitrogen_stress", "total_reward"]]),
        "", "## 固定规则选点", "",
        f"- Step: {int(chosen['checkpoint_step'])}；Yield: {chosen['final_grain_kg_ha']:.1f} kg/ha；I: {chosen['action_irrigation_total']:.1f} mm；N: {chosen['action_fertilizer_total']:.1f} kg/ha；Reward: {chosen['total_reward']:.1f}。",
        f"- Run dir: `{run_dir.relative_to(ROOT)}`。", "- 原生WP_ET、NLCM和严格基线判定由离线评估补充。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__": main()
