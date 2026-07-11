from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from custom_double_dqn import CustomDoubleDQN
import run_hla_baseline_relative_dqn_checkpoint_015_12 as hla
import run_hla_unified_dqn_long_train_015_09 as base
from run_hla_official_reward_restart_smoke import install_official_reward_module, prepare_case_at


OUT_ROOT = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_double_dqn_seed1_020_04"
DOC = ROOT / "docs" / "2026-07-10_020_04_hla2010_double_dqn_seed1_record.md"


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in df.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.3f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_record(run_dir: Path, timesteps: int, checkpoint_interval: int, null_baseline: float, summary: pd.DataFrame) -> None:
    selected = summary.sort_values(["total_reward", "checkpoint_step"], ascending=[False, True]).iloc[0]
    view = summary[
        [
            "checkpoint_step", "action_irrigation_total", "action_fertilizer_total",
            "final_grain_kg_ha", "final_biomass_kg_ha", "max_water_stress",
            "max_nitrogen_stress", "total_reward",
        ]
    ]
    lines = [
        "# 020_04 HLA2010 Double DQN seed1 严格单变量对照记录",
        "",
        "## 设置",
        "",
        "- 唯一改动：标准DQN TD target改为Double DQN TD target。",
        "- Seed: 1。",
        f"- Timesteps: {timesteps}；checkpoint interval: {checkpoint_interval}。",
        f"- HLA2010 null baseline: {null_baseline:.1f} kg/ha。",
        "- reward、输入、IC、9动作、预算和所有其他超参数与015_12 seed1相同。",
        "- 选择规则：total_reward最大；并列时取最早checkpoint。",
        "",
        "## Checkpoint结果",
        "",
        markdown_table(view),
        "",
        "## 固定规则选点",
        "",
        f"- Step: {int(selected['checkpoint_step'])}。",
        f"- Yield: {selected['final_grain_kg_ha']:.1f} kg/ha。",
        f"- Irrigation: {selected['action_irrigation_total']:.1f} mm。",
        f"- Nitrogen: {selected['action_fertilizer_total']:.1f} kg/ha。",
        f"- Reward: {selected['total_reward']:.1f}。",
        "",
        f"- Run dir: `{run_dir.relative_to(ROOT)}`。",
        "- 原生WP_ET/NLCM和严格基线判定由离线评估补充。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--checkpoint-interval", type=int, required=True)
    args = parser.parse_args()
    if args.timesteps <= 0 or args.checkpoint_interval <= 0:
        raise ValueError("timesteps and checkpoint interval must be positive")
    if args.timesteps % args.checkpoint_interval != 0:
        raise ValueError("timesteps must be divisible by checkpoint interval")

    base.configure_shared_settings()
    install_official_reward_module()
    null_baseline = hla.get_null_baseline_yield(2010)
    run_dir = OUT_ROOT / f"double_dqn_seed1_{args.timesteps}steps"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    prepare_case_at(2010, run_dir)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))

    original_base_make_env = base.make_env
    base.make_env = lambda current_env_args: hla.make_env(current_env_args, null_baseline)
    train_env = hla.make_env(env_args, null_baseline)
    all_daily = []
    all_summary = []
    try:
        model = CustomDoubleDQN(
            "MlpPolicy",
            train_env,
            verbose=0,
            seed=1,
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
        previous = 0
        for checkpoint in range(args.checkpoint_interval, args.timesteps + 1, args.checkpoint_interval):
            model.learn(
                total_timesteps=checkpoint - previous,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            previous = checkpoint
            model_dir = run_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(model_dir / f"double_dqn_checkpoint_{checkpoint}"))
            daily, summary = base.evaluate_model(model, env_args, checkpoint, run_dir)
            all_daily.append(daily)
            all_summary.append(summary)
    finally:
        base.make_env = original_base_make_env
        train_env.close()

    daily_df = pd.concat(all_daily, ignore_index=True)
    summary_df = pd.DataFrame(all_summary)
    daily_df.to_csv(run_dir / "double_dqn_eval_daily.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(run_dir / "checkpoint_summary.csv", index=False, encoding="utf-8-sig")
    figure = run_dir / "figures" / f"hla2010_double_dqn_seed1_{args.timesteps}steps.png"
    base.plot_checkpoints(daily_df, summary_df, figure, 2010, 1, args.timesteps)
    write_record(run_dir, args.timesteps, args.checkpoint_interval, null_baseline, summary_df)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
