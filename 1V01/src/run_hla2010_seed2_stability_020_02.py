from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_hla_baseline_relative_dqn_checkpoint_015_12 as original


OUT_ROOT = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "hla2010_seed2_stability_020_02"
)
DOC = PROJECT_ROOT / "docs" / "2026-07-10_020_02_hla2010_seed2_stability_record.md"


def write_record(
    year: int,
    seed: int,
    timesteps: int,
    checkpoint_interval: int,
    null_baseline: float,
    run_dir: Path,
    summary: pd.DataFrame,
) -> None:
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
    selected = view.sort_values(
        ["total_reward", "checkpoint_step"], ascending=[False, True]
    ).iloc[0]
    lines = [
        "# 020_02 HLA2010 seed2 稳定性复核记录",
        "",
        "## 固定设置",
        "",
        f"- Seed: {seed}",
        f"- Timesteps: {timesteps}",
        f"- Checkpoint interval: {checkpoint_interval}",
        f"- Null baseline yield: {null_baseline:.1f} kg/ha",
        "- 算法、reward、动作、预算、IC 和输入链路均复用 015_12。",
        "- checkpoint 选择规则：total_reward 最大；并列时取最早 checkpoint。",
        "- 本记录不允许依据产量或图形人工改选 checkpoint。",
        "",
        "## 输出",
        "",
        f"- Run dir: `{run_dir.relative_to(PROJECT_ROOT)}`",
        f"- Summary: `{(run_dir / 'checkpoint_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- Daily: `{(run_dir / 'dqn_eval_daily.csv').relative_to(PROJECT_ROOT)}`",
        "",
        "## Checkpoint 结果",
        "",
        original.markdown_table(view),
        "",
        "## 预先固定规则选中的 checkpoint",
        "",
        f"- Step: {int(selected['checkpoint_step'])}",
        f"- Yield: {selected['final_grain_kg_ha']:.1f} kg/ha",
        f"- Irrigation: {selected['action_irrigation_total']:.1f} mm",
        f"- Nitrogen: {selected['action_fertilizer_total']:.1f} kg/ha",
        f"- Reward: {selected['total_reward']:.1f}",
        "",
        "DSSAT 原生 WP_ET、NLCM 及相对基线的严格成功判定由后续离线评估脚本补充。",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
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

    original.OUT_ROOT = OUT_ROOT
    original.write_record = write_record
    sys.argv = [
        sys.argv[0],
        "--year", "2010",
        "--timesteps", str(args.timesteps),
        "--seed", "2",
        "--checkpoint-interval", str(args.checkpoint_interval),
    ]
    original.main()


if __name__ == "__main__":
    main()
