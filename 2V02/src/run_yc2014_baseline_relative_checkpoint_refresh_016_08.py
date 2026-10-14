from __future__ import annotations

import argparse
from pathlib import Path

import run_yc2014_baseline_relative_dqn_015_10 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_baseline_relative_checkpoint_refresh_016_08"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-05_016_08_yc2014_baseline_relative_checkpoint_refresh_record.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()

    base.OUT_DIR = OUT_DIR
    base.DOC_PATH = DOC_PATH
    base.configure_yc_module(args.seed, args.timesteps)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        _daily, summary_df = base.run_checkpoint_train(args.timesteps, args.seed, args.checkpoint_interval)
        fig_path = (
            OUT_DIR
            / f"seed{args.seed}"
            / "dqn_baseline_relative_checkpoint"
            / "figures"
            / "yc2014_baseline_relative_checkpoint_summary.png"
        )
        base.plot_checkpoint_summary(summary_df, fig_path)
        best = summary_df.sort_values(["total_reward", "final_grain_kg_ha"], ascending=False).iloc[0].to_dict()
        base.write_record(best)
        print(summary_df.to_string(index=False))
    else:
        _daily, summary = base.train_and_eval(args.timesteps, args.seed)
        base.write_record(summary)
        print(summary)


if __name__ == "__main__":
    main()
