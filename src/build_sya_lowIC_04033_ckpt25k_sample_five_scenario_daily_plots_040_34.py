from __future__ import annotations

from pathlib import Path

import build_sya_lowIC_04028_ckpt75k_sample_five_scenario_daily_plots_040_30 as plot04030


ROOT = Path(__file__).resolve().parents[1]
TASK = "040_34_sya_lowIC_04033_ckpt25k_sample_five_scenario_daily_plots"
OUT = ROOT / "benchmark_results" / TASK
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
SNAP_DIR = OUT / "snapshots"
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"


def main() -> int:
    plot04030.TASK = TASK
    plot04030.OUT = OUT
    plot04030.FIG_DIR = FIG_DIR
    plot04030.TAB_DIR = TAB_DIR
    plot04030.SNAP_DIR = SNAP_DIR
    plot04030.DOC = DOC
    plot04030.PROMPT = PROMPT
    plot04030.CHECKPOINT = 25_000
    plot04030.PPO04028_EVAL = (
        ROOT
        / "benchmark_results"
        / "040_33_sya_lowIC_ppo_i240_swfac_guardrail_coarse_actions"
        / "evaluation"
        / "040_33_checkpoint_validation_summary.csv"
    )
    plot04030.LABELS["rl_candidate"] = "040_33 MaskablePPO ckpt25k"
    plot04030.PLOT_TAG = "040_33 MaskablePPO ckpt25k"
    plot04030.FILE_TAG = "040_34_lowIC_04033_ckpt25k"
    plot04030.TABLE_TAG = "040_34_sya_2014_2017_2022_lowIC_04033_ckpt25k"
    return plot04030.main()


if __name__ == "__main__":
    raise SystemExit(main())
