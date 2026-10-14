"""040_52: full multi-year retrain of 040_40, with the DAP90 gate OFF.

This is the direct action item from 040_51: the frozen 040_40 checkpoint,
when replayed with the DAP<=90/195mm gate disabled, wanted to irrigate
earlier in all 10/10 validation years. That's a clean, no-retraining signal
that the gate is constraining the policy. This task turns that into an
actual usable result: retrain from scratch with the SAME multi-year
training data, reward, discrete action levels, and PPO hyperparameters as
040_40, changing ONLY the gate.

It does this by reusing 040_40's own training pipeline directly (it already
supports running a named variant via --suffix, landing in a separate
benchmark_results/docs path so it never overwrites 040_40's original
checkpoint100000 results). This avoids re-deriving env/training code and
avoids the single-year training noise that made 040_49/040_50 inconclusive
-- this trains on the full multi-year split, exactly like 040_40 did.

Usage
-----
    python run_sya_lowIC_ppo_dap90_gate_off_full_multiyear_040_52.py --dry-run
    python run_sya_lowIC_ppo_dap90_gate_off_full_multiyear_040_52.py
    python run_sya_lowIC_ppo_dap90_gate_off_full_multiyear_040_52.py --timesteps 100000

Output lands at:
    benchmark_results/040_40_sya_lowIC_ppo_yield_guardrail_v3_dap90_gate_off/
    docs/040_40_sya_lowIC_ppo_yield_guardrail_v3_dap90_gate_off_record.md

That record is produced by 040_40's own add_04040_summaries/write_clean_record,
so it is directly comparable in format to 040_40's original record and to
040_42/040_43/040_44-style checkpoint evaluations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_sya_lowIC_ppo_yield_guardrail_v3_040_40 as base04040
import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as base04036

SUFFIX = "dap90_gate_off"
GATE_DAP_END_OFF = 0  # dap<=0 never true for dap>=1 -> gate never fires


def apply_gate_off() -> dict:
    before = base04036.LATE_RESERVE_DAP_END
    base04036.LATE_RESERVE_DAP_END = GATE_DAP_END_OFF
    return {"before": before, "after": base04036.LATE_RESERVE_DAP_END}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timesteps", type=int, default=base04040.DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--checkpoint-steps", type=str, default=None)
    args = parser.parse_args()

    gate_state = apply_gate_off()
    print(f"[040_52] gate state: {gate_state}")

    checkpoint_steps = base04040.parse_checkpoint_steps(args.checkpoint_steps, args.timesteps)
    if args.dry_run:
        base04040.dry_run(args.timesteps, checkpoint_steps, SUFFIX)
    else:
        base04040.run_training(args.timesteps, checkpoint_steps, SUFFIX)


if __name__ == "__main__":
    main()
