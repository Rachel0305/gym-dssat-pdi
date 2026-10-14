"""Isolated 072 wrapper around the proven 068 smoke engine.

Only the registered 072 plan is loaded; the 068 implementation is not edited.
This wrapper exposes dry-run/smoke only and refuses long phases.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_068_hl_yc_seed1_stability_validation as base

PLAN = ROOT / "configs" / "072_hl_yc_seed1_smoke.json"
base.PLAN_PATH = PLAN


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["HL", "YC"], required=True)
    parser.add_argument("--phase", choices=["dry-run", "smoke", "validate25k"], required=True)
    args = parser.parse_args()
    if args.phase == "validate25k":
        raise SystemExit("072 is smoke-only; 25K/100K are forbidden")
    plan = base.load_plan()
    common, spec = base.experiment(plan, args.site)
    cfg = base.cfg_for(common, spec, "smoke")
    if args.phase == "dry-run":
        print(json.dumps(base.preflight(cfg), ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(base.phase_run(cfg, "smoke"), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
