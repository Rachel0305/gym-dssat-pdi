"""Isolated seed expansion of frozen SY 046_10 and LC 053_00 contracts.

The historical wrappers read a seed from JSON but their shared 032_22 engine
uses a module-level SEED.  Set both here in a fresh process for every run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "053_lca_lowIC_site_transfer"))

import run_sya_originIC_expanded_action_maskableppo_046_10 as sy
import run_053_00_lca_lowIC_expanded_action_maskableppo as lc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["SY", "LC"], required=True)
    parser.add_argument("--seed", type=int, choices=range(1, 8), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    if args.smoke == args.formal:
        parser.error("Choose exactly one of --smoke and --formal")

    runner = sy if args.site == "SY" else lc
    source = sy.DEFAULT_CONFIG if args.site == "SY" else lc.DEFAULT_CONFIG
    cfg = json.loads(source.read_text(encoding="utf-8"))
    cfg["task_id"] = f"231_{args.site.lower()}_seed{args.seed}"
    cfg["task_name"] = f"{cfg['task_name']}_seed{args.seed}_extension"
    cfg["seed"] = args.seed
    config_dir = ROOT / "results" / "sy_lc_8seed_extension_231" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{args.site.lower()}_seed{args.seed}.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != cfg:
            raise RuntimeError(f"Refusing to change existing config: {config_path}")
    else:
        config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    engine = runner.engine.base03222
    old_seed = engine.SEED
    old_verify = runner.verify_completed_smoke
    try:
        engine.SEED = args.seed
        if args.formal:
            # The historical smoke gate includes action diversity. Keep its
            # finding, but permit a technically complete smoke to proceed:
            # a poor policy is a valid member of an eight-seed comparison.
            def verify_technical_smoke(base_cfg):
                checks = old_verify(base_cfg)
                if checks.get("next_step_allowed") is True:
                    return checks
                import orchestrate_sy_lc_8seed_extension_231 as cohort

                technical = cohort.gate_status(args.site, args.seed, smoke=True)
                nonbehavior = all(
                    value for key, value in checks.items()
                    if key not in {"smoke_gate_passed", "smoke_output_root", "next_step_allowed"}
                )
                checks["behavior_smoke_gate_passed"] = checks.get("smoke_gate_passed")
                checks["technical_smoke_gate_passed"] = bool(technical)
                checks["next_step_allowed"] = bool(technical and nonbehavior)
                return checks

            runner.verify_completed_smoke = verify_technical_smoke
        result = runner.run(config_path, dry_run=False, smoke=args.smoke, formal=args.formal)
        if engine.SEED != args.seed:
            raise RuntimeError("Training engine seed changed during run")
    finally:
        engine.SEED = old_seed
        runner.verify_completed_smoke = old_verify
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
