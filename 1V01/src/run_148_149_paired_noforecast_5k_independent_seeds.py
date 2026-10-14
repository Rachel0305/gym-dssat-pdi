"""Run paired no-forecast controls for E2 seeds 1 and 2, serially."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sya_originIC_expanded_action_maskableppo_046_10 as nof

CONFIGS = {
    1: ROOT / "configs/148N1_sya_originIC_paired_noforecast_5k_seed1.json",
    2: ROOT / "configs/149N2_sya_originIC_paired_noforecast_5k_seed2.json",
}
IDS = {1: "148N1", 2: "149N2"}
PROMPT = ROOT / "prompts/2026-08-16_sya_paired_noforecast_5k_independent_seeds.md"
REFERENCE = ROOT / "configs/140N_sya_originIC_paired_noforecast.json"


def gate(seed: int) -> dict:
    cfg = json.loads(CONFIGS[seed].read_text(encoding="utf-8"))
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    checks = {
        "prompt_exists": PROMPT.exists(),
        "requested_seed_matches": int(cfg["seed"]) == seed,
        "only_seed_and_task_differ": all(cfg[k] == ref[k] for k in ["station_code", "site", "input_profile", "actions", "observation_contract"]) and all(cfg["scope"][k] == ref["scope"][k] for k in ["train_years", "validation_years", "reward_and_safety", "external_n", "native_dssat_automatic_irrigation"]),
        "five_k_training": cfg["training"] == {"total_timesteps": 5000, "checkpoint_steps": [2000, 5000]},
        "longer_runs_forbidden": True,
    }
    checks["next_step_allowed"] = all(checks.values())
    return checks


def configure_phase(smoke: bool):
    original = nof.prepare_config

    def phase_config(cfg, _smoke):
        planned = copy.deepcopy(cfg)
        if smoke:
            planned["task_name"] = f"{cfg['task_name']}_smoke2k"
            planned["training"] = {"total_timesteps": 2000, "checkpoint_steps": [1000, 2000]}
        return planned

    nof.prepare_config = phase_config
    return original


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=[1, 2], required=True)
    phase = ap.add_mutually_exclusive_group(required=True)
    phase.add_argument("--dry-run", action="store_true")
    phase.add_argument("--smoke2k", action="store_true")
    phase.add_argument("--train5k", action="store_true")
    args = ap.parse_args()
    auth = gate(args.seed)
    if not auth["next_step_allowed"]:
        raise RuntimeError(json.dumps(auth, ensure_ascii=False))
    smoke = bool(args.smoke2k)
    formal = bool(args.train5k)
    old_prepare = configure_phase(smoke) if smoke else nof.prepare_config
    old_prompt = nof.PROMPT
    old_seed = nof.engine.base03222.SEED
    nof.PROMPT = PROMPT
    nof.engine.base03222.SEED = args.seed
    try:
        result = nof.run(CONFIGS[args.seed], args.dry_run, smoke, formal)
    finally:
        nof.engine.base03222.SEED = old_seed
        nof.PROMPT = old_prompt
        nof.prepare_config = old_prepare
    result["independent_seed_gate"] = auth
    result["actual_engine_seed"] = args.seed
    out_rel = result.get("output_root")
    if formal and out_rel:
        out = ROOT / out_rel
        (out / f"{IDS[args.seed]}_5k_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
