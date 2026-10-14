"""Read-only forecast-swap audit for 118 SY potential-shaping checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import forecast_engineered_observation_056_057 as forecast
from audit_111_112_sy_lc_rolling7d_counterfactual import action_map, collect_states, counterfactual_decisions
from run_118_sy_forecast_potential_shaping_5k10k import CONFIG, PROMPT, LONG_TASK_NAME, SMOKE_TASK_NAME, patch_potential_contract, patch_stage_schedule


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke2k", "long"], required=True)
    args = parser.parse_args()
    patch_potential_contract()
    patch_stage_schedule(args.stage == "long")
    cfg = forecast.read_json(CONFIG)
    base_name = LONG_TASK_NAME if args.stage == "long" else SMOKE_TASK_NAME
    run = ROOT / "benchmark_results" / f"118_00_{base_name}"
    if not run.exists():
        raise FileNotFoundError(run)
    checkpoints = [5000, 10000] if args.stage == "long" else [2000]
    out = run / "audits" / "118_00_counterfactual"
    out.mkdir(parents=True, exist_ok=True)
    old = {key: getattr(forecast.engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_names = dict(forecast.engine.base03222.SITE_NAMES)
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_make_env = forecast.engine.base03222.base.make_env
    summaries: dict[str, dict] = {}
    try:
        forecast.engine.TASK_ID = "118_00"
        forecast.engine.TASK_NAME = f"{base_name}_counterfactual"
        forecast.engine.BASE_OUT = out
        forecast.engine.BASE_DOC = PROMPT
        forecast.engine.PROMPT = PROMPT
        forecast.engine.LOWIC_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        forecast.engine.STATION = "SYA"
        forecast.engine.SITES = ["SYA"]
        forecast.engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        forecast.engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        forecast.engine.base03222.SITE_NAMES["SYA"] = "SY"
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        runtime_config = forecast.engine.load_config()
        selection = forecast.engine.base03222.build_selection(forecast.engine.base03222.load_split())
        env_config = forecast.direct_ppo.build_env_config(runtime_config, selection)
        make_env = forecast.make_forecast_env_factory(old_make_env, cfg)
        for checkpoint in checkpoints:
            model_path = next(run.glob(f"models/SYA/*seed0_ckpt{checkpoint}.zip"), None)
            if model_path is None:
                summaries[str(checkpoint)] = {"status": "missing_model"}
                continue
            model = MaskablePPO.load(str(model_path), device="cpu")
            states = collect_states(model, make_env, runtime_config, env_config, "SYA", 0)
            probe = make_env(runtime_config, env_config, "SYA", 2014, 0, f"118_grid_probe_{checkpoint}", evaluation=True)
            try:
                grid = action_map(probe)
            finally:
                probe.close()
            decisions = counterfactual_decisions(states, model, grid, 0)
            decisions.to_csv(out / f"checkpoint_{checkpoint}_decisions.csv", index=False, encoding="utf-8-sig")
            summaries[str(checkpoint)] = {
                "model": str(model_path.relative_to(ROOT)).replace("\\", "/"),
                "n_decision_states": int(len(decisions)),
                "swap_action_change_rate": float(decisions["swap_changed"].mean()) if len(decisions) else 0.0,
                "shuffled_action_change_rate": float(decisions["shuffled_changed"].mean()) if len(decisions) else 0.0,
                "swap_changes_after_dap1": int(decisions.loc[(decisions["dap"] > 1) & decisions["swap_changed"]].shape[0]),
                "shuffled_changes_after_dap1": int(decisions.loc[(decisions["dap"] > 1) & decisions["shuffled_changed"]].shape[0]),
                "all_variant_actions_legal": bool(decisions[["real_legal", "swap_legal", "shuffled_legal"]].all().all()) if len(decisions) else False,
            }
        (out / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
        forecast.engine.base03222.base.make_env = old_make_env
        forecast.engine.base03222.SITE_NAMES.clear()
        forecast.engine.base03222.SITE_NAMES.update(old_names)
        for key, value in old.items():
            setattr(forecast.engine, key, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
