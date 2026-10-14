"""Read-only forecast-swap audit for SY 117 checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import forecast_engineered_observation_056_057 as forecast
from audit_111_112_sy_lc_rolling7d_counterfactual import action_map, collect_states, counterfactual_decisions
from run_115_116_sy_lc_delayed_reward_smoke2k import patch_contract

CFG = ROOT / "configs/117_00_sya_originIC_forecast_delayed_reward_5k10k.json"
RUN = ROOT / "benchmark_results/117_00_sya_originIC_forecast_delayed_reward_5k10k"


def main() -> int:
    patch_contract()
    cfg = forecast.read_json(CFG)
    summary_path = RUN / "evaluation/117_00_checkpoint_validation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    old = {key: getattr(forecast.engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_names = dict(forecast.engine.base03222.SITE_NAMES); old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT; old_make_env = forecast.engine.base03222.base.make_env
    out = RUN / "audits/117_00_counterfactual"
    out.mkdir(parents=True, exist_ok=True)
    all_summary: dict[str, dict] = {}
    try:
        forecast.engine.TASK_ID = "117_00"; forecast.engine.TASK_NAME = "sya_forecast_delayed_reward_counterfactual"; forecast.engine.BASE_OUT = out; forecast.engine.PROMPT = ROOT / cfg["prompt"]; forecast.engine.LOWIC_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]; forecast.engine.STATION = "SYA"; forecast.engine.SITES = ["SYA"]; forecast.engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"])); forecast.engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"])); forecast.engine.base03222.SITE_NAMES["SYA"] = "SY"; ppo_safe_rendering.MULTISITE_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        runtime_config = forecast.engine.load_config(); selection = forecast.engine.base03222.build_selection(forecast.engine.base03222.load_split()); env_config = forecast.direct_ppo.build_env_config(runtime_config, selection); make_env = forecast.make_forecast_env_factory(old_make_env, cfg)
        for checkpoint in [5000, 10000]:
            model_path = next(RUN.glob(f"models/SYA/*seed0_ckpt{checkpoint}.zip"), None)
            if model_path is None:
                all_summary[str(checkpoint)] = {"status": "missing_model"}; continue
            model = MaskablePPO.load(str(model_path), device="cpu")
            states = collect_states(model, make_env, runtime_config, env_config, "SYA", 0)
            probe = make_env(runtime_config, env_config, "SYA", 2014, 0, f"117_grid_probe_{checkpoint}", evaluation=True)
            try: grid = action_map(probe)
            finally: probe.close()
            decisions = counterfactual_decisions(states, model, grid, 0)
            decisions.to_csv(out / f"checkpoint_{checkpoint}_decisions.csv", index=False, encoding="utf-8-sig")
            all_summary[str(checkpoint)] = {"model": str(model_path.relative_to(ROOT)).replace("\\", "/"), "n_decision_states": int(len(decisions)), "swap_action_change_rate": float(decisions["swap_changed"].mean()), "shuffled_action_change_rate": float(decisions["shuffled_changed"].mean()), "swap_changes_after_dap1": int(decisions.loc[(decisions["dap"] > 1) & decisions["swap_changed"]].shape[0]), "all_variant_actions_legal": bool(decisions[["real_legal", "swap_legal", "shuffled_legal"]].all().all())}
        (out / "summary.json").write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(all_summary, ensure_ascii=False, indent=2))
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root; forecast.engine.base03222.base.make_env = old_make_env; forecast.engine.base03222.SITE_NAMES.clear(); forecast.engine.base03222.SITE_NAMES.update(old_names)
        for key, value in old.items(): setattr(forecast.engine, key, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
