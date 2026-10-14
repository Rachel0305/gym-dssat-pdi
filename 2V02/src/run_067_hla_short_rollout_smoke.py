"""Run the pre-registered HL short-rollout 2K mechanism smoke in isolation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_064():
    path = ROOT / "src/064_hla_lowIC_tuning/run_064_hla_ppo_kwargs_smoke.py"
    spec = importlib.util.spec_from_file_location("runner064_for_067", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    cfg_path = ROOT / "configs/067_hla_short_rollout_smoke.json"
    plan = json.loads(cfg_path.read_text(encoding="utf-8"))
    # The 064 runner's preflight expects the original A/B/C plan shape; keep
    # the new intervention isolated while reusing its audited execution path.
    plan["fixed_ppo"] = {"learning_rate": 3e-4, "gamma": 1.0, "gae_lambda": 1.0, "n_steps": 144, "batch_size": 144, "n_epochs": 5, "ent_coef": 0.01, "clip_range": 0.2, "net_arch": [64, 64]}
    plan["actions"] = {"irrigation_levels_mm": [0.0, 15.0, 30.0, 45.0], "nitrogen_levels_kg_ha": [0.0, 40.0, 80.0, 120.0]}
    plan["observation_contract"] = {"base": "046_02_raw_observation", "normalization_enabled": False, "weather_forecast_enabled": False}
    plan["scope"] = {"train_years": [2004,2005,2006,2007,2008,2009,2010,2011,2012,2013], "validation_years": [2014,2015,2016,2017,2018,2019,2020,2021,2022,2023]}
    candidate = {"id": "D_short_rollout", "override": plan["ppo_overrides"]}
    runner = load_064()
    print(json.dumps(runner.run_candidate(plan, candidate, dry_run=True), indent=2, ensure_ascii=False))
    result = runner.run_candidate(plan, candidate, dry_run=False, attempt="short_rollout")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
