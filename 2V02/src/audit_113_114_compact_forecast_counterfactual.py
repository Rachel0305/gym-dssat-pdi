"""Read-only same-state forecast swap audit for compact 113/114 models."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import forecast_engineered_observation_056_057 as forecast
from audit_111_112_sy_lc_rolling7d_counterfactual import action_map, collect_states, counterfactual_decisions
from run_113_114_sy_lc_compact_forecast_smoke2k import CONFIGS, EXPECTED_IDS, patch_contract

INFO = {"sy": {"station": "SYA", "site": "SY", "profile": "originIC", "task": "113_00", "smoke": "benchmark_results/113_00_sya_originIC_compact_rolling7d_forecast_smoke2k"}, "lc": {"station": "LCA", "site": "LC", "profile": "lowIC", "task": "114_00", "smoke": "benchmark_results/114_00_lca_lowIC_compact_rolling7d_forecast_smoke2k"}}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["sy", "lc"], required=True)
    args = parser.parse_args()
    patch_contract()
    info = INFO[args.site]
    cfg = forecast.read_json(CONFIGS[args.site])
    preflight = forecast.preflight(cfg, ROOT / cfg["prompt"])
    if not preflight["next_step_allowed"]:
        raise RuntimeError(preflight["issues"])
    smoke = ROOT / info["smoke"]
    model_path = next(smoke.glob(f"models/{info['station']}/*seed0_ckpt2000.zip"), None)
    if model_path is None:
        raise FileNotFoundError(f"missing smoke model under {smoke}")
    out = ROOT / f"benchmark_results/{info['task']}_compact_forecast_counterfactual_audit"
    if out.exists():
        raise FileExistsError(f"refuse to overwrite {rel(out)}")
    out.mkdir(parents=True)
    old = {key: getattr(forecast.engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_names = dict(forecast.engine.base03222.SITE_NAMES)
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_make_env = forecast.engine.base03222.base.make_env
    try:
        forecast.engine.TASK_ID = info["task"]; forecast.engine.TASK_NAME = f"{info['site']}_compact_forecast_counterfactual_audit"; forecast.engine.BASE_OUT = out; forecast.engine.PROMPT = ROOT / cfg["prompt"]; forecast.engine.LOWIC_INPUT_ROOT = forecast.INPUT_PROFILES[info["profile"]]; forecast.engine.STATION = info["station"]; forecast.engine.SITES = [info["station"]]; forecast.engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"])); forecast.engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"])); forecast.engine.base03222.SITE_NAMES[info["station"]] = info["site"]; ppo_safe_rendering.MULTISITE_INPUT_ROOT = forecast.INPUT_PROFILES[info["profile"]]
        runtime_config = forecast.engine.load_config(); selection = forecast.engine.base03222.build_selection(forecast.engine.base03222.load_split()); env_config = forecast.direct_ppo.build_env_config(runtime_config, selection); make_env = forecast.make_forecast_env_factory(old_make_env, cfg); model = MaskablePPO.load(str(model_path), device="cpu")
        states = collect_states(model, make_env, runtime_config, env_config, info["station"], int(cfg["seed"])); states.to_json(out / "state_trace.json", orient="records", force_ascii=False)
        probe = make_env(runtime_config, env_config, info["station"], 2014, int(cfg["seed"]), "compact_grid_probe", evaluation=True)
        try: grid = action_map(probe)
        finally: probe.close()
        decisions = counterfactual_decisions(states, model, grid, int(cfg["seed"])); decisions.to_csv(out / "counterfactual_decisions.csv", index=False, encoding="utf-8-sig")
        summary = {"status": "completed_read_only_compact_forecast_counterfactual", "model": rel(model_path), "preflight": preflight, "n_decision_states": int(len(decisions)), "swap_action_change_rate": float(decisions["swap_changed"].mean()) if len(decisions) else 0.0, "shuffled_action_change_rate": float(decisions["shuffled_changed"].mean()) if len(decisions) else 0.0, "swap_changes_after_dap1": int(decisions.loc[(decisions["dap"] > 1) & decisions["swap_changed"]].shape[0]), "all_variant_actions_legal": bool(decisions[["real_legal", "swap_legal", "shuffled_legal"]].all().all()) if len(decisions) else False, "next_step_long_training_allowed": False}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root; forecast.engine.base03222.base.make_env = old_make_env; forecast.engine.base03222.SITE_NAMES.clear(); forecast.engine.base03222.SITE_NAMES.update(old_names)
        for key, value in old.items(): setattr(forecast.engine, key, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
