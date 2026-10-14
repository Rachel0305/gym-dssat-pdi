"""Read-only forecast-swap audit for the 111/112 rolling 7-day smoke models."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import forecast_engineered_observation_056_057 as forecast
from run_111_112_sy_lc_rolling7d_perfect_forecast_smoke2k import CONFIGS, EXPECTED_IDS, patch_contract

YEARS = list(range(2014, 2024))
SITE_INFO = {
    "sy": {"station": "SYA", "site": "SY", "input_profile": "originIC", "out": "benchmark_results/111_00_sya_originIC_rolling7d_perfect_forecast_smoke2k_smoke2k", "task": "111_00"},
    "lc": {"station": "LCA", "site": "LC", "input_profile": "lowIC", "out": "benchmark_results/112_00_lca_lowIC_rolling7d_perfect_forecast_smoke2k_smoke2k", "task": "112_00"},
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def action_map(env) -> dict[int, tuple[float, float]]:
    return {int(idx): (float(row.get("amir", 0.0)), float(row.get("anfer", 0.0))) for idx, row in enumerate(getattr(env, "grid"))}


def collect_states(model, make_env, runtime_config, env_config, station: str, seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    for year in YEARS:
        env = make_env(runtime_config, env_config, station, year, seed, f"rolling7d_cf_{station}_{year}", evaluation=True)
        try:
            grid = action_map(env)
            obs, info = env.reset()
            done, step = False, 0
            while not done and step < int(runtime_config["runtime"]["max_steps"]):
                latest = forecast.direct_ppo.latest_observation_dict(env, obs, info)
                dap_raw = forecast.direct_ppo.scalar(latest.get("dap", step + 1), step + 1)
                dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step + 1
                mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                action = int(np.asarray(action).reshape(-1)[0])
                vector = np.asarray(obs, dtype=np.float32).reshape(-1)
                fc_len = len(forecast.FORECAST_FEATURE_NAMES)
                base_obs = vector[:-fc_len].copy()
                real_fc = vector[-fc_len:].copy()
                obs_next, _reward, terminated, truncated, info = env.step(action)
                action_info = dict(getattr(env, "last_action_info", {}) or {})
                rows.append({"year": year, "dap": dap, "step": step, "actual_action": action, "mask": mask.astype(int).tolist(), "base_obs": base_obs.tolist(), "real_forecast": real_fc.tolist(), "forecast_observation_date": str(action_info.get("forecast_observation_date", "")), "real_i": grid[action][0], "real_n": grid[action][1]})
                obs, done = obs_next, bool(terminated or truncated)
                step += 1
        finally:
            env.close()
    return pd.DataFrame(rows)


def counterfactual_decisions(states: pd.DataFrame, model, grid: dict[int, tuple[float, float]], seed: int) -> pd.DataFrame:
    work = states[states["dap"].ge(1)].copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    sources: dict[int, tuple[int, int]] = {}
    for _dap, idxs in work.groupby("dap").groups.items():
        ids = list(idxs)
        if len(ids) < 2:
            for i in ids:
                sources[i] = (i, i)
            continue
        rotated = ids[1:] + ids[:1]
        shuffled = list(rng.permutation(ids))
        if all(a == b for a, b in zip(ids, shuffled)):
            shuffled = rotated
        sources.update({i: (rotated[pos], shuffled[pos]) for pos, i in enumerate(ids)})
    rows: list[dict] = []
    fc_len = len(forecast.FORECAST_FEATURE_NAMES)
    for idx, row in work.iterrows():
        swap_idx, shuffled_idx = sources[idx]
        mask = np.asarray(row["mask"], dtype=bool)
        base = np.asarray(row["base_obs"], dtype=np.float32)
        preds: dict[str, int] = {}
        for name, source in [("real", idx), ("swap", swap_idx), ("shuffled", shuffled_idx)]:
            forecast_vector = np.asarray(work.loc[source, "real_forecast"], dtype=np.float32)
            if len(forecast_vector) != fc_len:
                raise RuntimeError("forecast vector length mismatch")
            action, _ = model.predict(np.concatenate([base, forecast_vector]), action_masks=mask, deterministic=True)
            preds[name] = int(np.asarray(action).reshape(-1)[0])
        rows.append({"year": int(row["year"]), "dap": int(row["dap"]), "real_action": preds["real"], "swap_action": preds["swap"], "shuffled_action": preds["shuffled"], "swap_changed": preds["real"] != preds["swap"], "shuffled_changed": preds["real"] != preds["shuffled"], "real_legal": bool(mask[preds["real"]]), "swap_legal": bool(mask[preds["swap"]]), "shuffled_legal": bool(mask[preds["shuffled"]]), "real_i": grid[preds["real"]][0], "real_n": grid[preds["real"]][1], "swap_i": grid[preds["swap"]][0], "swap_n": grid[preds["swap"]][1], "shuffled_i": grid[preds["shuffled"]][0], "shuffled_n": grid[preds["shuffled"]][1]})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["sy", "lc"], required=True)
    args = parser.parse_args()
    patch_contract()
    info = SITE_INFO[args.site]
    cfg = forecast.read_json(CONFIGS[args.site])
    preflight = forecast.preflight(cfg, ROOT / cfg["prompt"])
    if not preflight["next_step_allowed"]:
        raise RuntimeError(preflight["issues"])
    smoke = ROOT / info["out"]
    model_path = next(smoke.glob(f"models/{info['station']}/*seed0_ckpt2000.zip"), None)
    if model_path is None:
        raise FileNotFoundError(f"missing smoke model under {smoke}")
    out = ROOT / f"benchmark_results/{info['task']}_rolling7d_counterfactual_audit"
    if out.exists():
        raise FileExistsError(f"refuse to overwrite {rel(out)}")
    out.mkdir(parents=True)
    shutil.copy2(CONFIGS[args.site], out / CONFIGS[args.site].name)
    old = {key: getattr(forecast.engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_names = dict(forecast.engine.base03222.SITE_NAMES)
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_make_env = forecast.engine.base03222.base.make_env
    try:
        forecast.engine.TASK_ID = info["task"]
        forecast.engine.TASK_NAME = f"{info['site']}_rolling7d_counterfactual_audit"
        forecast.engine.BASE_OUT = out
        forecast.engine.PROMPT = ROOT / cfg["prompt"]
        forecast.engine.LOWIC_INPUT_ROOT = forecast.INPUT_PROFILES[info["input_profile"]]
        forecast.engine.STATION = info["station"]
        forecast.engine.SITES = [info["station"]]
        forecast.engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        forecast.engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        forecast.engine.base03222.SITE_NAMES[info["station"]] = info["site"]
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = forecast.INPUT_PROFILES[info["input_profile"]]
        runtime_config = forecast.engine.load_config()
        selection = forecast.engine.base03222.build_selection(forecast.engine.base03222.load_split())
        env_config = forecast.direct_ppo.build_env_config(runtime_config, selection)
        make_env = forecast.make_forecast_env_factory(old_make_env, cfg)
        model = MaskablePPO.load(str(model_path), device="cpu")
        states = collect_states(model, make_env, runtime_config, env_config, info["station"], int(cfg["seed"]))
        states.to_json(out / "state_trace.json", orient="records", force_ascii=False)
        probe = make_env(runtime_config, env_config, info["station"], YEARS[0], int(cfg["seed"]), "rolling7d_grid_probe", evaluation=True)
        try:
            grid = action_map(probe)
        finally:
            probe.close()
        decisions = counterfactual_decisions(states, model, grid, int(cfg["seed"]))
        decisions.to_csv(out / "counterfactual_decisions.csv", index=False, encoding="utf-8-sig")
        summary = {"status": "completed_read_only_rolling7d_counterfactual", "model": rel(model_path), "preflight": preflight, "n_decision_states": int(len(decisions)), "swap_action_change_rate": float(decisions["swap_changed"].mean()) if len(decisions) else 0.0, "shuffled_action_change_rate": float(decisions["shuffled_changed"].mean()) if len(decisions) else 0.0, "swap_changes_after_dap1": int(decisions.loc[(decisions["dap"] > 1) & decisions["swap_changed"]].shape[0]), "shuffled_changes_after_dap1": int(decisions.loc[(decisions["dap"] > 1) & decisions["shuffled_changed"]].shape[0]), "all_variant_actions_legal": bool(decisions[["real_legal", "swap_legal", "shuffled_legal"]].all().all()) if len(decisions) else False, "next_step_long_training_allowed": False}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
        forecast.engine.base03222.base.make_env = old_make_env
        forecast.engine.base03222.SITE_NAMES.clear(); forecast.engine.base03222.SITE_NAMES.update(old_names)
        for key, value in old.items(): setattr(forecast.engine, key, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
