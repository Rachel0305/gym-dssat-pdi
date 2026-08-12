"""Read-only 057_00 forecast-sensitivity audit; it never trains or steps counterfactual actions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

import ppo_safe_rendering
import forecast_engineered_observation_056_057 as forecast


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "057_00_lca_lowIC_forecast_engineered_maskableppo.json"
SMOKE = ROOT / "benchmark_results" / "057_00_lca_lowIC_forecast_engineered_maskableppo_smoke2k"
OUT = ROOT / "benchmark_results" / "057_00_lca_lowIC_forecast_engineered_maskableppo_smoke2k_counterfactual_audit"
YEARS = list(range(2014, 2024))
SEED = 0


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def action_map(env) -> dict[int, tuple[float, float]]:
    grid = list(getattr(env, "grid"))
    return {
        int(idx): (float(row.get("amir", 0.0)), float(row.get("anfer", 0.0)))
        for idx, row in enumerate(grid)
    }


def run_actual_states(model, make_env, runtime_config, env_config, cfg):
    """Collect real decision states; only the actual masked action is executed."""
    rows = []
    for year in YEARS:
        env = make_env(runtime_config, env_config, "LCA", year, SEED, f"057_00_cf_audit_{year}", evaluation=True)
        try:
            grid = action_map(env)
            obs, info = env.reset()
            done, step = False, 0
            last_grnwt = np.nan
            total_i, total_n = 0.0, 0.0
            while not done and step < int(runtime_config["runtime"]["max_steps"]):
                latest = forecast.direct_ppo.latest_observation_dict(env, obs, info)
                dap_raw = forecast.direct_ppo.scalar(latest.get("dap", step + 1), step + 1)
                dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step + 1
                mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                actual_action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                actual_action = int(np.asarray(actual_action).reshape(-1)[0])
                base = np.asarray(obs, dtype=np.float32).reshape(-1)[:-len(forecast.FORECAST_FEATURE_NAMES)].copy()
                real_fc = np.asarray(obs, dtype=np.float32).reshape(-1)[-len(forecast.FORECAST_FEATURE_NAMES):].copy()
                obs_next, _reward, terminated, truncated, info = env.step(actual_action)
                action_info = dict(env.last_action_info)
                safe_i = float(action_info.get("safe_action_amir", 0.0))
                safe_n = float(action_info.get("safe_action_anfer", 0.0))
                total_i += safe_i
                total_n += safe_n
                post = forecast.direct_ppo.latest_observation_dict(env, obs_next, info)
                last_grnwt = forecast.direct_ppo.scalar(post.get("grnwt", np.nan), np.nan)
                rows.append({
                    "year": year, "dap": dap, "step": step, "actual_action": actual_action,
                    "legal_action_count": int(mask.sum()), "mask": mask.astype(int).tolist(),
                    "base_obs": base.tolist(), "real_forecast": real_fc.tolist(),
                    "requested_irrigation_mm": grid[actual_action][0], "requested_nitrogen_kg_ha": grid[actual_action][1],
                    "safe_irrigation_mm": safe_i, "safe_nitrogen_kg_ha": safe_n,
                    "safety_triggered": bool(action_info.get("action_safety_triggered", False)),
                    "forecast_observation_date": str(action_info.get("forecast_observation_date", "")),
                    "forecast_mode": str(action_info.get("forecast_mode_056_057", "")),
                    "post_grnwt": last_grnwt,
                })
                obs, done = obs_next, bool(terminated or truncated)
                step += 1
            rows.append({"year": year, "dap": -1, "step": -1, "episode_final_grnwt": last_grnwt,
                         "episode_total_irrigation": total_i, "episode_total_n": total_n,
                         "episode_pfp_n": last_grnwt / total_n if total_n > 0 else np.nan})
        finally:
            env.close()
    return pd.DataFrame(rows)


def assess_predictions(model, states: pd.DataFrame, grid: dict[int, tuple[float, float]]) -> pd.DataFrame:
    work = states[states["dap"].ge(1)].copy().reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    sources = {}
    for dap, idx in work.groupby("dap").groups.items():
        ids = list(idx)
        years = work.loc[ids, "year"].astype(int).tolist()
        if len(ids) < 2:
            for i in ids:
                sources[i] = (i, i)
            continue
        swap_ids = ids[1:] + ids[:1]
        shuffled = list(rng.permutation(ids))
        if all(a == b for a, b in zip(ids, shuffled)):
            shuffled = ids[1:] + ids[:1]
        sources.update({i: (swap_ids[pos], shuffled[pos]) for pos, i in enumerate(ids)})
    rows = []
    for i, row in work.iterrows():
        swap_i, shuffled_i = sources[i]
        mask = np.asarray(row["mask"], dtype=bool)
        base = np.asarray(row["base_obs"], dtype=np.float32)
        variants = {"real": i, "swap": swap_i, "shuffled": shuffled_i}
        predictions = {}
        for label, source_i in variants.items():
            fc = np.asarray(work.loc[source_i, "real_forecast"], dtype=np.float32)
            obs = np.concatenate([base, fc]).astype(np.float32)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            predictions[label] = int(np.asarray(action).reshape(-1)[0])
        swap_source_year = int(work.loc[swap_i, "year"])
        shuffled_source_year = int(work.loc[shuffled_i, "year"])
        rows.append({
            "year": int(row.year), "dap": int(row.dap), "legal_action_count": int(mask.sum()),
            "real_action": predictions["real"], "swap_action": predictions["swap"], "shuffled_action": predictions["shuffled"],
            "swap_source_year": swap_source_year, "shuffled_source_year": shuffled_source_year,
            "swap_changed": bool(predictions["real"] != predictions["swap"]),
            "shuffled_changed": bool(predictions["real"] != predictions["shuffled"]),
            "real_i": grid[predictions["real"]][0], "real_n": grid[predictions["real"]][1],
            "swap_i": grid[predictions["swap"]][0], "swap_n": grid[predictions["swap"]][1],
            "shuffled_i": grid[predictions["shuffled"]][0], "shuffled_n": grid[predictions["shuffled"]][1],
            "real_legal": bool(mask[predictions["real"]]), "swap_legal": bool(mask[predictions["swap"]]),
            "shuffled_legal": bool(mask[predictions["shuffled"]]),
            "real_rain_next7_norm": float(np.asarray(row.real_forecast)[1]),
            "swap_rain_next7_norm": float(np.asarray(work.loc[swap_i, "real_forecast"])[1]),
            "shuffled_rain_next7_norm": float(np.asarray(work.loc[shuffled_i, "real_forecast"])[1]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite: {rel(OUT)}")
    model_path = SMOKE / "models" / "LCA" / "LCA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    cfg = forecast.validate_forecast_config(forecast.read_json(CFG_PATH), expected_task_id="057_00")
    preflight = forecast.preflight(cfg, ROOT / "prompts" / "057_00_lca_lowIC_forecast_engineered_maskableppo.md")
    if not preflight["next_step_allowed"]:
        raise RuntimeError(preflight["issues"])
    OUT.mkdir(parents=True)
    shutil.copy2(CFG_PATH, OUT / CFG_PATH.name)
    old = {key: getattr(forecast.engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_names, old_root = dict(forecast.engine.base03222.SITE_NAMES), ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_make_env = forecast.engine.base03222.base.make_env
    try:
        forecast.engine.TASK_ID = "057_00"
        forecast.engine.TASK_NAME = "lca_lowIC_forecast_engineered_maskableppo_smoke2k_counterfactual_audit"
        forecast.engine.BASE_OUT = OUT
        forecast.engine.LOWIC_INPUT_ROOT = forecast.INPUT_PROFILES["lowIC"]
        forecast.engine.STATION, forecast.engine.SITES = "LCA", ["LCA"]
        forecast.engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        forecast.engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        forecast.engine.base03222.SITE_NAMES["LCA"] = "LC"
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = forecast.INPUT_PROFILES["lowIC"]
        runtime_config = forecast.engine.load_config()
        selection = forecast.engine.base03222.build_selection(forecast.engine.base03222.load_split())
        env_config = forecast.direct_ppo.build_env_config(runtime_config, selection)
        make_env = forecast.make_forecast_env_factory(old_make_env, cfg)
        model = MaskablePPO.load(model_path)
        states = run_actual_states(model, make_env, runtime_config, env_config, cfg)
        states.to_json(OUT / "057_00_counterfactual_state_trace.json", orient="records", force_ascii=False)
        grid = {i: (float(a), float(n)) for i, (a, n) in enumerate(zip([], []))}
        probe = make_env(runtime_config, env_config, "LCA", YEARS[0], SEED, "057_00_cf_grid_probe", evaluation=True)
        try:
            grid = action_map(probe)
        finally:
            probe.close()
        decisions = assess_predictions(model, states, grid)
        decisions.to_csv(OUT / "057_00_counterfactual_decisions.csv", index=False, encoding="utf-8-sig")
        episodes = states[states["dap"].eq(-1)].copy()
        episodes.to_csv(OUT / "057_00_actual_rollout_endpoints.csv", index=False, encoding="utf-8-sig")
        summary = {
            "status": "completed_read_only_counterfactual_action_audit",
            "model": rel(model_path), "preflight": preflight, "years": YEARS, "seed": SEED,
            "counterfactual_actions_executed_in_dssat": False,
            "real_rollout_yield_mean": float(episodes["episode_final_grnwt"].mean()),
            "real_rollout_pfp_n_mean": float(episodes["episode_pfp_n"].mean()),
            "wp_et_available": False,
            "n_decision_states": int(len(decisions)),
            "swap_action_change_rate": float(decisions["swap_changed"].mean()),
            "shuffled_action_change_rate": float(decisions["shuffled_changed"].mean()),
            "swap_changes_after_dap1": int(decisions.loc[(decisions.dap > 1) & decisions.swap_changed].shape[0]),
            "shuffled_changes_after_dap1": int(decisions.loc[(decisions.dap > 1) & decisions.shuffled_changed].shape[0]),
            "all_variant_actions_legal": bool(decisions[["real_legal", "swap_legal", "shuffled_legal"]].all().all()),
            "next_step_100k_allowed": False,
            "gate_reason": "Counterfactual action test is diagnostic only; no 2K noninferiority in yield/PFP_N and WP_ET unavailable.",
        }
        (OUT / "057_00_counterfactual_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
        forecast.engine.base03222.base.make_env = old_make_env
        forecast.engine.base03222.SITE_NAMES.clear(); forecast.engine.base03222.SITE_NAMES.update(old_names)
        for key, value in old.items():
            setattr(forecast.engine, key, value)


if __name__ == "__main__":
    main()
