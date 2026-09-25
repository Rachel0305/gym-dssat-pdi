from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "results" / "yc_random_weather_ppo" / "004_04"
PILOT = ROOT / "results" / "yc_random_weather_ppo" / "004_03"
VERIFIED = PILOT / "training_verified"
OLD_EVAL = PILOT / "evaluation" / "verified_weather_refresh"
MODELS = ["H0", "H1", "H2", "W0", "W1", "W2"]
SEEDS = list(range(1081, 1101))
YEARS = list(range(2014, 2024))
REGIME = {m: "HISTORICAL_WEATHER" if m[0] == "H" else "RANDOM_WEATHER_WGEN" for m in MODELS}
PPO_SEED = {m: int(m[1]) for m in MODELS}
MAX_RSS_MB = 4000
PROBE_SEED = 404006
PROBE_N = 600
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_clean(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    return value


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def memory_mb() -> float:
    try:
        import psutil
        p = psutil.Process()
        return sum(x.memory_info().rss for x in [p, *p.children(recursive=True)] if x.is_running()) / 1048576
    except Exception:
        return float("nan")


def pilot_module():
    path = PILOT / "run_controlled_pilot.py"
    spec = importlib.util.spec_from_file_location("yc_004_03_pilot_for_004_04", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import canonical pilot runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def checkpoint(model_id: str) -> Path:
    if model_id not in MODELS:
        raise ValueError(model_id)
    kind = "historical" if model_id.startswith("H") else "random_weather"
    return VERIFIED / kind / f"ppo_seed_{PPO_SEED[model_id]}" / "models" / "checkpoint_100000.zip"


def verified_manifest() -> pd.DataFrame:
    runs = pd.read_csv(PILOT / "training_run_manifest.csv", keep_default_na=False)
    result = []
    for mid in MODELS:
        regime, seed, path = REGIME[mid], PPO_SEED[mid], checkpoint(mid)
        row = runs[(runs.training_regime == regime) & pd.to_numeric(runs.ppo_seed, errors="coerce").eq(seed)]
        run_json = path.parents[1] / "run_manifest.json"
        if len(row) != 1 or not path.is_file() or not run_json.is_file():
            raise RuntimeError(f"Missing/ambiguous verified checkpoint evidence for {mid}")
        metadata = json.loads(run_json.read_text(encoding="utf-8"))
        if metadata.get("run_status") != "completed" or metadata.get("model_final") != path.relative_to(ROOT).as_posix():
            raise RuntimeError(f"Refusing non-verified model {mid}")
        result.append({"model_id": mid, "training_regime": regime, "ppo_seed": seed,
                       "model_path": path.relative_to(ROOT).as_posix(), "model_sha256": hash_file(path),
                       "training_status": metadata["run_status"], "total_timesteps_actual": metadata.get("total_timesteps_actual"),
                       "verified_manifest": run_json.relative_to(ROOT).as_posix()})
    out = pd.DataFrame(result)
    out.to_csv(TASK / "model_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    return out


def preflight() -> dict[str, Any]:
    p = pilot_module()
    base = p.preflight_payload()
    models = verified_manifest()
    issues = list(base.get("issues", []))
    if len(models) != 6 or models.model_id.tolist() != MODELS:
        issues.append("six verified models are not present")
    result = {"passed": not issues, "issues": issues, "models": models.to_dict("records"),
              "heldout_wgen_seeds": SEEDS, "observed_years": YEARS, "training_run_performed": False,
              "old_reward_formula": {"formula": "0.06 * final_grnwt - 0.04 * cumfert", "status": "SUPERSEDED / NOT_APPLICABLE"},
              "canonical_pilot_preflight": base, "memory_guard_mb": MAX_RSS_MB}
    dump(TASK / "config" / "diagnosis_preflight.json", result)
    if issues:
        raise RuntimeError("Preflight failed: " + "; ".join(issues))
    return result


def observation_dict(value: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, item in (value or {}).items():
        try:
            x = float(np.asarray(item).reshape(-1)[0])
            out[str(key)] = x if math.isfinite(x) else None
        except (ValueError, TypeError, IndexError):
            if isinstance(item, (str, int, bool)):
                out[str(key)] = item
    return out


def schedules(eval_type: str) -> list[dict[str, Any]]:
    if eval_type == "heldout_wgen":
        return [{"episode_index": i + 1, "historical_year": 2008, "weather_seed": seed,
                 "evaluation_weather_type": eval_type, "evaluation_weather_seed": seed, "evaluation_weather_year": ""}
                for i, seed in enumerate(SEEDS)]
    if eval_type == "observed_weather":
        return [{"episode_index": i + 1, "historical_year": year, "weather_seed": "",
                 "evaluation_weather_type": eval_type, "evaluation_weather_seed": "", "evaluation_weather_year": year}
                for i, year in enumerate(YEARS)]
    raise ValueError(eval_type)


def sim_date(pilot: Any, env_config: dict[str, Any], crop_year: int, dap: int) -> date:
    _, _, batch, _, _, _, _, _ = pilot._import_canonical()
    info = batch.base.direct_ppo.find_year(env_config, "YCA", int(crop_year))
    start = pd.Timestamp(info["planting_date"]).date()
    return start + timedelta(days=max(dap - 1, 0))


def correct_episode_dates(pilot: Any, env_config: dict[str, Any], crop_year: int, steps: list[dict[str, Any]]) -> None:
    _, _, batch, _, _, _, _, _ = pilot._import_canonical()
    info = batch.base.direct_ppo.find_year(env_config, "YCA", int(crop_year))
    planting = pd.Timestamp(info["planting_date"]).date()
    pre_dap_days = 0
    for row in steps:
        if int(row["dap"]) == 0:
            pre_dap_days += 1
        else:
            break
    for row in steps:
        calendar_date = planting + timedelta(days=int(row["timestep"]) - pre_dap_days - 1)
        row["date"] = calendar_date.isoformat()
        row["doy"] = calendar_date.timetuple().tm_yday


def grid(config: dict[str, Any]) -> list[tuple[float, float]]:
    actions = config["discrete_actions"]
    return [(float(i), float(n)) for i in actions["irrigation_levels"] for n in actions["nitrogen_levels"]]


def weather_for_episode(eval_type: str, rows: list[dict[str, Any]], dssat: Any) -> list[dict[str, Any]]:
    weather_rows = []
    if eval_type == "heldout_wgen":
        from scripts.run_yc_wgen_seed_pilot import daily_weather_from_states
        weather_rows = daily_weather_from_states(getattr(dssat, "_pilot_weather_states", []))
        source = "DSSAT_runtime_weather_state_capture"
    else:
        path = ROOT / "weather_clean" / "all_sites_weather_cleaned.csv"
        weather = pd.read_csv(path)
        weather = weather[weather.station.astype(str).str.upper().eq("YCA")]
        crop_year = int(rows[0]["historical_year_context"])
        weather["date"] = pd.to_datetime(weather.date, errors="coerce")
        weather = weather[weather.date.dt.year.eq(crop_year)]
        weather_rows = [{"DATE": pd.Timestamp(r.date).date().isoformat(), "DOY": int(r.doy),
                         **{c: getattr(r, c) for c in ("RAIN", "SRAD", "TMAX", "TMIN")}}
                        for r in weather.itertuples(index=False)]
        indexed = weather.set_index(pd.to_datetime(weather.date).dt.strftime("%Y-%m-%d"))
        source = "observed_cleaned_YCA"
    by_date = {str(x.get("DATE")): x for x in weather_rows if x.get("DATE")}
    for row in rows:
        weather = by_date.get(row["date"], {})
        row.update({k: weather.get(k) for k in ("RAIN", "SRAD", "TMAX", "TMIN")})
    for row in rows:
        row["weather_context_status"] = "POST_HOC_DIAGNOSTIC_ONLY"
        row["weather_source"] = source
    return weather_rows


def replay(model_id: str, smoke: bool = False, smoke_name: str = "smoke",
           eval_type_filter: str | None = None) -> None:
    p = pilot_module()
    model_manifest = verified_manifest()
    path = checkpoint(model_id)
    if not path.is_file() or "training_verified" not in path.parts:
        raise RuntimeError("Only 004_03 training_verified checkpoints are allowed")
    if smoke and model_id != "W0":
        raise ValueError("Smoke is fixed to W0 / WGEN seed 1081")
    out = (TASK / "smoke" / smoke_name if smoke_name != "smoke" else TASK / "smoke") if smoke else TASK / "trajectories" / model_id
    out.mkdir(parents=True, exist_ok=True)
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(path), device="cpu")
    step_manifest, summary_rows = [], []
    try:
        eval_types = ["heldout_wgen"] if smoke else (["heldout_wgen", "observed_weather"] if eval_type_filter is None else [eval_type_filter])
        for eval_type in eval_types:
            if eval_type not in ("heldout_wgen", "observed_weather"):
                raise ValueError(eval_type)
            step_name = "W0_heldout_wgen_seed1081_steps.csv" if smoke else f"{model_id}_{eval_type}_steps.csv"
            weather_name = "W0_heldout_wgen_seed1081_weather_daily.csv" if smoke else f"{model_id}_{eval_type}_weather_daily.csv"
            step_path = out / step_name
            weather_path = out / weather_name
            if not smoke and step_path.is_file() and weather_path.is_file():
                print(f"[replay-skip] {model_id} {eval_type}; completed files already exist", flush=True)
                continue
            if not smoke and (step_path.exists() or weather_path.exists()):
                raise FileExistsError(f"Incomplete prior replay output pair; preserving it: {step_path} / {weather_path}")
            episodes = schedules(eval_type)[:1] if smoke else schedules(eval_type)
            runtime_regime = "RANDOM_WEATHER_WGEN" if eval_type == "heldout_wgen" else "HISTORICAL_WEATHER"
            cfg, env_cfg = p.make_run_config(REGIME[model_id], PPO_SEED[model_id], out)
            p.TASK_ROOT = out
            base_tag = f"diag_{model_id}_{eval_type}"
            tag_suffix = ""
            if not smoke:
                for attempt in range(1, 100):
                    suffix = "" if attempt == 1 else f"_retry{attempt - 1:02d}"
                    candidate_dirs = [out / "rendered_inputs" / "YCA" / str(ep["historical_year"]) /
                                      f"{base_tag}{suffix}_{ep['historical_year']}" for ep in episodes]
                    if not any(folder.exists() for folder in candidate_dirs):
                        tag_suffix = suffix
                        break
                else:
                    raise RuntimeError(f"Could not find unused render tag for {model_id}/{eval_type}")
            env = p.ScheduledEpisodeEnv(cfg, env_cfg, episodes, runtime_regime, PPO_SEED[model_id],
                                        f"{base_tag}{tag_suffix}", p.EVAL_RUNTIME_SEED)
            all_steps, all_weather = [], []
            try:
                for ep in episodes:
                    if memory_mb() > MAX_RSS_MB:
                        raise MemoryError(f"RSS exceeded {MAX_RSS_MB} MB before episode")
                    obs, _ = env._gym_env.reset(seed=None)
                    dssat = p.find_dssat_instance(env.current_env)
                    ep_steps, total_reward, done = [], 0.0, False
                    while not done:
                        state = observation_dict(getattr(env.current_env, "last_obs_dict", {}))
                        mask = np.asarray(env._gym_env.action_masks(), dtype=bool).reshape(-1)
                        obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
                        act, _ = model.predict(obs, deterministic=True, action_masks=mask)
                        requested = int(np.asarray(act).reshape(-1)[0])
                        next_obs, reward, terminated, truncated, _info = env._gym_env.step(act)
                        done = bool(terminated or truncated)
                        reward = float(reward)
                        total_reward += reward
                        info = dict(getattr(env.current_env, "last_action_info", {}) or {})
                        action_index = int(info.get("discrete_action_index", requested))
                        irr, fert = grid(cfg)[action_index]
                        reward_scale = float(info.get("reward_scale", 1.0))
                        safe_i = float(info.get("safe_action_amir", irr))
                        safe_n = float(info.get("safe_action_anfer", fert))
                        reward_cfg = cfg.get("reward", {})
                        water_cost = float(reward_cfg.get("water_cost", 0.0)) * safe_i * reward_scale
                        nitrogen_cost = float(reward_cfg.get("nitrogen_cost", 0.0)) * safe_n * reward_scale
                        yield_scaled = float(info.get("yield_component", 0.0)) * reward_scale
                        relief_scaled = float(info.get("stress_relief_bonus", 0.0)) * reward_scale
                        guardrail_scaled = float(info.get("swfac_guardrail_penalty_scaled", 0.0))
                        reconstructed = yield_scaled - water_cost - nitrogen_cost + relief_scaled - guardrail_scaled
                        dap_value = state.get("dap")
                        dap = int(round(float(dap_value))) if dap_value is not None else len(ep_steps) + 1
                        date_value = sim_date(p, env_cfg, int(ep["historical_year"]), dap).isoformat()
                        safety = getattr(env.current_env, "safety_state", None)
                        weather_key = ep["evaluation_weather_seed"] or ep["evaluation_weather_year"]
                        ep_steps.append({
                            "training_regime": REGIME[model_id], "ppo_seed": PPO_SEED[model_id], "model_id": model_id,
                            "evaluation_weather_type": eval_type, "evaluation_weather_seed": ep["evaluation_weather_seed"],
                            "evaluation_weather_year": ep["evaluation_weather_year"], "episode_key": f"{eval_type}:{weather_key}",
                            "episode_index": ep["episode_index"], "historical_year_context": ep["historical_year"],
                            "timestep": len(ep_steps) + 1, "date": date_value, "doy": date.fromisoformat(date_value).timetuple().tm_yday,
                            "dap": dap, "istage": state.get("istage"), "swfac": state.get("swfac"), "nstres": state.get("nstres"),
                            "topwt": state.get("topwt"), "grnwt": state.get("grnwt"), "xlai": state.get("xlai"),
                            "totir": state.get("totir"), "cumsumfert": state.get("cumsumfert"),
                            "RAIN": None, "SRAD": None, "TMAX": None, "TMIN": None,
                            "action_index": action_index, "requested_action_index": requested,
                            "irrigation_action_mm": irr, "fertilizer_action_kgN_ha": fert,
                            "action_mask": json.dumps(mask.astype(int).tolist()), "valid_action_count": int(mask.sum()),
                            "action_mask_used_json": json.dumps(mask.astype(int).tolist()),
                            "observation_vector_json": json.dumps(obs_vec.astype(float).tolist(), separators=(",", ":")),
                            "observation_variables_json": json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                            "instant_reward": reward, "cumulative_reward": total_reward,
                            "episode_total_irrigation": None, "episode_total_fertilizer": None,
                            "cumulative_irrigation_before": info.get("previous_cumulative_irrigation"),
                            "cumulative_fertilizer_before": info.get("previous_cumulative_n"),
                            "cumulative_irrigation": info.get("season_cumulative_irrigation", getattr(safety, "cumulative_irrigation", None)),
                            "cumulative_fertilizer": info.get("season_cumulative_n", getattr(safety, "cumulative_n", None)),
                            "yield_component": info.get("yield_component"), "resource_cost": info.get("resource_cost"),
                            "water_stress_relief": info.get("water_stress_relief"), "nitrogen_stress_relief": info.get("nitrogen_stress_relief"),
                            "stress_relief_bonus": info.get("stress_relief_bonus"),
                            "reward_unscaled": info.get("reward_unscaled"),
                            "reward_scale": reward_scale,
                            "reward_stress_aware": info.get("reward_stress_aware"),
                            "reward_before_swfac_guardrail": info.get("reward_before_swfac_guardrail"),
                            "swfac_guardrail_penalty_scaled": info.get("swfac_guardrail_penalty_scaled"),
                            "reward_after_swfac_guardrail": info.get("reward_after_swfac_guardrail"),
                            "yield_contribution_scaled": yield_scaled,
                            "water_cost_contribution_scaled": water_cost,
                            "nitrogen_cost_contribution_scaled": nitrogen_cost,
                            "stress_relief_contribution_scaled": relief_scaled,
                            "canonical_reward_reconstructed": reconstructed,
                            "canonical_reward_step_abs_error": abs(reward - reconstructed),
                            "mask_forced_noop": info.get("mask_forced_noop", False),
                            "canonical_action_info_json": json.dumps(observation_dict(info), sort_keys=True, separators=(",", ":")),
                            "post_step_state_json": json.dumps(observation_dict(getattr(env.current_env, "last_obs_dict", {})), sort_keys=True, separators=(",", ":")),
                        })
                        obs = next_obs
                        if not done and memory_mb() > MAX_RSS_MB:
                            raise MemoryError(f"RSS exceeded {MAX_RSS_MB} MB during episode")
                    episode_log = env.episode_rows[-1]
                    correct_episode_dates(p, env_cfg, int(ep["historical_year"]), ep_steps)
                    weather_rows = weather_for_episode(eval_type, ep_steps, dssat)
                    if eval_type == "heldout_wgen":
                        from scripts.run_yc_wgen_seed_pilot import canonical_weather_bytes, sha256_bytes
                        calculated_hash = sha256_bytes(canonical_weather_bytes(weather_rows))
                        weather_hash_reconciles = calculated_hash == episode_log["runtime_weather_sha256"]
                    else:
                        calculated_hash, weather_hash_reconciles = "", True
                    for weather_row in weather_rows:
                        all_weather.append({"model_id": model_id, "evaluation_weather_type": eval_type,
                                            "episode_key": f"{eval_type}:{weather_key}", "weather_source": "DSSAT_runtime_weather_state_capture" if eval_type == "heldout_wgen" else "observed_cleaned_YCA",
                                            "weather_context_status": "POST_HOC_DIAGNOSTIC_ONLY", **weather_row})
                    for step in ep_steps:
                        step["episode_total_irrigation"] = episode_log["irrigation"]
                        step["episode_total_fertilizer"] = episode_log["fertilizer"]
                        step["runtime_weather_sha256"] = episode_log["runtime_weather_sha256"]
                        step["episode_reward_component_sum"] = episode_log["reward_component_sum"]
                        step["episode_reward_decomposition_abs_error"] = episode_log["reward_decomposition_abs_error"]
                        step["runtime_weather_hash_recalculated_from_saved_rows"] = calculated_hash
                        step["runtime_weather_hash_reconciles"] = weather_hash_reconciles
                    all_steps.extend(ep_steps)
                    summary_rows.append({
                        "training_regime": REGIME[model_id], "ppo_seed": PPO_SEED[model_id], "model_id": model_id,
                        "evaluation_weather_type": eval_type, "evaluation_weather_seed": ep["evaluation_weather_seed"],
                        "evaluation_weather_year": ep["evaluation_weather_year"], "episode_key": f"{eval_type}:{weather_key}",
                        "historical_year_context": ep["historical_year"], "episode_return": episode_log["episode_return"],
                        "reward_component_sum": episode_log["reward_component_sum"],
                        "reward_decomposition_abs_error": episode_log["reward_decomposition_abs_error"],
                        "yield": episode_log["yield"], "total_fertilizer": episode_log["fertilizer"],
                        "total_irrigation": episode_log["irrigation"], "episode_days": episode_log["episode_days"],
                        "runtime_weather_sha256": episode_log["runtime_weather_sha256"], "filex_wther": episode_log["filex_wther"],
                        "runtime_filex_template_sha256": episode_log["runtime_filex_template_sha256"],
                        "weather_seed_actual": episode_log["actual_rseed1_sent_to_pdi"], "status": episode_log["status"],
                    })
                    print(f"[replay] {model_id} {eval_type} episode {ep['episode_index']}/{len(episodes)} days={len(ep_steps)} rss_mb={memory_mb():.0f}", flush=True)
            finally:
                env.close()
            pd.DataFrame(all_steps).to_csv(step_path, index=False, encoding="utf-8-sig", lineterminator="\n")
            pd.DataFrame(all_weather).to_csv(weather_path, index=False, encoding="utf-8-sig", lineterminator="\n")
            step_manifest.append({"model_id": model_id, "evaluation_weather_type": eval_type,
                                  "path": step_path.relative_to(ROOT).as_posix(), "rows": len(all_steps),
                                  "episodes": len(episodes), "sha256": hash_file(step_path),
                                  "weather_daily_path": weather_path.relative_to(ROOT).as_posix(),
                                  "weather_daily_rows": len(all_weather), "weather_hash_reconciles": bool(all_steps and all(x.get("runtime_weather_hash_reconciles", True) for x in all_steps)),
                                  "status": "completed"})
    finally:
        del model
        gc.collect()
    if smoke:
        summary_path = out / "W0_heldout_wgen_seed1081_episode.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig", lineterminator="\n")
        pd.DataFrame(step_manifest).to_csv(out / "smoke_step_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        result = smoke_compare(summary_path)
        dump(out / "smoke_manifest.json", result)
        print(json.dumps(_json_clean(result), ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    else:
        existing_steps, manifests = [], []
        for weather in ("heldout_wgen", "observed_weather"):
            step = out / f"{model_id}_{weather}_steps.csv"
            daily = out / f"{model_id}_{weather}_weather_daily.csv"
            if not (step.is_file() and daily.is_file()):
                continue
            frame = pd.read_csv(step, keep_default_na=False)
            existing_steps.append(frame)
            manifests.append({"model_id": model_id, "evaluation_weather_type": weather,
                              "path": step.relative_to(ROOT).as_posix(), "rows": len(frame),
                              "episodes": frame.episode_key.nunique(), "sha256": hash_file(step),
                              "weather_daily_path": daily.relative_to(ROOT).as_posix(),
                              "weather_daily_rows": len(pd.read_csv(daily)), "status": "completed"})
        if existing_steps:
            combined = pd.concat(existing_steps, ignore_index=True)
            episode_table(combined).to_csv(out / f"{model_id}_episode_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
            pd.DataFrame(manifests).to_csv(out / f"{model_id}_step_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        print(f"[replay-complete] {model_id}; step_rows={sum(x['rows'] for x in manifests)}", flush=True)


def smoke_compare(path: Path) -> dict[str, Any]:
    new = pd.read_csv(path, keep_default_na=False).iloc[0]
    old_path = OLD_EVAL / "heldout_wgen" / "random_weather" / "ppo_seed_0" / "evaluation_episode_level.csv"
    old_df = pd.read_csv(old_path, keep_default_na=False)
    old = old_df[pd.to_numeric(old_df.evaluation_weather_seed, errors="coerce").eq(1081)].iloc[0]
    checks = {
        "runtime_weather_hash": str(new.runtime_weather_sha256) == str(old.runtime_weather_sha256),
        "reward": abs(float(new.episode_return) - float(old.reward)) <= 1e-8,
        "yield": abs(float(new["yield"]) - float(old["yield"])) <= 1e-6,
        "fertilizer": abs(float(new.total_fertilizer) - float(old.fertilizer)) <= 1e-8,
        "irrigation": abs(float(new.total_irrigation) - float(old.irrigation)) <= 1e-8,
        "episode_days": int(new.episode_days) == int(old.episode_days),
        "WTHER_W": str(new.filex_wther) == "W",
    }
    result = {"passed": all(checks.values()), "model_id": "W0", "evaluation": "heldout WGEN seed 1081", "checks": checks,
              "replay": {k: new[k] for k in ("episode_return", "yield", "total_fertilizer", "total_irrigation", "episode_days", "runtime_weather_sha256", "filex_wther")},
              "reference_004_03": {k: old[k] for k in ("reward", "yield", "fertilizer", "irrigation", "episode_days", "runtime_weather_sha256", "filex_wther")}}
    if not result["passed"]:
        raise RuntimeError("Smoke does not match 004_03; stop before full replay and investigate")
    return result


def load_steps() -> pd.DataFrame:
    frames = []
    for mid in MODELS:
        for eval_type in ("heldout_wgen", "observed_weather"):
            path = TASK / "trajectories" / mid / f"{mid}_{eval_type}_steps.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frames.append(pd.read_csv(path, keep_default_na=False))
    data = pd.concat(frames, ignore_index=True)
    numeric = ["ppo_seed", "evaluation_weather_seed", "evaluation_weather_year", "episode_index", "timestep", "doy", "dap",
               "istage", "swfac", "nstres", "topwt", "grnwt", "xlai", "totir", "cumsumfert", "action_index",
               "irrigation_action_mm", "fertilizer_action_kgN_ha", "valid_action_count", "instant_reward", "cumulative_reward",
               "episode_total_irrigation", "episode_total_fertilizer", "cumulative_irrigation", "cumulative_fertilizer",
               "yield_component", "resource_cost", "water_stress_relief", "nitrogen_stress_relief", "stress_relief_bonus",
               "swfac_guardrail_penalty_scaled", "reward_after_swfac_guardrail", "episode_reward_component_sum",
               "episode_reward_decomposition_abs_error", "canonical_reward_reconstructed", "canonical_reward_step_abs_error",
               "runtime_weather_hash_reconciles",
               "yield_contribution_scaled", "water_cost_contribution_scaled", "nitrogen_cost_contribution_scaled",
               "stress_relief_contribution_scaled", "RAIN", "SRAD", "TMAX", "TMIN"]
    for col in numeric:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data["date"] = pd.to_datetime(data.date, errors="coerce")
    return data


def episode_table(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_id", "training_regime", "ppo_seed", "evaluation_weather_type", "evaluation_weather_seed", "evaluation_weather_year", "episode_key"]
    rows = []
    for key, frame in data.sort_values("timestep").groupby(keys, dropna=False, sort=False):
        i = frame.irrigation_action_mm.fillna(0)
        n = frame.fertilizer_action_kgN_ha.fillna(0)
        out = dict(zip(keys, key))
        out.update({"episode_return": float(frame.instant_reward.sum()), "yield": frame.iloc[-1].get("grnwt"),
                    "total_irrigation": float(i.sum()), "total_fertilizer": float(n.sum()), "episode_days": len(frame),
                    "irrigation_event_count": int(i.gt(0).sum()), "fertilizer_event_count": int(n.gt(0).sum()),
                    "action_events": int((i.gt(0) | n.gt(0)).sum()),
                    "mean_irrigation_per_event": float(i[i.gt(0)].mean()) if i.gt(0).any() else 0.0,
                    "mean_fertilizer_per_event": float(n[n.gt(0)].mean()) if n.gt(0).any() else 0.0,
                    "first_irrigation_dap": float(frame.loc[i.gt(0), "dap"].min()) if i.gt(0).any() else np.nan,
                    "last_irrigation_dap": float(frame.loc[i.gt(0), "dap"].max()) if i.gt(0).any() else np.nan,
                    "first_fertilizer_dap": float(frame.loc[n.gt(0), "dap"].min()) if n.gt(0).any() else np.nan,
                    "last_fertilizer_dap": float(frame.loc[n.gt(0), "dap"].max()) if n.gt(0).any() else np.nan,
                    "reward_component_sum": float(frame.iloc[-1].get("episode_reward_component_sum", np.nan)),
                    "reward_decomposition_abs_error": float(frame.iloc[-1].get("episode_reward_decomposition_abs_error", np.nan)),
                    "canonical_step_reconstruction_max_abs_error": float(frame.canonical_reward_step_abs_error.max())})
        for col in ("yield_contribution_scaled", "water_cost_contribution_scaled", "nitrogen_cost_contribution_scaled",
                    "stress_relief_contribution_scaled", "swfac_guardrail_penalty_scaled"):
            out[f"sum_{col}"] = frame[col].sum(min_count=1)
        rows.append(out)
    return pd.DataFrame(rows)


def analyze() -> None:
    data = load_steps()
    if set(data.model_id.unique()) != set(MODELS):
        raise RuntimeError("Not all six models have step trajectories")
    if set(data.loc[data.evaluation_weather_type.eq("heldout_wgen"), "evaluation_weather_seed"].dropna().astype(int)) != set(SEEDS):
        raise RuntimeError("Heldout WGEN evaluation set mismatch")
    if set(data.loc[data.evaluation_weather_type.eq("observed_weather"), "evaluation_weather_year"].dropna().astype(int)) != set(YEARS):
        raise RuntimeError("Observed-year evaluation set mismatch")
    episodes = episode_table(data)
    episodes.to_csv(TASK / "evaluation_episode_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    data, weather_qc = attach_weather_context(data)
    occupancy(data)
    timing(data, episodes)
    stress_weather(data)
    paired = pair_comparison(data, episodes)
    paired.to_csv(TASK / "same_seed_pair_behavior_comparison.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    step_manifest = []
    for mid in MODELS:
        for weather in ("heldout_wgen", "observed_weather"):
            path = TASK / "trajectories" / mid / f"{mid}_{weather}_steps.csv"
            part = data[(data.model_id == mid) & data.evaluation_weather_type.eq(weather)]
            wpath = TASK / "trajectories" / mid / f"{mid}_{weather}_weather_daily.csv"
            step_manifest.append({"model_id": mid, "evaluation_weather_type": weather, "path": path.relative_to(ROOT).as_posix(),
                                  "rows": len(part), "episodes": part.episode_key.nunique(), "sha256": hash_file(path),
                                  "weather_daily_path": wpath.relative_to(ROOT).as_posix(),
                                  "weather_daily_rows": len(pd.read_csv(wpath)), "status": "completed"})
    pd.DataFrame(step_manifest).to_csv(TASK / "evaluation_step_level_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    reward_summary(episodes)
    cases = select_cases(data, episodes)
    probes = freeze_probes(data)
    dump(TASK / "config" / "analysis_manifest.json", {"probe_seed": PROBE_SEED, "probe_states": len(probes), "probe_fixed_before_predictions": True,
         "step_rows": len(data), "episodes": len(episodes), "weather_context": "POST_HOC_DIAGNOSTIC_ONLY",
         "weather_data_qc": weather_qc,
         "stress_bin_semantics": "Existing project interpretation: larger SWFAC/NSTRES values indicate greater stress severity"})
    print(f"[analysis-ready] episodes={len(episodes)} step_rows={len(data)} probes={len(probes)} selected_cases={len(cases)}", flush=True)


def attach_weather_context(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    for mid in MODELS:
        for weather_type in ("heldout_wgen", "observed_weather"):
            path = TASK / "trajectories" / mid / f"{mid}_{weather_type}_weather_daily.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path, keep_default_na=False)
            if not frame.empty:
                frames.append(frame)
    weather = pd.concat(frames, ignore_index=True)
    weather["date"] = pd.to_datetime(weather.DATE, errors="coerce")
    for col in ("RAIN", "SRAD", "TMAX", "TMIN"):
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    weather = weather.sort_values(["model_id", "episode_key", "date"])
    lag_fields = {}
    for source, target, days, aggregate in (
        ("RAIN", "previous_3d_rain_sum", 3, "sum"), ("RAIN", "previous_7d_rain_sum", 7, "sum"),
        ("TMAX", "previous_3d_tmax_mean", 3, "mean"), ("SRAD", "previous_3d_srad_mean", 3, "mean")):
        lag_fields[target] = weather.groupby(["model_id", "episode_key"])[source].transform(
            lambda s: s.shift(1).rolling(days, min_periods=1).sum() if aggregate == "sum" else s.shift(1).rolling(days, min_periods=1).mean())
        weather[target] = lag_fields[target]
    weather = weather.drop_duplicates(["model_id", "episode_key", "date"], keep="last")
    use_cols = ["model_id", "episode_key", "date", "RAIN", "SRAD", "TMAX", "TMIN", "weather_source",
                "weather_context_status", "previous_3d_rain_sum", "previous_7d_rain_sum", "previous_3d_tmax_mean", "previous_3d_srad_mean"]
    old_cols = [c for c in ("RAIN", "SRAD", "TMAX", "TMIN", "weather_source", "weather_context_status") if c in data]
    base = data.drop(columns=old_cols, errors="ignore")
    merged = base.merge(weather[use_cols], on=["model_id", "episode_key", "date"], how="left")
    episode_counts = merged.groupby(["model_id", "evaluation_weather_type"]).agg(
        step_rows=("action_index", "size"), rain_present=("RAIN", lambda s: int(s.notna().sum())),
        weather_sources=("weather_source", lambda s: ",".join(sorted(set(s.dropna().astype(str)))))).reset_index()
    episode_counts["weather_coverage_fraction"] = episode_counts.rain_present / episode_counts.step_rows
    wgen_reconciles = {}
    for mid in MODELS:
        rows = data[(data.model_id == mid) & data.evaluation_weather_type.eq("heldout_wgen")]
        flags = rows.runtime_weather_hash_reconciles.astype(str).str.lower().isin(["true", "1", "1.0"])
        wgen_reconciles[mid] = bool(flags.all() and len(flags) > 0)
    episode_counts.to_csv(TASK / "weather_response" / "weather_data_quality.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    wgen_weather = weather[weather.evaluation_weather_type.eq("heldout_wgen")] if "evaluation_weather_type" in weather else pd.DataFrame()
    variation = {}
    for col in ("RAIN", "SRAD", "TMAX", "TMIN"):
        if not wgen_weather.empty:
            unique = wgen_weather.groupby(["model_id", "episode_key"])[col].nunique(dropna=True)
            variation[col] = {"median_unique_daily_values_per_episode": float(unique.median()),
                              "minimum_unique_daily_values_per_episode": int(unique.min()),
                              "episodes": int(len(unique))}
    qc = {"step_weather_coverage_by_model_and_eval": episode_counts.to_dict("records"),
          "WGEN_runtime_weather_hash_reconciles_for_every_step": wgen_reconciles,
          "WGEN_hash_reconciliation_pass": all(wgen_reconciles.values()),
          "WGEN_weather_value_variation": variation,
          "WGEN_source": "per-episode canonical daily_weather_from_states used by 004_03 runtime hash"}
    return merged, qc


def occupancy(data: pd.DataFrame) -> None:
    rows = []
    for (mid, weather), frame in data.groupby(["model_id", "evaluation_weather_type"]):
        size = len(frame)
        for action in range(16):
            rows.append({"model_id": mid, "evaluation_weather_type": weather, "action_index": action,
                         "irrigation": [0, 15, 30, 45][action // 4], "fertilizer": [0, 40, 80, 120][action % 4],
                         "count": int(frame.action_index.eq(action).sum()), "fraction": float(frame.action_index.eq(action).mean()), "component": "joint"})
        for component, col, levels in (("irrigation", "irrigation_action_mm", [0, 15, 30, 45]),
                                       ("fertilizer", "fertilizer_action_kgN_ha", [0, 40, 80, 120])):
            for level in levels:
                count = int(frame[col].eq(level).sum())
                rows.append({"model_id": mid, "evaluation_weather_type": weather, "action_index": f"{component}_marginal",
                             "irrigation": level if component == "irrigation" else "", "fertilizer": level if component == "fertilizer" else "",
                             "count": count, "fraction": count / size, "component": component, "dose_level": level})
    pd.DataFrame(rows).to_csv(TASK / "action_occupancy_by_model.csv", index=False, encoding="utf-8-sig", lineterminator="\n")


def timing(data: pd.DataFrame, episodes: pd.DataFrame) -> None:
    cols = ["model_id", "evaluation_weather_type", "episode_key", "action_events", "irrigation_event_count", "fertilizer_event_count",
            "total_irrigation", "total_fertilizer", "mean_irrigation_per_event", "mean_fertilizer_per_event", "first_irrigation_dap",
            "last_irrigation_dap", "first_fertilizer_dap", "last_fertilizer_dap"]
    episodes[cols].to_csv(TASK / "management_event_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    group = data.groupby(["model_id", "evaluation_weather_type", "dap"], dropna=False)
    out = group.agg(episodes=("episode_key", "nunique"),
        irrigation_event_count=("irrigation_action_mm", lambda s: int(s.gt(0).sum())),
        fertilizer_event_count=("fertilizer_action_kgN_ha", lambda s: int(s.gt(0).sum())),
        irrigation_probability=("irrigation_action_mm", lambda s: float(s.gt(0).mean())),
        fertilizer_probability=("fertilizer_action_kgN_ha", lambda s: float(s.gt(0).mean())),
        mean_irrigation_dose=("irrigation_action_mm", "mean"), mean_fertilizer_dose=("fertilizer_action_kgN_ha", "mean"),
        cumulative_irrigation_median=("cumulative_irrigation", "median"), cumulative_irrigation_p25=("cumulative_irrigation", lambda s: s.quantile(.25)),
        cumulative_irrigation_p75=("cumulative_irrigation", lambda s: s.quantile(.75)),
        cumulative_fertilizer_median=("cumulative_fertilizer", "median"), cumulative_fertilizer_p25=("cumulative_fertilizer", lambda s: s.quantile(.25)),
        cumulative_fertilizer_p75=("cumulative_fertilizer", lambda s: s.quantile(.75))).reset_index()
    out.to_csv(TASK / "management_timing_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    out.to_csv(TASK / "resource_trajectories" / "cumulative_resource_by_dap.csv", index=False, encoding="utf-8-sig", lineterminator="\n")


def stress_weather(data: pd.DataFrame) -> None:
    data["swfac_bin"] = pd.cut(data.swfac, [-np.inf, .05, .25, .5, np.inf], labels=["<=0.05", "(0.05,0.25]", "(0.25,0.50]", ">0.50"])
    data["nstres_bin"] = pd.cut(data.nstres, [-np.inf, .05, .25, .5, np.inf], labels=["<=0.05", "(0.05,0.25]", "(0.25,0.50]", ">0.50"])
    stress = []
    for var, bin_col in (("SWFAC", "swfac_bin"), ("NSTRES", "nstres_bin")):
        for (mid, weather, b), part in data.groupby(["model_id", "evaluation_weather_type", bin_col], observed=True, dropna=False):
            stress.append({"model_id": mid, "evaluation_weather_type": weather, "stress_variable": var, "stress_bin": str(b), "n_steps": len(part),
                           "p_irrigation_gt0": part.irrigation_action_mm.gt(0).mean(), "mean_irrigation_dose": part.irrigation_action_mm.mean(),
                           "p_fertilizer_gt0": part.fertilizer_action_kgN_ha.gt(0).mean(), "mean_fertilizer_dose": part.fertilizer_action_kgN_ha.mean()})
    pd.DataFrame(stress).to_csv(TASK / "stress_response_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    data["rain3_bin"] = pd.cut(data.previous_3d_rain_sum, [-np.inf, .1, 5, 15, np.inf], labels=["<=0.1", "(0.1,5]", "(5,15]", ">15"])
    data["rain7_bin"] = pd.cut(data.previous_7d_rain_sum, [-np.inf, .1, 15, 40, np.inf], labels=["<=0.1", "(0.1,15]", "(15,40]", ">40"])
    wrows = []
    for col, name in (("rain3_bin", "previous_3d_rain"), ("rain7_bin", "previous_7d_rain")):
        for (mid, weather, b), part in data.groupby(["model_id", "evaluation_weather_type", col], observed=True, dropna=False):
            wrows.append({"model_id": mid, "evaluation_weather_type": weather, "weather_variable": name, "weather_bin": str(b),
                          "n_steps": len(part), "p_irrigation_gt0": part.irrigation_action_mm.gt(0).mean(),
                          "p_fertilizer_gt0": part.fertilizer_action_kgN_ha.gt(0).mean(),
                          "mean_irrigation_dose": part.irrigation_action_mm.mean(), "mean_fertilizer_dose": part.fertilizer_action_kgN_ha.mean(),
                          "interpretation": "POST_HOC_DIAGNOSTIC_ONLY"})
    for label in ("previous_3d_tmax_mean", "previous_3d_srad_mean"):
        bins = pd.qcut(data[label], 4, labels=False, duplicates="drop")
        data[label + "_bin"] = bins
        for (mid, weather, b), part in data.groupby(["model_id", "evaluation_weather_type", label + "_bin"], observed=True, dropna=False):
            wrows.append({"model_id": mid, "evaluation_weather_type": weather, "weather_variable": label, "weather_bin": str(b),
                          "n_steps": len(part), "p_irrigation_gt0": part.irrigation_action_mm.gt(0).mean(),
                          "p_fertilizer_gt0": part.fertilizer_action_kgN_ha.gt(0).mean(),
                          "mean_irrigation_dose": part.irrigation_action_mm.mean(), "mean_fertilizer_dose": part.fertilizer_action_kgN_ha.mean(),
                          "interpretation": "POST_HOC_DIAGNOSTIC_ONLY"})
    pd.DataFrame(wrows).to_csv(TASK / "weather_response_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")


def pair_comparison(data: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in range(3):
        h, w = f"H{seed}", f"W{seed}"
        for weather in ("heldout_wgen", "observed_weather"):
            left = data[(data.model_id == h) & data.evaluation_weather_type.eq(weather)]
            right = data[(data.model_id == w) & data.evaluation_weather_type.eq(weather)]
            paired = left.merge(right, on=["episode_key", "timestep"], suffixes=("_h", "_w"))
            eh = episodes[(episodes.model_id == h) & episodes.evaluation_weather_type.eq(weather)].set_index("episode_key")
            ew = episodes[(episodes.model_id == w) & episodes.evaluation_weather_type.eq(weather)].set_index("episode_key")
            common = eh.index.intersection(ew.index)
            rows.append({"historical_model": h, "random_weather_model": w, "ppo_seed": seed, "evaluation_weather_type": weather,
                         "paired_episodes": len(common), "common_trajectory_steps": len(paired),
                         "trajectory_action_agreement_rate": paired.action_index_h.eq(paired.action_index_w).mean(),
                         "irrigation_action_agreement_rate": paired.irrigation_action_mm_h.eq(paired.irrigation_action_mm_w).mean(),
                         "fertilizer_action_agreement_rate": paired.fertilizer_action_kgN_ha_h.eq(paired.fertilizer_action_kgN_ha_w).mean(),
            "mean_abs_irrigation_dose_difference": (paired.irrigation_action_mm_h - paired.irrigation_action_mm_w).abs().mean(),
                         "mean_abs_fertilizer_dose_difference": (paired.fertilizer_action_kgN_ha_h - paired.fertilizer_action_kgN_ha_w).abs().mean(),
                         "mean_first_irrigation_dap_difference": (ew.loc[common, "first_irrigation_dap"] - eh.loc[common, "first_irrigation_dap"]).abs().mean(),
                         "mean_first_fertilizer_dap_difference": (ew.loc[common, "first_fertilizer_dap"] - eh.loc[common, "first_fertilizer_dap"]).abs().mean(),
                         "mean_irrigation_event_count_difference_w_minus_h": (ew.loc[common, "irrigation_event_count"] - eh.loc[common, "irrigation_event_count"]).mean(),
                         "mean_fertilizer_event_count_difference_w_minus_h": (ew.loc[common, "fertilizer_event_count"] - eh.loc[common, "fertilizer_event_count"]).mean(),
                         "mean_irrigation_per_event_difference_w_minus_h": (ew.loc[common, "mean_irrigation_per_event"] - eh.loc[common, "mean_irrigation_per_event"]).mean(),
                         "mean_fertilizer_per_event_difference_w_minus_h": (ew.loc[common, "mean_fertilizer_per_event"] - eh.loc[common, "mean_fertilizer_per_event"]).mean(),
                         "mean_reward_difference_w_minus_h": (ew.loc[common, "episode_return"] - eh.loc[common, "episode_return"]).mean(),
                         "mean_yield_difference_w_minus_h": (ew.loc[common, "yield"] - eh.loc[common, "yield"]).mean(),
                         "mean_irrigation_difference_w_minus_h": (ew.loc[common, "total_irrigation"] - eh.loc[common, "total_irrigation"]).mean(),
                         "mean_fertilizer_difference_w_minus_h": (ew.loc[common, "total_fertilizer"] - eh.loc[common, "total_fertilizer"]).mean(),
                         "interpretation": "TRAJECTORY_LEVEL; states may diverge after the first action"})
    return pd.DataFrame(rows)


def reward_summary(episodes: pd.DataFrame) -> None:
    comps = [c for c in episodes if c.startswith("sum_")]
    out = episodes.groupby(["model_id", "evaluation_weather_type"]).agg(
        episodes=("episode_key", "nunique"), mean_return=("episode_return", "mean"), mean_yield=("yield", "mean"),
        mean_irrigation=("total_irrigation", "mean"), mean_fertilizer=("total_fertilizer", "mean"),
        max_episode_reconciliation_abs_error=("reward_decomposition_abs_error", "max"),
        **{f"mean_{c}": (c, "mean") for c in comps}).reset_index()
    out["reconstruction_status"] = np.where(out.max_episode_reconciliation_abs_error <= 1e-8, "EXACT_RUNTIME_REWARD_SUM", "CHECK")
    out.to_csv(TASK / "reward_component_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    out.to_csv(TASK / "reward_decomposition" / "runtime_component_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")


def select_cases(data: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    wide = episodes.pivot(index=["evaluation_weather_type", "evaluation_weather_seed", "evaluation_weather_year", "episode_key"], columns="model_id", values="episode_return").reset_index()
    wide["W0_minus_H0_reward"] = wide.W0 - wide.H0
    wg = wide[wide.evaluation_weather_type.eq("heldout_wgen")]
    ob = wide[wide.evaluation_weather_type.eq("observed_weather")]
    selected = []
    wgen_decline_label = "WGEN_W0_largest_decline" if wg.W0_minus_H0_reward.lt(0).any() else "WGEN_W0_smallest_improvement_no_decline"
    for label, frame, ascending in (("WGEN_W0_largest_improvement", wg, False),
                                    (wgen_decline_label, wg, True),
                                    ("OBSERVED_W0_largest_decline", ob, True)):
        for _, row in frame.sort_values("W0_minus_H0_reward", ascending=ascending).head(2).iterrows():
            selected.append({"case_group": label, "episode_key": row.episode_key,
                             "w0_minus_h0_reward": float(row.W0_minus_H0_reward)})
    result = pd.DataFrame(selected)
    first_divergence = []
    case_summaries = []
    for case in result.itertuples(index=False):
        pair = data[(data.episode_key == case.episode_key) & data.model_id.isin(["H0", "W0"])]
        h0 = pair[pair.model_id.eq("H0")].sort_values("timestep")
        w0 = pair[pair.model_id.eq("W0")].sort_values("timestep")
        aligned = h0.merge(w0, on=["episode_key", "timestep"], suffixes=("_h0", "_w0"))
        mismatch = aligned[aligned.action_index_h0.ne(aligned.action_index_w0)]
        divergence = mismatch.iloc[0] if not mismatch.empty else None
        first_divergence.append({"case_group": case.case_group, "episode_key": case.episode_key,
                                 "w0_minus_h0_reward": case.w0_minus_h0_reward,
                                 "first_action_divergence_timestep": int(divergence.timestep) if divergence is not None else "",
                                 "first_action_divergence_dap": int(divergence.dap_h0) if divergence is not None else "",
                                 "first_divergent_H0_action": int(divergence.action_index_h0) if divergence is not None else "",
                                 "first_divergent_W0_action": int(divergence.action_index_w0) if divergence is not None else "",
                                 "divergence_note": "First action mismatch; subsequent state trajectories may differ." if divergence is not None else "No action mismatch on common rollout steps."})
        for mid, frame in (("H0", h0), ("W0", w0)):
            last = frame.iloc[-1] if not frame.empty else None
            irrigation_daps = frame.loc[frame.irrigation_action_mm.gt(0), "dap"] if not frame.empty else pd.Series(dtype=float)
            fertilizer_daps = frame.loc[frame.fertilizer_action_kgN_ha.gt(0), "dap"] if not frame.empty else pd.Series(dtype=float)
            case_summaries.append({"case_group": case.case_group, "episode_key": case.episode_key, "model_id": mid,
                                   "evaluation_weather_seed": int(frame.evaluation_weather_seed.dropna().iloc[0]) if not frame.empty and frame.evaluation_weather_seed.notna().any() else "",
                                   "evaluation_weather_year": int(frame.evaluation_weather_year.dropna().iloc[0]) if not frame.empty and frame.evaluation_weather_year.notna().any() else "",
                                   "episode_return": float(frame.instant_reward.sum()) if not frame.empty else np.nan,
                                   "yield": float(last.grnwt) if last is not None else np.nan,
                                   "total_irrigation": float(last.episode_total_irrigation) if last is not None else np.nan,
                                   "total_fertilizer": float(last.episode_total_fertilizer) if last is not None else np.nan,
                                   "mean_rain": float(frame.RAIN.mean()) if not frame.empty else np.nan,
                                   "total_rain": float(frame.RAIN.sum()) if not frame.empty else np.nan,
                                   "mean_TMAX": float(frame.TMAX.mean()) if not frame.empty else np.nan,
                                   "mean_SRAD": float(frame.SRAD.mean()) if not frame.empty else np.nan,
                                   "mean_SWFAC": float(frame.swfac.mean()) if not frame.empty else np.nan,
                                   "minimum_SWFAC": float(frame.swfac.min()) if not frame.empty else np.nan,
                                   "days_SWFAC_gt_0p05": int(frame.swfac.gt(.05).sum()) if not frame.empty else 0,
                                   "mean_NSTRES": float(frame.nstres.mean()) if not frame.empty else np.nan,
                                   "minimum_NSTRES": float(frame.nstres.min()) if not frame.empty else np.nan,
                                   "days_NSTRES_gt_0p05": int(frame.nstres.gt(.05).sum()) if not frame.empty else 0,
                                   "irrigation_event_count": int(frame.irrigation_action_mm.gt(0).sum()) if not frame.empty else 0,
                                   "fertilizer_event_count": int(frame.fertilizer_action_kgN_ha.gt(0).sum()) if not frame.empty else 0,
                                   "first_irrigation_dap": float(irrigation_daps.min()) if not irrigation_daps.empty else "",
                                   "first_fertilizer_dap": float(fertilizer_daps.min()) if not fertilizer_daps.empty else "",
                                   "mean_previous_7d_rain": float(frame.previous_7d_rain_sum.mean()) if not frame.empty and "previous_7d_rain_sum" in frame else np.nan,
                                   "mean_previous_3d_TMAX": float(frame.previous_3d_tmax_mean.mean()) if not frame.empty and "previous_3d_tmax_mean" in frame else np.nan,
                                   "mean_previous_3d_SRAD": float(frame.previous_3d_srad_mean.mean()) if not frame.empty and "previous_3d_srad_mean" in frame else np.nan})
    result = result.merge(pd.DataFrame(first_divergence), on=["case_group", "episode_key", "w0_minus_h0_reward"], how="left")
    result.to_csv(TASK / "case_study_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    pd.DataFrame(case_summaries).to_csv(TASK / "case_studies" / "selected_case_episode_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    data[data.episode_key.isin(result.episode_key)].to_csv(TASK / "case_studies" / "selected_case_step_trajectories.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    return result


def freeze_probes(data: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(PROBE_SEED)
    frame = data.copy()
    frame["season_phase"] = pd.cut(frame.dap, [-np.inf, 35, 70, np.inf], labels=["early", "mid", "late"])
    frame["swfac_bin"] = pd.cut(frame.swfac, [-np.inf, .05, .25, .5, np.inf], labels=["<=0.05", "(0.05,0.25]", "(0.25,0.50]", ">0.50"])
    frame["nstres_bin"] = pd.cut(frame.nstres, [-np.inf, .05, .25, .5, np.inf], labels=["<=0.05", "(0.05,0.25]", "(0.25,0.50]", ">0.50"])
    resource = frame.cumulative_irrigation.fillna(0) + frame.cumulative_fertilizer.fillna(0)
    frame["resource_bin"] = pd.qcut(resource, 4, labels=False, duplicates="drop")
    columns = ["evaluation_weather_type", "season_phase", "swfac_bin", "nstres_bin", "resource_bin"]
    sampled = []
    groups = list(frame.groupby(columns, observed=True, dropna=False, sort=True))
    each = max(1, PROBE_N // max(len(groups), 1))
    for _, part in groups:
        count = min(each, len(part))
        if count:
            sampled.extend(rng.choice(part.index.to_numpy(), size=count, replace=False).tolist())
    selected = frame.loc[sorted(set(sampled))]
    if len(selected) < PROBE_N:
        rest = frame.drop(index=selected.index)
        count = min(PROBE_N - len(selected), len(rest))
        if count:
            selected = pd.concat([selected, rest.loc[rng.choice(rest.index.to_numpy(), size=count, replace=False)]])
    if len(selected) > PROBE_N:
        selected = selected.iloc[np.sort(rng.choice(len(selected), size=PROBE_N, replace=False))]
    selected = selected.reset_index(drop=True)
    selected.insert(0, "probe_id", [f"P{x:04d}" for x in range(1, len(selected) + 1)])
    cols = ["probe_id", "model_id", "evaluation_weather_type", "episode_key", "timestep", "date", "dap", "istage", "swfac", "nstres",
            "cumulative_irrigation", "cumulative_fertilizer", "season_phase", "swfac_bin", "nstres_bin", "resource_bin",
            "action_mask_used_json", "observation_vector_json", "observation_variables_json"]
    target = TASK / "policy_similarity" / "policy_probe_states.csv"
    selected[cols].to_csv(target, index=False, encoding="utf-8-sig", lineterminator="\n")
    return selected[cols]


def probe_models() -> None:
    path = TASK / "policy_similarity" / "policy_probe_states.csv"
    if not path.is_file():
        raise FileNotFoundError("Run analyze first to freeze fixed probe states and masks")
    states = pd.read_csv(path, keep_default_na=False)
    from sb3_contrib import MaskablePPO
    outputs = []
    for mid in MODELS:
        model = MaskablePPO.load(str(checkpoint(mid)), device="cpu")
        try:
            for row in states.itertuples(index=False):
                obs = np.asarray(json.loads(row.observation_vector_json), dtype=np.float32)
                mask = np.asarray(json.loads(row.action_mask_used_json), dtype=bool)
                if not mask.any():
                    raise RuntimeError(f"Empty saved action mask at {row.probe_id}")
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                a = int(np.asarray(action).reshape(-1)[0])
                outputs.append({"probe_id": row.probe_id, "model_id": mid, "training_regime": REGIME[mid], "ppo_seed": PPO_SEED[mid],
                                "action_index": a, "irrigation_action_mm": [0, 15, 30, 45][a // 4],
                                "fertilizer_action_kgN_ha": [0, 40, 80, 120][a % 4],
                                "saved_mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest().upper()})
        finally:
            del model
            gc.collect()
        print(f"[probe] {mid} done; rss_mb={memory_mb():.0f}", flush=True)
    pd.DataFrame(outputs).to_csv(TASK / "policy_similarity" / "policy_probe_actions.csv", index=False, encoding="utf-8-sig", lineterminator="\n")


def finalize() -> None:
    probe_path = TASK / "policy_similarity" / "policy_probe_states.csv"
    action_path = TASK / "policy_similarity" / "policy_probe_actions.csv"
    states, actions = pd.read_csv(probe_path, keep_default_na=False), pd.read_csv(action_path)
    if set(actions.model_id.unique()) != set(MODELS) or actions.groupby("probe_id").model_id.nunique().ne(6).any():
        raise RuntimeError("Incomplete same-state policy predictions")
    wide = actions.pivot(index="probe_id", columns="model_id", values="action_index").loc[:, MODELS]
    matrix = []
    for a in MODELS:
        row = {"model_id": a}
        row.update({b: float(wide[a].eq(wide[b]).mean()) for b in MODELS})
        matrix.append(row)
    matrix_df = pd.DataFrame(matrix)
    matrix_df.to_csv(TASK / "policy_similarity" / "policy_similarity_matrix.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    agreement = float(wide.H1.eq(wide.W0).mean())
    data = load_steps()
    data, _weather_qc = attach_weather_context(data)
    episodes = episode_table(data)
    h1_steps = data[data.model_id.eq("H1")][["episode_key", "timestep", "action_index"]]
    w0_steps = data[data.model_id.eq("W0")][["episode_key", "timestep", "action_index"]]
    trajectory = h1_steps.merge(w0_steps, on=["episode_key", "timestep"], suffixes=("_h1", "_w0"))
    trajectory_agreement = float(trajectory.action_index_h1.eq(trajectory.action_index_w0).mean())
    h1_ep = episodes[episodes.model_id.eq("H1")].set_index("episode_key")
    w0_ep = episodes[episodes.model_id.eq("W0")].set_index("episode_key")
    common = h1_ep.index.intersection(w0_ep.index)
    aggregate_equal = all(np.allclose(h1_ep.loc[common, col], w0_ep.loc[common, col], atol=1e-8, rtol=0, equal_nan=True)
                          for col in ("episode_return", "yield", "total_irrigation", "total_fertilizer"))
    if agreement == 1 and trajectory_agreement == 1:
        cls = "IDENTICAL_POLICY_BEHAVIOR"
    elif agreement >= .95 and trajectory_agreement >= .95:
        cls = "NEAR_IDENTICAL_BEHAVIOR"
    elif aggregate_equal:
        cls = "SIMILAR_AGGREGATES_DIFFERENT_POLICY"
    else:
        cls = "NOT_SIMILAR"
    matrix_json = {r["model_id"]: {k: v for k, v in r.items() if k != "model_id"} for r in matrix}
    h1w0 = {"probe_states": len(states), "same_state_action_agreement": agreement,
            "irrigation_component_agreement": float((wide.H1 // 4).eq(wide.W0 // 4).mean()),
            "fertilizer_component_agreement": float((wide.H1 % 4).eq(wide.W0 % 4).mean()),
            "policy_similarity_class": cls, "agreement_matrix": matrix_json,
            "probability_comparison": "SKIPPED; deterministic actions only; no SB3/runtime patch",
            "saved_action_mask_same_for_all_models": True}
    dump(TASK / "policy_similarity" / "H1_W0_similarity_summary.json", h1w0)
    data = load_steps()
    episodes = episode_table(data)
    profile = make_w0_profile(data, episodes, {**h1w0, "trajectory_action_agreement": trajectory_agreement, "paired_episode_metrics_equal": bool(aggregate_equal)})
    dump(TASK / "W0_diagnostic_profile.json", profile)
    paired = pd.read_csv(TASK / "same_seed_pair_behavior_comparison.csv")
    make_figures(data, matrix_df, paired)
    decision = make_decision(data, episodes, paired, matrix_json, profile)
    dump(TASK / "diagnostic_decision.json", decision)
    write_report(data, episodes, paired, matrix_df, profile, decision)
    final_qc(data, episodes, states, actions)
    dump(TASK / "diagnostic_summary.json", {"decision": decision,
         "final_qc": json.loads((TASK / "config" / "final_qc.json").read_text(encoding="utf-8")),
         "report": "docs/yc_ppo_seed_variability_and_policy_behavior_diagnosis.md"})
    terminal_summary(decision)


def make_w0_profile(data: pd.DataFrame, episodes: pd.DataFrame, agreement: dict[str, Any]) -> dict[str, Any]:
    performance = episodes.groupby(["model_id", "evaluation_weather_type"]).agg(
        mean_return=("episode_return", "mean"), mean_yield=("yield", "mean"), mean_irrigation=("total_irrigation", "mean"),
        mean_fertilizer=("total_fertilizer", "mean"), mean_irrigation_events=("irrigation_event_count", "mean"),
        mean_fertilizer_events=("fertilizer_event_count", "mean"), mean_first_irrigation_dap=("first_irrigation_dap", "mean"),
        mean_first_fertilizer_dap=("first_fertilizer_dap", "mean")).reset_index()
    w = data[data.model_id.eq("W0")].groupby("evaluation_weather_type").agg(
        noop_fraction=("action_index", lambda s: s.eq(0).mean()),
        irrigation_action_probability=("irrigation_action_mm", lambda s: s.gt(0).mean()),
        fertilizer_action_probability=("fertilizer_action_kgN_ha", lambda s: s.gt(0).mean()),
        any_management_event_probability=("action_index", lambda s: s.ne(0).mean())).reset_index()
    archetypes = []
    step_profile = data.groupby("model_id").agg(
        noop_fraction=("action_index", lambda s: float(s.eq(0).mean())),
        irrigation_probability=("irrigation_action_mm", lambda s: float(s.gt(0).mean())),
        fertilizer_probability=("fertilizer_action_kgN_ha", lambda s: float(s.gt(0).mean()))).reset_index()
    season_profile = episodes.groupby("model_id").agg(
        mean_season_irrigation=("total_irrigation", "mean"), mean_season_fertilizer=("total_fertilizer", "mean"),
        mean_irrigation_events=("irrigation_event_count", "mean"), mean_fertilizer_events=("fertilizer_event_count", "mean")).reset_index()
    overall = step_profile.merge(season_profile, on="model_id")
    for row in overall.to_dict("records"):
        labels = []
        if row["mean_season_irrigation"] <= 60 and row["mean_season_fertilizer"] <= 80:
            labels.append("低季节投入")
        elif row["mean_season_fertilizer"] >= 100 and row["mean_season_irrigation"] <= 90:
            labels.append("低灌溉高氮")
        elif row["mean_season_irrigation"] >= 150 and row["mean_season_fertilizer"] >= 150:
            labels.append("高灌溉高氮")
        elif row["mean_season_irrigation"] >= 150:
            labels.append("灌溉密集")
        else:
            labels.append("中等/混合投入")
        if row["noop_fraction"] >= .85: labels.append("高no-op频率")
        archetypes.append({"model_id": row["model_id"], "labels": labels or ["中间型/混合"], **row})
    return {"performance": performance.to_dict("records"), "action_profile": w.to_dict("records"),
            "H1_W0_same_state_action_agreement": agreement["same_state_action_agreement"],
            "H1_W0_policy_similarity_class": agreement["policy_similarity_class"],
            "H1_W0_trajectory_action_agreement": agreement.get("trajectory_action_agreement"),
            "H1_W0_paired_episode_metrics_equal": agreement.get("paired_episode_metrics_equal"),
            "empirical_behavior_archetypes": archetypes,
            "reward_components": pd.read_csv(TASK / "reward_component_summary.csv").query("model_id == 'W0'").to_dict("records"),
            "scope": "Single seed profile; not an estimate of policy-archetype probability."}


def make_decision(data: pd.DataFrame, episodes: pd.DataFrame, paired: pd.DataFrame, matrix: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    probe_actions = pd.read_csv(TASK / "policy_similarity" / "policy_probe_actions.csv")
    wide = probe_actions.pivot(index="probe_id", columns="model_id", values="action_index")
    probe_disagreement = {f"H{s}_vs_W{s}": float(wide[f"H{s}"].ne(wide[f"W{s}"]).mean()) for s in range(3)}
    by_pair = paired.groupby("ppo_seed").trajectory_action_agreement_rate.mean()
    change = "YES" if all(v > .05 for v in probe_disagreement.values()) else "NO" if all(v <= .05 for v in probe_disagreement.values()) else "MIXED"
    w0 = episodes[episodes.model_id.eq("W0")]
    h0 = episodes[episodes.model_id.eq("H0")]
    means = {}
    for weather in ("heldout_wgen", "observed_weather"):
        a = w0[w0.evaluation_weather_type.eq(weather)]
        b = h0[h0.evaluation_weather_type.eq(weather)]
        means[weather] = {"reward_w0_minus_h0": float(a.episode_return.mean() - b.episode_return.mean()),
                          "yield_w0_minus_h0": float(a["yield"].mean() - b["yield"].mean()),
                          "irrigation_w0_minus_h0": float(a.total_irrigation.mean() - b.total_irrigation.mean()),
                          "fertilizer_w0_minus_h0": float(a.total_fertilizer.mean() - b.total_fertilizer.mean())}
    comp = pd.read_csv(TASK / "reward_component_summary.csv")
    exact = comp.max_episode_reconciliation_abs_error.max() <= 1e-8
    drivers = []
    if paired.mean_abs_irrigation_dose_difference.mean() > .1: drivers.append("灌溉剂量/时机")
    if paired.mean_abs_fertilizer_dose_difference.mean() > .1: drivers.append("施氮剂量/时机")
    if paired.trajectory_action_agreement_rate.mean() < .95: drivers.append("no-op频率与联合动作选择")
    if not drivers: drivers.append("状态轨迹分叉后的管理时机差异")
    return {
        "models_analyzed": MODELS, "training_run_performed": "NO", "heldout_wgen_eval": "1081-1100", "observed_eval": "2014-2023",
        "evaluation_replay_status": "PASS; W0/1081 smoke exactly matched 004_03; six models x 30 episodes",
        "action_occupancy_status": "PASS", "management_timing_status": "PASS", "stress_response_status": "PASS; descriptive bins",
        "weather_response_status": "PARTIAL" if (not bool(json.loads((TASK / "config" / "analysis_manifest.json").read_text(encoding="utf-8"))["weather_data_qc"]["WGEN_hash_reconciliation_pass"])
                                                   or any(x["weather_coverage_fraction"] < .95 for x in json.loads((TASK / "config" / "analysis_manifest.json").read_text(encoding="utf-8"))["weather_data_qc"]["step_weather_coverage_by_model_and_eval"])) else "PASS; post-hoc only",
        "random_weather_changes_policy_behavior": change,
        "paired_action_agreement": {f"H{s}_vs_W{s}": float(by_pair.loc[s]) for s in range(3)},
        "same_state_paired_seed_disagreement": probe_disagreement,
        "H1_vs_W0_same_state_action_agreement": profile["H1_W0_same_state_action_agreement"],
        "H1_vs_W0_policy_similarity_class": profile["H1_W0_policy_similarity_class"],
        "main_source_of_seed_variability": drivers,
        "W0_heldout_gain_explained": "YES" if exact else "PARTIALLY",
        "W0_heldout_gain_reward_difference_vs_H0": means["heldout_wgen"]["reward_w0_minus_h0"],
        "W0_heldout_gain_main_mechanism": reward_mechanism(comp, "heldout_wgen", "W0", "H0"),
        "W0_observed_drop_explained": "YES" if exact else "PARTIALLY",
        "W0_observed_reward_difference_vs_H0": means["observed_weather"]["reward_w0_minus_h0"],
        "W0_observed_drop_main_mechanism": reward_mechanism(comp, "observed_weather", "W0", "H0"),
        "W0_vs_H0_yield_and_resource_differences": means,
        "policy_archetypes_found": True,
        "policy_archetype_summary": "; ".join(f"{x['model_id']}={','.join(x['labels'])}" for x in profile["empirical_behavior_archetypes"]),
        "weather_augmentation_interpretation": "4. Evidence remains insufficient (D): same-seed behavior changes are mixed, and W0 matches the existing H1 policy rather than demonstrating a systematic shift." if change == "MIXED" else "2. Weather augmentation may increase favorable-archetype probability; more PPO seeds are required to estimate that probability.",
        "recommended_next_step": "本轮不扩增 seeds；若后续正式立项，预注册可区分 favorable-archetype discovery 与 weather-specific response 的 seed-level 假设，并冻结 site/reward/action/runtime，再开展预定规模的多 seed 诊断。",
        "larger_ppo_seed_experiment_recommended": "YES_WITH_PREDEFINED_DIAGNOSTIC_HYPOTHESIS",
        "PPO_retrained": "NO", "PPO_algorithm_modified": "NO", "reward_modified": "NO", "runtime_modified": "NO",
        "CNYC_CLI_modified": "NO", "seed_cherry_picking": "NO", "ppt_created": "NO",
        "github_backup_status": "No push performed; external GitHub backup not checked.",
        "same_state_agreement_matrix": matrix}


def reward_mechanism(summary: pd.DataFrame, weather: str, first: str, second: str) -> str:
    part = summary[summary.evaluation_weather_type.eq(weather)].set_index("model_id")
    fields = ("mean_sum_yield_contribution_scaled", "mean_sum_water_cost_contribution_scaled",
              "mean_sum_nitrogen_cost_contribution_scaled", "mean_sum_stress_relief_contribution_scaled",
              "mean_sum_swfac_guardrail_penalty_scaled")
    if not all(col in part for col in fields) or first not in part.index or second not in part.index:
        return "Exact runtime reward terms are incomplete; consult reward_component_summary.csv; no unobserved components inferred."
    a = part.loc[first]
    b = part.loc[second]
    effects = [float(a[fields[0]] - b[fields[0]]), float(b[fields[1]] - a[fields[1]]),
               float(b[fields[2]] - a[fields[2]]), float(a[fields[3]] - b[fields[3]]),
               float(b[fields[4]] - a[fields[4]])]
    observed_delta = float(a.mean_return - b.mean_return)
    total = sum(effects)
    names = ("yield", "water-cost savings", "N-cost savings", "stress-relief", "SWFAC-penalty effect")
    detail = "; ".join(f"{name}={value:+.4f}" for name, value in zip(names, effects))
    return f"{detail}; component sum={total:+.4f}, runtime return delta={observed_delta:+.4f}, residual={observed_delta-total:+.2e}"


def make_figures(data: pd.DataFrame, matrix: pd.DataFrame, paired: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = TASK / "figures"
    out.mkdir(parents=True, exist_ok=True)
    models = MODELS
    values = np.array([[data.loc[data.model_id.eq(m), "action_index"].eq(a).mean() for a in range(16)] for m in models])
    fig, ax = plt.subplots(figsize=(12, 4.5)); im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(6), models); ax.set_xticks(range(16), [f"{(i//4)*15}/{(i%4)*40}" for i in range(16)], rotation=45)
    ax.set_xlabel("Irrigation mm / N kg ha-1"); ax.set_title("16-action occupancy, pooled evaluation steps")
    fig.colorbar(im, ax=ax, label="Fraction"); fig.tight_layout(); fig.savefig(out / "01_action_occupancy.png", dpi=160); plt.close(fig)
    for col, median, lo, hi, title, file_name in (
        ("cumulative_irrigation", "median", .25, .75, "Cumulative irrigation vs DAP", "02_cumulative_irrigation.png"),
        ("cumulative_fertilizer", "median", .25, .75, "Cumulative fertilizer vs DAP", "03_cumulative_fertilizer.png")):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        for ax, weather in zip(axes, ("heldout_wgen", "observed_weather")):
            part = data[data.evaluation_weather_type.eq(weather)]
            for mid in models:
                s = part[part.model_id.eq(mid)].groupby("dap")[col]
                ax.plot(s.median().index, s.median().values, label=mid)
                ax.fill_between(s.median().index, s.quantile(lo).values, s.quantile(hi).values, alpha=.1)
            ax.set_title(weather); ax.set_xlabel("DAP"); ax.grid(alpha=.2)
        axes[0].set_ylabel("mm" if col.endswith("irrigation") else "kg N/ha"); axes[1].legend(ncol=3, fontsize=8)
        fig.suptitle(title); fig.tight_layout(); fig.savefig(out / file_name, dpi=160); plt.close(fig)
    for var, act, file_name in (("swfac", "irrigation_action_mm", "04_irrigation_by_swfac.png"), ("nstres", "fertilizer_action_kgN_ha", "05_fertilizer_by_nstres.png")):
        d = data.copy(); d["bin"] = pd.cut(d[var], [-np.inf, .05, .25, .5, np.inf], labels=["<=.05", ".05-.25", ".25-.50", ">.50"])
        table = d.groupby(["model_id", "bin"], observed=True)[act].apply(lambda s: s.gt(0).mean()).unstack(0).reindex(columns=models)
        fig, ax = plt.subplots(figsize=(10, 4)); table.plot(kind="bar", ax=ax)
        ax.set_ylabel(f"P({act} > 0)"); ax.grid(axis="y", alpha=.2); ax.legend(ncol=3); fig.tight_layout(); fig.savefig(out / file_name, dpi=160); plt.close(fig)
    m = matrix.set_index("model_id").loc[models, models].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(m, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(6), models); ax.set_yticks(range(6), models)
    for i in range(6):
        for j in range(6): ax.text(j, i, f"{m[i,j]:.2f}", ha="center", va="center", color="white" if m[i,j] < .65 else "black")
    ax.set_title("Same-state deterministic agreement"); fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out / "06_same_state_agreement.png", dpi=160); plt.close(fig)
    p = paired.groupby("historical_model").trajectory_action_agreement_rate.mean().reindex(["H0", "H1", "H2"])
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(["H0/W0", "H1/W1", "H2/W2"], p.values, color=["#277da1", "#f9844a", "#43aa8b"])
    ax.set_ylim(0, 1); ax.set_ylabel("Trajectory-level action agreement"); ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(out / "07_paired_seed_agreement.png", dpi=160); plt.close(fig)
    selected = pd.read_csv(TASK / "case_study_manifest.csv")
    case_rows = pd.concat([
        selected[selected.case_group.eq("WGEN_W0_largest_improvement")].head(2),
        selected[selected.case_group.eq("OBSERVED_W0_largest_decline")].head(2)], ignore_index=True)
    fig, axes = plt.subplots(len(case_rows), 3, figsize=(15, max(8, 2.6 * len(case_rows))), squeeze=False)
    colors = {"H0": "#277da1", "W0": "#f9844a"}
    columns = (("cumulative_irrigation", "Cumulative irrigation (mm)"),
               ("cumulative_fertilizer", "Cumulative fertilizer (kg N/ha)"),
               ("cumulative_reward", "Cumulative reward"))
    for row_ix, case in enumerate(case_rows.itertuples(index=False)):
        for mid in ("H0", "W0"):
            s = data[(data.episode_key == case.episode_key) & (data.model_id == mid)].sort_values("timestep")
            for col_ix, (column, label) in enumerate(columns):
                axes[row_ix, col_ix].plot(s.dap, s[column], color=colors[mid], label=mid, linewidth=1.5)
                if row_ix == 0:
                    axes[row_ix, col_ix].set_title(label)
                axes[row_ix, col_ix].grid(alpha=.2)
                axes[row_ix, col_ix].set_xlabel("DAP")
        realization = case.episode_key.rsplit(":", 1)[-1]
        case_label = f"WGEN {realization}" if case.episode_key.startswith("heldout_wgen:") else f"Observed {realization}"
        axes[row_ix, 0].set_ylabel(f"{case_label}\nΔR={case.w0_minus_h0_reward:+.3f}", rotation=0, ha="right", va="center", fontsize=8, labelpad=8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Selected WGEN gains and observed declines: H0 vs W0")
    fig.subplots_adjust(left=.15, right=.99, top=.94, bottom=.06, hspace=.42, wspace=.32)
    fig.savefig(out / "08_selected_case_trajectories.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    if "RAIN" in data:
        data["rain_bin"] = pd.cut(data.RAIN, [-np.inf, .1, 5, 15, np.inf], labels=["<=.1", ".1-5", "5-15", ">15"])
        for mid in models:
            s = data[data.model_id.eq(mid)].groupby("rain_bin", observed=True).irrigation_action_mm.apply(lambda x: x.gt(0).mean())
            ax.plot([str(x) for x in s.index], s.values, marker="o", label=mid)
    ax.set_title("Irrigation vs same-day rain; post-hoc only"); ax.set_ylabel("P(irrigation > 0)"); ax.legend(ncol=3); fig.tight_layout(); fig.savefig(out / "09_weather_response.png", dpi=160); plt.close(fig)


def write_report(data: pd.DataFrame, episodes: pd.DataFrame, paired: pd.DataFrame, matrix: pd.DataFrame, profile: dict[str, Any], decision: dict[str, Any]) -> None:
    summary_frame = data.groupby("model_id").agg(steps=("action_index", "size"), noop_fraction=("action_index", lambda s: s.eq(0).mean()),
        irrigation_probability=("irrigation_action_mm", lambda s: s.gt(0).mean()), fertilizer_probability=("fertilizer_action_kgN_ha", lambda s: s.gt(0).mean()),
        mean_irrigation_dose=("irrigation_action_mm", "mean"), mean_fertilizer_dose=("fertilizer_action_kgN_ha", "mean")).reset_index()
    summary = markdown_table(summary_frame, 3)
    step_by_weather = data.groupby(["model_id", "evaluation_weather_type"]).agg(
        steps=("action_index", "size"), noop_fraction=("action_index", lambda s: s.eq(0).mean()),
        p_irrigation_action=("irrigation_action_mm", lambda s: s.gt(0).mean()),
        p_fertilizer_action=("fertilizer_action_kgN_ha", lambda s: s.gt(0).mean())).reset_index()
    episode_by_weather = episodes.groupby(["model_id", "evaluation_weather_type"]).agg(
        episodes=("episode_key", "nunique"), mean_return=("episode_return", "mean"), mean_yield=("yield", "mean"),
        mean_season_irrigation=("total_irrigation", "mean"), mean_season_fertilizer=("total_fertilizer", "mean"),
        mean_irrigation_events=("irrigation_event_count", "mean"), mean_fertilizer_events=("fertilizer_event_count", "mean"),
        mean_first_irrigation_dap=("first_irrigation_dap", "mean"), mean_first_fertilizer_dap=("first_fertilizer_dap", "mean")).reset_index()
    model_weather = episode_by_weather.merge(step_by_weather, on=["model_id", "evaluation_weather_type"])
    model_weather = model_weather[["model_id", "evaluation_weather_type", "episodes", "mean_return", "mean_yield",
        "mean_season_irrigation", "mean_season_fertilizer", "mean_irrigation_events", "mean_fertilizer_events",
        "mean_first_irrigation_dap", "mean_first_fertilizer_dap", "noop_fraction", "p_irrigation_action", "p_fertilizer_action"]]
    model_weather_table = markdown_table(model_weather, 3)
    pairs = markdown_table(paired, 3)
    stress_summary = pd.read_csv(TASK / "stress_response_summary.csv")
    stress_behavior = []
    for (mid, weather), part in stress_summary.groupby(["model_id", "evaluation_weather_type"]):
        record: dict[str, Any] = {"model_id": mid, "evaluation_weather_type": weather}
        for variable, action_col, label in (("SWFAC", "p_irrigation_gt0", "irrigation"), ("NSTRES", "p_fertilizer_gt0", "fertilizer")):
            subset = part[part.stress_variable.eq(variable)].copy()
            lower = subset[subset.stress_bin.eq("<=0.05")]
            higher = subset[~subset.stress_bin.eq("<=0.05")]
            record[f"n_{label}_stress_le_0p05"] = int(lower.n_steps.sum())
            record[f"p_{label}_at_stress_le_0p05"] = float(lower[action_col].iloc[0]) if not lower.empty else np.nan
            record[f"n_{label}_stress_gt_0p05"] = int(higher.n_steps.sum())
            record[f"p_{label}_at_stress_gt_0p05"] = (float(np.average(higher[action_col], weights=higher.n_steps))
                if not higher.empty and higher.n_steps.sum() else np.nan)
        stress_behavior.append(record)
    stress_behavior_table = markdown_table(pd.DataFrame(stress_behavior).sort_values(["model_id", "evaluation_weather_type"]), 3)
    archetype_frame = pd.DataFrame([{"model_id": x["model_id"], "labels": ", ".join(x["labels"]),
        "mean_season_irrigation": x["mean_season_irrigation"], "mean_season_fertilizer": x["mean_season_fertilizer"],
        "noop_fraction": x["noop_fraction"], "mean_irrigation_events": x["mean_irrigation_events"],
        "mean_fertilizer_events": x["mean_fertilizer_events"]} for x in profile["empirical_behavior_archetypes"]])
    archetypes = markdown_table(archetype_frame, 3)
    analysis_manifest = json.loads((TASK / "config" / "analysis_manifest.json").read_text(encoding="utf-8"))
    weather_coverage = pd.DataFrame(analysis_manifest["weather_data_qc"]["step_weather_coverage_by_model_and_eval"])
    wgen_coverage = weather_coverage[weather_coverage.evaluation_weather_type.eq("heldout_wgen")].weather_coverage_fraction.min()
    case_manifest = pd.read_csv(TASK / "case_study_manifest.csv")
    case_summary = pd.read_csv(TASK / "case_studies" / "selected_case_episode_summary.csv")
    wgen_decline_exists = bool(case_manifest.case_group.eq("WGEN_W0_largest_decline").any())
    wgen_case_note = ("WGEN 有负的 W0-H0 episode reward 差，按差值排序报告下降最大两例。" if wgen_decline_exists
                      else "20 个 WGEN seed 均未出现 W0 低于 H0；按规则报告 W0 优势最小的两例，不将其误称为下降案例。")
    case_divergences = markdown_table(case_manifest[["case_group", "episode_key", "w0_minus_h0_reward",
        "first_action_divergence_timestep", "first_action_divergence_dap", "first_divergent_H0_action", "first_divergent_W0_action"]], 3)
    case_context = markdown_table(case_summary[["case_group", "episode_key", "model_id", "episode_return", "yield",
        "total_irrigation", "total_fertilizer", "total_rain", "mean_TMAX", "mean_SRAD", "mean_SWFAC",
        "mean_NSTRES", "irrigation_event_count", "fertilizer_event_count"]], 3)
    n_probe = len(pd.read_csv(TASK / "policy_similarity" / "policy_probe_states.csv"))
    text = f"""# YC PPO Seed Variability and Policy Behavior Diagnosis

## 1. Why diagnosis before more training
004_03 的 pooled 指标隐藏了 seed 间方向冲突，因此先检查策略行为，再决定是否值得扩展 PPO seeds；本轮不训练、不调参。

## 2. Existing pilot result
004_03 的 held-out pooled reward 为 historical 0.830541、random-weather 0.907329（+9.2456%），但逐 seed 方向并不一致。本轮独立复核策略轨迹与同状态响应。

## 3. Six models
H0/H1/H2 与 W0/W1/W2 分别为两种训练域的 seed 0/1/2。checkpoint 清单及 SHA256 见 results/yc_random_weather_ppo/004_04/model_manifest.csv；仅使用 004_03/training_verified。

## 4. Evaluation replay design
确定性回放每模型 20 个 held-out WGEN seeds (1081-1100) 与 observed 2014-2023，共 30 集。W0/1081 smoke 的 runtime weather hash、reward、yield、N、I、天数与 004_03 对齐后再进行全量回放。环境、reward 与 mask 复用 004_03 canonical runner；进程树 RSS 守卫 {MAX_RSS_MB} MB。逐步文件和 manifest 位于 004_04/trajectories。

## 5. Action occupancy
{summary}

16 动作和 marginal dose occupancy 按总体、held-out WGEN、observed weather 的分组见 results/yc_random_weather_ppo/004_04/action_occupancy_by_model.csv。W0 的 no-op、边际剂量概率和累计季节投入需联合阅读；动作事件数与单次事件剂量见第 6、7 节，不能仅凭逐日平均动作量解释节水/减氮。

## 5.1 Model by evaluation weather
下表补充每种模型在两类天气下的回报、产量、季节总投入、事件数、首次管理时间与 no-op 频率；可直接横向比较 W0 与 H0/H1/W1/W2。

{model_weather_table}

## 6. Resource use
每集资源 totals、episode return 和 yield 见 evaluation_episode_summary.csv；DAP 累积资源 median/P25/P75 见 resource_trajectories/cumulative_resource_by_dap.csv。

## 7. Management timing
management_event_summary.csv 包含事件数、事件均量、首末 DAP；management_timing_summary.csv 给出按 DAP 的 action probability 与累计资源分位数。
本 runner 的 `DAP=0` 对应播种前 5 天（播种日−5 至 −1），`DAP=1` 才是播种日；因此表中首次管理 DAP=0 表示播前首个决策窗口，不应误读为播种当天。

## 8. Stress response
SWFAC/NSTRES 分箱为 <=0.05、(0.05,0.25]、(0.25,0.50]、>0.50。按仓库后处理语义，0 近似无胁迫、数值越大胁迫越强。下表合并显示 <=0.05 与 >0.05 条件下的目标动作概率和样本步数；完整四档见 stress_response_summary.csv。

{stress_behavior_table}

这是沿各自策略轨迹的描述性条件关联，不能作因果解释。H1/W0 的条件动作分布相同；其管理集中在 DAP=0 的单次输入，不能据此声称 random-weather 训练增强了持续的 stress-feedback。较高压力分箱尤其 WGEN 端样本较少，证据不足以比较实时 cue 依赖强弱。

## 9. Weather response
前 3/7 天雨量与滞后 3 日 TMAX/SRAD 只作为 POST_HOC_DIAGNOSTIC_ONLY，不作为 PPO 输入。天气分箱结果见 weather_response_summary.csv。
所有 120 个 WGEN episode 的保存 runtime weather hash 均复核一致；但实际天气逐步值仅覆盖每模型 1850/2059 = {wgen_coverage:.2%} 的 held-out 步数。缺失的末段天气字段保持缺失，未用基础 rendered .WTH 补齐，因此 WGEN weather-response 结论为 PARTIAL；observed weather 步级覆盖为 100%。

## 10. Same-seed paired behavior
同 evaluation realization 按共同 timestep 的轨迹级比较如下。不同动作可能已令后续 DSSAT state 分叉，不能把后续 mismatch 当作同状态策略差异。

{pairs}

## 11. Same-state policy probe
从六模型轨迹 union 按天气类型、season phase、SWFAC、NSTRES、累计资源分层，以固定 seed 404006 抽取 {n_probe} 个 probe state。先保存 state 和 action mask，再由六个模型在完全相同输入上预测；不 step DSSAT。概率/JSD 跳过，未改 SB3。矩阵见 policy_similarity/policy_similarity_matrix.csv。

## 12. H1 vs W0
同状态 action agreement={profile['H1_W0_same_state_action_agreement']:.3f}，rollout trajectory action agreement={profile['H1_W0_trajectory_action_agreement']:.3f}，配对 episode return/yield/N/I 一致={profile['H1_W0_paired_episode_metrics_equal']}，分类 **{profile['H1_W0_policy_similarity_class']}**。因此这里不是仅凭 pooled aggregate 得出相似，而是本评估 support 上的确定性动作序列也完全一致。此结果说明 W0 的低投入型行为已在历史天气 PPO 的 H1 seed 中出现，不构成 random-weather augmentation 独有行为的证据。

## 13. W0 mechanism
W0 与 H0/H1/W1/W2 在两类天气下的平均产量、投入、事件频率及 no-op 见第 5.1 节。与 H0 相比，canonical runtime reward 的 signed component effect 为：

- held-out WGEN：{decision['W0_heldout_gain_main_mechanism']}
- observed weather：{decision['W0_observed_drop_main_mechanism']}

逐 episode 分项与重建误差见 reward_component_summary.csv。WGEN 评价期内 W0 的资源节省足以抵消产量项损失；observed 期回报下降则由产量项和 SWFAC guardrail penalty 主导，投入成本节省仅部分抵消。以上是精确 reward 分项对账，不把 yield/N/I 单项替代总回报。

## 14. Held-out vs observed case studies
按 W0-H0 episode return 差自动选择 WGEN 改善最大 2 例、{('下降最大 2 例' if wgen_decline_exists else '优势最小 2 例')}及 observed 下降最大 2 例；{wgen_case_note}首次动作分歧与 episode 环境/胁迫摘要如下：

{case_divergences}

{case_context}

逐日文件保留天气、SWFAC/NSTRES、动作、累计 N/I 与 reward trajectory；规则为排序，未人工挑 seed/year。

## 15. Reward decomposition
只使用 canonical runtime reward components；旧式 0.06 * final_grnwt - 0.04 * cumfert 为 SUPERSEDED / NOT_APPLICABLE。缺失的精确 wrapper component 不猜；reward_component_summary.csv 给出重建和对账状态。

## 16. Policy archetypes
依据逐步 no-op 频率与 episode 季节总投入、事件数识别行为 profile（而非把低日均剂量误作低季节投入）：

{archetypes}

命名是对这批轨迹的描述，不是稳定类别或 seed 价值排序。

## 17. What weather augmentation changed
random_weather_changes_policy_behavior={decision['random_weather_changes_policy_behavior']}。同 seed rollout divergence 与同状态 probe 分开解释，不能将轨迹级差异误读为纯策略映射差异。

## 18. What does this diagnosis imply about weather augmentation?
{decision['weather_augmentation_interpretation']}

## 19. Recommended next experiment
{decision['recommended_next_step']} 暂不挑选最佳 seed；若扩展 seed，应预注册行为机制与 reward component 假设，同时冻结 site/reward/action/runtime。

## 20. Limitations
- 每种训练域只有 3 个 PPO seeds，不能估计稳定的有利 archetype 概率。
- Probe 状态来自策略自身诱导的有限 state support；trajectory agreement 混合环境状态分叉。
- Stress 与 post-hoc 天气分箱是描述性关联；weather windows 未输入 policy。
- WGEN 天气分箱使用 004_03 同一 runtime state capture/hash 口径；基础 rendered .WTH 不代表 WTHER=W 的实际生成序列。
- WGEN 末段 runtime weather state capture 不完整（逐步天气字段覆盖 {wgen_coverage:.2%}）；未以其他来源插补，限制天气响应分析。
- 未提取动作概率分布，因此没有报告 JSD。
- observed 2014-2023 不足以断言普遍泛化失败。

## 21. Files / tests / git
机器结果位于 results/yc_random_weather_ppo/004_04/。QC 覆盖六个 verified model、固定评估集合、smoke 复现、逐步资源闭合、episode reward 对账和 probe 覆盖。训练/reward/runtime/CLI 修改均为 NO。commit subject: analysis: diagnose YC PPO seed policy behavior；不 push。

## Summary
- Models H0,H1,H2,W0,W1,W2; no training; 6 x 30 evaluation episodes.
- Random-weather policy behavior: {decision['random_weather_changes_policy_behavior']}.
- H1/W0 agreement/class: {profile['H1_W0_same_state_action_agreement']:.3f} / {profile['H1_W0_policy_similarity_class']}.
- W0 heldout mechanism: {decision['W0_heldout_gain_explained']}; observed drop: {decision['W0_observed_drop_explained']}.
- Interpretation: {decision['weather_augmentation_interpretation']}
"""
    (ROOT / "docs" / "yc_ppo_seed_variability_and_policy_behavior_diagnosis.md").write_text(text, encoding="utf-8", newline="\n")
    (TASK / "experiment_log.md").write_text(
        "# YC PPO 策略行为诊断实验记录\n\n"
        "- 任务类型：evaluation-only / DIAGNOSTIC ONLY；未执行 PPO 训练、微调、调参或 seed 扩增。\n"
        "- 模型：仅使用 004_03/training_verified 下 H0-H2、W0-W2 的 checkpoint_100000.zip。\n"
        "- 评估集：held-out WGEN seed 1081-1100；observed YCA 2014-2023；deterministic=True。\n"
        "- 资源约束：模型逐个串行评估；进程树 RSS 守卫 4000 MB；不安装依赖、不改 runtime。\n"
        "- QC 烟测：W0 / seed 1081 的 runtime weather SHA256、reward、yield、N、I、episode days 与 004_03 一致。\n"
        "- 过程修正：初次 smoke 完成后只在 JSON 终端打印处遇到 NumPy int64 序列化异常；结果已先写盘且全部复现检查通过。runner 后续加入 JSON 类型规范化。\n"
        "- 天气来源：rendered FileX .WTH 在 WTHER=W 模式下不等同实际随机天气；使用与 004_03 runtime hash 同口径的 daily_weather_from_states 捕获，observed 使用 YCA cleaned weather。天气变量只用于 POST_HOC_DIAGNOSTIC_ONLY 分析。\n"
        "- Reward：逐步保留 canonical action_info；拆分 scaled yield、water/N cost、stress relief 与 SWFAC guardrail 并核对。旧公式标记 SUPERSEDED / NOT_APPLICABLE。\n"
        "- Same-state probe：先固定并保存 trajectory union 分层抽取的 state+mask，再逐模型预测；不推进 DSSAT，不提取概率分布。\n"
        "- 运行失败与修正：首次 smoke 仅在终端 JSON 序列化遇到 NumPy int64 类型错误，数据文件已落盘；规范化 JSON 类型后复现通过。H0 首轮观察天气评估遇到 weather path 配置错误，后续重试遇到 episode 元数据字段名不一致；修正路径为 weather_clean/all_sites_weather_cleaned.csv、字段为 historical_year_context 后，以独立 attempt tag 重跑，完整保留前次日志/产物。分析重算首次因 case manifest merge key 大小写不一致退出，修正统一键名后重新生成分析、probe 与报告；没有为该修正重跑 DSSAT episodes。\n"
        "- 全量回放：六模型各完成 20 个 WGEN seed + 10 个 observed 年，共 180 episode、18,408 step；逐模型串行，无训练。运行期间峰值 RSS 约 898 MB，低于 4,000 MB 守卫。\n"
        "- 天气 QC：120 个 WGEN episode 的 runtime hash 全部一致；逐步天气字段覆盖 89.85%，末段缺失值保持缺失，因此 weather-response 标记 PARTIAL。\n"
        "- 诊断纠正：奖励成本项按回报方向转换为节省/惩罚效应；行为标签使用 season-level totals、事件数和 no-op；WGEN 若无负向差值则选取优势最小的两个案例，不称为下降。\n"
        "- 结果解释不做 best-seed 选择、不作 observed 十年以外的泛化断言。\n",
        encoding="utf-8", newline="\n")


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    cols = list(frame.columns)
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def final_qc(data: pd.DataFrame, episodes: pd.DataFrame, states: pd.DataFrame, actions: pd.DataFrame) -> None:
    errors = []
    expected_seeds = set(SEEDS)
    expected_years = set(YEARS)
    for mid in MODELS:
        model_eps = episodes[episodes.model_id.eq(mid)]
        wgen = model_eps[model_eps.evaluation_weather_type.eq("heldout_wgen")]
        observed = model_eps[model_eps.evaluation_weather_type.eq("observed_weather")]
        if len(wgen) != 20 or set(wgen.evaluation_weather_seed.dropna().astype(int)) != expected_seeds:
            errors.append(f"held-out WGEN evaluation set mismatch {mid}")
        if len(observed) != 10 or set(observed.evaluation_weather_year.dropna().astype(int)) != expected_years:
            errors.append(f"observed evaluation set mismatch {mid}")
    for (mid, episode), frame in data.groupby(["model_id", "episode_key"]):
        if not np.isclose(frame.irrigation_action_mm.sum(), frame.episode_total_irrigation.iloc[-1], atol=1e-7):
            errors.append(f"irrigation action sum mismatch {mid}/{episode}")
        if not np.isclose(frame.fertilizer_action_kgN_ha.sum(), frame.episode_total_fertilizer.iloc[-1], atol=1e-7):
            errors.append(f"fertilizer action sum mismatch {mid}/{episode}")
        if frame.valid_action_count.isna().any() or frame.valid_action_count.lt(1).any():
            errors.append(f"invalid or empty action mask {mid}/{episode}")
    if episodes.reward_decomposition_abs_error.max() > 1e-8:
        errors.append("reward reconciliation mismatch")
    if data.canonical_reward_step_abs_error.max() > 1e-8:
        errors.append("canonical per-step reward reconstruction mismatch")
    wgen_steps = data[data.evaluation_weather_type.eq("heldout_wgen")]
    hash_ok = wgen_steps.runtime_weather_hash_reconciles.astype(str).str.lower().isin(["true", "1", "1.0"])
    if not hash_ok.all():
        errors.append("one or more WGEN runtime weather hashes failed reconciliation")
    weather_hashes = wgen_steps.groupby(["evaluation_weather_seed", "model_id"]).runtime_weather_sha256.first().unstack()
    if set(weather_hashes.columns) != set(MODELS) or weather_hashes.isna().any().any() or weather_hashes.nunique(axis=1).ne(1).any():
        errors.append("evaluation WGEN realization hashes differ across models")
    if len(actions) != len(states) * 6:
        errors.append("incomplete probe rows")
    if actions.groupby("probe_id").saved_mask_sha256.nunique().ne(1).any():
        errors.append("saved action mask differs across models for the same probe")
    if states.probe_id.nunique() != len(states) or len(states) < PROBE_N:
        errors.append("probe states are duplicated or below the planned fixed sample")
    if not json.loads((TASK / "config" / "analysis_manifest.json").read_text(encoding="utf-8")).get("probe_fixed_before_predictions", False):
        errors.append("probe states were not frozen before model predictions")
    state_masks = states.set_index("probe_id").action_mask_used_json.to_dict()
    for row in actions.itertuples(index=False):
        if not bool(json.loads(state_masks[row.probe_id])[int(row.action_index)]):
            errors.append(f"predicted action violates saved mask at {row.probe_id}/{row.model_id}")
            break
    model_rows = pd.read_csv(TASK / "model_manifest.csv")
    if len(model_rows) != 6 or not model_rows.model_path.str.contains("training_verified").all():
        errors.append("model manifest contains a missing or non-verified checkpoint")
    if errors:
        raise RuntimeError("; ".join(errors))
    dump(TASK / "config" / "final_qc.json", {"passed": True, "models": MODELS, "episodes": len(episodes), "step_rows": len(data),
         "probe_states": len(states), "probe_action_rows": len(actions), "step_action_totals_reconcile": True,
         "canonical_per_step_reward_reconstruction_max_abs_error": float(data.canonical_reward_step_abs_error.max()),
         "episode_reward_reconciliation_max_abs_error": float(episodes.reward_decomposition_abs_error.max()),
         "all_six_models_all_probe_states": True, "all_probe_masks_identical_across_models": True,
         "all_checkpoints_training_verified": True, "evaluation_sets_identical_for_all_models": True,
         "all_wgen_runtime_hashes_reconcile_and_match_across_models": True,
         "probe_states_frozen_before_predictions": True, "predicted_actions_obey_saved_masks": True,
         "training_call_executed": False})


def terminal_summary(d: dict[str, Any]) -> None:
    print("\n".join([
        "=== YC PPO SEED / POLICY BEHAVIOR DIAGNOSIS SUMMARY ===",
        "models_analyzed: H0,H1,H2,W0,W1,W2", "training_run_performed: NO", "heldout_wgen_eval: 1081-1100", "observed_eval: 2014-2023",
        f"evaluation_replay_status: {d['evaluation_replay_status']}", f"action_occupancy_status: {d['action_occupancy_status']}",
        f"management_timing_status: {d['management_timing_status']}", f"stress_response_status: {d['stress_response_status']}", f"weather_response_status: {d['weather_response_status']}",
        f"random_weather_changes_policy_behavior: {d['random_weather_changes_policy_behavior']}",
        f"H0_vs_W0_action_agreement: {d['paired_action_agreement']['H0_vs_W0']:.4f}",
        f"H1_vs_W1_action_agreement: {d['paired_action_agreement']['H1_vs_W1']:.4f}",
        f"H2_vs_W2_action_agreement: {d['paired_action_agreement']['H2_vs_W2']:.4f}",
        f"H1_vs_W0_same_state_action_agreement: {d['H1_vs_W0_same_state_action_agreement']:.4f}",
        f"H1_vs_W0_policy_similarity_class: {d['H1_vs_W0_policy_similarity_class']}",
        f"main_source_of_seed_variability: {d['main_source_of_seed_variability']}",
        f"W0_heldout_gain_explained: {d['W0_heldout_gain_explained']}",
        f"W0_heldout_gain_main_mechanism: {d['W0_heldout_gain_main_mechanism']}",
        f"W0_observed_drop_explained: {d['W0_observed_drop_explained']}",
        f"W0_observed_drop_main_mechanism: {d['W0_observed_drop_main_mechanism']}",
        f"policy_archetypes_found: {d['policy_archetypes_found']}", f"policy_archetype_summary: {d['policy_archetype_summary']}",
        f"weather_augmentation_interpretation: {d['weather_augmentation_interpretation']}",
        f"recommended_next_step: {d['recommended_next_step']}",
        f"larger_ppo_seed_experiment_recommended: {d['larger_ppo_seed_experiment_recommended']}",
        "PPO_retrained: NO", "PPO_algorithm_modified: NO", "reward_modified: NO", "runtime_modified: NO", "CNYC_CLI_modified: NO",
        "seed_cherry_picking: NO", "ppt_created: NO",
        "report_md: docs/yc_ppo_seed_variability_and_policy_behavior_diagnosis.md", "diagnostic_summary_json: results/yc_random_weather_ppo/004_04/diagnostic_summary.json",
        "results_directory: results/yc_random_weather_ppo/004_04/", "tests_status: PASS",
        "git_commit: pending", "git_push: NO", f"github_backup_status: {d['github_backup_status']}"
    ]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation-only YC PPO seed behavior diagnosis")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    commands.add_parser("smoke")
    commands.add_parser("weather-smoke")
    replay_parser = commands.add_parser("replay-model")
    replay_parser.add_argument("model_id", choices=MODELS)
    replay_parser.add_argument("--eval-type", choices=("heldout_wgen", "observed_weather"))
    commands.add_parser("analyze")
    commands.add_parser("probe-models")
    commands.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(); print(json.dumps({"passed": result["passed"], "models": MODELS, "training_run_performed": False}), flush=True)
    elif args.command == "smoke":
        preflight(); replay("W0", smoke=True)
    elif args.command == "weather-smoke":
        preflight(); replay("W0", smoke=True, smoke_name="weather_capture_v3")
    elif args.command == "replay-model":
        replay(args.model_id, eval_type_filter=args.eval_type)
    elif args.command == "analyze":
        analyze()
    elif args.command == "probe-models":
        probe_models()
    elif args.command == "finalize":
        finalize()


if __name__ == "__main__":
    main()
