from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/yc_random_weather_ppo/004_09_observation_mask_action_audit"
REPORT = ROOT / "docs/yc_random_weather_004_09_observation_mask_action_audit.md"
FROZEN = ROOT / "results/yc_random_weather_ppo/004_05"
REPLAY = ROOT / "results/yc_random_weather_ppo/004_07_02_seed345_etcp_replay"
RUNNER_PATH = ROOT / "src/run_004_07_02_yc_seed345_etcp_replay.py"
PROMPT = ROOT / "prompt_02/004_09_yc_observation_mask_action_audit.md"
CONFIG = FROZEN / "config/canonical_yc_ppo_config.json"
YEARS = list(range(2014, 2024))
SEEDS = [3, 4, 5]
SMOKE_SEED, SMOKE_YEAR = 5, 2019
REGIME = "RANDOM_WEATHER_WGEN"
ACTION_DIM = 16
RSS_LIMIT_MB = 2048.0
PALETTE = {3: "#2A9D55", 4: "#2878B5", 5: "#D9822B"}
LINESTYLE = {3: ("-", "o"), 4: ("--", "s"), 5: ("-.", "^")}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite evidence: {rel(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def next_attempt_path(base: Path) -> Path:
    if not base.exists():
        return base
    attempt = 1
    while True:
        candidate = base.with_name(f"{base.stem}_retry_{attempt:02d}{base.suffix}")
        if not candidate.exists():
            return candidate
        attempt += 1


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def process_tree_rss_mb() -> float:
    import psutil

    process = psutil.Process()
    rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.Error:
            pass
    return rss / (1024 * 1024)


def load_runtime():
    runner = import_module(RUNNER_PATH, "yc_00409_replay_runner")
    runner.TASK_ROOT = OUT
    pilot = runner.load_runtime()
    pilot.MAX_PROCESS_TREE_RSS_MB = RSS_LIMIT_MB
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        raise
    return runner, pilot


def frozen_model_path(seed: int) -> Path:
    return FROZEN / "training/random_weather" / f"ppo_seed_{seed}/models/checkpoint_100000.zip"


def load_reference(runner, seed: int) -> tuple[dict[int, pd.DataFrame], dict[int, dict[str, Any]]]:
    _, episode_path, step_path, _ = runner.frozen_paths(seed, YEARS[0])
    episodes = pd.read_csv(episode_path, keep_default_na=False)
    episodes = episodes[
        episodes.training_regime.eq(REGIME)
        & episodes.evaluation_weather_type.eq("observed_weather")
        & pd.to_numeric(episodes.ppo_seed, errors="coerce").eq(seed)
        & pd.to_numeric(episodes.evaluation_weather_year, errors="coerce").isin(YEARS)
    ].copy()
    if len(episodes) != len(YEARS) or episodes.evaluation_weather_year.astype(int).nunique() != len(YEARS):
        raise RuntimeError(f"Frozen episode coverage incomplete for seed {seed}: {len(episodes)}")
    required = [
        "training_regime", "ppo_seed", "evaluation_weather_type", "evaluation_weather_year", "timestep",
        "dap", "action_index", "irrigation_action_mm", "fertilizer_action_kgN_ha", "instant_reward",
    ]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(step_path, usecols=required, chunksize=50000):
        selected = chunk[
            chunk.training_regime.eq(REGIME)
            & chunk.evaluation_weather_type.eq("observed_weather")
            & pd.to_numeric(chunk.ppo_seed, errors="coerce").eq(seed)
            & pd.to_numeric(chunk.evaluation_weather_year, errors="coerce").isin(YEARS)
        ].copy()
        if len(selected):
            parts.append(selected)
    if not parts:
        raise RuntimeError(f"No frozen step trace found for seed {seed}")
    steps = pd.concat(parts, ignore_index=True)
    for column in ("evaluation_weather_year", "timestep", "dap", "action_index", "irrigation_action_mm", "fertilizer_action_kgN_ha", "instant_reward"):
        steps[column] = pd.to_numeric(steps[column], errors="coerce")
    steps = steps.sort_values(["evaluation_weather_year", "timestep"]).reset_index(drop=True)
    by_year: dict[int, pd.DataFrame] = {}
    ep_by_year: dict[int, dict[str, Any]] = {}
    for year in YEARS:
        group = steps[steps.evaluation_weather_year.eq(year)].copy().reset_index(drop=True)
        ep = episodes[pd.to_numeric(episodes.evaluation_weather_year, errors="coerce").eq(year)]
        if len(ep) != 1 or len(group) != int(float(ep.iloc[0].episode_days)):
            raise RuntimeError(f"Frozen step/episode closure failed for seed={seed} year={year}")
        if group.timestep.duplicated().any() or not np.array_equal(group.timestep.to_numpy(), np.arange(1, len(group) + 1)):
            raise RuntimeError(f"Unexpected timestep index for seed={seed} year={year}")
        by_year[year] = group
        ep_by_year[year] = ep.iloc[0].to_dict()
    return by_year, ep_by_year


def schema_for_observation(pilot, env, observation: Any) -> list[dict[str, Any]]:
    raw = pilot.find_dssat_instance(env.current_env)
    variables = list(getattr(raw, "observation_variables", []) or [])
    state = getattr(raw, "observation", None)
    values: list[tuple[str, int, float]] = []
    for variable in variables:
        if not isinstance(state, dict) or variable not in state:
            raise RuntimeError(f"Cannot resolve observation field from raw environment: {variable}")
        arr = np.asarray(state[variable]).reshape(-1)
        if arr.size == 0:
            raise RuntimeError(f"Empty raw observation field: {variable}")
        for index, value in enumerate(arr):
            numeric = float(value)
            values.append((str(variable), index, numeric))
    obs = np.asarray(observation).reshape(-1)
    if len(values) != obs.size:
        raise RuntimeError(f"Observation schema mismatch: expanded raw fields={len(values)} vector={obs.size}")
    schema = []
    for dim, (variable, index, initial_value) in enumerate(values):
        display = variable if sum(1 for row in values if row[0] == variable) == 1 else f"{variable}[{index}]"
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", display).strip("_").lower() or f"dim_{dim:02d}"
        schema.append({"dimension": dim, "field": variable, "field_index": index, "display_name": display, "column_suffix": safe})
    if len({row["column_suffix"] for row in schema}) != len(schema):
        for row in schema:
            row["column_suffix"] = f"{row['column_suffix']}__d{row['dimension']:02d}"
    return schema


def cat_arrays(distribution) -> tuple[np.ndarray, np.ndarray]:
    categorical = getattr(distribution, "distribution", None)
    if categorical is None or not hasattr(categorical, "probs") or not hasattr(categorical, "logits"):
        raise RuntimeError(f"Unexpected MaskablePPO action distribution: {type(distribution).__name__}")
    logits = categorical.logits.detach().cpu().numpy().reshape(-1).astype(float)
    probabilities = categorical.probs.detach().cpu().numpy().reshape(-1).astype(float)
    return logits, probabilities


def entropy_bits(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def persist_episode_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite episode trace: {rel(path)}")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def run_episode(runner, pilot, seed: int, year: int, expected_steps: pd.DataFrame, expected_episode: dict[str, Any], expected_schema: list[dict[str, Any]] | None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    from sb3_contrib import MaskablePPO

    model_path = frozen_model_path(seed)
    run_root = OUT / "evaluation/runtime" / f"seed_{seed}" / f"YCA_{year}"
    trace_path = OUT / "episodes" / f"decision_trace_seed{seed}_{year}.csv"
    if run_root.exists() or trace_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior episode output: {rel(run_root)} or {rel(trace_path)}")
    run_root.mkdir(parents=True)
    config, env_config = pilot.make_run_config(REGIME, seed, run_root)
    schedule = [{"episode_index": 1, "historical_year": year, "weather_seed": ""}]
    run_tag = f"eval_observed_weather_RANDOM_WEATHER_WGEN_{seed}"
    env = pilot.ScheduledEpisodeEnv(
        config, env_config, schedule, "HISTORICAL_WEATHER", seed, run_tag,
        pilot.EVAL_RUNTIME_SEED, max_cached_years=1,
    )
    model = None
    rows: list[dict[str, Any]] = []
    rss_peak = process_tree_rss_mb()
    total_reward = 0.0
    try:
        model = MaskablePPO.load(str(model_path), device="cpu")
        observation, _ = env._gym_env.reset(seed=None)
        schema = schema_for_observation(pilot, env, observation)
        if expected_schema is not None and schema != expected_schema:
            raise RuntimeError(f"Observation schema drift at seed={seed} year={year}")
        obs_columns = [f"obs__{row['column_suffix']}" for row in schema]
        input_columns = [f"input__{row['column_suffix']}" for row in schema]
        if len(expected_steps) != int(expected_episode["episode_days"]):
            raise RuntimeError("Formal trace step count changed before replay")

        for offset, formal in enumerate(expected_steps.itertuples(index=False)):
            obs_np = np.asarray(observation).reshape(-1).astype(np.float32, copy=False)
            if len(obs_np) != len(schema):
                raise RuntimeError(f"Observation dimension changed at timestep={offset + 1}")
            masks = np.asarray(env._gym_env.action_masks(), dtype=bool).reshape(-1)
            if masks.size != ACTION_DIM or not masks.any():
                raise RuntimeError(f"Invalid action mask shape/content at timestep={offset + 1}: {masks.shape}")
            obs_tensor, _ = model.policy.obs_to_tensor(observation)
            with __import__("torch").no_grad():
                features = model.policy.extract_features(obs_tensor, model.policy.pi_features_extractor)
                latent_pi = model.policy.mlp_extractor.forward_actor(features)
                actor_logits = model.policy.action_net(latent_pi).detach().cpu().numpy().reshape(-1).astype(float)
                unmasked = model.policy.get_distribution(obs_tensor)
                _, raw_probs = cat_arrays(unmasked)
                masked = model.policy.get_distribution(obs_tensor, action_masks=masks)
                _, masked_probs = cat_arrays(masked)
            raw_logits = actor_logits
            masked_logits = np.where(masks, raw_logits, -1e8)
            if any(array.size != ACTION_DIM for array in (raw_logits, raw_probs, masked_logits, masked_probs)):
                raise RuntimeError("Policy output dimension does not match 16-action contract")
            invalid_probs = masked_probs[~masks]
            if (invalid_probs.size and np.max(np.abs(invalid_probs)) > 1e-7) or not np.isclose(masked_probs.sum(), 1.0, atol=1e-6):
                raise RuntimeError(f"Masked probabilities invalid at timestep={offset + 1}")
            stable_raw = np.exp(raw_logits - np.max(raw_logits))
            stable_raw /= stable_raw.sum()
            stable_masked = np.exp(masked_logits - np.max(masked_logits))
            stable_masked /= stable_masked.sum()
            if not np.allclose(stable_raw, raw_probs, rtol=1e-5, atol=1e-7) or not np.allclose(stable_masked, masked_probs, rtol=1e-5, atol=1e-7):
                raise RuntimeError(f"Captured actor logits do not reproduce distribution probabilities at timestep={offset + 1}")
            predicted, _ = model.predict(observation, deterministic=True, action_masks=masks)
            action = int(np.asarray(predicted).reshape(-1)[0])
            masked_argmax = int(masked_probs.argmax())
            if action != masked_argmax or not bool(masks[action]):
                raise RuntimeError(f"Predict/masked-distribution action mismatch at timestep={offset + 1}: {action}/{masked_argmax}")
            formal_action = int(formal.action_index)
            if action != formal_action:
                raise RuntimeError(f"Formal action mismatch seed={seed} year={year} timestep={offset + 1}: {action} != {formal_action}")
            if int(formal.timestep) != offset + 1:
                raise RuntimeError(f"Unexpected reference timestep at replay offset {offset}")

            raw_order = np.argsort(raw_logits)[::-1]
            legal_ids = np.flatnonzero(masks)
            legal_order = legal_ids[np.argsort(masked_logits[legal_ids])[::-1]]
            raw_top = int(raw_order[0])
            masked_top = int(legal_order[0])
            has_legal_alternative = len(legal_order) > 1
            masked_second = int(legal_order[1]) if has_legal_alternative else None
            tensor_np = obs_tensor.detach().cpu().numpy().reshape(-1).astype(float)
            if tensor_np.size != obs_np.size or not np.allclose(tensor_np, obs_np, rtol=0, atol=1e-7, equal_nan=True):
                raise RuntimeError("Raw observation differs from actual policy input tensor")
            row = {
                "seed": seed, "year": year, "timestep": int(formal.timestep), "dap": float(formal.dap),
                "date": "", "doy": np.nan, "calendar_date_available": False, "action_index": action,
                "formal_action_index": formal_action, "action_matches_formal": True,
                "formal_irrigation_action_mm": float(formal.irrigation_action_mm),
                "formal_fertilizer_action_kgN_ha": float(formal.fertilizer_action_kgN_ha),
                "mask_legal_count": int(masks.sum()), "mask_json": json.dumps(masks.astype(int).tolist()),
                "raw_argmax_action": raw_top, "raw_argmax_masked_out": not bool(masks[raw_top]),
                "masked_argmax_action": masked_top, "model_predict_action": action,
                "predict_distribution_action_match": action == masked_top,
                "chosen_masked_probability": float(masked_probs[action]),
                "masked_top1_probability": float(masked_probs[masked_top]),
                "masked_top2_probability": float(masked_probs[masked_second]) if has_legal_alternative else np.nan,
                "masked_logit_margin": float(masked_logits[masked_top] - masked_logits[masked_second]) if has_legal_alternative else np.nan,
                "masked_probability_margin": float(masked_probs[masked_top] - masked_probs[masked_second]) if has_legal_alternative else np.nan,
                "raw_entropy_bits": entropy_bits(raw_probs), "masked_entropy_bits": entropy_bits(masked_probs),
                "raw_logits_json": json.dumps(raw_logits.tolist(), separators=(",", ":")),
                "raw_probabilities_json": json.dumps(raw_probs.tolist(), separators=(",", ":")),
                "masked_logits_json": json.dumps(masked_logits.tolist(), separators=(",", ":")),
                "masked_probabilities_json": json.dumps(masked_probs.tolist(), separators=(",", ":")),
                "raw_observation_json": json.dumps(obs_np.astype(float).tolist(), separators=(",", ":")),
                "policy_input_json": json.dumps(tensor_np.tolist(), separators=(",", ":")),
                "raw_equals_policy_input": True,
                "step_reward": None,
            }
            row.update({column: float(value) for column, value in zip(obs_columns, obs_np)})
            row.update({column: float(value) for column, value in zip(input_columns, tensor_np)})
            row.update({f"mask__a{index:02d}": int(value) for index, value in enumerate(masks)})
            row.update({f"raw_logit__a{index:02d}": float(value) for index, value in enumerate(raw_logits)})
            row.update({f"raw_prob__a{index:02d}": float(value) for index, value in enumerate(raw_probs)})
            row.update({f"masked_logit__a{index:02d}": float(value) for index, value in enumerate(masked_logits)})
            row.update({f"masked_prob__a{index:02d}": float(value) for index, value in enumerate(masked_probs)})

            pre_state = dict(getattr(env.current_env, "last_obs_dict", {}) or {})
            runtime_dap = pre_state.get("dap")
            if runtime_dap is not None and math.isfinite(float(runtime_dap)) and abs(float(runtime_dap) - float(formal.dap)) > 1e-6:
                raise RuntimeError(f"Runtime pre-action DAP differs from formal trace at timestep={offset + 1}")
            row["runtime_pre_action_dap"] = float(runtime_dap) if runtime_dap is not None and math.isfinite(float(runtime_dap)) else np.nan
            next_observation, reward, terminated, truncated, _ = env._gym_env.step(action)
            action_info = dict(getattr(env.current_env, "last_action_info", {}) or {})
            if "safe_action_amir" not in action_info or "safe_action_anfer" not in action_info:
                raise RuntimeError(f"Applied action amounts unavailable at timestep={offset + 1}")
            actual_irrigation = float(action_info["safe_action_amir"] or 0.0)
            actual_nitrogen = float(action_info["safe_action_anfer"] or 0.0)
            if abs(actual_irrigation - float(formal.irrigation_action_mm)) > 1e-8 or abs(actual_nitrogen - float(formal.fertilizer_action_kgN_ha)) > 1e-8:
                raise RuntimeError(f"Decoded applied amount differs from formal trace at timestep={offset + 1}")
            row["irrigation_action_mm"] = actual_irrigation
            row["fertilizer_action_kgN_ha"] = actual_nitrogen
            row["mask_forced_noop"] = bool(action_info.get("mask_forced_noop", False))
            row["step_reward"] = float(reward)
            row["formal_step_reward"] = float(formal.instant_reward)
            row["step_reward_abs_error"] = abs(float(reward) - float(formal.instant_reward))
            total_reward += float(reward)
            rows.append(row)
            observation = next_observation
            if (offset + 1) % 10 == 0:
                rss_now = process_tree_rss_mb()
                rss_peak = max(rss_peak, rss_now)
                print(f"[rss] seed={seed} year={year} step={offset + 1} tree_mb={rss_now:.1f}", flush=True)
                if rss_now >= RSS_LIMIT_MB:
                    raise MemoryError(f"RSS reached {RSS_LIMIT_MB:.0f} MiB; stopped replay")
            if terminated or truncated:
                if offset + 1 != len(expected_steps):
                    raise RuntimeError(f"Replay terminated early at step {offset + 1}/{len(expected_steps)}")
                break
        persist_episode_trace(trace_path, rows)
        if len(rows) != len(expected_steps):
            raise RuntimeError(f"Replay step count mismatch: {len(rows)} != {len(expected_steps)}")
        replay_episode = dict(env.episode_rows[-1])
        if not bool(pd.Series([r["predict_distribution_action_match"] for r in rows]).all()):
            raise RuntimeError("Not all selected actions agree with the masked distribution")
        if not np.allclose([r["dap"] for r in rows], expected_steps.dap.to_numpy(float), rtol=0, atol=1e-8):
            raise RuntimeError("DAP sequence changed relative to frozen trace")
        irrigation = float(sum(row["irrigation_action_mm"] for row in rows))
        nitrogen = float(sum(row["fertilizer_action_kgN_ha"] for row in rows))
        yield_value = float(replay_episode["yield"])
        expected_yield = float(expected_episode["yield"])
        expected_irrigation = float(expected_episode["irrigation"])
        expected_nitrogen = float(expected_episode["fertilizer"])
        reward_error = abs(total_reward - float(expected_episode["reward"]))
        gates = {
            "schema_dimension_closed": len(obs_columns) == len(input_columns) == len(schema) == len(np.asarray(observation).reshape(-1)),
            "applied_action_amounts_match_formal": all(abs(row["irrigation_action_mm"] - float(formal.irrigation_action_mm)) <= 1e-8 and abs(row["fertilizer_action_kgN_ha"] - float(formal.fertilizer_action_kgN_ha)) <= 1e-8 for row, formal in zip(rows, expected_steps.itertuples(index=False))),
            "action_trace_exact": all(row["action_index"] == int(formal_action) for row, formal_action in zip(rows, expected_steps.action_index)),
            "dap_trace_exact": np.allclose([r["dap"] for r in rows], expected_steps.dap.to_numpy(float), atol=1e-8, rtol=0),
            "predict_equals_masked_argmax": all(row["predict_distribution_action_match"] for row in rows),
            "yield_within_1kg_ha": abs(yield_value - expected_yield) <= 1.0,
            "irrigation_within_1mm": abs(irrigation - expected_irrigation) <= 1.0,
            "nitrogen_within_1kg_ha": abs(nitrogen - expected_nitrogen) <= 1.0,
            "canonical_reward_within_1e-4": reward_error <= 1e-4,
            "step_rewards_within_1e-6": max(row["step_reward_abs_error"] for row in rows) <= 1e-6,
            "runtime_pre_action_dap_matches": all(not math.isfinite(row["runtime_pre_action_dap"]) or abs(row["runtime_pre_action_dap"] - row["dap"]) <= 1e-6 for row in rows),
            "rss_below_2048_mib": rss_peak < RSS_LIMIT_MB,
        }
        input_provenance = runner.check_rendered_inputs(seed, year)
        gates["observed_weather_and_filex_hashes_match"] = all(item["identical"] for item in input_provenance["checks"].values())
        gates = {key: bool(value) for key, value in gates.items()}
        if not all(gates.values()):
            raise RuntimeError(f"Replay reconciliation gates failed seed={seed} year={year}: {gates}")
        summary = {
            "seed": seed, "year": year, "checkpoint": 100000, "model_path": rel(model_path),
            "model_sha256": sha256_file(model_path), "steps": len(rows),
            "formal_yield_kg_ha": expected_yield, "replay_yield_kg_ha": yield_value,
            "yield_abs_error_kg_ha": abs(yield_value - expected_yield),
            "formal_irrigation_mm": expected_irrigation, "replay_irrigation_mm": irrigation,
            "irrigation_abs_error_mm": abs(irrigation - expected_irrigation),
            "formal_nitrogen_kg_ha": expected_nitrogen, "replay_nitrogen_kg_ha": nitrogen,
            "nitrogen_abs_error_kg_ha": abs(nitrogen - expected_nitrogen),
            "formal_reward": float(expected_episode["reward"]), "replay_reward": total_reward,
            "reward_abs_error": reward_error, "raw_argmax_masked_out_steps": int(sum(row["raw_argmax_masked_out"] for row in rows)),
            "mean_legal_action_count": float(np.mean([row["mask_legal_count"] for row in rows])),
            "mean_masked_entropy_bits": float(np.mean([row["masked_entropy_bits"] for row in rows])),
            "action_signature": ";".join(
                f"{int(row['dap'])}:a{row['action_index']}:I{row['irrigation_action_mm']:g}/N{row['fertilizer_action_kgN_ha']:g}"
                for row in rows if row["irrigation_action_mm"] > 0 or row["fertilizer_action_kgN_ha"] > 0
            ),
            "raw_action_trace_sha256": hashlib.sha256(json.dumps([row["action_index"] for row in rows]).encode()).hexdigest().upper(),
            "observation_dimensions": len(schema), "observation_schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest().upper(),
            "rendered_input_provenance": input_provenance, "rss_peak_mb": rss_peak,
            "trace_path": rel(trace_path), "status": "passed", "gates": gates,
        }
        return summary, rows, schema
    except Exception:
        if rows and not trace_path.exists():
            persist_episode_trace(trace_path, rows)
        raise
    finally:
        try:
            env.close()
        finally:
            if model is not None:
                del model
            gc.collect()


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    p = np.asarray(left, dtype=float)
    q = np.asarray(right, dtype=float)
    p = p / max(p.sum(), 1e-300)
    q = q / max(q.sum(), 1e-300)
    middle = 0.5 * (p + q)
    def kl(a, b):
        keep = a > 0
        return float(np.sum(a[keep] * np.log2(a[keep] / b[keep])))
    return max(0.0, 0.5 * kl(p, middle) + 0.5 * kl(q, middle))


def pairwise_values(matrix: np.ndarray, metric) -> list[float]:
    return [float(metric(matrix[left], matrix[right])) for left, right in itertools.combinations(range(len(matrix)), 2)]


def summarize_decisions(trace: pd.DataFrame, schema: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    obs_cols = [f"obs__{row['column_suffix']}" for row in schema]
    raw_prob_cols = [f"raw_prob__a{i:02d}" for i in range(ACTION_DIM)]
    masked_prob_cols = [f"masked_prob__a{i:02d}" for i in range(ACTION_DIM)]
    mask_cols = [f"mask__a{i:02d}" for i in range(ACTION_DIM)]

    common_keys: dict[int, list[tuple[int, int]]] = {}
    for seed in SEEDS:
        per_year = []
        for year in YEARS:
            part = trace.loc[trace.seed.eq(seed) & trace.year.eq(year)]
            per_year.append(set(zip(part.dap.round().astype(int), part.dap_occurrence.astype(int))))
        common = sorted(set.intersection(*per_year)) if per_year else []
        common_keys[seed] = common

    global_scales: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        values = trace.loc[trace.seed.eq(seed), obs_cols].to_numpy(float)
        scales = np.nanstd(values, axis=0)
        scales[~np.isfinite(scales) | (scales < 1e-8)] = 1.0
        global_scales[seed] = scales

    observation_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for dap, occurrence in common_keys[seed]:
            group = trace.loc[
                trace.seed.eq(seed) & trace.dap.round().eq(dap) & trace.dap_occurrence.eq(occurrence)
            ].sort_values("year")
            if group.year.nunique() != len(YEARS) or len(group) != len(YEARS):
                continue
            for index, column in enumerate(obs_cols):
                vals = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
                finite = vals[np.isfinite(vals)]
                scale = global_scales[seed][index]
                observation_rows.append({
                    "seed": seed, "dap": dap, "dap_occurrence": occurrence,
                    "feature": schema[index]["display_name"], "feature_column": column,
                    "n_years": int(len(finite)), "mean": float(np.mean(finite)) if len(finite) else np.nan,
                    "std": float(np.std(finite, ddof=0)) if len(finite) else np.nan,
                    "min": float(np.min(finite)) if len(finite) else np.nan,
                    "max": float(np.max(finite)) if len(finite) else np.nan,
                    "range": float(np.ptp(finite)) if len(finite) else np.nan,
                    "range_in_global_sd": float(np.ptp(finite) / scale) if len(finite) else np.nan,
                    "values_by_year_json": json.dumps({str(int(year)): float(value) for year, value in group[["year", column]].itertuples(index=False, name=None) if np.isfinite(float(value))}, sort_keys=True),
                })
            vectors = group[obs_cols].to_numpy(float) / global_scales[seed]
            obs_dist = pairwise_values(vectors, lambda a, b: float(np.sqrt(np.nanmean(np.square(a - b)))))
            masks = group[mask_cols].to_numpy(bool)
            raw_probs = group[raw_prob_cols].to_numpy(float)
            masked_probs = group[masked_prob_cols].to_numpy(float)
            actions = group.action_index.to_numpy(int)
            raw_argmax = group.raw_argmax_action.to_numpy(int)
            modal_count = int(pd.Series(actions).value_counts().max())
            mask_hamming = pairwise_values(masks.astype(float), lambda a, b: float(np.mean(a != b)))
            mask_jaccard = pairwise_values(masks.astype(bool), lambda a, b: float((a & b).sum() / (a | b).sum()) if (a | b).any() else 1.0)
            raw_js = pairwise_values(raw_probs, js_divergence)
            masked_js = pairwise_values(masked_probs, js_divergence)
            logit_margins = pd.to_numeric(group.masked_logit_margin, errors="coerce").dropna()
            probability_margins = pd.to_numeric(group.masked_probability_margin, errors="coerce").dropna()
            same_action = pd.Series(actions).nunique() == 1
            distribution_changed = bool(np.mean(masked_js) >= 0.01)
            near_boundary = bool(
                (not logit_margins.empty and logit_margins.min() <= 0.25)
                or (not probability_margins.empty and probability_margins.min() <= 0.05)
            )
            policy_rows.append({
                "seed": seed, "dap": dap, "dap_occurrence": occurrence,
                "n_years": int(group.year.nunique()), "n_decisions": int(len(group)),
                "action_unique_count": int(pd.Series(actions).nunique()), "action_modal_share": modal_count / len(actions),
                "actions_by_year_json": json.dumps({str(int(row.year)): int(row.action_index) for row in group.itertuples()}, sort_keys=True),
                "raw_argmax_unique_count": int(pd.Series(raw_argmax).nunique()),
                "raw_argmax_masked_out_rate": float(group.raw_argmax_masked_out.mean()),
                "choice_opportunity_years": int(group.mask_legal_count.gt(1).sum()),
                "raw_argmax_masked_out_rate_when_alternatives": float(group.loc[group.mask_legal_count.gt(1), "raw_argmax_masked_out"].mean()) if group.mask_legal_count.gt(1).any() else np.nan,
                "forced_single_action_share": float(group.mask_legal_count.eq(1).mean()),
                "legal_action_count_mean": float(group.mask_legal_count.mean()),
                "legal_action_count_min": int(group.mask_legal_count.min()),
                "mask_unique_count": int(pd.DataFrame(masks).drop_duplicates().shape[0]),
                "mask_pairwise_hamming_mean": float(np.mean(mask_hamming)),
                "mask_pairwise_jaccard_mean": float(np.mean(mask_jaccard)),
                "observation_pairwise_rms_z_mean": float(np.mean(obs_dist)),
                "observation_pairwise_rms_z_max": float(np.max(obs_dist)),
                "observation_unique_vector_count_1e-6": int(group[obs_cols].round(6).drop_duplicates().shape[0]),
                "raw_policy_pairwise_js_mean": float(np.mean(raw_js)),
                "masked_policy_pairwise_js_mean": float(np.mean(masked_js)),
                "chosen_probability_mean": float(group.chosen_masked_probability.mean()),
                "chosen_probability_std": float(group.chosen_masked_probability.std(ddof=0)),
                "chosen_probability_range": float(group.chosen_masked_probability.max() - group.chosen_masked_probability.min()),
                "masked_logit_margin_min": float(logit_margins.min()) if not logit_margins.empty else np.nan,
                "masked_logit_margin_mean": float(logit_margins.mean()) if not logit_margins.empty else np.nan,
                "masked_logit_margin_range": float(logit_margins.max() - logit_margins.min()) if not logit_margins.empty else np.nan,
                "masked_probability_margin_min": float(probability_margins.min()) if not probability_margins.empty else np.nan,
                "masked_probability_margin_range": float(probability_margins.max() - probability_margins.min()) if not probability_margins.empty else np.nan,
                "same_final_action_but_policy_distribution_changed": bool(same_action and distribution_changed and group.mask_legal_count.min() > 1),
                "same_final_action_near_boundary": bool(same_action and distribution_changed and group.mask_legal_count.min() > 1 and near_boundary),
            })

    observation_summary = pd.DataFrame(observation_rows)
    policy_response = pd.DataFrame(policy_rows)
    annual_rows: list[dict[str, Any]] = []
    for (seed, year), group in trace.groupby(["seed", "year"], sort=True):
        annual_rows.append({
            "seed": int(seed), "year": int(year), "steps": int(len(group)),
            "action_unique_count": int(group.action_index.nunique()),
            "raw_argmax_masked_out_steps": int(group.raw_argmax_masked_out.sum()),
            "raw_argmax_masked_out_rate": float(group.raw_argmax_masked_out.mean()),
            "choice_opportunity_steps": int(group.mask_legal_count.gt(1).sum()),
            "raw_argmax_masked_out_rate_when_alternatives": float(group.loc[group.mask_legal_count.gt(1), "raw_argmax_masked_out"].mean()) if group.mask_legal_count.gt(1).any() else np.nan,
            "forced_single_action_steps": int(group.mask_legal_count.eq(1).sum()),
            "forced_single_action_share": float(group.mask_legal_count.eq(1).mean()),
            "legal_action_count_mean": float(group.mask_legal_count.mean()),
            "legal_action_count_min": int(group.mask_legal_count.min()),
            "chosen_probability_mean": float(group.chosen_masked_probability.mean()),
            "masked_logit_margin_mean": float(group.masked_logit_margin.mean()),
            "masked_entropy_mean_bits": float(group.masked_entropy_bits.mean()),
            "raw_entropy_mean_bits": float(group.raw_entropy_bits.mean()),
            "total_irrigation_mm": float(group.irrigation_action_mm.sum()),
            "total_nitrogen_kg_ha": float(group.fertilizer_action_kgN_ha.sum()),
            "raw_equals_policy_input_all_steps": bool(group.raw_equals_policy_input.all()),
            "all_actions_match_formal": bool(group.action_matches_formal.all()),
        })
    diagnostics = pd.DataFrame(annual_rows)
    common_counts = {str(seed): len(common_keys[seed]) for seed in SEEDS}
    common_daps = {str(seed): sorted({dap for dap, _ in common_keys[seed]}) for seed in SEEDS}
    return observation_summary, policy_response, diagnostics, {
        "common_daps": common_daps, "common_dap_occurrence_counts": common_counts,
        "common_dap_keys": {str(seed): [[dap, occurrence] for dap, occurrence in common_keys[seed]] for seed in SEEDS},
        "global_scales": {str(seed): global_scales[seed].tolist() for seed in SEEDS},
    }


def verdicts(policy: pd.DataFrame, annual: pd.DataFrame, seed5_trace: pd.DataFrame) -> dict[str, Any]:
    subset = policy[policy.seed.eq(5)]
    if len(subset) < 5 or subset.n_years.min() != len(YEARS):
        return {
            "seed5_observation_separability": "INSUFFICIENT",
            "seed5_mask_blocking": "INSUFFICIENT",
            "seed5_policy_sensitivity": "INSUFFICIENT",
            "seed5_argmax_boundary_evidence": "INSUFFICIENT",
            "next_step": "no tuning yet",
            "diagnostic_class": "INSUFFICIENT",
        }
    median_state_distance = float(subset.observation_pairwise_rms_z_mean.median())
    if median_state_distance >= 0.5:
        state_verdict = "HIGH"
    elif median_state_distance >= 0.1:
        state_verdict = "MODERATE"
    else:
        state_verdict = "LOW"
    eligible_steps = seed5_trace[seed5_trace.mask_legal_count.gt(1)]
    forced_single_share = float(seed5_trace.mask_legal_count.eq(1).mean())
    blocked_rate_all = float(seed5_trace.raw_argmax_masked_out.mean())
    blocked_rate = float(eligible_steps.raw_argmax_masked_out.mean()) if len(eligible_steps) else np.nan
    if blocked_rate >= 0.25:
        mask_verdict = "HIGH"
    elif blocked_rate >= 0.05:
        mask_verdict = "MODERATE"
    elif blocked_rate > 0:
        mask_verdict = "LOW"
    else:
        mask_verdict = "NONE"
    eligible_groups = subset[subset.legal_action_count_min.gt(1)]
    if len(eligible_groups) < 5:
        return {
            "seed5_observation_separability": state_verdict,
            "seed5_mask_blocking": mask_verdict,
            "seed5_policy_sensitivity": "INSUFFICIENT",
            "seed5_argmax_boundary_evidence": "INSUFFICIENT",
            "next_step": "no tuning yet",
            "diagnostic_class": "INSUFFICIENT: too few same-DAP groups with legal alternatives",
            "seed5_raw_argmax_masked_out_rate_when_alternatives": blocked_rate,
            "seed5_raw_argmax_masked_out_rate_all_steps": blocked_rate_all,
            "seed5_forced_single_action_share": forced_single_share,
        }
    median_js = float(eligible_groups.masked_policy_pairwise_js_mean.median())
    sensitivity = "HIGH" if median_js >= 0.1 else "MODERATE" if median_js >= 0.02 else "LOW"
    boundary_share = float(eligible_groups.same_final_action_near_boundary.mean())
    changed_same_share = float(eligible_groups.same_final_action_but_policy_distribution_changed.mean())
    if boundary_share >= 0.4:
        boundary = "STRONG"
    elif boundary_share >= 0.1:
        boundary = "MODERATE"
    elif changed_same_share > 0:
        boundary = "WEAK"
    else:
        boundary = "NONE"
    if state_verdict == "LOW":
        diagnostic_class, next_step = "A: observation 在共同 DAP 上区分度低/可能偏晚", "observation-action issue"
    elif mask_verdict in {"HIGH", "MODERATE"} and sensitivity == "LOW":
        diagnostic_class, next_step = "B+C: mask 常移除 raw 首选；剩余合法选择上的 policy 分布也很少随年改变", "observation-action issue"
    elif mask_verdict in {"HIGH", "MODERATE"}:
        diagnostic_class, next_step = "B: mask 经常移除 raw policy 首选动作（仅按存在合法替代动作的决策计）", "observation-action issue"
    elif state_verdict in {"HIGH", "MODERATE"} and sensitivity == "LOW":
        diagnostic_class, next_step = "C: 输入变化但 masked policy 分布基本不变", "PPO optimization"
    elif sensitivity in {"HIGH", "MODERATE"} and changed_same_share > 0:
        diagnostic_class, next_step = "D: policy 分布改变但离散 argmax 多数未切换", "no tuning yet"
    else:
        diagnostic_class, next_step = "混合/证据不足以唯一归类", "no tuning yet"
    return {
        "seed5_observation_separability": state_verdict,
        "seed5_mask_blocking": mask_verdict,
        "seed5_policy_sensitivity": sensitivity,
        "seed5_argmax_boundary_evidence": boundary,
        "next_step": next_step,
        "diagnostic_class": diagnostic_class,
        "seed5_median_common_dap_pairwise_rms_z": median_state_distance,
        "seed5_median_choice_capable_dap_masked_policy_js": median_js,
        "seed5_raw_argmax_masked_out_rate_when_alternatives": blocked_rate,
        "seed5_raw_argmax_masked_out_rate_all_steps": blocked_rate_all,
        "seed5_forced_single_action_share": forced_single_share,
        "seed5_choice_capable_dap_groups": int(len(eligible_groups)),
        "seed5_same_action_policy_changed_dap_share": changed_same_share,
        "seed5_same_action_near_boundary_dap_share": boundary_share,
        "thresholds_note": "State RMS-z: HIGH >=0.50, MODERATE >=0.10; mask blocking uses only steps with more than one legal action, HIGH >=25%, MODERATE >=5%; policy JS uses shared DAP-occurrence groups where every year has at least two legal actions, HIGH >=0.10, MODERATE >=0.02; boundary evidence only compares legal top-two actions.",
    }


def apply_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#667085", "axes.linewidth": 0.8,
        "grid.color": "#D9DEE3", "grid.linewidth": 0.65, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def make_figures(observation_summary: pd.DataFrame, policy: pd.DataFrame, schema: list[dict[str, Any]]) -> list[str]:
    apply_plot_style()
    folder = OUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    outputs = []

    event_daps = [dap for dap in [0, 8, 15, 31, 61] if not policy[(policy.seed.eq(5)) & (policy.dap.eq(dap)) & policy.dap_occurrence.eq(0)].empty]
    seed5_obs = observation_summary[
        observation_summary.seed.eq(5) & observation_summary.dap.isin(event_daps) & observation_summary.dap_occurrence.eq(0)
    ].copy()
    if not seed5_obs.empty:
        score = seed5_obs.groupby("feature").range_in_global_sd.max().sort_values(ascending=False)
        features = score.head(min(18, len(score))).index.tolist()
        matrix = np.full((len(features), len(event_daps)), np.nan)
        labels = []
        for r, feature in enumerate(features):
            part = seed5_obs[seed5_obs.feature.eq(feature)].set_index("dap")
            labels.append(feature)
            for c, dap in enumerate(event_daps):
                if dap in part.index:
                    matrix[r, c] = part.loc[dap, "range_in_global_sd"]
        fig, ax = plt.subplots(figsize=(11, max(4.2, 0.28 * len(features) + 1.8)), constrained_layout=True)
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="YlGnBu", vmin=0)
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        ax.set_xticks(np.arange(len(event_daps)), labels=[f"DAP {d}" for d in event_daps])
        ax.set_title("Seed 5 | Cross-year observation range at repeated action DAPs", loc="left", fontweight="bold")
        ax.set_xlabel("Decision day (common to 2014–2023)")
        ax.set_ylabel("Actual policy observation feature")
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025)
        cbar.set_label("Cross-year range / feature SD")
        path = folder / "observation_separation_seed5.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(rel(path))

    seed5 = policy[
        policy.seed.eq(5) & policy.dap.isin(event_daps) & policy.dap_occurrence.eq(0)
    ].sort_values("dap")
    if not seed5.empty:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
        x = np.arange(len(seed5))
        ax1.bar(x, seed5.chosen_probability_mean, color=PALETTE[5], alpha=0.9, width=0.65)
        ax1.errorbar(x, seed5.chosen_probability_mean, yerr=seed5.chosen_probability_std.fillna(0), fmt="none", ecolor="#344054", capsize=3, linewidth=1)
        ax1.set_ylim(0, 1.02)
        ax1.set_ylabel("Chosen action probability")
        ax1.set_title("Seed 5 | Masked policy output at repeated action DAPs", loc="left", fontweight="bold")
        ax1.grid(axis="y", alpha=0.7)
        ax2.bar(x, seed5.masked_logit_margin_min, color="#2878B5", alpha=0.9, width=0.65, label="Minimum legal top-1/top-2 logit margin")
        ax2.axhline(0.25, color="#D9822B", linestyle="--", linewidth=1.2, label="Near-boundary reference (0.25)")
        ax2.set_ylabel("Logit margin")
        ax2.set_xticks(x, [f"DAP {int(d)}" for d in seed5.dap])
        ax2.grid(axis="y", alpha=0.7)
        ax2.legend(frameon=False, loc="best")
        path = folder / "policy_probability_by_dap_seed5.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(rel(path))

    shared = policy[policy.dap_occurrence.eq(0)].copy()
    shared_daps = sorted(set.intersection(*(set(shared.loc[shared.seed.eq(s), "dap"].astype(int)) for s in SEEDS)))
    shared = shared[shared.dap.isin(shared_daps)]
    if not shared.empty:
        daps = sorted(shared.dap.unique().astype(int).tolist())
        labels = [f"S{s}" for s in SEEDS]
        blocked = np.zeros((len(SEEDS), len(daps)))
        actions = np.zeros_like(blocked)
        legal = np.zeros_like(blocked)
        for r, seed in enumerate(SEEDS):
            for c, dap in enumerate(daps):
                row = shared[(shared.seed.eq(seed)) & (shared.dap.eq(dap))]
                if len(row):
                    blocked[r, c] = float(row.iloc[0].raw_argmax_masked_out_rate_when_alternatives) if pd.notna(row.iloc[0].raw_argmax_masked_out_rate_when_alternatives) else np.nan
                    actions[r, c] = float(row.iloc[0].action_unique_count)
                    legal[r, c] = float(row.iloc[0].legal_action_count_mean)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, constrained_layout=True)
        im1 = ax1.imshow(blocked, aspect="auto", interpolation="nearest", cmap="Oranges", vmin=0, vmax=1)
        ax1.set_yticks(np.arange(len(labels)), labels=labels)
        ax1.set_title("Raw policy argmax removed when legal alternatives exist", loc="left", fontweight="bold")
        ax1.set_ylabel("PPO seed")
        fig.colorbar(im1, ax=ax1, fraction=0.025, pad=0.025, label="Share of choice-capable years")
        im2 = ax2.imshow(actions, aspect="auto", interpolation="nearest", cmap="Greens", vmin=1, vmax=ACTION_DIM)
        ax2.set_yticks(np.arange(len(labels)), labels=labels)
        tick_step = max(1, math.ceil(len(daps) / 18))
        tick_positions = np.arange(0, len(daps), tick_step)
        ax2.set_xticks(tick_positions, labels=[str(daps[index]) for index in tick_positions], rotation=0)
        ax2.set_xlabel("DAP shared by all ten years")
        ax2.set_ylabel("Distinct chosen actions")
        ax2.set_title("Chosen action diversity across years", loc="left", fontweight="bold")
        fig.colorbar(im2, ax=ax2, fraction=0.025, pad=0.025, label="Unique actions / 10 years")
        path = folder / "mask_and_argmax_diagnostics_seed345.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(rel(path))
    return outputs


def write_report(summaries: pd.DataFrame, observation_summary: pd.DataFrame, policy: pd.DataFrame, annual: pd.DataFrame,
                 schema: list[dict[str, Any]], analysis: dict[str, Any], provenance: dict[str, Any], figure_paths: list[str]) -> None:
    if REPORT.exists():
        raise FileExistsError(f"Refusing to overwrite report: {rel(REPORT)}")
    v = analysis["verdicts"]
    lines = [
        "# YC 004_09：observation-mask-action 链路审计", "",
        "## 结论先行", "",
        f"- 诊断类别：**{v['diagnostic_class']}**。", 
        f"- Seed 5 observation separability：**{v['seed5_observation_separability']}**；共同 DAP 的标准化跨年距离中位数 {v.get('seed5_median_common_dap_pairwise_rms_z', float('nan')):.3f}。",
        f"- Seed 5 mask blocking：**{v['seed5_mask_blocking']}**；raw policy 首选动作被 mask 禁止的决策比例 {v.get('seed5_raw_argmax_masked_out_rate', float('nan')):.1%}。",
        f"- Seed 5 policy sensitivity：**{v['seed5_policy_sensitivity']}**；共同 DAP masked policy pairwise JS 中位数 {v.get('seed5_median_common_dap_masked_policy_js', float('nan')):.4f} bits。",
        f"- Seed 5 argmax-boundary evidence：**{v['seed5_argmax_boundary_evidence']}**；保持同一动作但分布发生变化的 DAP 占比 {v.get('seed5_same_action_policy_changed_dap_share', float('nan')):.1%}，同时达到近边界口径的 DAP 占比 {v.get('seed5_same_action_near_boundary_dap_share', float('nan')):.1%}。",
        f"- 下一步：**{v['next_step']}**。上述阈值是本次诊断的操作性分级，不是普适统计检验。", "",
        "```text",
        f"seed5_observation_separability = {v['seed5_observation_separability']}",
        f"seed5_mask_blocking = {v['seed5_mask_blocking']}",
        f"seed5_policy_sensitivity = {v['seed5_policy_sensitivity']}",
        f"seed5_argmax_boundary_evidence = {v['seed5_argmax_boundary_evidence']}",
        f"next_step = {v['next_step']}",
        "```", "",
        "## 方法与复现边界", "",
        "- 固定 004_05 的 100K checkpoint，逐年应用于 2014–2023；仅 evaluation-only replay，没有训练、更新模型、改 reward、mask、动作空间或 DSSAT 输入。每个 season 串行运行，一个 CPU 模型/环境，cache=1，进程树 RSS 硬停止线 2048 MiB。",
        "- 在每个 step 调用环境 step 之前记录原始 observation、送入 actor 的 tensor、未加 mask logits/probabilities、实际 action mask、masked logits/probabilities 和 deterministic action。配置中 observation normalization 与 VecNormalize 均关闭；本次另逐步核对 raw observation 与 policy tensor 一致。",
        "- Figure style 沿用 YC 055_03 / 004_05 的白底、左对齐标题、紧凑轴、浅灰网格、PPO 绿色/蓝色/橙色 seed 色；仅图形内容按 observation-policy 链路调整。",
        "- 状态是各模型闭环 rollout 自身的决策前状态；跨 seed 的状态不等于同一外生轨迹。没有做 counterfactual state intervention，因此不做因果结论。", "",
        "## 覆盖与闭合核验", "",
        f"- 决策记录：{len(summaries)} 个 seed-year，{int(summaries.steps.sum())} 个 step；observation 维度 {len(schema)}；每个 seed 覆盖 {len(YEARS)} 年。",
        f"- 每步 action_index/DAP 与正式 trace 一致：{bool(summaries.gates_json.notna().all()) and bool(summaries.action_trace_exact.all()) if 'action_trace_exact' in summaries else '见 episode gates'}；predict 与 masked distribution argmax 一致：{bool(summaries.predict_distribution_action_match.all()) if 'predict_distribution_action_match' in summaries else '见逐步 trace'}。",
        f"- yield/I/N/reward 最大绝对闭合误差：{summaries.yield_abs_error_kg_ha.max():.6g} kg/ha / {summaries.irrigation_abs_error_mm.max():.6g} mm / {summaries.nitrogen_abs_error_kg_ha.max():.6g} kg/ha / {summaries.reward_abs_error.max():.6g}。",
        f"- RSS 峰值：{summaries.rss_peak_mb.max():.1f} MiB；最大值低于 2048 MiB 门槛。", "",
        "## Seed 5：共同 DAP 的 observation 与策略输出", "",
        "同一 DAP 的以下汇总覆盖 2014–2023 全部十年。RMS-z 是将每个 observation 维度按该 seed 全部验证决策的标准差缩放后，计算年份间 RMS 欧氏距离；JS divergence 比较 16 动作概率分布，单位 bit。", "",
        "| DAP | action 唯一数/10年 | 状态距离 RMS-z 均值 | masked policy JS 均值 | chosen prob 范围 | 最小 logit margin | raw 首选被 mask 比例 | 合法动作均值 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policy[policy.seed.eq(5)].itertuples():
        lines.append(f"| {int(row.dap)} | {row.action_unique_count}/10 | {row.observation_pairwise_rms_z_mean:.3f} | {row.masked_policy_pairwise_js_mean:.4f} | {row.chosen_probability_range:.3f} | {row.masked_logit_margin_min:.3f} | {row.raw_argmax_masked_out_rate:.1%} | {row.legal_action_count_mean:.1f} |")
    lines.extend(["", "### 十年总体与 seeds 对照", "", "| Seed | 公共 DAP 数 | 平均 raw 首选被 mask 比例 | 公共 DAP 平均 masked JS | 动作跨年发生变化的公共 DAP 数 |", "|---:|---:|---:|---:|---:|"])
    for seed in SEEDS:
        p = policy[policy.seed.eq(seed)]
        a = annual[annual.seed.eq(seed)]
        lines.append(f"| {seed} | {int(p.dap.nunique())} | {a.raw_argmax_masked_out_rate.mean():.1%} | {p.masked_policy_pairwise_js_mean.mean():.4f} | {int((p.action_unique_count > 1).sum())} |")
    lines.extend(["", "## 观察解释", ""])
    if v["diagnostic_class"].startswith("A:"):
        lines.append("- 在 seed 5 的共同决策日，实际输入本身区分度偏低；这支持优先追查信息何时进入当前 observation，而不是先调 reward。它不证明某个具体 weather feature 一定缺失。")
    elif v["diagnostic_class"].startswith("B:"):
        lines.append("- raw policy argmax 有一定比例被 action mask 移除，mask 对最终可选行为具有可观影响；这只是说明约束改变动作选择，不表示约束设计错误。")
    elif v["diagnostic_class"].startswith("C:"):
        lines.append("- 输入跨年可分，但合法集合仍开放且 masked policy 分布变化弱，符合策略在已见状态上的 action sensitivity/optimization 候选问题；本分析还不能识别训练动态根因。")
    elif v["diagnostic_class"].startswith("D:"):
        lines.append("- 策略分布存在跨年变化而最终动作保持不变，直接支持“同一 argmax 掩盖概率变化”的解释。若 margin 也接近零线，说明部分年靠近切换边界，但仍不证明换动作会提升农艺结果。")
    else:
        lines.append("- 现有决策链证据呈混合或不充分；不据此调整 reward、mask 或 observation。")
    top_obs = observation_summary[(observation_summary.seed.eq(5)) & (observation_summary.dap.isin([0, 8, 15, 31, 61]))]
    if not top_obs.empty:
        ranked = top_obs.groupby("feature").range_in_global_sd.max().sort_values(ascending=False).head(8)
        lines.append("- Seed 5 关键动作 DAP 上跨年变化较大的输入维度（range/global SD）：" + "; ".join(f"{name}={value:.2f}" for name, value in ranked.items()) + ".")
    lines.extend(["", "## 图表", ""])
    for item in figure_paths:
        lines.append(f"- `{item}`")
    lines.extend(["", "## 数据文件", "", "- 决策级原始记录（每步 observation、输入张量、mask、完整16动作分布）：`results/yc_random_weather_ppo/004_09_observation_mask_action_audit/decision_trace_seed345.csv`。", "- Episode闭合与输入校验：`results/yc_random_weather_ppo/004_09_observation_mask_action_audit/episode_reconciliation_seed345.csv`。", "- 同DAP observation逐变量统计：`results/yc_random_weather_ppo/004_09_observation_mask_action_audit/same_dap_observation_summary.csv`。", "- 同DAP policy/mask诊断：`results/yc_random_weather_ppo/004_09_observation_mask_action_audit/same_dap_policy_response.csv`。", "- 每年总体诊断：`results/yc_random_weather_ppo/004_09_observation_mask_action_audit/yearly_policy_diagnostics.csv`。", "- 观察 schema 与维度：`results/yc_random_weather_ppo/004_09_observation_mask_action_audit/observation_schema.json`。", "", "## 限制", "", "- 原始 step-level trace 未提供 calendar date/DOY；本报告按 timestep/DAP 对齐，不从事后 snapshot/weather 表倒推日期。汇总只比较每个固定策略自身 closed-loop rollout 的共同 DAP，不是相同外生状态下的跨 seed 直接实验。",
        "- 直接状态响应按已记录 observation、policy 输出和 mask 逐步描述；未做改变单一 observation 维度的 counterfactual 推理，也未评估另一动作会带来的产量/耗水后果。",
        "- 一个政策分布的概率变化不等于农艺上有用的变化；共同 DAP 经验阈值只用于定位下一步，不构成显著性检验。",
        f"- provenance：`{json.dumps(provenance, ensure_ascii=False, sort_keys=True)}`。", "", "未提交、未推送。"])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_report_final(summaries: pd.DataFrame, observation_summary: pd.DataFrame, policy: pd.DataFrame, annual: pd.DataFrame,
                       schema: list[dict[str, Any]], analysis: dict[str, Any], provenance: dict[str, Any],
                       figure_paths: list[str], replace: bool = False) -> None:
    if REPORT.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite report: {rel(REPORT)}")
    v = analysis["verdicts"]
    key_daps = [0, 8, 15, 31, 61]
    key = policy[policy.seed.eq(5) & policy.dap.isin(key_daps) & policy.dap_occurrence.eq(0)].sort_values("dap")
    feature_rows = observation_summary[
        observation_summary.seed.eq(5) & observation_summary.dap.isin(key_daps) & observation_summary.dap_occurrence.eq(0)
    ]
    ranked = feature_rows.groupby("feature").range_in_global_sd.max().sort_values(ascending=False).head(8) if not feature_rows.empty else pd.Series(dtype=float)
    lines = [
        "# YC 004_09：observation-mask-action 链路审计", "",
        "## 结论", "",
        f"- 主诊断：**{v['diagnostic_class']}**。",
        f"- Seed 5 observation separability：**{v['seed5_observation_separability']}**；跨年输入 RMS-z 中位数 {v.get('seed5_median_common_dap_pairwise_rms_z', float('nan')):.3f}。",
        f"- Seed 5 mask blocking：**{v['seed5_mask_blocking']}**；在存在合法替代动作的决策中 raw argmax 被 mask 的比例 {v.get('seed5_raw_argmax_masked_out_rate_when_alternatives', float('nan')):.1%}；仅一个合法动作的 step 占比 {v.get('seed5_forced_single_action_share', float('nan')):.1%}。",
        f"- Seed 5 policy sensitivity：**{v['seed5_policy_sensitivity']}**；合法动作机会组 masked-policy JS 中位数 {v.get('seed5_median_choice_capable_dap_masked_policy_js', float('nan')):.5f} bits。",
        f"- Seed 5 argmax-boundary evidence：**{v['seed5_argmax_boundary_evidence']}**；分布变化但动作不变的组 {v.get('seed5_same_action_policy_changed_dap_share', float('nan')):.1%}，近边界组 {v.get('seed5_same_action_near_boundary_dap_share', float('nan')):.1%}。",
        f"- next_step = {v['next_step']}。分级阈值是本次诊断口径，不是普适统计检验。", "",
        "```text",
        f"seed5_observation_separability = {v['seed5_observation_separability']}",
        f"seed5_mask_blocking = {v['seed5_mask_blocking']}",
        f"seed5_policy_sensitivity = {v['seed5_policy_sensitivity']}",
        f"seed5_argmax_boundary_evidence = {v['seed5_argmax_boundary_evidence']}",
        f"next_step = {v['next_step']}",
        "```", "",
        "## 证据口径", "",
        "- 固定 004_05 seed 3/4/5 的 100K checkpoint，2014–2023 全验证年；只做 evaluation-only replay，没有训练、改模型、reward、mask、动作空间或 DSSAT 输入。CPU 串行、每 episode 单模型/单环境、cache=1，RSS 停止线 2048 MiB。",
        "- 逐步记录了调用 env.step 前的 observation、传入 actor 的 tensor、unmasked logits/probabilities、16 维 mask、masked logits/probabilities、deterministic action，并与环境实际执行剂量、正式 trace 和 episode endpoint 对照。",
        "- 22 维实际 observation 字段：" + ", ".join(row["display_name"] for row in schema) + ". 可见输入包含 `sw[0]`–`sw[5]`、`swfac`、`nstres`、`wtdep`、`totir`、作物状态和 `tmax/srad`；没有 rainfall 字段。此前“没有直接 soil-water 输入”的前提需要修正：至少 `sw[0..5]` 是直接提供给 policy 的土壤水相关状态。",
        "- 观测归一化和 VecNormalize 均关闭，3048/3048 步 raw observation 与 policy tensor 一致。每年状态属于策略自身闭环轨迹，不是跨 seed 的同一外生轨迹；没有做 counterfactual intervention，故不作因果归因。",
        "- 同一 DAP 有重复 timestep 时按 `(DAP, dap_occurrence)` 配对；mask blocker 比例只在 legal action count > 1 时计算，top-2 margin 也只比较合法动作。", "",
        "## Smoke 与闭合", "",
        "- Seed 5 / 2019 smoke 通过后，串行完成 30/30 seed-year、3048/3048 steps。逐步 action index、DAP、实际 I/N 剂量与正式 trace 一致；predict 与 masked distribution argmax 全部一致。",
        f"- yield/I/N/reward 最大闭合误差：{summaries.yield_abs_error_kg_ha.max():.3g} kg/ha / {summaries.irrigation_abs_error_mm.max():.3g} mm / {summaries.nitrogen_abs_error_kg_ha.max():.3g} kg/ha / {summaries.reward_abs_error.max():.3g}；峰值 RSS {summaries.rss_peak_mb.max():.1f} MiB。",
        "- Smoke 前置调试中有 3 次未启动 DSSAT season 的失败尝试（目录初始化、正式 trace 日期列契约、列类型清单）；失败日志/manifest 保留。后续 QA 修正了重复 DAP 对齐和合法 top-2 margin；每次都复用已保存的逐步回放、不重跑模拟，旧派生版本保存在 `superseded_v1/`、`superseded_v2/`。", "",
        "## Seed 5 关键动作日", "",
        "以下为 DAP occurrence 0 的跨年配对；完整 89 个 DAP 的全部 occurrence 在 CSV 中。RMS-z 为 observation 年际 pairwise 标准化距离，JS 为 16 维 masked distribution 的 Jensen–Shannon divergence。",
        "| DAP | 实际动作唯一数/10年 | observation RMS-z | masked JS (bit) | chosen prob range | 合法 top-2 最小 logit margin | raw argmax 被 mask | legal action 数 min–mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in key.itertuples():
        margin = "NA" if pd.isna(row.masked_logit_margin_min) else f"{row.masked_logit_margin_min:.3f}"
        blocker = "NA" if pd.isna(row.raw_argmax_masked_out_rate_when_alternatives) else f"{row.raw_argmax_masked_out_rate_when_alternatives:.0%}"
        lines.append(f"| {int(row.dap)} | {row.action_unique_count}/10 | {row.observation_pairwise_rms_z_mean:.3f} | {row.masked_policy_pairwise_js_mean:.5f} | {row.chosen_probability_range:.3f} | {margin} | {blocker} | {row.legal_action_count_min}–{row.legal_action_count_mean:.1f} |")
    behavior = analysis.get("seed5_action_sequence_summary", {})
    event_text = "; ".join(
        f"DAP {row['dap']}: I={row['irrigation_mm']:g} mm, N={row['nitrogen_kg_ha']:g} kg/ha"
        for row in behavior.get("management_events", [])
    )
    lines.extend([
        "", "## Seed 5 完整动作序列与管理事件", "",
        f"- 十年逐日 `action_index` 完整序列有 {behavior.get('distinct_daily_action_sequences', 'NA')} 种；但只看实际施用（I>0 或 N>0）的管理事件序列，十年完全一致（唯一序列数 {behavior.get('distinct_management_event_sequences', 'NA')}）。因此‘固定 schedule’指实际管理事件，不等于每个空动作日的 action index 也完全相同。",
        f"- Seed 5 每年共同的 5 个实际事件：{event_text}。",
        "", "## Seed 对照", "", "| Seed | 公共 DAP-occurrence 组 | 多个合法动作时 raw argmax 被 mask | 单合法动作 step 占比 | 合法机会组 masked JS 中位数 | action 有跨年变化的组 |", "|---:|---:|---:|---:|---:|---:|"])
    for seed in SEEDS:
        p = policy[policy.seed.eq(seed)]
        eligible = p[p.legal_action_count_min.gt(1)]
        a = annual[annual.seed.eq(seed)]
        lines.append(f"| {seed} | {len(p)} | {a.raw_argmax_masked_out_rate_when_alternatives.mean():.1%} | {a.forced_single_action_share.mean():.1%} | {eligible.masked_policy_pairwise_js_mean.median():.5f} | {int((p.action_unique_count > 1).sum())} |")
    lines.extend(["", "## 解释与下一步", ""])
    if v["diagnostic_class"].startswith("B+C:"):
        lines.extend([
            "- 输入在共同决策时点能明显区分年份，但行为链呈混合的 B+C：有替代选择时 mask 经常移除 actor 的 raw 首选；另一方面，DAP 8/31/61 在十年均保留多种合法动作，masked 概率分布仍几乎相同。DAP 15 的 raw 首选被 mask，而 DAP 0 的首个决策单独分析后，不再把同日第二步 forced no-op 混入。",
            "- 这说明 fixed schedule 不是“PPO没看到状态”：实际输入含 `sw[0..5]` 等水分状态，且关键 DAP 的 observation 确有年际差异；但 policy 输出在多数可选择的重复动作日变化极少，mask 又在部分决策改变了候选集。仅凭当前记录还不能断言 mask 或优化中的哪一个是唯一根因。",
            "- 因而先做 observation/action-mask 的逐规则核查，尤其查看 raw argmax 被禁的动作类型和禁用原因；在核清哪些 mask 属于预期安全约束前，不改约束、不重调 reward，也不直接开启新训练。",
        ])
    elif v["diagnostic_class"].startswith("A:"):
        lines.append("- 实际 policy 输入自身在关键重复决策点区分度不足，先核对信息时序与 observation 映射，不先调 reward。")
    elif v["diagnostic_class"].startswith("C:"):
        lines.append("- observation 可区分年份且合法替代动作开放，但 masked policy 分布变化弱；这更像策略敏感性/优化候选问题，仍未识别训练根因。")
    elif v["diagnostic_class"].startswith("D:"):
        lines.append("- policy distribution 随状态变化但 argmax 保持，支持离散动作边界掩盖概率响应；本身不证明改变动作可提升农艺结果。")
    else:
        lines.append("- 现有证据混合或不足，不据此修改训练因素。")
    if not ranked.empty:
        lines.append("- 关键动作 DAP 跨年变化最大的输入特征（range / feature SD）：" + "; ".join(f"{name}={value:.2f}" for name, value in ranked.items()) + ".")
    lines.extend(["", "## 图表与数据", ""])
    for path in figure_paths:
        lines.append(f"- `{path}`")
    lines.extend([
        "- `results/yc_random_weather_ppo/004_09_observation_mask_action_audit/decision_trace_seed345.csv`：逐步 observation、mask、16 动作 logits/probabilities、action。",
        "- `results/yc_random_weather_ppo/004_09_observation_mask_action_audit/same_dap_observation_summary.csv`、`same_dap_policy_response.csv`、`yearly_policy_diagnostics.csv`：跨年 occurrence 对照和逐年诊断。",
        "- `results/yc_random_weather_ppo/004_09_observation_mask_action_audit/episode_reconciliation_seed345.csv`、`observation_schema.json`、`analysis_summary.json`、`input_provenance.json`：闭合、schema、结论和来源哈希。",
        "", "## 限制", "",
        "- 正式 step trace 未提供 calendar date/DOY；按 timestep/DAP 对齐，不从事后 snapshot 倒推日期。",
        "- 比较基于各 policy 的 closed-loop 状态，未在相同环境状态上做跨 seed 对照；没有单独改写/置换 observation，也未估算未选动作的收成结果。",
        f"- provenance：`{json.dumps(provenance, ensure_ascii=False, sort_keys=True)}`。", "", "未提交、未推送。",
    ])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_smoke(runner, pilot) -> None:
    manifest_path = next_attempt_path(OUT / "smoke_manifest.json")
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "episodes").mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "passed": False, "seed": SMOKE_SEED, "year": SMOKE_YEAR, "rss_limit_mb": RSS_LIMIT_MB, "started_utc": pd.Timestamp.utcnow().isoformat()}
    write_json(manifest_path, manifest)
    try:
        by_year, episodes = load_reference(runner, SMOKE_SEED)
        summary, rows, schema = run_episode(runner, pilot, SMOKE_SEED, SMOKE_YEAR, by_year[SMOKE_YEAR], episodes[SMOKE_YEAR], None)
        if len(schema) == 0 or len(rows) != len(by_year[SMOKE_YEAR]):
            raise RuntimeError("Smoke schema/step coverage is empty")
        schema_path = OUT / "observation_schema.json"
        write_json(schema_path, {"dimension": len(schema), "fields": schema, "normalization_config": {"observation_normalization": False, "vecnormalize": False}, "smoke_source": f"seed={SMOKE_SEED},year={SMOKE_YEAR}"})
        summary_path = OUT / "smoke_episode_reconciliation.csv"
        pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig", lineterminator="\n")
        manifest.update({"status": "passed", "passed": True, "completed_utc": pd.Timestamp.utcnow().isoformat(), "episode": summary, "observation_dimension": len(schema), "observation_schema_sha256": summary["observation_schema_sha256"], "gates": summary["gates"]})
        write_json(manifest_path, manifest, replace=True)
        print(f"[smoke-pass] seed={SMOKE_SEED} year={SMOKE_YEAR} steps={summary['steps']} obs_dim={len(schema)} RSS={summary['rss_peak_mb']:.1f} MiB", flush=True)
    except Exception as exc:
        manifest.update({"status": "failed", "passed": False, "completed_utc": pd.Timestamp.utcnow().isoformat(), "error": repr(exc), "traceback": traceback.format_exc()})
        write_json(manifest_path, manifest, replace=True)
        raise


def provenance_for(seed_summaries: pd.DataFrame, schema: list[dict[str, Any]]) -> dict[str, Any]:
    paths = [PROMPT, Path(__file__), CONFIG, ROOT / "src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py",
             FROZEN / "evaluation/evaluation_qc.json", REPLAY / "provenance.json", REPLAY / "finalization_manifest.json"]
    for seed in SEEDS:
        paths.extend([frozen_model_path(seed), FROZEN / f"training/random_weather/ppo_seed_{seed}/run_manifest.json",
                      FROZEN / f"evaluation/runs/observed_weather/random_weather/ppo_seed_{seed}/evaluation_episode_level.csv",
                      FROZEN / f"evaluation/runs/observed_weather/random_weather/ppo_seed_{seed}/evaluation_step_level.csv"])
    return {
        "created_utc": pd.Timestamp.utcnow().isoformat(), "prompt_sha256": sha256_file(PROMPT), "script_sha256": sha256_file(Path(__file__)),
        "source_hashes": {rel(path): sha256_file(path) for path in paths if path.is_file()},
        "model_hashes": {str(seed): sha256_file(frozen_model_path(seed)) for seed in SEEDS},
        "observation_schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest().upper(),
        "seeds": SEEDS, "years": YEARS, "checkpoint": 100000, "episode_count": int(len(seed_summaries)),
        "step_count": int(seed_summaries.steps.sum()), "rss_limit_mb": RSS_LIMIT_MB,
        "peak_rss_mb": float(seed_summaries.rss_peak_mb.max()), "runtime": {"device": "cpu", "serial_episodes": True, "environment_cache": 1, "normalization": "disabled per frozen config; raw/policy tensor equality checked"},
    }


def analyze_existing() -> None:
    manifest_path = OUT / "formal_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("No completed formal replay manifest; analyze phase will not run a replay")
    formal_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if formal_manifest.get("passed") is not True or formal_manifest.get("episode_count") != 30 or formal_manifest.get("step_count") != 3048:
        raise RuntimeError("Formal replay manifest is not a passed 30-episode/3048-step result")

    episode_dir = OUT / "episodes"
    trace_paths = sorted(episode_dir.glob("decision_trace_seed[345]_20*.csv"))
    if len(trace_paths) != 30:
        raise RuntimeError(f"Expected 30 saved episode traces; found {len(trace_paths)}")
    schema_path = OUT / "observation_schema.json"
    schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
    schema = schema_payload["fields"]
    if schema_payload.get("dimension") != len(schema) or len(schema) != 22:
        raise RuntimeError("Saved observation schema is incomplete or inconsistent")

    reconciliation_path = OUT / "episode_reconciliation_seed345.csv"
    summaries = pd.read_csv(reconciliation_path)
    if len(summaries) != 30 or summaries.duplicated(["seed", "year"]).any() or not summaries.status.eq("passed").all():
        raise RuntimeError("Episode reconciliation does not contain 30 unique passed seed-years")
    for column in ("yield_abs_error_kg_ha", "irrigation_abs_error_mm", "nitrogen_abs_error_kg_ha", "reward_abs_error", "rss_peak_mb", "steps", "seed", "year"):
        summaries[column] = pd.to_numeric(summaries[column], errors="raise")
    if int(summaries.steps.sum()) != 3048:
        raise RuntimeError("Saved reconciliation step count does not close to 3048")

    old_provenance_path = OUT / "input_provenance.json"
    old_provenance = json.loads(old_provenance_path.read_text(encoding="utf-8"))
    derived_paths = [
        OUT / "decision_trace_seed345.csv", OUT / "same_dap_observation_summary.csv",
        OUT / "same_dap_policy_response.csv", OUT / "yearly_policy_diagnostics.csv",
        OUT / "analysis_summary.json", OUT / "analysis_revision_manifest.json", old_provenance_path, REPORT,
        *(OUT / "figures").glob("*.png"),
    ]
    archive_index = 1
    archive = OUT / f"superseded_v{archive_index}"
    while archive.exists():
        archive_index += 1
        archive = OUT / f"superseded_v{archive_index}"
    for source in derived_paths:
        if source.is_file():
            destination = archive / source.resolve().relative_to(ROOT.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    trace_parts = [pd.read_csv(path) for path in trace_paths]
    trace = pd.concat(trace_parts, ignore_index=True).sort_values(["seed", "year", "timestep"]).reset_index(drop=True)
    if len(trace) != 3048 or trace.duplicated(["seed", "year", "timestep"]).any():
        raise RuntimeError(f"Saved step traces are incomplete or duplicated: {len(trace)} rows")
    for seed in SEEDS:
        for year in YEARS:
            episode = trace.loc[trace.seed.eq(seed) & trace.year.eq(year)]
            if episode.empty or len(episode) != int(summaries.loc[summaries.seed.eq(seed) & summaries.year.eq(year), "steps"].iloc[0]):
                raise RuntimeError(f"Trace/reconciliation step mismatch for seed={seed}, year={year}")
    trace["dap_occurrence"] = trace.groupby(["seed", "year", "dap"], sort=False).cumcount()
    mask_cols = [f"mask__a{i:02d}" for i in range(ACTION_DIM)]
    raw_logit_cols = [f"raw_logit__a{i:02d}" for i in range(ACTION_DIM)]
    masked_logit_cols = [f"masked_logit__a{i:02d}" for i in range(ACTION_DIM)]
    masked_prob_cols = [f"masked_prob__a{i:02d}" for i in range(ACTION_DIM)]
    masks = trace[mask_cols].to_numpy(bool)
    raw_logits = trace[raw_logit_cols].to_numpy(float)
    masked_logits = trace[masked_logit_cols].to_numpy(float)
    masked_probs = trace[masked_prob_cols].to_numpy(float)
    legal_counts = masks.sum(axis=1)
    if (legal_counts < 1).any() or not np.allclose(masked_probs.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("Saved action masks or masked probabilities are invalid")
    invalid_probabilities = masked_probs[~masks]
    if invalid_probabilities.size and np.max(np.abs(invalid_probabilities)) > 1e-7:
        raise RuntimeError("Saved masked probability is nonzero for an illegal action")
    legal_order = np.argsort(-np.where(masks, masked_logits, -np.inf), axis=1, kind="stable")
    row_ids = np.arange(len(trace))
    top1_ids = legal_order[:, 0]
    has_alternative = legal_counts > 1
    top2_ids = np.full(len(trace), -1, dtype=int)
    top2_ids[has_alternative] = legal_order[has_alternative, 1]
    raw_top_ids = raw_logits.argmax(axis=1)
    if not np.array_equal(trace.action_index.to_numpy(int), top1_ids):
        raise RuntimeError("Reconstructed legal masked argmax does not match the recorded action")
    trace["mask_legal_count"] = legal_counts
    trace["raw_argmax_action"] = raw_top_ids
    trace["raw_argmax_masked_out"] = ~masks[row_ids, raw_top_ids]
    trace["masked_argmax_action"] = top1_ids
    trace["masked_top1_probability"] = masked_probs[row_ids, top1_ids]
    trace["masked_top2_probability"] = np.nan
    trace.loc[has_alternative, "masked_top2_probability"] = masked_probs[row_ids[has_alternative], top2_ids[has_alternative]]
    trace["masked_logit_margin"] = np.nan
    trace.loc[has_alternative, "masked_logit_margin"] = masked_logits[row_ids[has_alternative], top1_ids[has_alternative]] - masked_logits[row_ids[has_alternative], top2_ids[has_alternative]]
    trace["masked_probability_margin"] = np.nan
    trace.loc[has_alternative, "masked_probability_margin"] = trace.loc[has_alternative, "masked_top1_probability"].to_numpy() - trace.loc[has_alternative, "masked_top2_probability"].to_numpy()
    trace.to_csv(OUT / "decision_trace_seed345.csv", index=False, encoding="utf-8-sig", lineterminator="\n")

    observation_summary, policy_response, annual, analysis = summarize_decisions(trace, schema)
    observation_summary.to_csv(OUT / "same_dap_observation_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    policy_response.to_csv(OUT / "same_dap_policy_response.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    annual.to_csv(OUT / "yearly_policy_diagnostics.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    analysis["verdicts"] = verdicts(policy_response, annual, trace.loc[trace.seed.eq(5)])
    analysis["decision_rules"] = {
        "alignment": "same policy, year, DAP, and within-DAP occurrence; only keys present in all ten years are compared",
        "observation_separability": "median same-key pairwise RMS standardized distance; HIGH >=0.50, MODERATE >=0.10",
        "mask_blocking": "unmasked actor argmax forbidden, calculated only where more than one action is legal; HIGH >=25%, MODERATE >=5%",
        "policy_sensitivity": "median same-key pairwise masked probability JS among keys with >=2 legal actions in every year; HIGH >=0.10 bits, MODERATE >=0.02 bits",
        "argmax_boundary": "only legal top-two margin; no margin when fewer than two legal actions",
    }
    analysis["coverage"] = {
        "seed_years": len(summaries), "steps": len(trace), "observation_dimensions": len(schema),
        "all_seed_year_coverage": 10, "all_raw_equals_policy_input": bool(trace.raw_equals_policy_input.all()),
        "all_predict_matches_masked_argmax": bool(trace.predict_distribution_action_match.all()),
        "all_actions_match_formal_trace": bool(trace.action_matches_formal.all()),
        "dap_occurrence_alignment": True,
    }
    seed5_events = trace.loc[trace.seed.eq(5) & (trace.irrigation_action_mm.gt(0) | trace.fertilizer_action_kgN_ha.gt(0))]
    event_signatures = {
        int(year): tuple(zip(part.dap.astype(int), part.irrigation_action_mm.astype(float), part.fertilizer_action_kgN_ha.astype(float)))
        for year, part in seed5_events.groupby("year", sort=True)
    }
    unique_event_signatures = list(dict.fromkeys(event_signatures.values()))
    if len(unique_event_signatures) != 1:
        raise RuntimeError("Seed 5 applied irrigation/N event schedule is not identical across years")
    analysis["seed5_action_sequence_summary"] = {
        "distinct_daily_action_sequences": int(trace.loc[trace.seed.eq(5)].groupby("year").action_index.apply(tuple).nunique()),
        "distinct_management_event_sequences": len(unique_event_signatures),
        "management_events": [
            {"dap": int(dap), "irrigation_mm": float(irrigation), "nitrogen_kg_ha": float(nitrogen)}
            for dap, irrigation, nitrogen in unique_event_signatures[0]
        ],
    }
    figure_paths = make_figures(observation_summary, policy_response, schema)
    provenance = provenance_for(summaries, schema)
    provenance["rendered_input_hash_checks"] = old_provenance.get("rendered_input_hash_checks", {})
    provenance["analysis_revision"] = {
        "revision": "v3_dap_occurrence_legal_top2_recomputed",
        "reason": "corrected repeated-DAP alignment and recomputed all top-two fields from saved legal masks and logits",
        "source_trace_count": len(trace_paths), "source_step_count": len(trace),
        "additional_simulation_replay": False, "superseded_outputs": rel(archive),
    }
    write_json(OUT / "analysis_summary.json", analysis, replace=True)
    write_json(OUT / "input_provenance.json", provenance, replace=True)
    write_json(OUT / "analysis_revision_manifest.json", {
        "status": "passed", "passed": True, "revision": "v3_dap_occurrence_legal_top2_recomputed",
        "completed_utc": pd.Timestamp.utcnow().isoformat(), "additional_simulation_replay": False,
        "input_manifest": rel(manifest_path), "source_episode_trace_count": len(trace_paths),
        "source_step_count": len(trace), "superseded_outputs": rel(archive),
        "verdicts": analysis["verdicts"],
    }, replace=True)
    write_report_final(summaries, observation_summary, policy_response, annual, schema, analysis, provenance, figure_paths, replace=True)
    print(f"[analysis-pass] episodes={len(summaries)} steps={len(trace)} replay=none verdict={analysis['verdicts']['diagnostic_class']} report={rel(REPORT)}", flush=True)


def run_formal(runner, pilot) -> None:
    smoke_paths = sorted(OUT.glob("smoke_manifest*.json"))
    passed_smokes = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in smoke_paths]
    passed_smokes = [(path, payload) for path, payload in passed_smokes if payload.get("passed") is True]
    if not passed_smokes:
        raise FileNotFoundError("No passed seed 5 / 2019 smoke manifest is available")
    smoke_path, smoke = passed_smokes[-1]
    manifest_path = OUT / "formal_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Refusing to overwrite prior formal attempt")
    manifest = {"status": "running", "passed": False, "seeds": SEEDS, "years": YEARS, "rss_limit_mb": RSS_LIMIT_MB, "started_utc": pd.Timestamp.utcnow().isoformat(), "reused_smoke_seed_year": [SMOKE_SEED, SMOKE_YEAR], "smoke_manifest_used": rel(smoke_path), "smoke_attempt_history": [rel(path) for path in smoke_paths]}
    write_json(manifest_path, manifest)
    schema = json.loads((OUT / "observation_schema.json").read_text(encoding="utf-8"))["fields"]
    records = [json.loads(json.dumps(smoke["episode"]))]
    try:
        references = {seed: load_reference(runner, seed) for seed in SEEDS}
        episode_rows = []
        for seed in SEEDS:
            for year in YEARS:
                if seed == SMOKE_SEED and year == SMOKE_YEAR:
                    continue
                expected_steps, expected_episodes = references[seed]
                summary, _, episode_schema = run_episode(runner, pilot, seed, year, expected_steps[year], expected_episodes[year], schema)
                if episode_schema != schema:
                    raise RuntimeError("Schema mismatch across seed/year")
                records.append(summary)
                episode_rows.append(summary)
                partial = pd.DataFrame(records).sort_values(["seed", "year"])
                partial.to_csv(OUT / "formal_episode_progress.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
                print(f"[episode-pass] seed={seed} year={year} steps={summary['steps']} yield_err={summary['yield_abs_error_kg_ha']:.4g} I_err={summary['irrigation_abs_error_mm']:.4g} N_err={summary['nitrogen_abs_error_kg_ha']:.4g}", flush=True)

        summaries = pd.DataFrame(records).sort_values(["seed", "year"]).reset_index(drop=True)
        if len(summaries) != 30 or summaries.duplicated(["seed", "year"]).any():
            raise RuntimeError("Expected 30 unique seed-year replays")
        if int(summaries.steps.sum()) != 3048:
            raise RuntimeError(f"Unexpected total daily-step count: {int(summaries.steps.sum())}")
        if not (summaries.rss_peak_mb < RSS_LIMIT_MB).all():
            raise MemoryError("A replay exceeded RSS hard stop")
        episode_dir = OUT / "episodes"
        trace_parts = [pd.read_csv(path) for path in sorted(episode_dir.glob("decision_trace_seed*_*.csv"))]
        trace = pd.concat(trace_parts, ignore_index=True).sort_values(["seed", "year", "timestep"]).reset_index(drop=True)
        if len(trace) != 3048 or trace.duplicated(["seed", "year", "timestep"]).any():
            raise RuntimeError(f"Incomplete decision trace: {len(trace)} rows")
        trace["dap_occurrence"] = trace.groupby(["seed", "year", "dap"], sort=False).cumcount()
        trace_path = OUT / "decision_trace_seed345.csv"
        if trace_path.exists():
            raise FileExistsError("Refusing to overwrite combined decision trace")
        trace.to_csv(trace_path, index=False, encoding="utf-8-sig", lineterminator="\n")
        observation_summary, policy_response, annual, analysis = summarize_decisions(trace, schema)
        observation_summary.to_csv(OUT / "same_dap_observation_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        policy_response.to_csv(OUT / "same_dap_policy_response.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        annual.to_csv(OUT / "yearly_policy_diagnostics.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        v = verdicts(policy_response, annual, trace[trace.seed.eq(5)])
        analysis["verdicts"] = v
        analysis["decision_rules"] = {
            "observation_separability": "median same-DAP pairwise RMS standardized distance; HIGH >=0.50, MODERATE >=0.10",
            "mask_blocking": "share of decisions where unmasked argmax is forbidden; HIGH >=25%, MODERATE >=5%, LOW >0%, NONE=0",
            "policy_sensitivity": "median same-DAP pairwise masked probability JS; HIGH >=0.10 bits, MODERATE >=0.02 bits",
            "argmax_boundary": "share of shared DAP groups with unchanged chosen action, JS>=0.01 bits, and minimum logit margin<=0.25 or probability margin<=0.05; STRONG>=40%, MODERATE>=10%",
        }
        analysis["coverage"] = {"seed_years": len(summaries), "steps": len(trace), "observation_dimensions": len(schema), "all_seed_year_coverage": 10, "all_raw_equals_policy_input": bool(trace.raw_equals_policy_input.all()), "all_predict_matches_masked_argmax": bool(trace.predict_distribution_action_match.all()), "all_actions_match_formal_trace": bool(trace.action_matches_formal.all())}
        figure_paths = make_figures(observation_summary, policy_response, schema)
        prov = provenance_for(summaries, schema)
        prov["rendered_input_hash_checks"] = {f"seed_{int(row.seed)}_{int(row.year)}": row.rendered_input_provenance["checks"] for row in summaries.itertuples()}
        write_json(OUT / "analysis_summary.json", analysis)
        write_json(OUT / "input_provenance.json", prov)
        summaries["action_trace_exact"] = True
        summaries["predict_distribution_action_match"] = True
        summaries["gates_json"] = summaries.gates.map(lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True))
        summaries.drop(columns=["gates", "rendered_input_provenance"], errors="ignore").to_csv(OUT / "episode_reconciliation_seed345.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        write_report_final(summaries, observation_summary, policy_response, annual, schema, analysis, prov, figure_paths)
        manifest.update({"status": "passed", "passed": True, "completed_utc": pd.Timestamp.utcnow().isoformat(), "episode_count": len(summaries), "step_count": len(trace), "peak_rss_mb": float(summaries.rss_peak_mb.max()), "verdicts": v, "report_path": rel(REPORT)})
        write_json(manifest_path, manifest, replace=True)
        print(f"[formal-pass] episodes={len(summaries)} steps={len(trace)} peak_rss={summaries.rss_peak_mb.max():.1f} MiB report={rel(REPORT)}", flush=True)
    except Exception as exc:
        manifest.update({"status": "failed", "passed": False, "completed_utc": pd.Timestamp.utcnow().isoformat(), "episodes_completed": len(records), "error": repr(exc), "traceback": traceback.format_exc()})
        write_json(manifest_path, manifest, replace=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="YC fixed-model observation-mask-action audit; evaluation replay only")
    parser.add_argument("--phase", choices=["smoke", "formal", "analyze"], required=True)
    args = parser.parse_args()
    if args.phase == "smoke" and OUT.exists():
        allowed = {"logs", "episodes"}
        unexpected = [path.name for path in OUT.iterdir() if path.name not in allowed and not path.name.startswith("smoke_manifest")]
        episode_files = list((OUT / "episodes").glob("*.csv")) if (OUT / "episodes").exists() else []
        if unexpected or episode_files or (OUT / "evaluation").exists() or (OUT / "observation_schema.json").exists():
            raise FileExistsError(f"Refusing to reuse smoke output with replay evidence: {unexpected or [p.name for p in episode_files]}")
    if args.phase == "formal" and not OUT.is_dir():
        raise FileNotFoundError("Run --phase smoke first")
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    log_path = OUT / "logs" / f"{args.phase}.log"
    if log_path.exists():
        attempt = 1
        while True:
            candidate = OUT / "logs" / f"{args.phase}_retry_{attempt:02d}.log"
            if not candidate.exists():
                log_path = candidate
                break
            attempt += 1
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        original_out, original_err = sys.stdout, sys.stderr
        sys.stdout = Tee(original_out, log)
        sys.stderr = Tee(original_err, log)
        try:
            print(f"[start] phase={args.phase} output={rel(OUT)} rss_limit={RSS_LIMIT_MB:.0f} MiB", flush=True)
            if args.phase == "analyze":
                analyze_existing()
            else:
                runner, pilot = load_runtime()
            if args.phase == "smoke":
                run_smoke(runner, pilot)
            elif args.phase == "formal":
                run_formal(runner, pilot)
        except Exception:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            raise
        finally:
            sys.stdout, sys.stderr = original_out, original_err
            gc.collect()


if __name__ == "__main__":
    main()
