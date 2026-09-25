from __future__ import annotations

import argparse
import csv
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = ROOT / "results" / "yc_random_weather_ppo" / "004_03"
CONFIG_ROOT = TASK_ROOT / "config"
TRAIN_ROOT = TASK_ROOT / "training_verified"
EVAL_ROOT = TASK_ROOT / "evaluation"
EVAL_RUN_ROOT = EVAL_ROOT / "verified_weather_refresh"
SMOKE_ROOT = TASK_ROOT / "smoke"
CLI = ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_05_02" / "final" / "CNYC.CLI"
FITTING_WEATHER = ROOT / "results" / "yc_weather_gapfill_finalize" / "yc_wgen_fitting_weather_2004_2013.csv"
WGEN_QC = ROOT / "results" / "yc_random_weather_episode_validation" / "004_02" / "summary.json"
LOWIC_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
TRAIN_YEARS = list(range(2004, 2014))
OBSERVED_YEARS = list(range(2014, 2024))
WGEN_TRAIN_SEEDS = list(range(1001, 1081))
WGEN_EVAL_SEEDS = list(range(1081, 1101))
PPO_SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = 100_000
CHECKPOINT_STEPS = [25_000, 50_000, 75_000, 100_000]
SMOKE_TIMESTEPS = 432
MAX_SCHEDULE_EPISODES = TOTAL_TIMESTEPS
HISTORICAL_SCHEDULE_SEED = 64003
WGEN_SCHEDULE_SEED = 64004
RUNTIME_BOOTSTRAP_SEED = 66003
EVAL_RUNTIME_SEED = 66004
WGEN_EVAL_CROP_YEAR = 2008
MAX_PROCESS_TREE_RSS_MB = 6000
RECREATE_WGEN_ENV_PER_EPISODE = False
EXPECTED_CLI_SHA256 = "65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929"
EXPECTED_FITTING_SHA256 = "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34"
OLD_REWARD_FORMULA = "0.06 * final_grnwt - 0.04 * cumfert"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
RUNNER_DIR = ROOT / "src" / "055_yca_lowIC_site_transfer"
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))
CANONICAL_CONTEXT = None


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any, refuse_different: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if refuse_different and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite different task evidence: {rel(path)}")
        return
    path.write_text(content, encoding="utf-8", newline="\n")


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig" if not exists else "utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def generate_year_schedule(length: int = MAX_SCHEDULE_EPISODES) -> list[dict[str, int]]:
    rng = np.random.default_rng(HISTORICAL_SCHEDULE_SEED)
    rows: list[dict[str, int]] = []
    for block in range(math.ceil(length / len(TRAIN_YEARS))):
        for slot, year in enumerate(rng.permutation(TRAIN_YEARS).tolist()):
            if len(rows) == length:
                break
            rows.append({"episode_index": len(rows) + 1, "block": block + 1, "slot": slot + 1, "historical_year": int(year)})
    return rows


def generate_wgen_schedule(year_rows: list[dict[str, int]]) -> list[dict[str, int]]:
    rng = np.random.default_rng(WGEN_SCHEDULE_SEED)
    rows: list[dict[str, int]] = []
    for block in range(math.ceil(len(year_rows) / len(WGEN_TRAIN_SEEDS))):
        block_seeds = rng.permutation(WGEN_TRAIN_SEEDS).tolist()
        for slot, weather_seed in enumerate(block_seeds):
            if len(rows) == len(year_rows):
                break
            year_row = year_rows[len(rows)]
            rows.append(
                {
                    "episode_index": len(rows) + 1,
                    "weather_block": block + 1,
                    "weather_slot": slot + 1,
                    "historical_year_context": year_row["historical_year"],
                    "weather_seed": int(weather_seed),
                }
            )
    return rows


def _import_canonical():
    global CANONICAL_CONTEXT
    if CANONICAL_CONTEXT is not None:
        return CANONICAL_CONTEXT
    import run_055_00_yca_lowIC_expanded_action_maskableppo as runner
    import ppo_safe_rendering

    engine = runner.engine
    engine.STATION = "YCA"
    engine.SITES = ["YCA"]
    engine.LOWIC_INPUT_ROOT = LOWIC_INPUT_ROOT
    engine.BINARY_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
    engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
    engine.patch_base_module(TASK_ROOT, ROOT / "docs" / "yc_random_weather_ppo_controlled_pilot.md", TOTAL_TIMESTEPS, CHECKPOINT_STEPS)
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    batch = engine.base03222
    config = engine.load_config()
    config["seed"] = 0
    config["total_timesteps"] = TOTAL_TIMESTEPS
    config["paths"]["output_root"] = rel(TASK_ROOT)
    split = batch.load_split()
    selection = batch.build_selection(split)
    env_config = batch.base.direct_ppo.build_env_config(config, selection)
    CANONICAL_CONTEXT = (runner, engine, batch, ppo_safe_rendering, config, split, selection, env_config)
    return CANONICAL_CONTEXT


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def preflight_payload() -> dict[str, Any]:
    runner, engine, batch, _, config, split, selection, _ = _import_canonical()
    cfg_path = runner.DEFAULT_CONFIG
    source_yaml = ROOT / "benchmark_results" / "055_00_yca_lowIC_expanded_action_maskableppo" / "configs" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
    formal_cfg_path = ROOT / "configs" / "055_00_yca_lowIC_expanded_action_maskableppo.json"
    formal_result_path = ROOT / "benchmark_results" / "055_00_yca_lowIC_expanded_action_maskableppo" / "055_00_formal_result.json"
    source_yaml_config = batch.direct_ppo.load_yaml(source_yaml)
    formal_task_config = json.loads(formal_cfg_path.read_text(encoding="utf-8"))
    formal_result = json.loads(formal_result_path.read_text(encoding="utf-8"))
    source_yaml_timesteps = int(source_yaml_config["total_timesteps"])
    formal_task_timesteps = int(formal_task_config["training"]["total_timesteps"])
    formal_result_timesteps = int(formal_result["action_gate"]["final_checkpoint"])
    budget_provenance = {
        "source_yaml_declared_total_timesteps": source_yaml_timesteps,
        "source_yaml_role": "032_00 smoke/base template; its 5000-step value is not the 055_00 formal budget",
        "formal_task_config_total_timesteps": formal_task_timesteps,
        "formal_result_total_timesteps": formal_result_timesteps,
        "formal_result_budget_evidence": "action_gate.final_checkpoint",
        "effective_pilot_total_timesteps": int(config["total_timesteps"]),
        "authority": "055_00 formal task config and completed formal result",
    }
    qc = json.loads(WGEN_QC.read_text(encoding="utf-8"))
    cli_hash = sha256_file(CLI)
    fit_hash = sha256_file(FITTING_WEATHER)
    yca_split = split[split["station_code"].astype(str).eq("YCA")].copy()
    actual_train = sorted(pd.to_numeric(yca_split.loc[yca_split["split"].eq("train"), "year"]).astype(int).tolist())
    actual_eval = sorted(pd.to_numeric(yca_split.loc[yca_split["split"].eq("validation"), "year"]).astype(int).tolist())
    issues = []
    for path in (CLI, FITTING_WEATHER, LOWIC_INPUT_ROOT / "YC" / "CNYC0801.MZX", formal_cfg_path, source_yaml):
        if not path.is_file():
            issues.append(f"missing required input: {rel(path)}")
    if cli_hash != EXPECTED_CLI_SHA256:
        issues.append("frozen CNYC.CLI hash mismatch")
    if fit_hash != EXPECTED_FITTING_SHA256:
        issues.append("fitting-weather hash mismatch")
    if actual_train != TRAIN_YEARS or actual_eval != OBSERVED_YEARS:
        issues.append(f"YCA train/evaluation split mismatch: {actual_train} / {actual_eval}")
    if config["reward"].get("reward_type") != "harvest_yield_minus_water_nitrogen_cost_plus_stress_relief_scaled_0p001":
        issues.append("effective 055 reward type differs from expected canonical chain")
    if int(config.get("total_timesteps", -1)) != TOTAL_TIMESTEPS:
        issues.append("effective 055 training budget is not 100K")
    if formal_task_timesteps != TOTAL_TIMESTEPS or formal_result_timesteps != TOTAL_TIMESTEPS:
        issues.append("055_00 formal task config/result budget differs from the 100K pilot budget")
    if formal_result.get("next_step_allowed") is not True:
        issues.append("055_00 formal result did not authorize continuation")
    if qc.get("yc_random_weather_ready_for_ppo_pilot") != "YES" or qc.get("generation_status") != "PASS":
        issues.append("004_02 WGEN episode-quality gate did not pass")
    if set(WGEN_TRAIN_SEEDS) & set(WGEN_EVAL_SEEDS):
        issues.append("training and held-out WGEN seed pools overlap")
    if not selection[selection["station_code"].astype(str).eq("YCA")]["selected_for_train"].any():
        issues.append("YCA training selection is empty")
    return {
        "passed": not issues,
        "issues": issues,
        "canonical_source_config": rel(source_yaml),
        "canonical_source_yaml": rel(source_yaml),
        "canonical_formal_task_config": rel(formal_cfg_path),
        "canonical_formal_result": rel(formal_result_path),
        "training_budget_provenance": budget_provenance,
        "training_script": rel(ROOT / "src" / "055_yca_lowIC_site_transfer" / "run_055_00_yca_lowIC_expanded_action_maskableppo.py"),
        "effective_training_engine": rel(ROOT / "src" / "run_five_site_half_split_stress_aware_maskableppo_batch_032_22.py"),
        "training_years_actual": actual_train,
        "observed_years_actual": actual_eval,
        "total_timesteps": int(config["total_timesteps"]),
        "checkpoint_steps": CHECKPOINT_STEPS,
        "ppo": config["ppo"],
        "policy": {
            "class": "MaskablePPO",
            "policy_name": "MlpPolicy",
            "net_arch": config["ppo"]["net_arch"],
            "activation": "Tanh (SB3 MlpPolicy default)",
            "vf_coef": 0.5,
            "vf_coef_source": "MaskablePPO default; not passed by canonical ppo_kwargs",
        },
        "normalization": {"observation_normalization": False, "vecnormalize": False, "forecast_features": False},
        "action_grid": config["discrete_actions"],
        "action_safety": config["action_safety"],
        "reward": config["reward"],
        "effective_reward_wrapper": "LateIrrigationReserveMaskWrapper -> I240SwfacGuardrailWrapper -> stress-aware reward wrapper",
        "old_prompt_reward_formula": {"formula": OLD_REWARD_FORMULA, "status": "SUPERSEDED / NOT_APPLICABLE"},
        "input_root": rel(LOWIC_INPUT_ROOT),
        "frozen_cli": {"path": rel(CLI), "sha256": cli_hash},
        "fitting_weather": {"path": rel(FITTING_WEATHER), "sha256": fit_hash, "period": "2004-2013"},
        "004_02_weather_qc": {
            "status": qc.get("random_weather_episode_quality_status"),
            "ready": qc.get("yc_random_weather_ready_for_ppo_pilot"),
            "temperature_note": "synthetic TMIN mean approximately fitting observed +0.58 C",
            "full_year_wgen_validation": "DEFERRED / OPTIONAL ADDITIONAL VALIDATION",
        },
        "runtime": {
            "docker_container": "nifty_taussig",
            "dssat_binary": "/opt/dssat_pdi/run_dssat",
            "gym_dssat_pdi_version": _version("gym-dssat-pdi"),
            "stable_baselines3_version": _version("stable-baselines3"),
            "sb3_contrib_version": _version("sb3-contrib"),
            "torch_version": _version("torch"),
        },
        "seed_plan": {
            "ppo_seeds": PPO_SEEDS,
            "historical_weather_schedule_seed": HISTORICAL_SCHEDULE_SEED,
            "random_weather_schedule_seed": WGEN_SCHEDULE_SEED,
            "runtime_bootstrap_seed": RUNTIME_BOOTSTRAP_SEED,
            "evaluation_runtime_seed": EVAL_RUNTIME_SEED,
            "training_wgen_seed_pool": [min(WGEN_TRAIN_SEEDS), max(WGEN_TRAIN_SEEDS)],
            "heldout_wgen_seed_pool": [min(WGEN_EVAL_SEEDS), max(WGEN_EVAL_SEEDS)],
            "seed_roles_are_disjoint": True,
        },
        "run_config": config,
    }


def freeze_config() -> dict[str, Any]:
    payload = preflight_payload()
    if not payload["passed"]:
        raise RuntimeError("Preflight failed: " + "; ".join(payload["issues"]))
    years = generate_year_schedule()
    wgen = generate_wgen_schedule(years)
    historical_path = CONFIG_ROOT / "training_weather_schedule_historical.csv"
    random_path = CONFIG_ROOT / "training_weather_schedule_random.csv"
    if not CONFIG_ROOT.exists():
        CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    for path, rows in ((historical_path, years), (random_path, wgen)):
        if path.exists():
            existing = pd.read_csv(path)
            proposed = pd.DataFrame(rows)
            if not existing.equals(proposed):
                raise FileExistsError(f"Refusing to overwrite a different frozen schedule: {rel(path)}")
        else:
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
    eval_rows = []
    for seed in WGEN_EVAL_SEEDS:
        eval_rows.append({
            "evaluation_weather_type": "heldout_wgen",
            "evaluation_weather_seed": seed,
            "evaluation_weather_year": "",
            "crop_year_context": WGEN_EVAL_CROP_YEAR,
            "schedule_id": "heldout_wgen_1081_1100_v1",
        })
    for year in OBSERVED_YEARS:
        eval_rows.append({
            "evaluation_weather_type": "observed_weather",
            "evaluation_weather_seed": "",
            "evaluation_weather_year": year,
            "crop_year_context": year,
            "schedule_id": "observed_yc_2014_2023_v1",
        })
    pd.DataFrame(eval_rows).to_csv(CONFIG_ROOT / "evaluation_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    canonical = {
        "canonical_source_config": payload["canonical_source_config"],
        "canonical_source_yaml": payload["canonical_source_yaml"],
        "canonical_formal_task_config": payload["canonical_formal_task_config"],
        "canonical_formal_result": payload["canonical_formal_result"],
        "training_budget_provenance": payload["training_budget_provenance"],
        "training_script": payload["training_script"],
        "effective_training_engine": payload["effective_training_engine"],
        "config_freeze_status": "FROZEN",
        "superseded_reward_formula": payload["old_prompt_reward_formula"],
        "training": {"total_timesteps": TOTAL_TIMESTEPS, "checkpoint_steps": CHECKPOINT_STEPS},
        "algorithm": payload["policy"],
        "ppo_hyperparameters": payload["ppo"],
        "normalization": payload["normalization"],
        "reward": payload["reward"],
        "action_grid": payload["action_grid"],
        "action_safety": payload["action_safety"],
        "run_config": payload["run_config"],
        "input_root": payload["input_root"],
        "frozen_cli": payload["frozen_cli"],
        "runtime": payload["runtime"],
    }
    design = {
        "task": "004_03 YC random-weather PPO controlled pilot",
        "correction": "004_03_01 takes precedence over the superseded reward formula in 004_03",
        "canonical_source_config": payload["canonical_source_config"],
        "canonical_formal_task_config": payload["canonical_formal_task_config"],
        "canonical_formal_result": payload["canonical_formal_result"],
        "old_prompt_reward_formula": OLD_REWARD_FORMULA,
        "old_prompt_reward_status": "SUPERSEDED / NOT_APPLICABLE",
        "weather_augmentation_only_change": "YES",
        "weather_augmentation_only_change_verified": True,
        "training_budget_provenance": payload["training_budget_provenance"],
        "regimes": ["HISTORICAL_WEATHER", "RANDOM_WEATHER_WGEN"],
        "ppo_seeds": PPO_SEEDS,
        "total_models": 6,
        "training_years": TRAIN_YEARS,
        "shared_year_schedule": {
            "seed": HISTORICAL_SCHEDULE_SEED,
            "length": MAX_SCHEDULE_EPISODES,
            "design": "Each consecutive block is a seeded permutation of all ten years 2004-2013; identical across both regimes and all PPO seeds.",
        },
        "training_wgen_weather_seeds": WGEN_TRAIN_SEEDS,
        "wgen_weather_schedule": {
            "seed": WGEN_SCHEDULE_SEED,
            "length": MAX_SCHEDULE_EPISODES,
            "design": "Each consecutive block contains every seed 1001-1080 exactly once in a seeded permutation; identical across PPO seeds.",
        },
        "evaluation": {
            "heldout_wgen_seeds": WGEN_EVAL_SEEDS,
            "heldout_wgen_crop_year_context": WGEN_EVAL_CROP_YEAR,
            "observed_comparison_years": OBSERVED_YEARS,
            "observed_period_label": "independent comparison period; not certified as pristine final test set",
            "deterministic_actions": True,
        },
        "seed_roles": {
            "ppo_seed": PPO_SEEDS,
            "historical_weather_schedule_seed": HISTORICAL_SCHEDULE_SEED,
            "wgen_weather_schedule_seed": WGEN_SCHEDULE_SEED,
            "runtime_bootstrap_seed": RUNTIME_BOOTSTRAP_SEED,
            "evaluation_runtime_seed": EVAL_RUNTIME_SEED,
            "distinct": True,
        },
        "smoke": {"regime": "RANDOM_WEATHER_WGEN", "ppo_seed": 0, "timesteps": SMOKE_TIMESTEPS, "performance_statistics_excluded": True},
        "resource_guard": {"run_models_serially": True, "max_process_tree_rss_mb": MAX_PROCESS_TREE_RSS_MB, "stop_on_threshold": True},
        "software_changes": {"runtime_modified": False, "dssat_binary_modified": False, "cli_modified": False, "wgen_refit": False, "reward_modified": False},
    }
    write_json(CONFIG_ROOT / "canonical_yc_ppo_config.json", canonical)
    write_json(CONFIG_ROOT / "experiment_design.json", design)
    return {"passed": True, "canonical_config": rel(CONFIG_ROOT / "canonical_yc_ppo_config.json"), "design": rel(CONFIG_ROOT / "experiment_design.json"), "schedule_rows": len(years), "wgen_schedule_rows": len(wgen), "evaluation_rows": len(eval_rows)}


class SequenceWeatherRng:
    def __init__(self, weather_seed: int):
        self.weather_seed = int(weather_seed)
        self.calls = 0

    def randint(self, low: int, high: int, **_: Any) -> int:
        self.calls += 1
        return self.weather_seed


def set_wgen_filex_mode(args: dict[str, Any], year: int, weather_seed: int, run_tag: str) -> tuple[str, str]:
    source = Path(args["fileX_template_path"])
    source_text = source.read_text(encoding="utf-8", errors="replace")
    wgen_text, replacements = re.subn(
        r"(?m)^([ \t]*\d+[ \t]+ME[ \t]+)\S+",
        r"\g<1>W",
        source_text,
    )
    modes = re.findall(r"(?m)^\s*\d+\s+ME\s+(\S+)", wgen_text)
    if not replacements or not modes or any(mode != "W" for mode in modes):
        raise RuntimeError(f"Could not enforce WTHER=W in task-local FileX template: {rel(source)}")
    safe_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", run_tag)
    target_dir = TASK_ROOT / "runtime_templates" / safe_tag
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"YCA_{int(year)}_rseed1_{int(weather_seed)}.jinja2"
    if target.exists():
        if target.read_text(encoding="utf-8") != wgen_text:
            raise FileExistsError(f"Refusing to overwrite different WGEN template: {rel(target)}")
    else:
        target.write_text(wgen_text, encoding="utf-8", newline="\n")
    args["fileX_template_path"] = str(target)
    return str(target), sha256_file(target)


def find_dssat_instance(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (
            type(current).__name__ == "DssatPdi"
            and type(current).__module__.startswith("gym_dssat_pdi.")
            and all(hasattr(current, name) for name in ("_random_generator", "_rseed1", "_get_sockets_"))
        ):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("Could not locate DssatPdi instance through the existing wrapper chain")


def make_weather_env(config: dict[str, Any], env_config: dict[str, Any], year: int, regime: str, runtime_seed: int, run_tag: str, bootstrap_weather_seed: int | None):
    import gym
    import gym_dssat_pdi.envs.dssat_pdi as dssat_module
    from sb3_wrapper import GymDssatWrapper

    _, _, batch, rendering, _, _, _, _ = _import_canonical()
    direct = batch.base.direct_ppo
    year_info = direct.find_year(env_config, "YCA", int(year))
    args = rendering.build_env_args(
        station="YCA",
        year=int(year),
        planting_date=year_info["planting_date"],
        seed=int(runtime_seed),
        config=env_config,
        run_tag=run_tag,
        evaluation=run_tag.startswith("eval_"),
        mode=env_config.get("runtime", {}).get("mode", "all"),
        linked_management=True,
    )
    args["random_weather"] = regime == "RANDOM_WEATHER_WGEN"
    if args["random_weather"]:
        if bootstrap_weather_seed is None:
            raise RuntimeError("WGEN environment requires an explicit weather seed before runtime launch")
        wgen_template_path, wgen_template_hash = set_wgen_filex_mode(args, year, bootstrap_weather_seed, run_tag)
        auxiliary = list(args["auxiliary_file_paths"])
        if str(CLI) not in auxiliary:
            auxiliary.append(str(CLI))
        args["auxiliary_file_paths"] = auxiliary

    class_type = getattr(dssat_module, "DssatPdi", None)
    if class_type is None:
        from scripts.run_yc_wgen_seed_pilot import _runtime_class
        class_type = _runtime_class(dssat_module)
    original = class_type._get_sockets_
    socket_injections: list[dict[str, Any]] = []
    if args["random_weather"] and bootstrap_weather_seed is not None:
        def inject_before_server(instance: Any) -> Any:
            instance._rseed1 = int(bootstrap_weather_seed)
            socket_injections.append({"instance_id": id(instance), "rseed1_": int(instance._rseed1)})
            return original(instance)
        class_type._get_sockets_ = inject_before_server
    try:
        raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **args).unwrapped
    finally:
        class_type._get_sockets_ = original
    if args["random_weather"] and bootstrap_weather_seed is not None:
        raw._random_generator = SequenceWeatherRng(int(bootstrap_weather_seed))
        raw._pilot_filex_wther = "W"
        raw._pilot_filex_template_path = wgen_template_path
        raw._pilot_filex_template_sha256 = wgen_template_hash
        raw._pilot_weather_states = []
        original_get_state = raw._get_state

        def capture_weather_state(*state_args: Any, **state_kwargs: Any) -> Any:
            result = original_get_state(*state_args, **state_kwargs)
            state = getattr(raw, "_state", None)
            if isinstance(state, dict):
                raw._pilot_weather_states.append(copy.deepcopy(state))
            return result

        raw._get_state = capture_weather_state
    wrapped = GymDssatWrapper(raw)
    result = batch.base.StressAwareDiscreteWrapper(wrapped, config)
    if args["random_weather"]:
        dssat = find_dssat_instance(result)
        dssat._pilot_socket_injection_trace = socket_injections
        dssat._pilot_constructor_rseed1 = int(dssat._rseed1)
    return result


class ScheduledEpisodeEnv:
    def __init__(self, config: dict[str, Any], env_config: dict[str, Any], schedule: list[dict[str, Any]], regime: str, ppo_seed: int, run_tag: str, runtime_seed: int, max_cached_years: int = 10):
        import gymnasium as gym

        class _Env(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self, outer):
                super().__init__()
                self.outer = outer
                first = outer.schedule[0]
                bootstrap_seed = int(first["weather_seed"]) if outer.regime == "RANDOM_WEATHER_WGEN" else None
                first_env = outer._env_for(int(first["historical_year"]), bootstrap_seed)
                self.action_space = first_env.action_space
                self.observation_space = first_env.observation_space

            def reset(self, *, seed=None, options=None):
                super().reset(seed=None)
                return self.outer._reset(seed, options)

            def step(self, action):
                return self.outer._step(action)

            def action_masks(self):
                return self.outer.current_env.action_masks()

            def close(self):
                return self.outer.close()

            def __getattr__(self, name):
                return getattr(self.outer.current_env, name)

        if not schedule:
            raise ValueError("An explicit non-empty episode schedule is required")
        self.config = config
        self.env_config = env_config
        self.schedule = schedule
        self.regime = regime
        self.ppo_seed = int(ppo_seed)
        self.run_tag = run_tag
        self.runtime_seed = int(runtime_seed)
        self.max_cached_years = int(max_cached_years)
        self.envs: dict[int, Any] = {}
        self.env_use_order: list[int] = []
        self.current_env: Any = None
        self.current_schedule: dict[str, Any] | None = None
        self.next_index = 0
        self.episode_rows: list[dict[str, Any]] = []
        self.episode_reward = 0.0
        self.episode_component_sum = 0.0
        self.episode_days = 0
        self.env_reset_seed_seen: int | None = None
        self._gym_env = _Env(self)

    def _env_for(self, year: int, bootstrap_weather_seed: int | None):
        if year in self.envs:
            if year in self.env_use_order:
                self.env_use_order.remove(year)
            self.env_use_order.append(year)
            return self.envs[year]
        env = make_weather_env(
            self.config,
            self.env_config,
            year,
            self.regime,
            self.runtime_seed,
            f"{self.run_tag}_{year}",
            bootstrap_weather_seed,
        )
        self.envs[year] = env
        self.env_use_order.append(year)
        while len(self.env_use_order) > self.max_cached_years:
            evicted = self.env_use_order.pop(0)
            old = self.envs.pop(evicted)
            old.close()
            gc.collect()
        return env

    def _reset(self, outer_seed: int | None, options: dict | None):
        if self.next_index >= len(self.schedule):
            raise RuntimeError(f"Episode schedule exhausted at episode {self.next_index + 1}")
        row = dict(self.schedule[self.next_index])
        self.next_index += 1
        self.current_schedule = row
        self.env_reset_seed_seen = None if outer_seed is None else int(outer_seed)
        year = int(row["historical_year"])
        expected_weather_seed = int(row["weather_seed"]) if self.regime == "RANDOM_WEATHER_WGEN" else None
        if RECREATE_WGEN_ENV_PER_EPISODE and expected_weather_seed is not None and year in self.envs:
            # WGEN is initialized with the DSSAT process; a new process is needed per seed.
            stale_env = self.envs.pop(year)
            if year in self.env_use_order:
                self.env_use_order.remove(year)
            stale_env.close()
            gc.collect()
        self.current_env = self._env_for(year, expected_weather_seed)
        dssat = find_dssat_instance(self.current_env)
        before_state = {
            "closed": bool(dssat.closed),
            "reset_counter": int(dssat.reset_counter),
            "rng_type": type(dssat._random_generator).__name__,
            "constructor_rseed1_": getattr(dssat, "_pilot_constructor_rseed1", None),
            "socket_injection_trace": getattr(dssat, "_pilot_socket_injection_trace", []),
        }
        if expected_weather_seed is not None:
            dssat._random_generator = SequenceWeatherRng(expected_weather_seed)
            dssat._pilot_weather_states = []
        obs, info = self.current_env.reset(seed=None)
        actual_seed = int(dssat._rseed1)
        if expected_weather_seed is not None and actual_seed != expected_weather_seed:
            raise RuntimeError(
                f"PDI rseed1 mismatch: scheduled={expected_weather_seed}, actual={actual_seed}; "
                f"before={before_state}; after_closed={dssat.closed}; after_reset_counter={dssat.reset_counter}; "
                f"after_rng={type(dssat._random_generator).__name__}; "
                f"adapter_calls={getattr(dssat._random_generator, 'calls', None)}"
            )
        if options:
            info = dict(info or {})
            info["reset_options_ignored"] = True
        info = dict(info or {})
        info.update({
            "active_year": year,
            "training_regime": self.regime,
            "ppo_seed": self.ppo_seed,
            "episode_index": int(row["episode_index"]),
            "weather_seed": expected_weather_seed,
            "rseed1_": actual_seed if expected_weather_seed is not None else "",
            "outer_reset_seed_received": self.env_reset_seed_seen,
        })
        self.episode_reward = 0.0
        self.episode_component_sum = 0.0
        self.episode_days = 0
        self._active_weather_hash = self._runtime_weather_hash(dssat)
        return obs, info

    @staticmethod
    def _runtime_weather_hash(dssat: Any) -> str:
        states = getattr(dssat, "_pilot_weather_states", [])
        if not states:
            return ""
        from scripts.run_yc_wgen_seed_pilot import canonical_weather_bytes, daily_weather_from_states, sha256_bytes

        rows = daily_weather_from_states(states)
        return sha256_bytes(canonical_weather_bytes(rows)) if rows else ""

    def _step(self, action):
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        reward = float(reward)
        self.episode_reward += reward
        self.episode_days += 1
        action_info = dict(getattr(self.current_env, "last_action_info", {}) or {})
        component_reward = action_info.get("reward_after_swfac_guardrail", reward)
        self.episode_component_sum += float(component_reward)
        info = dict(info or {})
        if terminated or truncated:
            row = self.current_schedule or {}
            last_obs = dict(getattr(self.current_env, "last_obs_dict", {}) or {})
            safety = getattr(self.current_env, "safety_state", None)
            dssat = find_dssat_instance(self.current_env)
            actual_seed = int(dssat._rseed1)
            weather_seed = row.get("weather_seed") if self.regime == "RANDOM_WEATHER_WGEN" else None
            final_yield = _finite_or_none(last_obs.get("grnwt"))
            irrigation = _finite_or_none(getattr(safety, "cumulative_irrigation", None))
            fertilizer = _finite_or_none(getattr(safety, "cumulative_n", None))
            self.episode_rows.append({
                "training_regime": self.regime,
                "ppo_seed": self.ppo_seed,
                "episode_index": int(row.get("episode_index", self.next_index)),
                "historical_year": int(row.get("historical_year", 0)),
                "training_weather_source": "WGEN_random" if self.regime == "RANDOM_WEATHER_WGEN" else "observed_historical",
                "training_weather_seed": weather_seed if weather_seed is not None else "",
                "scheduled_rseed1_": weather_seed if weather_seed is not None else "",
                "actual_rseed1_sent_to_pdi": actual_seed if weather_seed is not None else "",
                "outer_reset_seed_received": self.env_reset_seed_seen if self.env_reset_seed_seen is not None else "",
                "episode_return": self.episode_reward,
                "reward_component_sum": self.episode_component_sum,
                "reward_decomposition_abs_error": abs(self.episode_reward - self.episode_component_sum),
                "yield": final_yield,
                "fertilizer": fertilizer,
                "irrigation": irrigation,
                "episode_days": self.episode_days,
                "runtime_weather_sha256": self._runtime_weather_hash(dssat),
                "filex_wther": getattr(dssat, "_pilot_filex_wther", ""),
                "runtime_filex_template_sha256": getattr(dssat, "_pilot_filex_template_sha256", ""),
                "status": "completed",
            })
        return obs, reward, terminated, truncated, info

    def close(self):
        for env in list(self.envs.values()):
            try:
                env.close()
            except Exception:
                pass
        self.envs.clear()
        self.env_use_order.clear()
        gc.collect()


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(np.asarray(value).item())
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, AttributeError):
        return None


def _tree_rss_mb() -> float:
    try:
        import psutil
        process = psutil.Process()
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total / (1024 * 1024)
    except Exception:
        return float("nan")


def schedule_rows(regime: str, limit: int | None = None) -> list[dict[str, Any]]:
    hist = pd.read_csv(CONFIG_ROOT / "training_weather_schedule_historical.csv").to_dict(orient="records")
    if regime == "HISTORICAL_WEATHER":
        rows = [{"episode_index": int(row["episode_index"]), "historical_year": int(row["historical_year"]), "weather_seed": ""} for row in hist]
    elif regime == "RANDOM_WEATHER_WGEN":
        random_rows = pd.read_csv(CONFIG_ROOT / "training_weather_schedule_random.csv").to_dict(orient="records")
        rows = [{
            "episode_index": int(row["episode_index"]),
            "historical_year": int(row["historical_year_context"]),
            "weather_seed": int(row["weather_seed"]),
        } for row in random_rows]
        if [r["historical_year"] for r in rows] != [int(r["historical_year"]) for r in hist]:
            raise RuntimeError("WGEN training schedule does not share the frozen historical-year context schedule")
    else:
        raise ValueError(regime)
    return rows[:limit] if limit is not None else rows


def make_run_config(regime: str, seed: int, out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    runner, engine, batch, _, cfg, split, selection, _ = _import_canonical()
    cfg = json.loads(json.dumps(cfg))
    cfg["seed"] = int(seed)
    cfg["total_timesteps"] = TOTAL_TIMESTEPS
    cfg["paths"]["output_root"] = rel(out)
    cfg["runtime"]["smoke_station"] = "YCA"
    cfg["runtime"]["max_steps"] = 260
    batch.base.direct_ppo.OUTPUT_ROOT = out
    env_config = batch.base.direct_ppo.build_env_config(cfg, selection)
    env_config["paths"]["output_root"] = rel(out)
    return cfg, env_config


class TrainingCallback:
    def __init__(self, env: ScheduledEpisodeEnv, run_dir: Path, checkpoints: list[int], smoke: bool = False):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__(verbose=0)

            def _on_step(self) -> bool:
                step = int(self.model.num_timesteps)
                pending = [target for target in outer.checkpoints if target not in outer.saved and step >= target]
                for target in pending:
                    path = run_dir / "models" / f"checkpoint_{target}.zip"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(path))
                    outer.saved.append(target)
                    outer.actual_checkpoint_steps[target] = step
                    print(f"[checkpoint] {run_dir.name} requested={target} actual={step}", flush=True)
                return not outer.stop_for_memory

            def _on_rollout_end(self) -> None:
                outer.flush_episodes()
                rss = _tree_rss_mb()
                step = int(self.model.num_timesteps)
                append_csv(run_dir / "resource_usage.csv", [{"timesteps": step, "process_tree_rss_mb": rss}])
                if math.isfinite(rss) and rss >= MAX_PROCESS_TREE_RSS_MB:
                    outer.stop_for_memory = True
                    print(f"[resource-stop] {run_dir.name} step={step} process_tree_rss_mb={rss:.1f}", flush=True)
                if step - outer.last_progress_step >= 5000:
                    print(
                        f"[progress] {run_dir.name} timesteps={step} episodes={len(outer.env.episode_rows)} "
                        f"process_tree_rss_mb={rss:.1f}",
                        flush=True,
                    )
                    outer.last_progress_step = step

        self.env = env
        self.run_dir = run_dir
        self.checkpoints = list(checkpoints)
        self.saved: list[int] = []
        self.actual_checkpoint_steps: dict[int, int] = {}
        self.flushed = 0
        self.stop_for_memory = False
        self.last_progress_step = 0
        self.callback = _Callback()

    def flush_episodes(self) -> None:
        new_rows = self.env.episode_rows[self.flushed:]
        if new_rows:
            append_csv(self.run_dir / "training_episode_summary.csv", new_rows)
            self.flushed += len(new_rows)


def train_model(regime: str, ppo_seed: int, smoke: bool = False, smoke_attempt: str = "smoke") -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    import torch

    suffix = smoke_attempt if smoke else f"ppo_seed_{ppo_seed}"
    run_dir = SMOKE_ROOT / suffix if smoke else TRAIN_ROOT / ("historical" if regime == "HISTORICAL_WEATHER" else "random_weather") / suffix
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing training attempt: {rel(run_dir)}")
    run_dir.mkdir(parents=True, exist_ok=True)
    timesteps = SMOKE_TIMESTEPS if smoke else TOTAL_TIMESTEPS
    checkpoint_targets = [timesteps] if smoke else CHECKPOINT_STEPS
    config, env_config = make_run_config(regime, ppo_seed, run_dir)
    effective_schedule = schedule_rows(regime, MAX_SCHEDULE_EPISODES)
    if smoke:
        effective_schedule = schedule_rows(regime, 256)
        for row in effective_schedule:
            row["historical_year"] = WGEN_EVAL_CROP_YEAR
    write_json(run_dir / "training_config.json", {
        "training_regime": regime,
        "ppo_seed": ppo_seed,
        "smoke_only": smoke,
        "total_timesteps_requested": timesteps,
        "checkpoint_steps": checkpoint_targets,
        "canonical_source": "055_00",
        "effective_ppo": config["ppo"],
        "reward": config["reward"],
        "action_grid": config["discrete_actions"],
        "action_safety": config["action_safety"],
        "normalization": {"observation": False, "vecnormalize": False},
        "schedule_seed": HISTORICAL_SCHEDULE_SEED if regime == "HISTORICAL_WEATHER" else WGEN_SCHEDULE_SEED,
        "recreate_wgen_environment_per_episode": RECREATE_WGEN_ENV_PER_EPISODE,
    })
    env = None
    status = "failed"
    started = time.time()
    error = None
    callback = None
    actual_steps = 0
    try:
        torch.set_num_threads(1)
        random.seed(ppo_seed)
        np.random.seed(ppo_seed)
        env = ScheduledEpisodeEnv(config, env_config, effective_schedule, regime, ppo_seed, f"{regime}_{suffix}", RUNTIME_BOOTSTRAP_SEED)
        callback = TrainingCallback(env, run_dir, checkpoint_targets, smoke)
        model = MaskablePPO(
            "MlpPolicy",
            env._gym_env,
            verbose=0,
            seed=int(ppo_seed),
            tensorboard_log=str(run_dir / "tensorboard"),
            **engine_ppo_kwargs(config),
        )
        model.learn(total_timesteps=timesteps, reset_num_timesteps=True, progress_bar=False, callback=callback.callback)
        actual_steps = int(model.num_timesteps)
        callback.flush_episodes()
        if not callback.saved or (not callback.stop_for_memory and not all((run_dir / "models" / f"checkpoint_{target}.zip").is_file() for target in checkpoint_targets)):
            raise RuntimeError(f"Expected checkpoints missing: {callback.saved}")
        status = "resource_limit_stopped" if callback.stop_for_memory else "completed"
    except Exception:
        error = traceback.format_exc()
        (run_dir / "failure_traceback.txt").write_text(error, encoding="utf-8")
    finally:
        if callback is not None:
            callback.flush_episodes()
        if env is not None:
            env.close()
    episode_path = run_dir / "training_episode_summary.csv"
    episode_count = len(pd.read_csv(episode_path)) if episode_path.is_file() else 0
    manifest = {
        "training_regime": regime,
        "ppo_seed": int(ppo_seed),
        "run_status": status,
        "model_dir": rel(run_dir / "models"),
        "model_final": rel(run_dir / "models" / f"checkpoint_{checkpoint_targets[-1]}.zip"),
        "total_timesteps_requested": timesteps,
        "total_timesteps_actual": actual_steps,
        "checkpoint_steps_requested": checkpoint_targets,
        "checkpoint_steps_actual": callback.actual_checkpoint_steps if callback else {},
        "completed_training_episodes": episode_count,
        "training_weather_schedule_id": "historical_years_64003_v1" if regime == "HISTORICAL_WEATHER" else "wgen_rseed1_64004_v1",
        "training_weather_source": "observed_historical" if regime == "HISTORICAL_WEATHER" else "WGEN_random",
        "elapsed_seconds": round(time.time() - started, 3),
        "max_process_tree_rss_mb": _max_rss(run_dir / "resource_usage.csv"),
        "memory_stop_threshold_mb": MAX_PROCESS_TREE_RSS_MB,
        "error": error,
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def engine_ppo_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    from run_055_00_yca_lowIC_expanded_action_maskableppo import engine
    return engine.base03222.base.ppo_kwargs(config)


def _max_rss(path: Path) -> float | None:
    if not path.is_file():
        return None
    values = pd.to_numeric(pd.read_csv(path).get("process_tree_rss_mb"), errors="coerce").dropna()
    return float(values.max()) if len(values) else None


def evaluate_one(model_path: Path, regime: str, ppo_seed: int, eval_type: str) -> list[dict[str, Any]]:
    from sb3_contrib import MaskablePPO

    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if eval_type == "heldout_wgen":
        episodes = [{"episode_index": i + 1, "historical_year": WGEN_EVAL_CROP_YEAR, "weather_seed": seed, "evaluation_weather_type": "heldout_wgen", "evaluation_weather_seed": seed, "evaluation_weather_year": ""} for i, seed in enumerate(WGEN_EVAL_SEEDS)]
    elif eval_type == "observed_weather":
        episodes = [{"episode_index": i + 1, "historical_year": year, "weather_seed": "", "evaluation_weather_type": "observed_weather", "evaluation_weather_seed": "", "evaluation_weather_year": year} for i, year in enumerate(OBSERVED_YEARS)]
    else:
        raise ValueError(eval_type)
    random_weather = eval_type == "heldout_wgen"
    eval_regime = "RANDOM_WEATHER_WGEN" if random_weather else "HISTORICAL_WEATHER"
    if not random_weather:
        # Keep each observed validation episode on its named year rather than the training schedule.
        schedule = [{"episode_index": row["episode_index"], "historical_year": row["historical_year"], "weather_seed": ""} for row in episodes]
    else:
        schedule = [{"episode_index": row["episode_index"], "historical_year": row["historical_year"], "weather_seed": row["weather_seed"]} for row in episodes]
    run_dir = EVAL_RUN_ROOT / eval_type / ("historical" if regime == "HISTORICAL_WEATHER" else "random_weather") / f"ppo_seed_{ppo_seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evaluation attempt: {rel(run_dir)}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg, env_config = make_run_config(regime, ppo_seed, run_dir)
    eval_env = ScheduledEpisodeEnv(cfg, env_config, schedule, eval_regime, ppo_seed, f"eval_{eval_type}_{regime}_{ppo_seed}", EVAL_RUNTIME_SEED)
    model = MaskablePPO.load(str(model_path), device="cpu")
    rows: list[dict[str, Any]] = []
    try:
        for row in episodes:
            obs, _info = eval_env._gym_env.reset(seed=None)
            total_reward = 0.0
            days = 0
            while True:
                action, _ = model.predict(obs, deterministic=True, action_masks=eval_env._gym_env.action_masks())
                obs, reward, terminated, truncated, _info = eval_env._gym_env.step(action)
                total_reward += float(reward)
                days += 1
                if terminated or truncated:
                    episode_log = eval_env.episode_rows[-1]
                    rows.append({
                        "training_regime": regime,
                        "ppo_seed": ppo_seed,
                        "model_path": rel(model_path),
                        "evaluation_weather_type": row["evaluation_weather_type"],
                        "evaluation_weather_seed": row["evaluation_weather_seed"],
                        "evaluation_weather_year": row["evaluation_weather_year"],
                        "crop_year_context": row["historical_year"],
                        "actual_rseed1_": episode_log["actual_rseed1_sent_to_pdi"],
                        "runtime_weather_sha256": episode_log["runtime_weather_sha256"],
                        "filex_wther": episode_log["filex_wther"],
                        "runtime_filex_template_sha256": episode_log["runtime_filex_template_sha256"],
                        "reward": total_reward,
                        "reward_component_sum": episode_log["reward_component_sum"],
                        "reward_decomposition_abs_error": episode_log["reward_decomposition_abs_error"],
                        "yield": episode_log["yield"],
                        "fertilizer": episode_log["fertilizer"],
                        "irrigation": episode_log["irrigation"],
                        "episode_days": days,
                        "status": "completed",
                    })
                    break
        pd.DataFrame(rows).to_csv(run_dir / "evaluation_episode_level.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        write_json(run_dir / "evaluation_manifest.json", {
            "training_regime": regime,
            "ppo_seed": ppo_seed,
            "evaluation_type": eval_type,
            "deterministic_actions": True,
            "episodes_expected": len(episodes),
            "episodes_completed": len(rows),
            "model_path": rel(model_path),
            "evaluation_weather_seeds": WGEN_EVAL_SEEDS if random_weather else [],
            "evaluation_years": OBSERVED_YEARS if not random_weather else [],
            "runtime_seed_role": "independent evaluation environment initialization only",
        })
    finally:
        eval_env.close()
    return rows


def _percent_change(new: float, old: float) -> float | None:
    if not math.isfinite(new) or not math.isfinite(old) or abs(old) < 1e-12:
        return None
    return 100.0 * (new - old) / abs(old)


def validate_evaluation_evidence() -> dict[str, Any]:
    issues: list[str] = []
    weather_hashes_by_run: dict[str, dict[int, str]] = {}
    expected_runs = {(regime, seed) for regime in ("HISTORICAL_WEATHER", "RANDOM_WEATHER_WGEN") for seed in PPO_SEEDS}
    found_runs: set[tuple[str, int]] = set()
    checked = []
    for regime, seed in sorted(expected_runs):
        regime_dir = "historical" if regime == "HISTORICAL_WEATHER" else "random_weather"
        for eval_type, expected_n in (("heldout_wgen", len(WGEN_EVAL_SEEDS)), ("observed_weather", len(OBSERVED_YEARS))):
            run_dir = EVAL_RUN_ROOT / eval_type / regime_dir / f"ppo_seed_{seed}"
            csv_path = run_dir / "evaluation_episode_level.csv"
            manifest_path = run_dir / "evaluation_manifest.json"
            if not csv_path.is_file() or not manifest_path.is_file():
                issues.append(f"missing evaluation CSV or manifest: {regime}/seed{seed}/{eval_type}")
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            frame = pd.read_csv(csv_path, keep_default_na=False)
            found_runs.add((regime, seed))
            if manifest.get("training_regime") != regime or int(manifest.get("ppo_seed", -1)) != seed:
                issues.append(f"evaluation manifest identity mismatch: {regime}/seed{seed}/{eval_type}")
            if manifest.get("deterministic_actions") is not True:
                issues.append(f"deterministic evaluation not recorded: {regime}/seed{seed}/{eval_type}")
            if len(frame) != expected_n or int(manifest.get("episodes_completed", -1)) != expected_n:
                issues.append(f"wrong evaluation row count: {regime}/seed{seed}/{eval_type}")
            if frame.empty or frame["status"].ne("completed").any():
                issues.append(f"incomplete evaluation episodes: {regime}/seed{seed}/{eval_type}")
            if not frame["training_regime"].eq(regime).all() or not pd.to_numeric(frame["ppo_seed"], errors="coerce").eq(seed).all():
                issues.append(f"evaluation rows have mismatched model identity: {regime}/seed{seed}/{eval_type}")
            if pd.to_numeric(frame["reward_decomposition_abs_error"], errors="coerce").fillna(float("inf")).gt(1e-8).any():
                issues.append(f"evaluation reward decomposition mismatch: {regime}/seed{seed}/{eval_type}")
            if eval_type == "heldout_wgen":
                seeds = pd.to_numeric(frame["evaluation_weather_seed"], errors="coerce").astype(int).tolist()
                actual = pd.to_numeric(frame["actual_rseed1_"], errors="coerce").astype(int).tolist()
                if sorted(seeds) != WGEN_EVAL_SEEDS or len(set(seeds)) != len(WGEN_EVAL_SEEDS):
                    issues.append(f"held-out WGEN seed set mismatch: {regime}/seed{seed}")
                if actual != seeds:
                    issues.append(f"actual evaluation rseed1_ does not equal requested seed: {regime}/seed{seed}")
                hashes = frame["runtime_weather_sha256"].astype(str)
                if hashes.eq("").any() or hashes.nunique() != expected_n:
                    issues.append(f"held-out WGEN seeds did not produce distinct runtime weather files: {regime}/seed{seed}")
                if not frame["filex_wther"].eq("W").all():
                    issues.append(f"held-out WGEN evaluation did not use FileX WTHER=W: {regime}/seed{seed}")
                weather_hashes_by_run[f"{regime}/seed{seed}"] = dict(zip(seeds, hashes))
                if not frame["evaluation_weather_year"].astype(str).eq("").all():
                    issues.append(f"WGEN evaluation unexpectedly has observed weather year labels: {regime}/seed{seed}")
            else:
                years = pd.to_numeric(frame["evaluation_weather_year"], errors="coerce").astype(int).tolist()
                if sorted(years) != OBSERVED_YEARS or len(set(years)) != len(OBSERVED_YEARS):
                    issues.append(f"observed-year evaluation set mismatch: {regime}/seed{seed}")
                if pd.to_numeric(frame["evaluation_weather_seed"], errors="coerce").notna().any():
                    issues.append(f"observed evaluation unexpectedly has a WGEN seed: {regime}/seed{seed}")
            checked.append({"training_regime": regime, "ppo_seed": seed, "evaluation_weather_type": eval_type, "episodes": len(frame), "status": "passed" if not issues else "checked"})
    if found_runs != expected_runs:
        issues.append(f"evaluation model set mismatch: found {sorted(found_runs)}")
    if weather_hashes_by_run:
        reference = next(iter(weather_hashes_by_run.values()))
        if any(hashes != reference for hashes in weather_hashes_by_run.values()):
            issues.append("held-out weather files differ across models for the same evaluation seed")
    result = {
        "passed": not issues,
        "issues": issues,
        "expected_episode_rows": {"heldout_wgen": 6 * len(WGEN_EVAL_SEEDS), "observed_weather": 6 * len(OBSERVED_YEARS)},
        "training_eval_weather_seed_intersection": sorted(set(WGEN_TRAIN_SEEDS) & set(WGEN_EVAL_SEEDS)),
        "checked_runs": checked,
    }
    write_json(TASK_ROOT / "evaluation_qc_verified.json", result)
    return result


def summarize_evaluations() -> dict[str, Any]:
    qc = validate_evaluation_evidence()
    if not qc["passed"]:
        raise RuntimeError("Evaluation evidence gate failed: " + "; ".join(qc["issues"]))
    paths = list(EVAL_RUN_ROOT.glob("**/evaluation_episode_level.csv"))
    if not paths:
        raise FileNotFoundError("No evaluation episode-level CSVs exist")
    frames = [pd.read_csv(path, keep_default_na=False) for path in paths]
    all_eval = pd.concat(frames, ignore_index=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    all_eval.to_csv(EVAL_ROOT / "evaluation_episode_level.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    if all_eval["status"].ne("completed").any():
        raise RuntimeError("Evaluation contains incomplete episodes")
    wgen = all_eval[all_eval["evaluation_weather_type"].eq("heldout_wgen")].copy()
    obs = all_eval[all_eval["evaluation_weather_type"].eq("observed_weather")].copy()
    expected_wgen = 6 * len(WGEN_EVAL_SEEDS)
    expected_obs = 6 * len(OBSERVED_YEARS)
    if len(wgen) != expected_wgen or len(obs) != expected_obs:
        raise RuntimeError(f"Expected {expected_wgen} WGEN and {expected_obs} observed rows, got {len(wgen)} and {len(obs)}")
    model_keys = ["training_regime", "ppo_seed"]
    wgen_rows = []
    for (regime, seed), group in wgen.groupby(model_keys):
        reward = pd.to_numeric(group["reward"], errors="coerce")
        yield_values = pd.to_numeric(group["yield"], errors="coerce")
        fert = pd.to_numeric(group["fertilizer"], errors="coerce")
        irrigation = pd.to_numeric(group["irrigation"], errors="coerce")
        mean_reward = float(reward.mean())
        wgen_rows.append({
            "training_regime": regime, "ppo_seed": int(seed), "n_weather_seeds": int(len(group)),
            "mean_reward": mean_reward, "median_reward": float(reward.median()), "sd_reward": float(reward.std(ddof=1)),
            "cv_reward": float(reward.std(ddof=1) / abs(mean_reward)) if abs(mean_reward) > 1e-12 else "",
            "p10_reward": float(reward.quantile(0.10)), "minimum_reward": float(reward.min()),
            "mean_yield": float(yield_values.mean()), "p10_yield": float(yield_values.quantile(0.10)),
            "mean_fertilizer": float(fert.mean()), "p90_fertilizer": float(fert.quantile(0.90)),
            "mean_irrigation": float(irrigation.mean()),
        })
    wgen_summary = pd.DataFrame(wgen_rows).sort_values(model_keys)
    obs_rows = []
    for (regime, seed), group in obs.groupby(model_keys):
        obs_rows.append({
            "training_regime": regime, "ppo_seed": int(seed), "n_observed_years": int(len(group)),
            "mean_reward": float(pd.to_numeric(group["reward"], errors="coerce").mean()),
            "median_reward": float(pd.to_numeric(group["reward"], errors="coerce").median()),
            "mean_yield": float(pd.to_numeric(group["yield"], errors="coerce").mean()),
            "mean_fertilizer": float(pd.to_numeric(group["fertilizer"], errors="coerce").mean()),
            "mean_irrigation": float(pd.to_numeric(group["irrigation"], errors="coerce").mean()),
        })
    obs_summary = pd.DataFrame(obs_rows).sort_values(model_keys)
    paired_rows = []
    for seed in PPO_SEEDS:
        hist = wgen_summary[(wgen_summary.training_regime == "HISTORICAL_WEATHER") & (wgen_summary.ppo_seed == seed)].iloc[0]
        rand = wgen_summary[(wgen_summary.training_regime == "RANDOM_WEATHER_WGEN") & (wgen_summary.ppo_seed == seed)].iloc[0]
        paired_rows.append({
            "ppo_seed": seed,
            "historical_mean_reward": hist.mean_reward,
            "random_weather_mean_reward": rand.mean_reward,
            "reward_difference": rand.mean_reward - hist.mean_reward,
            "reward_difference_pct": _percent_change(rand.mean_reward, hist.mean_reward),
            "historical_mean_yield": hist.mean_yield,
            "random_weather_mean_yield": rand.mean_yield,
            "yield_difference_pct": _percent_change(rand.mean_yield, hist.mean_yield),
            "historical_mean_fertilizer": hist.mean_fertilizer,
            "random_weather_mean_fertilizer": rand.mean_fertilizer,
            "fertilizer_difference_pct": _percent_change(rand.mean_fertilizer, hist.mean_fertilizer),
            "historical_p10_reward": hist.p10_reward,
            "random_weather_p10_reward": rand.p10_reward,
            "p10_reward_difference_pct": _percent_change(rand.p10_reward, hist.p10_reward),
            "historical_mean_irrigation": hist.mean_irrigation,
            "random_weather_mean_irrigation": rand.mean_irrigation,
        })
    paired = pd.DataFrame(paired_rows)
    pooled_rows = []
    for regime, group in wgen.groupby("training_regime"):
        for metric in ("reward", "yield", "fertilizer", "irrigation"):
            pooled_rows.append({"training_regime": regime, "metric": metric, "mean": float(pd.to_numeric(group[metric], errors="coerce").mean()), "p10": float(pd.to_numeric(group[metric], errors="coerce").quantile(.10)), "p90": float(pd.to_numeric(group[metric], errors="coerce").quantile(.90)), "n": int(len(group))})
    pooled = pd.DataFrame(pooled_rows)
    h_obs = obs_summary[obs_summary.training_regime.eq("HISTORICAL_WEATHER")].set_index("ppo_seed")
    r_obs = obs_summary[obs_summary.training_regime.eq("RANDOM_WEATHER_WGEN")].set_index("ppo_seed")
    observed_directions = {str(seed): "better" if r_obs.loc[seed, "mean_reward"] > h_obs.loc[seed, "mean_reward"] else ("equal" if math.isclose(r_obs.loc[seed, "mean_reward"], h_obs.loc[seed, "mean_reward"], abs_tol=1e-12) else "worse") for seed in PPO_SEEDS}
    reward_paired_positive = int((paired["reward_difference"] > 0).sum())
    pooled_mean = {regime: float(pd.to_numeric(group["reward"], errors="coerce").mean()) for regime, group in wgen.groupby("training_regime")}
    pooled_yield = {regime: float(pd.to_numeric(group["yield"], errors="coerce").mean()) for regime, group in wgen.groupby("training_regime")}
    pooled_fert = {regime: float(pd.to_numeric(group["fertilizer"], errors="coerce").mean()) for regime, group in wgen.groupby("training_regime")}
    pooled_p10 = {regime: float(pd.to_numeric(group["reward"], errors="coerce").quantile(.10)) for regime, group in wgen.groupby("training_regime")}
    reward_pct = _percent_change(pooled_mean["RANDOM_WEATHER_WGEN"], pooled_mean["HISTORICAL_WEATHER"])
    yield_pct = _percent_change(pooled_yield["RANDOM_WEATHER_WGEN"], pooled_yield["HISTORICAL_WEATHER"])
    fert_pct = _percent_change(pooled_fert["RANDOM_WEATHER_WGEN"], pooled_fert["HISTORICAL_WEATHER"])
    p10_pct = _percent_change(pooled_p10["RANDOM_WEATHER_WGEN"], pooled_p10["HISTORICAL_WEATHER"])
    observed_all_worse = all(direction == "worse" for direction in observed_directions.values())
    go_checks = {
        "positive_mean_reward_at_least_2_of_3_ppo_seeds": reward_paired_positive >= 2,
        "pooled_mean_reward_difference_positive": pooled_mean["RANDOM_WEATHER_WGEN"] > pooled_mean["HISTORICAL_WEATHER"],
        "pooled_mean_yield_not_down_more_than_2pct": yield_pct is not None and yield_pct >= -2.0,
        "pooled_mean_fertilizer_not_up_more_than_5pct": fert_pct is not None and fert_pct <= 5.0,
        "pooled_p10_reward_not_down_more_than_5pct": p10_pct is not None and p10_pct >= -5.0,
        "observed_comparison_not_worse_for_all_3_seeds": not observed_all_worse,
    }
    if all(go_checks.values()):
        decision = "GO_SIGNAL"
    elif reward_paired_positive <= 1 and pooled_mean["RANDOM_WEATHER_WGEN"] <= pooled_mean["HISTORICAL_WEATHER"]:
        decision = "NO_SIGNAL"
    else:
        decision = "MIXED_SIGNAL"
    pd.DataFrame(wgen_rows).to_csv(EVAL_ROOT / "heldout_wgen_model_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    obs_summary.to_csv(EVAL_ROOT / "observed_weather_model_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    paired.to_csv(EVAL_ROOT / "paired_ppo_seed_comparison.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    pooled.to_csv(EVAL_ROOT / "pooled_regime_comparison.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    decision_payload = {
        "pilot_decision": decision,
        "operational_go_checks": go_checks,
        "ppo_seed_reward_direction": {str(int(row.ppo_seed)): "positive" if row.reward_difference > 0 else ("equal" if math.isclose(row.reward_difference, 0.0, abs_tol=1e-12) else "negative") for row in paired.itertuples(index=False)},
        "ppo_seeds_with_positive_reward_change": reward_paired_positive,
        "pooled_heldout_wgen": {
            "historical_mean_reward": pooled_mean["HISTORICAL_WEATHER"],
            "random_weather_mean_reward": pooled_mean["RANDOM_WEATHER_WGEN"],
            "mean_reward_difference_pct": reward_pct,
            "historical_mean_yield": pooled_yield["HISTORICAL_WEATHER"],
            "random_weather_mean_yield": pooled_yield["RANDOM_WEATHER_WGEN"],
            "mean_yield_difference_pct": yield_pct,
            "historical_mean_fertilizer": pooled_fert["HISTORICAL_WEATHER"],
            "random_weather_mean_fertilizer": pooled_fert["RANDOM_WEATHER_WGEN"],
            "mean_fertilizer_difference_pct": fert_pct,
            "historical_p10_reward": pooled_p10["HISTORICAL_WEATHER"],
            "random_weather_p10_reward": pooled_p10["RANDOM_WEATHER_WGEN"],
            "p10_reward_difference_pct": p10_pct,
        },
        "observed_2014_2023_direction_by_ppo_seed": observed_directions,
        "n_heldout_wgen_rows": len(wgen),
        "n_observed_rows": len(obs),
        "statistical_claim_limit": "Descriptive pilot only; no p<0.05 success claim.",
        "larger_experiment_recommendation": "YES" if decision == "GO_SIGNAL" else ("NO" if decision == "NO_SIGNAL" else "NEEDS_DIAGNOSIS"),
    }
    write_json(TASK_ROOT / "pilot_decision.json", decision_payload)
    write_figures(wgen, obs, paired)
    write_chinese_report(decision_payload, wgen_summary, obs_summary, paired, pooled, all_eval)
    return decision_payload


def write_figures(wgen: pd.DataFrame, obs: pd.DataFrame, paired: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = TASK_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    display_regime = lambda value: value.replace("_", "\n")
    order = [(regime, seed) for regime in ("HISTORICAL_WEATHER", "RANDOM_WEATHER_WGEN") for seed in PPO_SEEDS]
    label_values = [f"{display_regime(regime)}\nseed {seed}" for regime, seed in order]
    for metric, filename, ylabel in (("reward", "heldout_wgen_reward_by_model.png", "Episode return"), ("yield", "heldout_wgen_yield_by_model.png", "Yield (kg/ha)"), ("fertilizer", "heldout_wgen_fertilizer_by_model.png", "Fertilizer (kg N/ha)")):
        values = [pd.to_numeric(wgen[(wgen.training_regime.eq(regime)) & (wgen.ppo_seed.eq(seed))][metric], errors="coerce").dropna().to_numpy() for regime, seed in order]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.boxplot(values, tick_labels=label_values, showmeans=True)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Held-out WGEN {metric} by trained model")
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(figure_dir / filename, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color="black", linewidth=1)
    ax.bar(paired["ppo_seed"].astype(str), paired["reward_difference"], color=["#3b82f6" if x > 0 else "#e76f51" for x in paired["reward_difference"]])
    ax.set_xlabel("Paired PPO seed")
    ax.set_ylabel("Random-weather minus historical mean reward")
    ax.set_title("Paired reward difference on the same 20 WGEN seeds")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "paired_reward_difference_by_ppo_seed.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for regime, color in (("HISTORICAL_WEATHER", "#3b82f6"), ("RANDOM_WEATHER_WGEN", "#e76f51")):
        subset = wgen[wgen.training_regime.eq(regime)]
        for seed in PPO_SEEDS:
            rewards = pd.to_numeric(subset[subset.ppo_seed.eq(seed)]["reward"], errors="coerce").sort_values().to_numpy()
            if len(rewards):
                ax.plot(np.linspace(0, 100, len(rewards)), rewards, alpha=.35, color=color)
    ax.set_xlabel("Weather-seed percentile")
    ax.set_ylabel("Episode return")
    ax.set_title("Held-out reward distribution across 20 WGEN seeds")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "heldout_wgen_reward_distribution.png", dpi=160)
    plt.close(fig)
    obs_order = [(regime, seed) for regime in ("HISTORICAL_WEATHER", "RANDOM_WEATHER_WGEN") for seed in PPO_SEEDS]
    obs_labels = [f"{display_regime(regime)}\nseed {seed}" for regime, seed in obs_order]
    obs_values = [pd.to_numeric(obs[(obs.training_regime.eq(regime)) & (obs.ppo_seed.eq(seed))]["reward"], errors="coerce").dropna().to_numpy() for regime, seed in obs_order]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.boxplot(obs_values, tick_labels=obs_labels, showmeans=True)
    ax.set_ylabel("Episode return")
    ax.set_title("Observed YC 2014-2023 independent comparison period")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "observed_2014_2023_reward_comparison.png", dpi=160)
    plt.close(fig)


def write_chinese_report(
    decision: dict[str, Any],
    wgen_summary: pd.DataFrame,
    obs_summary: pd.DataFrame,
    paired: pd.DataFrame,
    pooled: pd.DataFrame,
    all_eval: pd.DataFrame,
) -> None:
    report_path = ROOT / "docs" / "yc_random_weather_ppo_controlled_pilot.md"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {rel(report_path)}")
    config = json.loads((CONFIG_ROOT / "canonical_yc_ppo_config.json").read_text(encoding="utf-8"))
    design = json.loads((CONFIG_ROOT / "experiment_design.json").read_text(encoding="utf-8"))
    training = pd.read_csv(TASK_ROOT / "training_run_manifest.csv", keep_default_na=False)
    train_qc_candidates = sorted(TASK_ROOT.glob("training_qc_verified*.json"))
    train_qc = next((json.loads(path.read_text(encoding="utf-8")) for path in train_qc_candidates if json.loads(path.read_text(encoding="utf-8")).get("passed") is True), None)
    eval_qc = json.loads((TASK_ROOT / "evaluation_qc_verified.json").read_text(encoding="utf-8"))
    if train_qc is None or not eval_qc.get("passed"):
        raise RuntimeError("Cannot write report before training and evaluation QC pass")

    def fmt(value: Any, digits: int = 3) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "未报告"
        return "未报告" if not math.isfinite(number) else f"{number:,.{digits}f}"

    def pct(new: float, old: float) -> str:
        value = _percent_change(float(new), float(old))
        return "未定义" if value is None else f"{value:+.2f}%"

    def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
        head = "| " + " | ".join(headers) + " |"
        rule = "| " + " | ".join("---" for _ in headers) + " |"
        body = ["| " + " | ".join(str(cell).replace("|", "/") for cell in row) + " |" for row in rows]
        return "\n".join([head, rule, *body])

    ppo = config["ppo_hyperparameters"]
    reward_cfg = config["reward"]
    action = config["action_grid"]
    safety = config["action_safety"]
    wgen = all_eval[all_eval["evaluation_weather_type"].eq("heldout_wgen")].copy()
    pooled_outcomes = {
        regime: group for regime, group in wgen.groupby("training_regime")
    }
    hist_w = pooled_outcomes["HISTORICAL_WEATHER"]
    rand_w = pooled_outcomes["RANDOM_WEATHER_WGEN"]
    pooled_metric_rows = []
    for metric, label in (("reward", "Episode return"), ("yield", "产量 kg/ha"), ("fertilizer", "肥料 kg N/ha"), ("irrigation", "灌溉 mm")):
        old = float(pd.to_numeric(hist_w[metric], errors="coerce").mean())
        new = float(pd.to_numeric(rand_w[metric], errors="coerce").mean())
        pooled_metric_rows.append([label, fmt(old), fmt(new), pct(new, old)])
    hist_p10 = float(pd.to_numeric(hist_w["reward"], errors="coerce").quantile(.10))
    rand_p10 = float(pd.to_numeric(rand_w["reward"], errors="coerce").quantile(.10))
    pooled_metric_rows.append(["Reward P10", fmt(hist_p10), fmt(rand_p10), pct(rand_p10, hist_p10)])

    train_rows = []
    for row in training.sort_values(["training_regime", "ppo_seed"]).itertuples(index=False):
        regime_label = "历史天气" if row.training_regime == "HISTORICAL_WEATHER" else "WGEN 随机天气"
        train_rows.append([
            regime_label,
            str(int(row.ppo_seed)),
            f"{int(row.total_timesteps_requested):,}/{int(row.total_timesteps_actual):,}",
            str(int(row.completed_training_episodes)),
            fmt(float(row.max_process_tree_rss_mb), 1),
            fmt(float(row.elapsed_seconds) / 60.0, 1),
        ])

    robustness_rows = []
    outcome_rows = []
    for row in wgen_summary.sort_values(["training_regime", "ppo_seed"]).itertuples(index=False):
        regime_label = "历史天气" if row.training_regime == "HISTORICAL_WEATHER" else "WGEN 随机天气"
        robustness_rows.append([
            regime_label, str(int(row.ppo_seed)), fmt(row.mean_reward), fmt(row.median_reward),
            fmt(row.sd_reward), fmt(100 * row.cv_reward, 1), fmt(row.p10_reward), fmt(row.minimum_reward),
        ])
        outcome_rows.append([
            regime_label, str(int(row.ppo_seed)), fmt(row.mean_yield), fmt(row.p10_yield),
            fmt(row.mean_fertilizer), fmt(row.p90_fertilizer), fmt(row.mean_irrigation),
        ])

    paired_rows = []
    for row in paired.sort_values("ppo_seed").itertuples(index=False):
        direction = "改善" if row.reward_difference > 0 else ("相同" if math.isclose(row.reward_difference, 0.0, abs_tol=1e-12) else "下降")
        paired_rows.append([
            str(int(row.ppo_seed)), fmt(row.historical_mean_reward), fmt(row.random_weather_mean_reward),
            fmt(row.reward_difference), direction, f"{row.reward_difference_pct:+.2f}%",
            f"{row.yield_difference_pct:+.2f}%", f"{row.fertilizer_difference_pct:+.2f}%",
            f"{row.p10_reward_difference_pct:+.2f}%",
        ])

    observed_rows = []
    for seed in PPO_SEEDS:
        h = obs_summary[(obs_summary.training_regime == "HISTORICAL_WEATHER") & (obs_summary.ppo_seed == seed)].iloc[0]
        r = obs_summary[(obs_summary.training_regime == "RANDOM_WEATHER_WGEN") & (obs_summary.ppo_seed == seed)].iloc[0]
        direction = decision["observed_2014_2023_direction_by_ppo_seed"][str(seed)]
        observed_rows.append([
            str(seed), fmt(h.mean_reward), fmt(r.mean_reward), direction,
            fmt(h.mean_yield), fmt(r.mean_yield), fmt(h.mean_fertilizer), fmt(r.mean_fertilizer),
            fmt(h.mean_irrigation), fmt(r.mean_irrigation),
        ])

    max_reward_error = float(pd.to_numeric(all_eval["reward_decomposition_abs_error"], errors="coerce").max())
    wgen_qc_note = (
        f"训练的 WGEN 组每个 PPO seed 有 {train_qc['unique_daily_weather_hashes_per_wgen_run'].get('RANDOM_WEATHER_WGEN/seed0', 0)} 个不同逐日天气哈希；"
        f"三种 PPO seed 的 962 条 year/seed/weather 序列完全一致。QC 观察到 {train_qc['repeated_seed_year_pairs_with_multiple_runtime_realizations']} 个重复 year×seed pair 跨 schedule block 重现时有多个天气 realization；"
        "这些序列在三个 PPO seed 之间仍逐 episode 对齐，因此保留为随机发生器的复现边界 note，不把 seed 标签误称为唯一 realization ID。"
    )
    smoke_gate_path = SMOKE_ROOT / "smoke_gate_wther_w_cached_process_probe_01.json"
    smoke_gate = json.loads(smoke_gate_path.read_text(encoding="utf-8"))
    episode_training = [
        "# YC 随机天气 PPO 受控 Pilot 实验报告",
        "",
        "## 1. 假设",
        "本实验检验：在保持 YC PPO 算法、reward、状态/动作、管理约束、作物和训练预算不变时，仅增加 WGEN 随机天气训练多样性，是否改善未见天气表现或稳健性。该命题是待检验假设，不是既定事实。",
        "",
        "## 2. 为什么这是受控 Pilot",
        "比较 `HISTORICAL_WEATHER` 与 `RANDOM_WEATHER_WGEN`，每组 PPO seed 为 0、1、2；两组共用同一个 `055_00` canonical config 和相同 2004–2013 年份上下文顺序。首轮六模型虽然完成，但 FileX 实际为 `WTHER=M`，并未启用 WGEN；首轮模型、评估与汇总均留档但不纳入本报告。修复后只在任务生成模板副本设置 `WTHER=W`，不修改 frozen 输入或安装 runtime。",
        "",
        "## 3. Frozen PPO 配置",
        f"Canonical 来源：`{config['canonical_source_yaml']}`，正式 YC config `055_00`；训练入口 `{config['training_script']}`。算法为 MaskablePPO / MlpPolicy，网络 `{ppo['net_arch']}`、Tanh；`learning_rate={ppo['learning_rate']}`、`gamma={ppo['gamma']}`、`gae_lambda={ppo['gae_lambda']}`、`n_steps={ppo['n_steps']}`、`batch_size={ppo['batch_size']}`、`n_epochs={ppo['n_epochs']}`、`clip_range={ppo['clip_range']}`、`ent_coef={ppo['ent_coef']}`、`vf_coef={config['algorithm']['vf_coef']}`（算法默认值）。不使用观测归一化、VecNormalize 或天气预报特征。",
        f"Reward 继承 canonical stress-aware wrapper：yield 系数 {reward_cfg['yield_coef']}、water cost {reward_cfg['water_cost']}、N cost {reward_cfg['nitrogen_cost']}、水/N stress relief 系数 {reward_cfg['water_stress_relief_coef']}/{reward_cfg['nitrogen_stress_relief_coef']}，统一 scale {reward_cfg['reward_scale']}；SWFAC process penalty threshold={reward_cfg['swfac_guardrail_process_penalty']['threshold']}、coef={reward_cfg['swfac_guardrail_process_penalty']['coef']}。旧公式 `{config['superseded_reward_formula']['formula']}` 明确为 `SUPERSEDED / NOT_APPLICABLE`，未据此建模或改 reward。",
        f"动作网格为 irrigation `{action['irrigation_levels']}` mm × N `{action['nitrogen_levels']}` kg/ha；单日上限 I/N={safety['daily_irrigation_max']}/{safety['daily_n_max']}，季节 soft limit I/N={safety['season_irrigation_soft_limit']}/{safety['season_n_soft_limit']}，最短间隔 I/N={safety['min_days_between_irrigation']}/{safety['min_days_between_fertilization']} 天；保留 DAP90 后 45 mm irrigation reserve mask。",
        "",
        "## 4. Seed 分离",
        "PPO seed `0/1/2`、历史年份 schedule seed `64003`、WGEN weather schedule seed `64004`、运行初始化 seed `66003`、evaluation 初始化 seed `66004` 分别记录。随机训练 seed pool 为 `1001–1080`，按每 80 条覆盖全池一次的固定 schedule；实际 PDI `rseed1_` 与 frozen sequence 逐条一致。",
        f"{wgen_qc_note}",
        "",
        "## 5. 历史天气训练组",
        "年份为 2004–2013。每连续 10 个 episode 使用独立 schedule RNG `64003` 对十年做一次排列；三种 PPO seed 使用相同年份序列。",
        "",
        "## 6. WGEN 随机天气训练组",
        "年份上下文与历史组相同，唯一设计差异是使用已冻结 `CNYC.CLI` 的 WGEN。任务本地 FileX 副本将所有 METHODS.WTHER 显式置为 `W`，每次 reset 按 weather schedule 设置 seed，并从 DSSAT daily state 记录实际 `RAIN/SRAD/TMAX/TMIN` 序列哈希。",
        "",
        "## 7. 留出天气设计",
        "Held-out WGEN seed `1081–1100`，每个模型评估 20 个 realization，训练/评估 seed 交集为空。Observed 评估使用 YC 2014–2023，严格称为 `independent comparison period`；此前没有证据认证它是 pristine final test set。策略动作 deterministic。全年 WGEN validation 仍为 `DEFERRED`，且不是本 episode-level Pilot blocker。",
        "",
        "## 8. 训练完成情况",
        markdown_table(["天气组", "PPO seed", "请求/实际步数", "完整 episode", "峰值 RSS MB", "耗时 min"], train_rows),
        "六个模型均请求 100,000 steps，实际 100,080（144-step rollout 边界导致 +80），各自保存 25K/50K/75K/100K checkpoints；训练串行，峰值 RSS 约 0.93–1.06 GB，低于 6 GB stop threshold。",
        "",
        "## 9. Held-out WGEN 结果",
        markdown_table(["训练组", "PPO seed", "Mean reward", "Median", "SD", "CV %", "P10", "Minimum"], robustness_rows),
        "",
        "## 10. Observed 2014–2023 结果",
        markdown_table(["PPO seed", "历史 mean reward", "WGEN mean reward", "方向", "历史 mean yield", "WGEN mean yield", "历史 N", "WGEN N", "历史 I", "WGEN I"], observed_rows),
        "",
        "## 11. Yield / Fertilizer / Irrigation 分解",
        markdown_table(["指标（held-out pooled）", "历史天气", "WGEN 随机天气", "相对变化"], pooled_metric_rows),
        "收益提高不能脱离 PPO seed 差异及资源轨迹解读。NUE 未报告，因为本实验没有唯一 canonical NUE 计算链；不新建 NUE/WUE 定义。",
        "",
        "## 12. 稳健性与下尾",
        "报告 20 个 held-out weather seed 上每模型的 mean、median、SD、CV、P10、minimum、yield P10、fertilizer P90 和 mean irrigation。Pooled P10 reward 从历史组 " + fmt(hist_p10) + " 变为 WGEN 组 " + fmt(rand_p10) + f"（{pct(rand_p10, hist_p10)}）。这些是描述性稳健性指标，不是显著性检验。",
        "",
        "## 13. PPO seed 差异",
        markdown_table(["PPO seed", "历史 mean reward", "WGEN mean reward", "绝对差", "方向", "Reward %", "Yield %", "Fertilizer %", "P10 reward %"], paired_rows),
        "同一 seed 配对评估时，seed 0 reward 改善，seed 1、2 下降；Observed 2014–2023 方向为 2 个 worse、1 个 better。禁止挑最好 seed，本报告纳入全部三个 seed。",
        "",
        "## 14. Pilot 判定",
        f"判定为 **{decision['pilot_decision']}**。Pooled held-out reward 变化 {decision['pooled_heldout_wgen']['mean_reward_difference_pct']:+.2f}%，但仅 {decision['ppo_seeds_with_positive_reward_change']}/3 PPO seeds 改善，因此未满足 GO_SIGNAL 的至少 2/3 seed 条件；其余 pooled yield、fertilizer、P10 和 observed consistency guardrails 均通过。更大实验建议：`{decision['larger_experiment_recommendation']}`。",
        "",
        "## 15. 结果能说明什么、不能说明什么",
        "本 Pilot 的结果表明：当前配置下，WGEN 天气增强在 held-out WGEN 的 pooled reward 有描述性改善，但改善只出现在 1/3 PPO seeds，Observed comparison 也呈现 seed 异质性，因此不能称为稳定的总体改善。该结果不证明天气多样性解释了此前全部 YC PPO 不稳定，也不说明 PPO 算法有缺陷。只有 3 个 PPO seeds，不作 p<0.05 成功结论。",
        "",
        "## 16. 下一步",
        "按 `NEEDS_DIAGNOSIS` 处理：暂不扩成更大 seed/训练实验，不改 PPO/reward；先对 seed 0/1/2 做 action occupancy、管理剂量、训练轨迹和天气条件响应诊断，解释 seed 1/2 的 held-out reward 下降及 observed 差异，再预先冻结是否需要新的单因素实验。",
        "",
        "## 17. 文件变更",
        "本任务新增/更新 runner、canonical/设计/调度/天气 runtime correction 配置、训练与评估 manifest/汇总、QC、图和本报告；实验日志记录首轮 `WTHER=M` 无效尝试及修复。有效模型/逐 episode 日志保留在 `results/yc_random_weather_ppo/004_03/training_verified/` 与 `evaluation/verified_weather_refresh/`，大模型和逐日原始输出不纳入 Git。DSSAT runtime、binary、canonical reward、CNYC.CLI、PPO 算法均未修改。",
        "",
        "## 18. 测试与 QC",
        f"- Runner syntax compile、schedule self-test：PASS。\n- Seed smoke `{smoke_gate_path.name}`：`{'PASS' if smoke_gate.get('passed') else 'FAIL'}`，432/432 steps、4 个天气 seed/hash，WTHER=W，checkpoint 可加载。\n- Training QC：PASS，6/6 models、同配置/同步数，WGEN 实际天气序列跨 PPO seed 一致；训练/评估 seed overlap 为空。\n- Evaluation QC：PASS，{eval_qc['expected_episode_rows']['heldout_wgen']} 条 held-out WGEN、{eval_qc['expected_episode_rows']['observed_weather']} 条 observed；实际 WGEN seed 与天气 hash 匹配，确定性标记及 reward 分解通过。\n- Reward decomposition 最大绝对误差：{max_reward_error:.3g}。",
        "",
        "## 19. Git 状态",
        "仓库在本任务开始时已有无关工作区修改；这些改动保持原样。提交时仅选择 YC random-weather Pilot 相关源码、配置、中文实验记录和紧凑汇总/图；不 stage 模型 checkpoint、runtime 日志或无关文件。请求的本地提交 subject 为 `experiment: run YC random-weather PPO controlled pilot`；`git push = NO`。",
        "",
        "## 附：结果文件",
        "Machine-readable decision：`results/yc_random_weather_ppo/004_03/pilot_decision.json`。QC：`training_qc_verified_retry1.json`、`evaluation_qc_verified.json`。指标表及六张 PNG 位于 `results/yc_random_weather_ppo/004_03/evaluation/` 与 `figures/`。",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(episode_training), encoding="utf-8", newline="\n")


def write_aggregate_manifests() -> None:
    rows = []
    for path in sorted(TRAIN_ROOT.glob("**/run_manifest.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if rows:
        pd.DataFrame(rows).to_csv(TASK_ROOT / "training_run_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    episode_paths = sorted(TRAIN_ROOT.glob("**/training_episode_summary.csv"))
    if episode_paths:
        frames = [pd.read_csv(path, keep_default_na=False) for path in episode_paths]
        pd.concat(frames, ignore_index=True).to_csv(TASK_ROOT / "training_episode_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")


def validate_training_evidence() -> dict[str, Any]:
    manifest_paths = sorted(TRAIN_ROOT.glob("**/run_manifest.json"))
    issues: list[str] = []
    if len(manifest_paths) != 6:
        issues.append(f"expected six run manifests, found {len(manifest_paths)}")
    hist_schedule = pd.read_csv(CONFIG_ROOT / "training_weather_schedule_historical.csv", keep_default_na=False)
    wgen_schedule = pd.read_csv(CONFIG_ROOT / "training_weather_schedule_random.csv", keep_default_na=False)
    hist_years = pd.to_numeric(hist_schedule["historical_year"], errors="coerce").astype(int).tolist()
    wgen_years = pd.to_numeric(wgen_schedule["historical_year_context"], errors="coerce").astype(int).tolist()
    wgen_seeds = pd.to_numeric(wgen_schedule["weather_seed"], errors="coerce").astype(int).tolist()
    if hist_years != wgen_years:
        issues.append("historical and WGEN year-context schedules differ")
    if len(wgen_seeds) < TOTAL_TIMESTEPS or any(seed not in WGEN_TRAIN_SEEDS for seed in wgen_seeds):
        issues.append("WGEN schedule is incomplete or contains a seed outside 1001-1080")
    if set(wgen_seeds[:80]) != set(WGEN_TRAIN_SEEDS) or len(set(wgen_seeds[:80])) != 80:
        issues.append("first WGEN schedule block does not cover the full seed pool exactly once")
    if set(wgen_seeds[:80]) & set(WGEN_EVAL_SEEDS):
        issues.append("held-out WGEN seeds overlap the first training schedule block")

    run_summaries: list[dict[str, Any]] = []
    schedule_prefixes: dict[str, list[Any]] = {}
    wgen_weather_sequences: dict[str, list[tuple[int, int, str]]] = {}
    repeated_seed_year_weather_variations: set[tuple[int, int]] = set()
    config_signatures: set[str] = set()
    actual_timesteps: set[int] = set()
    expected_runs = {(regime, seed) for regime in ("HISTORICAL_WEATHER", "RANDOM_WEATHER_WGEN") for seed in PPO_SEEDS}
    seen_runs: set[tuple[str, int]] = set()
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        regime = str(manifest.get("training_regime"))
        seed = int(manifest.get("ppo_seed", -1))
        key = (regime, seed)
        seen_runs.add(key)
        if key not in expected_runs:
            issues.append(f"unexpected training run: {key}")
        if manifest.get("run_status") != "completed":
            issues.append(f"run did not complete: {key}")
        if int(manifest.get("total_timesteps_requested", -1)) != TOTAL_TIMESTEPS:
            issues.append(f"requested timestep mismatch: {key}")
        actual = int(manifest.get("total_timesteps_actual", -1))
        actual_timesteps.add(actual)
        checkpoint_steps = [int(x) for x in manifest.get("checkpoint_steps_requested", [])]
        if checkpoint_steps != CHECKPOINT_STEPS:
            issues.append(f"checkpoint contract mismatch: {key}")
        for step in CHECKPOINT_STEPS:
            checkpoint = ROOT / str(manifest.get("model_dir", "")) / f"checkpoint_{step}.zip"
            if not checkpoint.is_file():
                issues.append(f"missing checkpoint {step}: {key}")
        model_path = ROOT / str(manifest.get("model_final", ""))
        if not model_path.is_file():
            issues.append(f"missing final model: {key}")
        run_dir = manifest_path.parent
        episode_path = run_dir / "training_episode_summary.csv"
        config_path = run_dir / "training_config.json"
        if not episode_path.is_file() or not config_path.is_file():
            issues.append(f"missing episode log or training config: {key}")
            continue
        episode_df = pd.read_csv(episode_path, keep_default_na=False)
        training_config = json.loads(config_path.read_text(encoding="utf-8"))
        signature = json.dumps({name: training_config.get(name) for name in ("effective_ppo", "reward", "action_grid", "action_safety", "normalization", "total_timesteps_requested")}, sort_keys=True, ensure_ascii=False)
        config_signatures.add(signature)
        if training_config.get("ppo_seed") != seed or training_config.get("training_regime") != regime:
            issues.append(f"run config identity mismatch: {key}")
        if episode_df.empty or episode_df["status"].ne("completed").any():
            issues.append(f"empty or incomplete training episode log: {key}")
        if pd.to_numeric(episode_df["reward_decomposition_abs_error"], errors="coerce").fillna(float("inf")).gt(1e-8).any():
            issues.append(f"reward decomposition mismatch: {key}")
        years = pd.to_numeric(episode_df["historical_year"], errors="coerce").astype(int).tolist()
        if regime == "HISTORICAL_WEATHER":
            scheduled_years = hist_years[:len(episode_df)]
            if years != scheduled_years:
                issues.append(f"historical year sequence does not match frozen schedule: {key}")
            if pd.to_numeric(episode_df["training_weather_seed"], errors="coerce").notna().any():
                issues.append(f"historical run unexpectedly logs WGEN training seeds: {key}")
        else:
            scheduled_seeds = wgen_seeds[:len(episode_df)]
            actual_seeds = pd.to_numeric(episode_df["actual_rseed1_sent_to_pdi"], errors="coerce").astype(int).tolist()
            logged_seeds = pd.to_numeric(episode_df["training_weather_seed"], errors="coerce").astype(int).tolist()
            if years != wgen_years[:len(episode_df)]:
                issues.append(f"WGEN year context differs from shared schedule: {key}")
            if logged_seeds != scheduled_seeds or actual_seeds != scheduled_seeds:
                issues.append(f"WGEN seeds were not injected exactly from the frozen schedule: {key}")
            if any(item == seed for item in actual_seeds):
                issues.append(f"PPO seed was reused as a WGEN seed: {key}")
            schedule_prefixes[f"{regime}/seed{seed}"] = actual_seeds
            runtime_hashes = episode_df["runtime_weather_sha256"].astype(str).tolist()
            if any(not value for value in runtime_hashes):
                issues.append(f"WGEN training episode lacks runtime weather hash: {key}")
            if not episode_df["filex_wther"].astype(str).eq("W").all():
                issues.append(f"WGEN training episode did not use FileX WTHER=W: {key}")
            weather_rows = episode_df.assign(
                _year=years,
                _seed=actual_seeds,
                _weather_hash=runtime_hashes,
            )
            for (weather_year, weather_seed), group in weather_rows.groupby(["_year", "_seed"]):
                unique_hashes = group["_weather_hash"].unique().tolist()
                if len(unique_hashes) > 1:
                    repeated_seed_year_weather_variations.add((int(weather_year), int(weather_seed)))
            per_year = weather_rows.groupby("_year").agg(seed_count=("_seed", "nunique"), hash_count=("_weather_hash", "nunique"))
            if (per_year["seed_count"] > per_year["hash_count"]).any():
                issues.append(f"two or more WGEN seeds shared a daily weather sequence within a YC year: {key}")
            seed_collisions = weather_rows.groupby(["_year", "_weather_hash"])["_seed"].nunique()
            if seed_collisions.gt(1).any():
                issues.append(f"different WGEN seeds produced an identical daily weather sequence within a YC year: {key}")
            wgen_weather_sequences[f"{regime}/seed{seed}"] = list(zip(years, actual_seeds, runtime_hashes))
        run_summaries.append({
            "training_regime": regime,
            "ppo_seed": seed,
            "run_status": manifest.get("run_status"),
            "requested_timesteps": manifest.get("total_timesteps_requested"),
            "actual_timesteps": actual,
            "episodes": len(episode_df),
            "max_process_tree_rss_mb": manifest.get("max_process_tree_rss_mb"),
            "checkpoints_present": all((ROOT / str(manifest.get("model_dir", "")) / f"checkpoint_{step}.zip").is_file() for step in CHECKPOINT_STEPS),
        })
    if seen_runs != expected_runs:
        issues.append(f"run identity set mismatch: expected {sorted(expected_runs)}, found {sorted(seen_runs)}")
    if len(config_signatures) != 1:
        issues.append(f"effective model config differs across runs: {len(config_signatures)} signatures")
    if len(actual_timesteps) != 1 or next(iter(actual_timesteps), -1) < TOTAL_TIMESTEPS:
        issues.append(f"actual timestep budgets differ or are below target: {sorted(actual_timesteps)}")
    common_wgen_prefix_length = 0
    if len(schedule_prefixes) == 3:
        prefixes = list(schedule_prefixes.values())
        common_wgen_prefix_length = min(map(len, prefixes))
        shared_prefix = prefixes[0][:common_wgen_prefix_length]
        if any(prefix[:common_wgen_prefix_length] != shared_prefix for prefix in prefixes[1:]):
            issues.append("WGEN actual rseed1 common schedule prefix differs across PPO seeds")
    weather_sequence_equal = False
    if len(wgen_weather_sequences) == 3:
        sequences = list(wgen_weather_sequences.values())
        common_weather_length = min(map(len, sequences))
        weather_sequence_equal = all(sequence[:common_weather_length] == sequences[0][:common_weather_length] for sequence in sequences[1:])
        if not weather_sequence_equal:
            issues.append("year/seed/daily-weather sequence differs across random-weather PPO seeds")
    result = {
        "passed": not issues,
        "issues": issues,
        "run_count": len(run_summaries),
        "same_effective_config_across_all_runs": len(config_signatures) == 1,
        "same_actual_timestep_count": len(actual_timesteps) == 1,
        "actual_timesteps_values": sorted(actual_timesteps),
        "wgen_schedule_common_prefix_rows": common_wgen_prefix_length,
        "wgen_daily_weather_sequence_equal_across_ppo_seeds": weather_sequence_equal,
        "unique_daily_weather_hashes_per_wgen_run": {name: len({item[2] for item in sequence}) for name, sequence in wgen_weather_sequences.items()},
        "repeated_seed_year_pairs_with_multiple_runtime_realizations": len(repeated_seed_year_weather_variations),
        "episode_count_variation_is_trajectory_dependent": len({row["episodes"] for row in run_summaries if row["training_regime"] == "RANDOM_WEATHER_WGEN"}) > 1,
        "train_eval_wgen_seed_intersection": sorted(set(WGEN_TRAIN_SEEDS) & set(WGEN_EVAL_SEEDS)),
        "runs": sorted(run_summaries, key=lambda row: (row["training_regime"], row["ppo_seed"])),
    }
    for attempt in range(100):
        name = "training_qc_verified.json" if attempt == 0 else f"training_qc_verified_retry{attempt}.json"
        evidence_path = TASK_ROOT / name
        payload = {**result, "evidence_path": rel(evidence_path)}
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if not evidence_path.exists():
            write_json(evidence_path, payload)
            break
        if evidence_path.read_text(encoding="utf-8") == encoded:
            break
    else:
        raise RuntimeError("Could not allocate a non-overwriting verified training QC file")
    return result


def smoke_gate(smoke_attempt: str = "smoke") -> dict[str, Any]:
    run_dir = SMOKE_ROOT / smoke_attempt
    manifest_path = run_dir / "run_manifest.json"
    episode_path = run_dir / "training_episode_summary.csv"
    checkpoint = run_dir / "models" / f"checkpoint_{SMOKE_TIMESTEPS}.zip"
    checks = {"manifest_exists": manifest_path.is_file(), "episode_log_exists": episode_path.is_file(), "checkpoint_exists": checkpoint.is_file()}
    if all(checks.values()):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        episodes = pd.read_csv(episode_path, keep_default_na=False)
        from sb3_contrib import MaskablePPO
        MaskablePPO.load(str(checkpoint), device="cpu")
        wgen = episodes[episodes["training_regime"].eq("RANDOM_WEATHER_WGEN")]
        ppo_values = pd.to_numeric(wgen["ppo_seed"], errors="coerce").dropna().unique().tolist()
        actual_weather = pd.to_numeric(wgen["actual_rseed1_sent_to_pdi"], errors="coerce").dropna()
        outer_seeds = pd.to_numeric(wgen["outer_reset_seed_received"], errors="coerce").dropna().unique().tolist()
        checks.update({
            "run_completed": manifest.get("run_status") == "completed",
            "ppo_seed_is_zero": ppo_values == [0],
            "weather_seed_uses_training_pool": bool(len(actual_weather) and actual_weather.between(1001, 1080).all()),
            "at_least_two_weather_seeds_reached_pdi": int(actual_weather.nunique()) >= 2,
            "same_year_context_for_weather_diversity_check": bool(pd.to_numeric(wgen["historical_year"], errors="coerce").nunique() == 1),
            "distinct_weather_files_per_seed": bool(wgen["runtime_weather_sha256"].astype(str).nunique() == len(wgen) and wgen["runtime_weather_sha256"].astype(str).ne("").all()),
            "filex_wther_is_wgen": bool(wgen["filex_wther"].astype(str).eq("W").all()),
            "ppo_reset_seed_separate_from_weather_seed": all(int(seed) not in WGEN_TRAIN_SEEDS for seed in outer_seeds),
            "reward_decomposition_consistent": bool(pd.to_numeric(wgen["reward_decomposition_abs_error"], errors="coerce").fillna(float("inf")).le(1e-8).all()),
            "checkpoint_loadable": True,
            "actual_training_steps": manifest.get("total_timesteps_actual"),
            "completed_episodes": int(len(wgen)),
            "weather_seed_sequence": actual_weather.astype(int).tolist(),
        })
    checks["passed"] = all(value is True for key, value in checks.items() if key not in {"actual_training_steps", "completed_episodes", "weather_seed_sequence"})
    write_json(SMOKE_ROOT / f"smoke_gate_{smoke_attempt}.json", checks)
    return checks


def run_all_training() -> list[dict[str, Any]]:
    outcomes = []
    for regime in ("HISTORICAL_WEATHER", "RANDOM_WEATHER_WGEN"):
        for seed in PPO_SEEDS:
            print(f"[train] regime={regime} ppo_seed={seed} start serial", flush=True)
            result = train_model(regime, seed)
            outcomes.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            write_aggregate_manifests()
            if result["run_status"] != "completed":
                write_json(TASK_ROOT / "training_failure_stop.json", {"stopped_after": result, "reason": "run incomplete or resource threshold reached; no automatic retry"})
                break
        if outcomes and outcomes[-1]["run_status"] != "completed":
            break
    return outcomes


def run_all_evaluations() -> None:
    qc = validate_training_evidence()
    if not qc["passed"]:
        raise RuntimeError("Training evidence gate failed: " + "; ".join(qc["issues"]))
    manifest_path = TASK_ROOT / "training_run_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    if len(manifest) != 6 or manifest["run_status"].ne("completed").any():
        raise RuntimeError("Evaluation is gated on all six complete training runs")
    for row in manifest.itertuples(index=False):
        for eval_type in ("heldout_wgen", "observed_weather"):
            print(f"[eval] model={row.training_regime}/seed{row.ppo_seed} type={eval_type} start serial", flush=True)
            evaluate_one(ROOT / str(row.model_final), row.training_regime, int(row.ppo_seed), eval_type)


def run_self_test() -> dict[str, Any]:
    years = generate_year_schedule(100)
    wgen = generate_wgen_schedule(years)
    checks = {
        "historical_schedule_length": len(years) == 100,
        "year_schedule_reproducible": years == generate_year_schedule(100),
        "each_year_block_balanced": all(sorted(r["historical_year"] for r in years[start:start + 10]) == TRAIN_YEARS for start in range(0, 100, 10)),
        "wgen_schedule_reproducible": wgen == generate_wgen_schedule(years),
        "each_wgen_block_covers_full_pool": all(sorted(r["weather_seed"] for r in wgen[start:start + 80]) == WGEN_TRAIN_SEEDS for start in range(0, 80, 80)),
        "weather_train_eval_disjoint": not set(WGEN_TRAIN_SEEDS).intersection(WGEN_EVAL_SEEDS),
        "ppo_seeds_disjoint_from_weather_pool": not set(PPO_SEEDS).intersection(WGEN_TRAIN_SEEDS),
        "sequence_rng_returns_explicit_seed": SequenceWeatherRng(1050).randint(0, 50000) == 1050,
    }
    checks["passed"] = all(checks.values())
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--self-test", action="store_true")
    actions.add_argument("--freeze-config", action="store_true")
    actions.add_argument("--smoke", action="store_true")
    actions.add_argument("--train-all", action="store_true")
    actions.add_argument("--validate-training", action="store_true")
    actions.add_argument("--evaluate-all", action="store_true")
    actions.add_argument("--summarize", action="store_true")
    parser.add_argument("--smoke-attempt", default="smoke")
    args = parser.parse_args()
    if args.self_test:
        result = run_self_test()
    elif args.freeze_config:
        result = freeze_config()
    elif args.smoke:
        result = train_model("RANDOM_WEATHER_WGEN", 0, smoke=True, smoke_attempt=args.smoke_attempt)
        if result["run_status"] == "completed":
            result["smoke_gate"] = smoke_gate(args.smoke_attempt)
        write_json(SMOKE_ROOT / f"{args.smoke_attempt}_run_result.json", result)
    elif args.train_all:
        result = run_all_training()
    elif args.validate_training:
        result = validate_training_evidence()
    elif args.evaluate_all:
        run_all_evaluations()
        result = {"status": "evaluations_completed"}
    else:
        result = summarize_evaluations()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.self_test:
        return 0 if result["passed"] else 1
    if args.freeze_config:
        return 0 if result["passed"] else 1
    if args.smoke:
        return 0 if result.get("smoke_gate", {}).get("passed", False) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
