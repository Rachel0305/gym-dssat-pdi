from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/yc_random_weather_ppo/004_03"
FROZEN = ROOT / "results/yc_random_weather_ppo/004_05"
OUT_BASE = ROOT / "results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining"
OUT = OUT_BASE / "attempt_05"
TREATMENT_EVAL_ROOT = OUT / "evaluation_final/treatment_105"
TREATMENT_EVAL_RETRY_ROOT = OUT / "evaluation_final_retry_03/treatment_105"
CONTROL_EVAL_ROOT = OUT / "control_replay_final/control_75"
REPORT = ROOT / "docs/yc_random_weather_004_13_105mm_early_cap_paired_retraining.md"
PROMPT = ROOT / "prompt_02/004_13_yc_105mm_early_cap_paired_retraining.md"
PILOT_PATH = BASE / "run_controlled_pilot.py"
WRAPPER_PATH = FROZEN / "run_yc_weather_augmentation_multi_seed_archetype_004_05.py"
REPLAY_PATH = ROOT / "src/run_004_07_02_yc_seed345_etcp_replay.py"
YEARS = list(range(2014, 2024))
SEEDS = list(range(8))
REGIME = "RANDOM_WEATHER_WGEN"
EARLY_CAP_MM = 105.0
CANONICAL_EARLY_CAP_MM = 75.0
RSS_LIMIT_MB = 2048.0
TOL = 1e-5
PALETTE = {"control_75": "#2878B5", "treatment_105": "#D9822B"}
WEATHER_HASH_CACHE: dict[int, dict[str, tuple[int, set[str], bool]]] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    for attempt in range(7):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(0.05 * (2 ** attempt))


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
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total / (1024 * 1024)


def assert_new_run() -> None:
    if OUT.exists() or REPORT.exists():
        raise FileExistsError(f"Refusing to overwrite existing 004_13 output: {OUT} or {REPORT}")
    if not PROMPT.is_file():
        raise FileNotFoundError(PROMPT)


def prepare_dirs() -> None:
    for part in ("config", "training/random_weather", "smoke", "evaluation", "control_replay",
                 "logs", "progress", "summaries", "figures", "provenance"):
        (OUT / part).mkdir(parents=True, exist_ok=True)


def load_context():
    pilot = import_module(PILOT_PATH, "yc_00413_frozen_pilot")
    pilot.TASK_ROOT = OUT
    pilot.CONFIG_ROOT = OUT / "config"
    pilot.TRAIN_ROOT = OUT / "training"
    pilot.EVAL_ROOT = OUT / "evaluation"
    pilot.EVAL_RUN_ROOT = OUT / "evaluation/runs"
    pilot.SMOKE_ROOT = OUT / "smoke"
    pilot.PPO_SEEDS = SEEDS
    pilot.MAX_PROCESS_TREE_RSS_MB = RSS_LIMIT_MB
    context = pilot._import_canonical()
    engine = context[1]
    staged = engine.base04036.base04026
    wrapper = import_module(WRAPPER_PATH, "yc_00405_capture_for_00413")
    wrapper.OUT = FROZEN
    wrapper.BASE = BASE
    wrapper.install_step_capture(pilot)
    replay = import_module(REPLAY_PATH, "yc_00407_replay_for_00413")
    helper = replay.load_summary_tools()
    return pilot, wrapper, engine, staged, replay, helper


def copy_frozen_config() -> dict[str, str]:
    source_names = (
        "canonical_yc_ppo_config.json",
        "experiment_design.json",
        "training_weather_schedule_historical.csv",
        "training_weather_schedule_random.csv",
        "evaluation_manifest.csv",
    )
    hashes = {}
    for name in source_names:
        source = FROZEN / "config" / name
        source_03 = BASE / "config" / name
        target = OUT / "config" / name
        if not source.is_file() or not source_03.is_file():
            raise FileNotFoundError(source if not source.is_file() else source_03)
        if sha256_file(source) != sha256_file(source_03):
            raise RuntimeError(f"004_03/004_05 frozen config mismatch: {name}")
        shutil.copy2(source, target)
        if sha256_file(target) != sha256_file(source):
            raise RuntimeError(f"Copied config hash mismatch: {name}")
        hashes[name] = sha256_file(target)
    return hashes


def frozen_control_models() -> pd.DataFrame:
    path = FROZEN / "all_model_manifest_0_7.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    frame = frame[(frame.training_regime == REGIME) & pd.to_numeric(frame.ppo_seed, errors="coerce").isin(SEEDS)].copy()
    if len(frame) != len(SEEDS) or set(pd.to_numeric(frame.ppo_seed).astype(int)) != set(SEEDS):
        raise RuntimeError("004_05 control manifest does not contain exactly random-weather seeds 0-7")
    for row in frame.itertuples(index=False):
        model = ROOT / str(row.model_path)
        if not model.is_file() or sha256_file(model) != str(row.model_sha256).upper():
            raise RuntimeError(f"004_05 seed {row.ppo_seed} checkpoint missing/hash mismatch: {model}")
    return frame.sort_values("ppo_seed")


def frozen_control_model_path(seed: int) -> Path:
    models = pd.read_csv(FROZEN / "all_model_manifest_0_7.csv", keep_default_na=False)
    selected = models[(models.model_id.astype(str) == f"W{seed}") & (models.training_regime == REGIME)]
    if len(selected) != 1:
        raise RuntimeError(f"Missing unique 004_05 control model manifest for seed {seed}")
    return ROOT / str(selected.iloc[0].model_path)


def frozen_observed_input_dir(seed: int, year: int) -> Path:
    if seed <= 2:
        root = BASE / "evaluation/observed_weather/random_weather"
    else:
        root = FROZEN / "evaluation/runs/observed_weather/random_weather"
    return root / f"ppo_seed_{seed}" / "rendered_inputs/YCA" / str(year) / f"eval_observed_weather_RANDOM_WEATHER_WGEN_{seed}_{year}"


def treatment_eval_root(seed: int) -> Path:
    return TREATMENT_EVAL_ROOT if seed <= 2 else TREATMENT_EVAL_RETRY_ROOT


def reference_weather_hash(seed: int, episode_key: str) -> str:
    if seed not in WEATHER_HASH_CACHE:
        path = FROZEN / "evaluation_step_level_all_models.csv"
        columns = ["model_id", "episode_key", "runtime_weather_sha256", "runtime_weather_hash_reconciles"]
        collected: dict[str, tuple[int, set[str], bool]] = {}
        for chunk in pd.read_csv(path, usecols=columns, keep_default_na=False, chunksize=50000):
            selected = chunk[chunk.model_id.astype(str) == f"W{seed}"]
            for key, part in selected.groupby("episode_key", sort=False):
                rows, hashes, reconciled = collected.get(str(key), (0, set(), True))
                hashes.update(str(v).strip().upper() for v in part.runtime_weather_sha256 if str(v).strip())
                reconciled = reconciled and part.runtime_weather_hash_reconciles.astype(str).str.lower().isin(["true", "1"]).all()
                collected[str(key)] = (rows + len(part), hashes, bool(reconciled))
        WEATHER_HASH_CACHE[seed] = collected
    rows, hashes, all_reconciled = WEATHER_HASH_CACHE[seed].get(episode_key, (0, set(), False))
    if rows == 0 or len(hashes) != 1 or not all_reconciled:
        if seed not in (0, 1, 2):
            reference_hashes = [reference_weather_hash(reference_seed, episode_key) for reference_seed in (0, 1, 2)]
            if len(set(reference_hashes)) == 1:
                return reference_hashes[0]
        raise RuntimeError(f"004_05 step-level weather hash is missing/non-unique/unreconciled for W{seed}/{episode_key}")
    return next(iter(hashes))


def compare_observed_input_hashes(seed: int, year: int, eval_root: Path) -> dict[str, Any]:
    old_dir = frozen_observed_input_dir(seed, year)
    run_tag = f"eval_observed_weather_RANDOM_WEATHER_WGEN_{seed}_{year}"
    new_dir = eval_root / "runs/observed_weather/random_weather" / f"ppo_seed_{seed}" / "rendered_inputs/YCA" / str(year) / run_tag
    names = (f"CNYC{str(year)[-2:]}01.WTH", f"YCA_{year}_{run_tag}.jinja2")
    checks = {}
    for name in names:
        old_path, new_path = old_dir / name, new_dir / name
        if not old_path.is_file() or not new_path.is_file():
            raise FileNotFoundError(f"Observed evaluation input missing: {old_path} or {new_path}")
        old_hash, new_hash = sha256_file(old_path), sha256_file(new_path)
        checks[name] = {"frozen_sha256": old_hash, "replay_sha256": new_hash, "identical": old_hash == new_hash}
    return {"checks": checks, "passed": all(row["identical"] for row in checks.values())}


def preflight(pilot, staged, copied_hashes: dict[str, str]) -> dict[str, Any]:
    config_path = FROZEN / "config/canonical_yc_ppo_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    design = json.loads((FROZEN / "config/experiment_design.json").read_text(encoding="utf-8"))
    if float(staged.EARLY_CUM_IRRIGATION_CAP) != CANONICAL_EARLY_CAP_MM:
        raise RuntimeError(f"Expected canonical early cap 75, found {staged.EARLY_CUM_IRRIGATION_CAP}")
    if float(staged.MID_CUM_IRRIGATION_CAP) != 150.0:
        raise RuntimeError(f"Expected canonical mid cap 150, found {staged.MID_CUM_IRRIGATION_CAP}")
    safety = config["action_safety"]
    expected_contract = {
        "mid_cap_mm": 150.0,
        "late_pre_dap90_cap_mm": 195.0,
        "minimum_irrigation_interval_days": 7,
        "minimum_nitrogen_interval_days": 7,
        "season_irrigation_limit_mm": 240.0,
        "season_nitrogen_limit_kg_ha": 250.0,
        "daily_irrigation_limit_mm": 45.0,
        "daily_nitrogen_limit_kg_ha": 120.0,
        "training_steps_requested": 100000,
        "wgen_schedule_seed": 64004,
        "wgen_training_seed_range": [1001, 1080],
    }
    actual = {
        "mid_cap_mm": float(staged.MID_CUM_IRRIGATION_CAP),
        "late_pre_dap90_cap_mm": float(safety["late_irrigation_reserve_mask"]["pre_late_cum_irrigation_cap_mm"]),
        "minimum_irrigation_interval_days": int(safety["min_days_between_irrigation"]),
        "minimum_nitrogen_interval_days": int(safety["min_days_between_fertilization"]),
        "season_irrigation_limit_mm": float(safety["season_irrigation_soft_limit"]),
        "season_nitrogen_limit_kg_ha": float(safety["season_n_soft_limit"]),
        "daily_irrigation_limit_mm": float(safety["daily_irrigation_max"]),
        "daily_nitrogen_limit_kg_ha": float(safety["daily_n_max"]),
        "training_steps_requested": int(config["training"]["total_timesteps"]),
        "wgen_schedule_seed": int(design["seed_roles"]["wgen_weather_schedule_seed"]),
        "wgen_training_seed_range": [min(design["training_wgen_weather_seeds"]), max(design["training_wgen_weather_seeds"])],
    }
    if actual != expected_contract:
        raise RuntimeError(f"Frozen mask/training contract differs: actual={actual}")
    for name in ("action_grid", "algorithm", "ppo_hyperparameters", "reward", "runtime"):
        if name == "runtime":
            continue
        if name == "action_grid" and config[name] != json.loads((BASE / "config/canonical_yc_ppo_config.json").read_text(encoding="utf-8"))[name]:
            raise RuntimeError("Canonical action grid differs from 004_03")
    random_schedule = pd.read_csv(OUT / "config/training_weather_schedule_random.csv")
    historical_schedule = pd.read_csv(OUT / "config/training_weather_schedule_historical.csv")
    if len(random_schedule) != 100000 or len(historical_schedule) != 100000:
        raise RuntimeError("Frozen WGEN/year training schedules do not contain 100K rows")
    seeds = pd.to_numeric(random_schedule.weather_seed, errors="coerce")
    if seeds.isna().any() or set(seeds.astype(int).unique()) != set(range(1001, 1081)):
        raise RuntimeError("Frozen WGEN schedule does not cover seeds 1001-1080")
    if not np.array_equal(pd.to_numeric(random_schedule.historical_year_context), pd.to_numeric(historical_schedule.historical_year)):
        raise RuntimeError("WGEN and historical training schedules have different year contexts")
    control_models = frozen_control_models()
    episode_path = FROZEN / "all_evaluation_episode_level_0_7.csv"
    step_path = FROZEN / "evaluation_step_level_all_models.csv"
    episodes = pd.read_csv(episode_path, keep_default_na=False)
    controls = episodes[(episodes.training_regime == REGIME) & episodes.model_id.astype(str).str.match(r"W[0-7]")]
    if len(controls) != 8 * 30:
        raise RuntimeError(f"Expected 240 formal WGEN PPO control episodes; found {len(controls)}")
    if not step_path.is_file() or step_path.stat().st_size == 0:
        raise FileNotFoundError(step_path)
    eval_qc = json.loads((FROZEN / "evaluation_qc.json").read_text(encoding="utf-8"))
    if eval_qc.get("passed") is not True:
        raise RuntimeError("004_05 evaluation QC is not passed")
    return {
        "status": "passed",
        "canonical_config_sha256": sha256_file(config_path),
        "copied_config_hashes": copied_hashes,
        "control_seed_count": len(control_models),
        "control_episode_count": len(controls),
        "control_step_trace_sha256": sha256_file(step_path),
        "control_episode_sha256": sha256_file(episode_path),
        "mask_contract": {"canonical_early_cap_mm": 75.0, **actual},
        "schedule": {
            "training_schedule_seed": 64004,
            "training_wgen_seeds": [1001, 1080],
            "schedule_rows": len(random_schedule),
            "schedule_sha256": copied_hashes["training_weather_schedule_random.csv"],
        },
        "runtime_versions": {
            "python": sys.version.split()[0],
            "torch": pilot._version("torch"),
            "stable_baselines3": pilot._version("stable-baselines3"),
            "sb3_contrib": pilot._version("sb3-contrib"),
            "gym_dssat_pdi": pilot._version("gym-dssat-pdi"),
        },
        "baseline_model_hashes": {f"seed_{int(r.ppo_seed)}": str(r.model_sha256).upper() for r in control_models.itertuples(index=False)},
    }


def run_mask_probe(pilot, staged, engine) -> dict[str, Any]:
    probe_root = OUT / "smoke/mask_probe"
    probe_root.mkdir(parents=True, exist_ok=False)
    cfg, env_cfg = pilot.make_run_config(REGIME, 0, probe_root)
    schedule = [{"episode_index": 1, "historical_year": 2008, "weather_seed": 1001}]
    env = pilot.ScheduledEpisodeEnv(cfg, env_cfg, schedule, REGIME, 0, "00413_mask_probe", pilot.RUNTIME_BOOTSTRAP_SEED, max_cached_years=1)
    staged.EARLY_CUM_IRRIGATION_CAP = 105.0
    actions = [(float(i), float(n)) for i in engine.BINARY_IRRIGATION_LEVELS for n in engine.BINARY_NITROGEN_LEVELS]
    lookup = {pair: idx for idx, pair in enumerate(actions)}
    noop_id = lookup[(0.0, 0.0)]
    i45_id = lookup[(45.0, 0.0)]
    i30_id = lookup[(30.0, 0.0)]
    observations, _ = env._gym_env.reset(seed=None)
    applied = []
    try:
        for _ in range(30):
            state = env.current_env.safety_state
            cum_i = float(state.cumulative_irrigation)
            dap = float(env.current_env._dap())
            mask105 = np.asarray(env._gym_env.action_masks(), dtype=bool).reshape(-1)
            if abs(cum_i) < 1e-8 and mask105[i45_id]:
                action = i45_id
            elif abs(cum_i - 45.0) < 1e-8 and dap <= 30 and mask105[i30_id]:
                action = i30_id
            elif abs(cum_i - 75.0) < 1e-8 and dap <= 30:
                staged.EARLY_CUM_IRRIGATION_CAP = 75.0
                mask75 = np.asarray(env._gym_env.action_masks(), dtype=bool).reshape(-1)
                staged.EARLY_CUM_IRRIGATION_CAP = 105.0
                mask105 = np.asarray(env._gym_env.action_masks(), dtype=bool).reshape(-1)
                changed = np.flatnonzero(mask105 != mask75).tolist()
                if bool(mask105[i30_id]) and not bool(mask75[i30_id]):
                    newly_legal = [idx for idx in changed if mask105[idx] and not mask75[idx]]
                    for idx in newly_legal:
                        dose_i, _ = actions[idx]
                        if dose_i <= 0 or cum_i + dose_i <= 75.0 or cum_i + dose_i > 105.0:
                            raise RuntimeError(f"Unexpected mask change at action {idx}: {actions[idx]}")
                    if any(mask75[idx] and not mask105[idx] for idx in range(len(mask105))):
                        raise RuntimeError("105 mm mask unexpectedly removed a canonical-legal action")
                    return {
                        "passed": True, "dap": dap, "cumulative_irrigation_mm": cum_i,
                        "canonical_early_cap_mm": 75.0, "treatment_early_cap_mm": 105.0,
                        "newly_legal_action_indices": newly_legal,
                        "newly_legal_actions": [actions[idx] for idx in newly_legal],
                        "all_mask_changes_explained_by_early_cap": True,
                        "mask_before_hash": hashlib.sha256(mask75.tobytes()).hexdigest().upper(),
                        "mask_after_hash": hashlib.sha256(mask105.tobytes()).hexdigest().upper(),
                    }
                action = noop_id
            else:
                action = noop_id
            if not mask105[action]:
                raise RuntimeError(f"Probe attempted illegal action {action} at DAP {dap}")
            observations, _, terminated, truncated, _ = env._gym_env.step(action)
            applied.append({"dap_before": dap, "action": actions[action], "cumulative_irrigation_after": float(env.current_env.safety_state.cumulative_irrigation)})
            if terminated or truncated:
                break
        raise RuntimeError("Could not reach the exact 75 mm early-cap probe state")
    finally:
        env.close()


def install_eval_snapshot_hook(pilot, wrapper, replay, snapshot_root: Path, run_kind: str, seed: int) -> list[dict[str, Any]]:
    if not getattr(wrapper, "_yc00413_snapshot_hook_installed", False):
        original = pilot.ScheduledEpisodeEnv._step

        def tracked(self, action):
            result = original(self, action)
            if (result[2] or result[3]) and str(getattr(self, "run_tag", "")).startswith("eval_"):
                schedule = dict(self.current_schedule or {})
                domain = "heldout_wgen" if "heldout_wgen" in self.run_tag else "observed_weather"
                identity = int(schedule.get("weather_seed")) if domain == "heldout_wgen" else int(schedule.get("historical_year"))
                key = f"{domain}_{identity}"
                target = snapshot_root / run_kind / f"seed_{seed}" / key
                raw = pilot.find_dssat_instance(self.current_env)
                replay.snapshot_outputs(raw, target)
            if int(getattr(self, "episode_days", 0)) % 10 == 0:
                value = process_tree_rss_mb()
                if value >= RSS_LIMIT_MB:
                    raise MemoryError(f"Evaluation RSS {value:.1f} MiB reached hard limit {RSS_LIMIT_MB:.0f}")
            return result

        pilot.ScheduledEpisodeEnv._step = tracked
        wrapper._yc00413_snapshot_hook_installed = True
    return []


def management_events_for_yield(path: Path, target_yield: float) -> pd.DataFrame:
    columns = ["doy", "dap", "operation", "amount"]
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    lines = path.read_text(encoding="latin-1", errors="ignore").splitlines()
    starts = [i for i, line in enumerate(lines) if "DSSAT Cropping System Model" in line]
    if not starts:
        starts = [0]
    blocks = [lines[start:starts[n + 1] if n + 1 < len(starts) else len(lines)] for n, start in enumerate(starts)]
    event_pattern = re.compile(
        r"^\s*\d+\s+[A-Z]{3}\s+\d+,\s+\d{4}\s+(\d+)\s+\d+\s+(-?\d+)\s+MZ\s+.*?\b(Irrigation|Fertilizer)\s+([0-9.]+)",
        re.IGNORECASE,
    )
    harvest_pattern = re.compile(r"\bHarvest Yield\s+([0-9.]+)", re.IGNORECASE)
    candidates = []
    for block_index, block in enumerate(blocks):
        harvests = [float(match.group(1)) for line in block if (match := harvest_pattern.search(line))]
        if not harvests:
            continue
        actual_yield = harvests[-1]
        distance = abs(actual_yield - float(target_yield))
        if distance <= 1.0:
            candidates.append((distance, block_index, block))
    if not candidates:
        raise RuntimeError(f"No MgmtEvent.OUT block matches Summary.OUT yield {target_yield}: {path}")
    _, _, selected_block = min(candidates, key=lambda item: (item[0], -item[1]))
    rows = []
    for line in selected_block:
        match = event_pattern.search(line)
        if match:
            rows.append({
                "doy": int(match.group(1)), "dap": int(match.group(2)),
                "operation": match.group(3).lower(), "amount": float(match.group(4)),
            })
    return pd.DataFrame(rows, columns=columns).drop_duplicates().reset_index(drop=True)


def summary_enrichment(replay, helper, snapshot: Path, episode: dict[str, Any], steps: pd.DataFrame) -> dict[str, Any]:
    selected, _ = replay.selected_summary_row(snapshot, float(episode["yield"]))
    metrics = helper.summary_metrics(snapshot, float(episode["yield"]))
    events = management_events_for_yield(snapshot / "MgmtEvent.OUT", float(episode["yield"]))
    event_i = float(events.loc[events.operation.eq("irrigation"), "amount"].sum()) if len(events) else 0.0
    event_n = float(events.loc[events.operation.eq("fertilizer"), "amount"].sum()) if len(events) else 0.0
    action_i = float(pd.to_numeric(steps.irrigation_action_mm, errors="coerce").fillna(0).sum())
    action_n = float(pd.to_numeric(steps.fertilizer_action_kgN_ha, errors="coerce").fillna(0).sum())
    num = __import__("calculate_five_site_wue_nue_from_summary_019_10", fromlist=["num"]).num
    sy, si, sn = num(selected, "HWAM"), num(selected, "IRCM"), num(selected, "NICM")
    if sy is None or abs(float(sy) - float(episode["yield"])) > 1.0:
        raise RuntimeError("Summary.OUT yield does not close to evaluation episode")
    for label, actual, summary, actions in (("irrigation", event_i, si, action_i), ("nitrogen", event_n, sn, action_n)):
        if summary is None or abs(actual - actions) > 1.0 or abs(float(summary) - actions) > 1.0:
            raise RuntimeError(f"{label} closure failure: event={actual}, summary={summary}, actions={actions}")
    etcp = float(metrics["etcp_mm"])
    wp_et = float(metrics["wp_et_kg_m3"])
    pfp_n = float(metrics["pfp_n_kg_kg"])
    if not math.isfinite(etcp) or etcp <= 0 or not math.isfinite(wp_et) or (action_n > 0 and not math.isfinite(pfp_n)):
        raise RuntimeError("Summary.OUT did not yield finite ETCP/WP_ET/PFP_N")
    action_events = steps[
        (pd.to_numeric(steps.irrigation_action_mm, errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(steps.fertilizer_action_kgN_ha, errors="coerce").fillna(0) > 0)
    ].sort_values("timestep")
    signature = ";".join(
        f"{int(float(r.dap))}:I{float(r.irrigation_action_mm):g}/N{float(r.fertilizer_action_kgN_ha):g}"
        for r in action_events.itertuples(index=False)
    )
    return {
        "etcp_mm": etcp, "wp_et_kg_m3": wp_et, "pfp_n_kg_kg": pfp_n,
        "common_reward": 0.158 * float(episode["yield"]) - 1.1 * event_i - 1.58 * event_n,
        "management_event_count": int(len(events)),
        "irrigation_event_count": int((events.operation == "irrigation").sum()) if len(events) else 0,
        "nitrogen_event_count": int((events.operation == "fertilizer").sum()) if len(events) else 0,
        "management_action_days": int(len(action_events)),
        "irrigation_action_days": int((pd.to_numeric(action_events.irrigation_action_mm, errors="coerce").fillna(0) > 0).sum()) if len(action_events) else 0,
        "nitrogen_action_days": int((pd.to_numeric(action_events.fertilizer_action_kgN_ha, errors="coerce").fillna(0) > 0).sum()) if len(action_events) else 0,
        "action_signature": signature,
        "management_event_irrigation_mm": event_i,
        "management_event_nitrogen_kg_ha": event_n,
        "summary_out_yield": float(sy),
        "summary_out_irrigation_mm": float(si),
        "summary_out_nitrogen_kg_ha": float(sn),
    }


def old_control_episode(seed: int, episode: dict[str, Any], frame: pd.DataFrame) -> pd.Series:
    model_id = f"W{seed}"
    mask = (frame.model_id.astype(str) == model_id) & (frame.training_regime == REGIME)
    if episode["evaluation_weather_type"] == "observed_weather":
        mask &= pd.to_numeric(frame.evaluation_weather_year, errors="coerce").eq(int(episode["evaluation_weather_year"]))
    else:
        mask &= pd.to_numeric(frame.evaluation_weather_seed, errors="coerce").eq(int(episode["evaluation_weather_seed"]))
    matched = frame.loc[mask]
    if len(matched) != 1:
        raise RuntimeError(f"Cannot uniquely match official 004_05 control episode: W{seed}/{episode}")
    return matched.iloc[0]


def compare_control_replay(seed: int, episode: dict[str, Any], steps: pd.DataFrame,
                           old_episodes: pd.DataFrame, old_steps: pd.DataFrame) -> dict[str, Any]:
    old = old_control_episode(seed, episode, old_episodes)
    metric_checks = {}
    mappings = {
        "reward": "reward", "yield": "yield",
        "total_irrigation": "irrigation", "total_fertilizer": "fertilizer",
    }
    for old_name, new_name in mappings.items():
        a = pd.to_numeric(pd.Series([old[old_name]]), errors="coerce").iloc[0]
        b = pd.to_numeric(pd.Series([episode[new_name]]), errors="coerce").iloc[0]
        check_name = {"total_irrigation": "irrigation", "total_fertilizer": "fertilizer"}.get(old_name, old_name)
        metric_checks[check_name] = bool(pd.notna(a) and pd.notna(b) and abs(float(a) - float(b)) <= TOL)
    ref = old_steps[(old_steps.model_id.astype(str) == f"W{seed}") & (old_steps.episode_key.astype(str) == str(old.episode_key))].copy()
    current = steps[steps.episode_key.astype(str) == str(old.episode_key)].copy()
    ref = ref.sort_values("timestep").reset_index(drop=True)
    current = current.sort_values("timestep").reset_index(drop=True)
    action_match = len(ref) == len(current) and len(ref) > 0
    if action_match:
        for col in ("action_index", "irrigation_action_mm", "fertilizer_action_kgN_ha"):
            a = pd.to_numeric(ref[col], errors="coerce").to_numpy(float)
            b = pd.to_numeric(current[col], errors="coerce").to_numpy(float)
            if len(a) != len(b) or not np.allclose(a, b, atol=TOL, rtol=0, equal_nan=True):
                action_match = False
                break
    hash_match = True
    old_hash = str(old.get("runtime_weather_sha256", "") or "")
    new_hash = str(episode.get("runtime_weather_sha256", "") or "")
    if episode["evaluation_weather_type"] == "heldout_wgen":
        old_hash = reference_weather_hash(seed, str(old.episode_key))
        hash_match = bool(new_hash) and old_hash.upper() == new_hash.upper()
    elif old_hash and new_hash:
        hash_match = old_hash == new_hash
    return {
        "official_episode_key": str(old.episode_key),
        "official_metric_checks": metric_checks,
        "official_metrics_all_match": all(metric_checks.values()),
        "official_action_trace_match": bool(action_match),
        "official_runtime_weather_hash_match": bool(hash_match),
        "official_replay_exact": bool(all(metric_checks.values()) and action_match and hash_match),
        "official_model_id": f"W{seed}",
        "official_yield": float(old["yield"]),
        "official_irrigation": float(old["total_irrigation"]),
        "official_fertilizer": float(old["total_fertilizer"]),
        "official_reward": float(old["reward"]),
        "official_action_signature": ";".join(
            f"{int(float(r.dap))}:I{float(r.irrigation_action_mm):g}/N{float(r.fertilizer_action_kgN_ha):g}"
            for r in ref.sort_values("timestep").itertuples(index=False)
            if float(r.irrigation_action_mm or 0) > 0 or float(r.fertilizer_action_kgN_ha or 0) > 0
        ),
    }


def evaluate_group(group: str, seed: int) -> None:
    pilot, wrapper, _, staged, replay, helper = load_context()
    cap = EARLY_CAP_MM if group == "treatment_105" else CANONICAL_EARLY_CAP_MM
    staged.EARLY_CUM_IRRIGATION_CAP = cap
    staged.MID_CUM_IRRIGATION_CAP = 150.0
    eval_root = treatment_eval_root(seed) if group == "treatment_105" else CONTROL_EVAL_ROOT
    pilot.EVAL_RUN_ROOT = eval_root / "runs"
    pilot.EVAL_ROOT = eval_root
    pilot.CONFIG_ROOT = OUT / "config"
    model_path = (
        OUT / f"training/random_weather/ppo_seed_{seed}/models/checkpoint_100000.zip"
        if group == "treatment_105"
        else frozen_control_model_path(seed)
    )
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    snapshot_root = eval_root / "snapshots"
    if not eval_root.exists():
        eval_root.mkdir(parents=True)
    install_eval_snapshot_hook(pilot, wrapper, replay, snapshot_root, group, seed)
    outputs = []
    old_episodes = None
    old_steps = None
    if group == "control_75":
        official = pd.read_csv(FROZEN / "all_evaluation_episode_level_0_7.csv", keep_default_na=False)
        old_episodes = official[(official.model_id.astype(str) == f"W{seed}") & (official.training_regime == REGIME)].copy()
        columns = ["model_id", "episode_key", "timestep", "dap", "action_index", "irrigation_action_mm", "fertilizer_action_kgN_ha"]
        old_steps = pd.read_csv(FROZEN / "evaluation_step_level_all_models.csv", usecols=columns, keep_default_na=False)
        old_steps = old_steps[old_steps.model_id.astype(str) == f"W{seed}"].copy()
    for domain in ("observed_weather", "heldout_wgen"):
        wrapper.STEP_CAPTURE.clear()
        rows = pilot.evaluate_one(model_path, REGIME, seed, domain)
        step_frame = pd.DataFrame(wrapper.STEP_CAPTURE)
        expected = 10 if domain == "observed_weather" else 20
        if len(rows) != expected or step_frame.empty:
            raise RuntimeError(f"{group}/seed{seed}/{domain}: incomplete episodes or action trace")
        step_path = eval_root / "action_traces" / f"seed_{seed}_{domain}.csv"
        step_path.parent.mkdir(parents=True, exist_ok=True)
        step_frame.to_csv(step_path, index=False, encoding="utf-8-sig", lineterminator="\n")
        for row in rows:
            identity = int(row["evaluation_weather_year"]) if domain == "observed_weather" else int(row["evaluation_weather_seed"])
            key = f"{domain}_{identity}"
            episode_key = f"{domain}:{identity}"
            snapshot = snapshot_root / group / f"seed_{seed}" / key
            if not (snapshot / "Summary.OUT").is_file():
                raise FileNotFoundError(f"Missing DSSAT Summary.OUT snapshot: {snapshot}")
            part = step_frame[step_frame.episode_key.astype(str) == episode_key].copy()
            if part.empty:
                raise RuntimeError(f"Missing per-episode step trace: {key}")
            enriched = dict(row)
            enriched.update(summary_enrichment(replay, helper, snapshot, row, part))
            enriched.update({
                "group": group, "ppo_seed": seed, "evaluation_weather_type": domain,
                "evaluation_weather_year": identity if domain == "observed_weather" else "",
                "evaluation_weather_seed": identity if domain == "heldout_wgen" else "",
                "checkpoint_sha256": sha256_file(model_path),
                "summary_snapshot": rel(snapshot),
                "action_trace": rel(step_path),
            })
            if domain == "observed_weather":
                input_provenance = compare_observed_input_hashes(seed, identity, eval_root)
                if not input_provenance["passed"]:
                    raise RuntimeError(f"Observed input hashes differ from frozen evaluation: {group}/seed{seed}/{identity}")
                enriched["input_hash_checks"] = json.dumps(input_provenance["checks"], sort_keys=True)
                enriched["input_hash_checks_passed"] = True
            else:
                expected_weather_hash = reference_weather_hash(seed, f"heldout_wgen:{identity}")
                actual_weather_hash = str(row.get("runtime_weather_sha256", "")).upper()
                enriched["reference_runtime_weather_sha256"] = expected_weather_hash
                enriched["runtime_weather_hash_match"] = actual_weather_hash == expected_weather_hash
                if not enriched["runtime_weather_hash_match"]:
                    raise RuntimeError(f"Held-out WGEN hash differs from 004_05 step-level reference: {group}/seed{seed}/{identity}")
            if group == "control_75":
                reconciliation = compare_control_replay(seed, row, part, old_episodes, old_steps)
                enriched.update(reconciliation)
                enriched["replay_reward"] = enriched["reward"]
                enriched["replay_yield"] = enriched["yield"]
                enriched["replay_irrigation"] = enriched["irrigation"]
                enriched["replay_fertilizer"] = enriched["fertilizer"]
                enriched["action_signature_replay"] = enriched["action_signature"]
                enriched["yield"] = reconciliation["official_yield"]
                enriched["irrigation"] = reconciliation["official_irrigation"]
                enriched["fertilizer"] = reconciliation["official_fertilizer"]
                enriched["reward"] = reconciliation["official_reward"]
                enriched["common_reward"] = 0.158 * enriched["yield"] - 1.1 * enriched["irrigation"] - 1.58 * enriched["fertilizer"]
                enriched["action_signature"] = reconciliation["official_action_signature"]
                enriched["wp_et_kg_m3_replay"] = enriched["wp_et_kg_m3"]
                if not enriched["official_replay_exact"]:
                    enriched["wp_et_kg_m3"] = np.nan
                    enriched["pfp_n_kg_kg"] = np.nan
            outputs.append(enriched)
        wrapper.STEP_CAPTURE.clear()
        gc.collect()
    out_path = eval_root / f"seed_{seed}_evaluation_metrics.csv"
    pd.DataFrame(outputs).to_csv(out_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    atomic_json(eval_root / f"seed_{seed}_evaluation_manifest.json", {
        "group": group, "seed": seed, "status": "passed", "checkpoint": rel(model_path),
        "checkpoint_sha256": sha256_file(model_path), "early_cap_mm": cap,
        "episodes_observed": 10, "episodes_heldout_wgen": 20,
        "evaluation_metrics": rel(out_path), "peak_process_tree_rss_mb": process_tree_rss_mb(),
        "official_control_replay_exact": bool(all(x.get("official_replay_exact", False) for x in outputs)) if group == "control_75" else None,
    })
    print(f"[eval-pass] group={group} seed={seed} episodes=30 RSS={process_tree_rss_mb():.1f}MiB", flush=True)


def training_progress_watch(seed: int, stop: threading.Event) -> None:
    run_dir = OUT / f"training/random_weather/ppo_seed_{seed}"
    progress_path = OUT / f"progress/seed_{seed}.json"
    requested = [25000, 50000, 75000, 100000]
    while not stop.wait(5):
        checkpoints = {}
        for step in requested:
            path = run_dir / f"models/checkpoint_{step}.zip"
            if path.is_file():
                checkpoints[str(step)] = {"path": rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        resource_path = run_dir / "resource_usage.csv"
        peak = None
        if resource_path.is_file():
            try:
                values = pd.to_numeric(pd.read_csv(resource_path).process_tree_rss_mb, errors="coerce").dropna()
                peak = float(values.max()) if len(values) else None
            except Exception:
                peak = None
        episode_path = run_dir / "training_episode_summary.csv"
        episode_count = 0
        if episode_path.is_file():
            try:
                episode_count = sum(1 for _ in episode_path.open("r", encoding="utf-8-sig")) - 1
            except OSError:
                pass
        atomic_json(progress_path, {
            "seed": seed, "status": "training", "checkpoints": checkpoints,
            "completed_episodes_flushed": max(episode_count, 0), "peak_process_tree_rss_mb": peak,
            "updated_utc": pd.Timestamp.utcnow().isoformat(),
        })


def train_one(seed: int) -> None:
    if seed not in SEEDS:
        raise ValueError(seed)
    pilot, _, _, staged, _, _ = load_context()
    staged.EARLY_CUM_IRRIGATION_CAP = EARLY_CAP_MM
    staged.MID_CUM_IRRIGATION_CAP = 150.0
    run_dir = OUT / f"training/random_weather/ppo_seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Seed output already exists; refusing overwrite/retrain: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / f"progress/seed_{seed}.json", {
        "seed": seed, "status": "training", "early_cap_mm": EARLY_CAP_MM,
        "started_utc": pd.Timestamp.utcnow().isoformat(), "checkpoints": {},
    })
    stop = threading.Event()
    watcher = threading.Thread(target=training_progress_watch, args=(seed, stop), daemon=True)
    watcher.start()
    try:
        manifest = pilot.train_model(REGIME, seed, smoke=False)
    finally:
        stop.set()
        watcher.join(timeout=10)
    config_path = run_dir / "training_config.json"
    task_config = json.loads(config_path.read_text(encoding="utf-8"))
    task_config["004_13_runtime_mask_override"] = {
        "early_dap_end": 30, "early_cumulative_irrigation_cap_mm": EARLY_CAP_MM,
        "canonical_early_cap_mm": CANONICAL_EARLY_CAP_MM, "mid_cumulative_irrigation_cap_mm": 150,
        "source": "process-local override of engine.base04036.base04026.EARLY_CUM_IRRIGATION_CAP",
    }
    config_path.write_text(json.dumps(task_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    checkpoint = run_dir / "models/checkpoint_100000.zip"
    rss = float(manifest.get("max_process_tree_rss_mb") or 0)
    passed = (
        manifest.get("run_status") == "completed"
        and int(manifest.get("total_timesteps_requested", 0)) == 100000
        and int(manifest.get("total_timesteps_actual", 0)) >= 100000
        and checkpoint.is_file()
        and rss < RSS_LIMIT_MB
    )
    seed_manifest = {
        **manifest, "seed": seed, "passed": passed,
        "early_cap_mm": EARLY_CAP_MM, "mid_cap_mm": 150.0,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
        "training_config_sha256": sha256_file(config_path),
        "progress_evidence": rel(OUT / f"progress/seed_{seed}.json"),
        "formal_training_log": rel(OUT / f"logs/train_seed_{seed}.log"),
        "updated_utc": pd.Timestamp.utcnow().isoformat(),
    }
    atomic_json(run_dir / "004_13_seed_manifest.json", seed_manifest)
    atomic_json(OUT / f"progress/seed_{seed}.json", {
        "seed": seed, "status": "passed" if passed else "failed",
        "checkpoint_sha256": seed_manifest["checkpoint_sha256"], "peak_process_tree_rss_mb": rss,
        "completed_utc": pd.Timestamp.utcnow().isoformat(),
    })
    if not passed:
        raise RuntimeError(f"Training QC failed for seed {seed}; outputs retained and later seeds not started")
    print(f"[train-pass] seed={seed} steps={manifest['total_timesteps_actual']} RSS={rss:.1f}MiB sha256={seed_manifest['checkpoint_sha256']}", flush=True)


def evaluate_smoke(pilot, wrapper, replay, helper, model_path: Path) -> dict[str, Any]:
    pilot.EVAL_RUN_ROOT = OUT / "smoke/evaluation/runs"
    pilot.EVAL_ROOT = OUT / "smoke/evaluation"
    pilot.OBSERVED_YEARS = [2014]
    pilot.WGEN_EVAL_SEEDS = [1081]
    install_eval_snapshot_hook(pilot, wrapper, replay, OUT / "smoke/evaluation/snapshots", "smoke_105", 0)
    observations = []
    for domain in ("observed_weather", "heldout_wgen"):
        wrapper.STEP_CAPTURE.clear()
        rows = pilot.evaluate_one(model_path, REGIME, 0, domain)
        if len(rows) != 1:
            raise RuntimeError(f"Smoke {domain} evaluation did not return one episode")
        row = rows[0]
        key = "observed_weather_2014" if domain == "observed_weather" else "heldout_wgen_1081"
        episode_key = "observed_weather:2014" if domain == "observed_weather" else "heldout_wgen:1081"
        snapshot = OUT / "smoke/evaluation/snapshots/smoke_105/seed_0" / key
        steps = pd.DataFrame(wrapper.STEP_CAPTURE)
        steps.to_csv(OUT / f"smoke/evaluation/{domain}_step_trace.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        enrich = summary_enrichment(replay, helper, snapshot, row, steps[steps.episode_key.astype(str).eq(episode_key)])
        if domain == "heldout_wgen":
            formal_hash = reference_weather_hash(0, "heldout_wgen:1081")
            if formal_hash != str(row["runtime_weather_sha256"]).upper():
                raise RuntimeError("Smoke held-out WGEN runtime weather hash differs from 004_05 seed0/1081")
            enrich["reference_runtime_weather_sha256"] = formal_hash
            enrich["runtime_weather_hash_match"] = True
        else:
            input_provenance = compare_observed_input_hashes(0, 2014, OUT / "smoke/evaluation")
            if not input_provenance["passed"]:
                raise RuntimeError("Smoke observed rendered input hashes differ from the seed0 frozen source")
            enrich["input_hash_checks"] = input_provenance["checks"]
        observations.append({"domain": domain, "episode": row, "summary_metrics": enrich, "step_count": len(steps)})
        wrapper.STEP_CAPTURE.clear()
    peak = process_tree_rss_mb()
    if peak >= RSS_LIMIT_MB:
        raise MemoryError(f"Smoke evaluation RSS {peak:.1f} MiB reached {RSS_LIMIT_MB:.0f} MiB")
    return {"episodes": observations, "peak_process_tree_rss_mb": peak}


def debug_capture_existing_smoke() -> None:
    pilot, wrapper, _, staged, replay, _ = load_context()
    staged.EARLY_CUM_IRRIGATION_CAP = EARLY_CAP_MM
    staged.MID_CUM_IRRIGATION_CAP = 150.0
    debug_root = OUT / "smoke/debug_evaluation"
    if debug_root.exists():
        raise FileExistsError(f"Refusing to overwrite capture diagnostic: {debug_root}")
    debug_root.mkdir(parents=True)
    pilot.EVAL_RUN_ROOT = debug_root / "runs"
    pilot.EVAL_ROOT = debug_root
    pilot.OBSERVED_YEARS = [2014]
    model_path = OUT / "smoke/smoke_seed_0_earlycap105/models/checkpoint_432.zip"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    install_eval_snapshot_hook(pilot, wrapper, replay, debug_root / "snapshots", "smoke_debug", 0)
    wrapper.STEP_CAPTURE.clear()
    rows = pilot.evaluate_one(model_path, REGIME, 0, "observed_weather")
    frame = pd.DataFrame(wrapper.STEP_CAPTURE)
    frame.to_csv(debug_root / "step_capture.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    report = {
        "episode": rows[0] if rows else None,
        "step_rows": len(frame),
        "columns": frame.columns.tolist(),
        "episode_keys": frame.episode_key.astype(str).unique().tolist() if "episode_key" in frame else [],
        "sum_irrigation_action_mm": float(pd.to_numeric(frame.get("irrigation_action_mm"), errors="coerce").fillna(0).sum()) if len(frame) else None,
        "sum_fertilizer_action_kgN_ha": float(pd.to_numeric(frame.get("fertilizer_action_kgN_ha"), errors="coerce").fillna(0).sum()) if len(frame) else None,
        "first_rows": frame.head(5).to_dict(orient="records"),
    }
    atomic_json(debug_root / "capture_diagnostic.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def run_smoke() -> None:
    assert_new_run()
    prepare_dirs()
    copied = copy_frozen_config()
    pilot, wrapper, engine, staged, replay, helper = load_context()
    pre = preflight(pilot, staged, copied)
    staged.EARLY_CUM_IRRIGATION_CAP = 75.0
    probe = run_mask_probe(pilot, staged, engine)
    staged.EARLY_CUM_IRRIGATION_CAP = EARLY_CAP_MM
    smoke_training = pilot.train_model(REGIME, 0, smoke=True, smoke_attempt="smoke_seed_0_earlycap105")
    smoke_model = OUT / "smoke/smoke_seed_0_earlycap105/models/checkpoint_432.zip"
    if smoke_training.get("run_status") != "completed" or not smoke_model.is_file():
        raise RuntimeError("432-step 105 mm smoke training failed or checkpoint missing")
    from sb3_contrib import MaskablePPO
    smoke_loaded = MaskablePPO.load(str(smoke_model), device="cpu")
    del smoke_loaded
    smoke_eval = evaluate_smoke(pilot, wrapper, replay, helper, smoke_model)
    training_rss = float(smoke_training.get("max_process_tree_rss_mb") or 0)
    peak = max(training_rss, float(smoke_eval["peak_process_tree_rss_mb"]))
    if peak >= RSS_LIMIT_MB:
        raise MemoryError(f"Smoke peak RSS {peak:.1f} MiB reached {RSS_LIMIT_MB:.0f} MiB")
    smoke_manifest = {
        "task": "004_13 YC 105 mm early-cap paired retraining",
        "status": "passed", "smoke_passed": True,
        "completed_utc": pd.Timestamp.utcnow().isoformat(),
        "preflight": pre, "mask_probe": probe,
        "smoke_training": smoke_training,
        "smoke_checkpoint": rel(smoke_model), "smoke_checkpoint_sha256": sha256_file(smoke_model),
        "smoke_evaluation": smoke_eval, "peak_process_tree_rss_mb": peak,
        "rss_limit_mb": RSS_LIMIT_MB, "formal_8_seed_training_authorized_by_gate": True,
        "canonical_files_modified": False,
    }
    atomic_json(OUT / "smoke/smoke_manifest.json", smoke_manifest)
    atomic_json(OUT / "formal_manifest.json", {
        "task": "004_13 YC 105 mm early-cap paired retraining",
        "status": "smoke_passed_formal_running", "smoke_passed": True,
        "early_cap_mm": EARLY_CAP_MM, "seeds": SEEDS,
        "control_source": rel(FROZEN), "smoke_manifest": rel(OUT / "smoke/smoke_manifest.json"),
        "peak_smoke_rss_mb": peak, "rss_limit_mb": RSS_LIMIT_MB,
        "started_utc": pd.Timestamp.utcnow().isoformat(),
    })
    print(f"[smoke-pass] cap=105mm mask_probe=pass checkpoint=loadable eval=2 RSS={peak:.1f}MiB", flush=True)


def run_child(phase: str, seed: int | None = None, group: str | None = None, log_path: Path | None = None) -> None:
    command = [sys.executable, str(Path(__file__).resolve()), "--phase", phase]
    if seed is not None:
        command += ["--seed", str(seed)]
    if group is not None:
        command += ["--group", group]
    if log_path is None:
        if phase == "eval-one" and group == "treatment_105":
            log_path = OUT / "logs" / f"eval-one_treatment_105_repaired_{seed}.log"
        else:
            log_path = OUT / "logs" / f"{phase}_{group or ''}_{seed if seed is not None else ''}.log"
    if log_path.exists():
        original = log_path
        attempt = 2
        while log_path.exists():
            log_path = original.with_name(f"{original.stem}_retry_{attempt}{original.suffix}")
            attempt += 1
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(f"Child phase failed ({phase}, exit={completed.returncode}). Log: {rel(log_path)}\n" + "\n".join(tail))


def verify_completed_seed(seed: int) -> dict[str, Any]:
    run_dir = OUT / f"training/random_weather/ppo_seed_{seed}"
    manifest_path = run_dir / "004_13_seed_manifest.json"
    checkpoint = run_dir / "models/checkpoint_100000.zip"
    if not manifest_path.is_file() or not checkpoint.is_file():
        raise RuntimeError(f"Seed {seed} training is incomplete; preserving files and stopping")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("passed") is not True or manifest.get("run_status") != "completed" or sha256_file(checkpoint) != manifest.get("checkpoint_sha256"):
        raise RuntimeError(f"Seed {seed} completion/checkpoint integrity gate failed")
    return manifest


def run_formal() -> None:
    smoke_path = OUT / "smoke/smoke_manifest.json"
    if not smoke_path.is_file() or json.loads(smoke_path.read_text(encoding="utf-8")).get("smoke_passed") is not True:
        raise RuntimeError("Formal training blocked: smoke gate has not passed")
    for seed in SEEDS:
        run_dir = OUT / f"training/random_weather/ppo_seed_{seed}"
        if run_dir.exists() and any(run_dir.iterdir()):
            verify_completed_seed(seed)
            print(f"[train-reuse] seed={seed} verified complete", flush=True)
            continue
        log = OUT / f"logs/train_seed_{seed}.log"
        run_child("train-one", seed=seed, log_path=log)
        verified = verify_completed_seed(seed)
        print(f"[train-verified] seed={seed} sha256={verified['checkpoint_sha256']}", flush=True)
        atomic_json(OUT / "formal_manifest.json", {
            "task": "004_13 YC 105 mm early-cap paired retraining",
            "status": "training_in_progress", "smoke_passed": True,
            "early_cap_mm": EARLY_CAP_MM, "seeds": SEEDS,
            "completed_seeds": [s for s in SEEDS if (OUT / f"training/random_weather/ppo_seed_{s}/004_13_seed_manifest.json").is_file()],
            "current_seed": seed, "updated_utc": pd.Timestamp.utcnow().isoformat(),
            "rss_limit_mb": RSS_LIMIT_MB, "control_source": rel(FROZEN),
        })
    for seed in SEEDS:
        verify_completed_seed(seed)
    atomic_json(OUT / "formal_manifest.json", {
        "task": "004_13 YC 105 mm early-cap paired retraining",
        "status": "training_complete_evaluation_in_progress", "smoke_passed": True,
        "early_cap_mm": EARLY_CAP_MM, "seeds": SEEDS,
        "completed_seeds": SEEDS, "updated_utc": pd.Timestamp.utcnow().isoformat(),
        "rss_limit_mb": RSS_LIMIT_MB, "control_source": rel(FROZEN),
    })
    for group in ("treatment_105", "control_75"):
        for seed in SEEDS:
            path = (treatment_eval_root(seed) / f"seed_{seed}_evaluation_manifest.json"
                    if group == "treatment_105"
                    else CONTROL_EVAL_ROOT / f"seed_{seed}_evaluation_manifest.json")
            if path.is_file():
                old = json.loads(path.read_text(encoding="utf-8"))
                if old.get("status") == "passed" and old.get("seed") == seed:
                    print(f"[eval-reuse] group={group} seed={seed}", flush=True)
                    continue
            run_child("eval-one", seed=seed, group=group)
    analyze()


def pareto_flags(group: pd.DataFrame) -> pd.Series:
    flags = []
    values = group[["yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha"]].to_numpy(float)
    for idx, (y, i, n) in enumerate(values):
        dominated = any(
            yy >= y and ii <= i and nn <= n and (yy > y or ii < i or nn < n)
            for j, (yy, ii, nn) in enumerate(values) if j != idx
        )
        flags.append(not dominated)
    return pd.Series(flags, index=group.index)


def aggregate_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = []
    for group, root in (
        ("treatment_105", TREATMENT_EVAL_ROOT),
        ("control_75", CONTROL_EVAL_ROOT),
    ):
        frames = []
        for seed in SEEDS:
            seed_root = treatment_eval_root(seed) if group == "treatment_105" else root
            path = seed_root / f"seed_{seed}_evaluation_metrics.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frames.append(pd.read_csv(path, keep_default_na=False))
        frame = pd.concat(frames, ignore_index=True)
        frame["group"] = group
        for col in ("yield", "irrigation", "fertilizer", "reward", "etcp_mm", "wp_et_kg_m3", "pfp_n_kg_kg", "common_reward"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.rename(columns={
            "yield": "yield_kg_ha", "irrigation": "irrigation_mm",
            "fertilizer": "nitrogen_kg_ha", "reward": "canonical_reward",
        })
        groups.append(frame)
    episodes = pd.concat(groups, ignore_index=True, sort=False)
    seed_summary = episodes.groupby(["group", "ppo_seed", "evaluation_weather_type"], as_index=False).agg(
        n_episodes=("yield_kg_ha", "count"),
        yield_kg_ha=("yield_kg_ha", "mean"), yield_sd_kg_ha=("yield_kg_ha", "std"),
        irrigation_mm=("irrigation_mm", "mean"), nitrogen_kg_ha=("nitrogen_kg_ha", "mean"),
        wp_et_kg_m3=("wp_et_kg_m3", "mean"), pfp_n_kg_kg=("pfp_n_kg_kg", "mean"),
        common_reward=("common_reward", "mean"), canonical_reward=("canonical_reward", "mean"),
        management_action_days=("management_action_days", "mean"),
        irrigation_action_days=("irrigation_action_days", "mean"),
        nitrogen_action_days=("nitrogen_action_days", "mean"),
    )
    paired = seed_summary.pivot(index=["ppo_seed", "evaluation_weather_type"], columns="group")
    pair_rows = []
    metrics = ("yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "wp_et_kg_m3", "pfp_n_kg_kg", "common_reward", "canonical_reward", "management_action_days")
    for (seed, domain), row in paired.iterrows():
        out = {"ppo_seed": int(seed), "evaluation_weather_type": domain}
        for metric in metrics:
            control = row.get((metric, "control_75"), np.nan)
            treatment = row.get((metric, "treatment_105"), np.nan)
            out[f"control75_{metric}"] = float(control) if pd.notna(control) else np.nan
            out[f"treatment105_{metric}"] = float(treatment) if pd.notna(treatment) else np.nan
            out[f"delta_{metric}"] = float(treatment - control) if pd.notna(treatment) and pd.notna(control) else np.nan
        y0, i0, n0 = (out[f"control75_{x}"] for x in ("yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha"))
        y1, i1, n1 = (out[f"treatment105_{x}"] for x in ("yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha"))
        out["pareto_dominates_75"] = bool(y1 >= y0 and i1 <= i0 and n1 <= n0 and (y1 > y0 or i1 < i0 or n1 < n0))
        out["pareto_dominated_by_75"] = bool(y0 >= y1 and i0 <= i1 and n0 <= n1 and (y0 > y1 or i0 < i1 or n0 < n1))
        out["higher_yield_and_lower_N"] = bool(y1 > y0 and n1 < n0)
        out["canonical_reward_higher"] = bool(out["delta_canonical_reward"] > 0)
        pair_rows.append(out)
    paired_frame = pd.DataFrame(pair_rows)
    key = "evaluation_weather_year"
    episodes["episode_pair_key"] = np.where(
        episodes.evaluation_weather_type.eq("observed_weather"),
        pd.to_numeric(episodes.evaluation_weather_year, errors="coerce").astype("Int64").astype(str),
        pd.to_numeric(episodes.evaluation_weather_seed, errors="coerce").astype("Int64").astype(str),
    )
    treatment = episodes[episodes.group.eq("treatment_105")].copy()
    control = episodes[episodes.group.eq("control_75")].copy()
    join_keys = ["ppo_seed", "evaluation_weather_type", "episode_pair_key"]
    paired_episode = control.merge(treatment, on=join_keys, suffixes=("_75", "_105"), validate="one_to_one")
    episode_metrics = ("yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "wp_et_kg_m3", "pfp_n_kg_kg", "common_reward", "canonical_reward")
    for metric in episode_metrics:
        paired_episode[f"delta_{metric}"] = paired_episode[f"{metric}_105"] - paired_episode[f"{metric}_75"]
    paired_episode["pareto_dominates_75"] = (
        (paired_episode.yield_kg_ha_105 >= paired_episode.yield_kg_ha_75)
        & (paired_episode.irrigation_mm_105 <= paired_episode.irrigation_mm_75)
        & (paired_episode.nitrogen_kg_ha_105 <= paired_episode.nitrogen_kg_ha_75)
        & (
            (paired_episode.yield_kg_ha_105 > paired_episode.yield_kg_ha_75)
            | (paired_episode.irrigation_mm_105 < paired_episode.irrigation_mm_75)
            | (paired_episode.nitrogen_kg_ha_105 < paired_episode.nitrogen_kg_ha_75)
        )
    )
    paired_episode["higher_yield_and_lower_N"] = (
        (paired_episode.yield_kg_ha_105 > paired_episode.yield_kg_ha_75)
        & (paired_episode.nitrogen_kg_ha_105 < paired_episode.nitrogen_kg_ha_75)
    )
    overall = seed_summary.groupby(["group", "evaluation_weather_type"], as_index=False).agg(
        seed_count=("ppo_seed", "nunique"),
        yield_mean=("yield_kg_ha", "mean"), yield_between_seed_sd=("yield_kg_ha", "std"),
        irrigation_mean=("irrigation_mm", "mean"), nitrogen_mean=("nitrogen_kg_ha", "mean"),
        wp_et_mean=("wp_et_kg_m3", "mean"), pfp_n_mean=("pfp_n_kg_kg", "mean"),
        common_reward_mean=("common_reward", "mean"), canonical_reward_mean=("canonical_reward", "mean"),
    )
    archetypes = []
    for (group, seed, domain), part in episodes.groupby(["group", "ppo_seed", "evaluation_weather_type"]):
        mean_i, mean_n, mean_y = part.irrigation_mm.mean(), part.nitrogen_kg_ha.mean(), part.yield_kg_ha.mean()
        archetypes.append({
            "group": group, "ppo_seed": int(seed), "evaluation_weather_type": domain,
            "mean_irrigation_mm": mean_i, "mean_nitrogen_kg_ha": mean_n, "mean_yield_kg_ha": mean_y,
            "very_low_input": bool(mean_i <= 75 and mean_n <= 80),
            "yield_collapse": bool(domain == "observed_weather" and mean_y < 6500 and mean_i <= 75 and mean_n <= 80),
            "fixed_N240": bool((np.isclose(part.nitrogen_kg_ha, 240, atol=1e-6)).mean() >= 0.80),
            "n240_episode_fraction": float(np.isclose(part.nitrogen_kg_ha, 240, atol=1e-6).mean()),
            "control_replay_exact_episode_fraction": float(part.get("official_replay_exact", pd.Series(False, index=part.index)).astype(str).str.lower().isin(["true", "1"]).mean()) if group == "control_75" else np.nan,
        })
    archetype_frame = pd.DataFrame(archetypes)
    return episodes, seed_summary, paired_frame, overall, archetype_frame, paired_episode


def make_figures(episodes: pd.DataFrame, seed_summary: pd.DataFrame, paired: pd.DataFrame) -> list[str]:
    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9, "legend.fontsize": 8})
    paths = []
    deltas = ["yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "wp_et_kg_m3", "pfp_n_kg_kg", "common_reward"]
    labels = ["Yield (kg/ha)", "Irrigation (mm)", "Nitrogen (kg/ha)", "WP_ET (kg/m3)", "PFP_N", "Common reward"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), dpi=170)
    for ax, metric, label in zip(axes.flat, deltas, labels):
        for domain, marker in (("observed_weather", "o"), ("heldout_wgen", "s")):
            col = f"delta_{metric}"
            part = paired[paired.evaluation_weather_type.eq(domain)].sort_values("ppo_seed")
            ax.scatter(part.ppo_seed, part[col], marker=marker, s=34, color="#2878B5" if domain == "observed_weather" else "#D9822B",
                       label="Observed" if domain == "observed_weather" else "Held-out WGEN", alpha=.9, edgecolors="white", linewidths=.45)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_title(label); ax.set_xlabel("Paired PPO seed"); ax.set_xticks(SEEDS)
        ax.set_ylabel("105 mm − 75 mm"); ax.grid(axis="y", color="#D9DEE3", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend(frameon=False, loc="best")
    fig.suptitle("YC 004_13 paired seed outcomes", y=1.01)
    fig.tight_layout()
    path = figdir / "figure_seed_paired_metric_deltas.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(rel(path))

    fig, ax = plt.subplots(figsize=(7.2, 5.5), dpi=170)
    for domain, marker, color in (("observed_weather", "o", "#2878B5"), ("heldout_wgen", "s", "#D9822B")):
        part = paired[paired.evaluation_weather_type.eq(domain)]
        for row in part.itertuples(index=False):
            n_delta = getattr(row, "delta_nitrogen_kg_ha")
            ax.scatter(getattr(row, "delta_irrigation_mm"), getattr(row, "delta_yield_kg_ha"),
                       marker=marker, s=78, c=color, edgecolors="#1F2933" if n_delta <= 0 else "white",
                       linewidths=1.4 if n_delta <= 0 else .6, alpha=.9)
            ax.annotate(str(row.ppo_seed), (getattr(row, "delta_irrigation_mm"), getattr(row, "delta_yield_kg_ha")),
                        xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.axhline(0, color="#333333", linewidth=1); ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Δ irrigation (mm; left = water saving)"); ax.set_ylabel("Δ yield (kg/ha; up = gain)")
    ax.set_title("Paired water-yield trade-off\ncircle = observed, square = held-out WGEN; dark edge = ΔN ≤ 0")
    ax.grid(color="#D9DEE3", linewidth=.7); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); path = figdir / "figure_seed_paired_pareto_tradeoff.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(rel(path))

    summary_metrics = ["yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "wp_et_kg_m3", "pfp_n_kg_kg", "canonical_reward"]
    titles = ["Yield", "Irrigation", "Nitrogen", "WP_ET", "PFP_N", "Canonical reward"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), dpi=170)
    for ax, metric, title in zip(axes.flat, summary_metrics, titles):
        for idx, domain in enumerate(("observed_weather", "heldout_wgen")):
            for j, group in enumerate(("control_75", "treatment_105")):
                values = seed_summary[(seed_summary.group == group) & (seed_summary.evaluation_weather_type == domain)][metric].dropna()
                x = idx * 2.8 + j * .85
                ax.bar(x, values.mean(), width=.65, color=PALETTE[group], alpha=.85,
                       label=("75 mm" if j == 0 else "105 mm") if idx == 0 else None,
                       yerr=values.std(ddof=1) if len(values) > 1 else 0, capsize=3)
        ax.set_title(title); ax.set_xticks([.43, 3.23], ["Observed", "Held-out WGEN"])
        ax.grid(axis="y", color="#D9DEE3", linewidth=.7); ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Across-seed means ± between-seed SD", y=1.01)
    fig.tight_layout(); path = figdir / "figure_overall_domain_summary.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); paths.append(rel(path))
    return paths


def write_report(episodes: pd.DataFrame, seed_summary: pd.DataFrame, paired: pd.DataFrame,
                 overall: pd.DataFrame, archetypes: pd.DataFrame, paired_episode: pd.DataFrame,
                 figures: list[str], manifest: dict[str, Any]) -> None:
    observed = paired[paired.evaluation_weather_type.eq("observed_weather")]
    heldout = paired[paired.evaluation_weather_type.eq("heldout_wgen")]
    def counts(part):
        return (
            int(part.pareto_dominates_75.sum()), int(part.pareto_dominated_by_75.sum()),
            int((~part.pareto_dominates_75 & ~part.pareto_dominated_by_75).sum()),
            int(part.higher_yield_and_lower_N.sum()),
        )
    o_dom, o_bad, o_mix, o_yln = counts(observed)
    h_dom, h_bad, h_mix, h_yln = counts(heldout)
    observed_ep = paired_episode[paired_episode.evaluation_weather_type.eq("observed_weather")]
    heldout_ep = paired_episode[paired_episode.evaluation_weather_type.eq("heldout_wgen")]
    def mean_delta(part, metric):
        return float(part[f"delta_{metric}"].mean())
    def win_count(part, metric, direction):
        values = pd.to_numeric(part[f"delta_{metric}"], errors="coerce")
        return int((values > 0 if direction == "higher" else values < 0).sum())
    obs_delta = {metric: mean_delta(observed, metric) for metric in (
        "yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "wp_et_kg_m3",
        "pfp_n_kg_kg", "common_reward", "canonical_reward",
    )}
    heldout_delta = {metric: mean_delta(heldout, metric) for metric in obs_delta}
    pfp_coverage = {}
    pfp_seed_coverage = {}
    for domain in ("observed_weather", "heldout_wgen"):
        for group in ("control_75", "treatment_105"):
            part = episodes[(episodes.evaluation_weather_type == domain) & (episodes.group == group)]
            pfp_coverage[(domain, group)] = int(pd.to_numeric(part.pfp_n_kg_kg, errors="coerce").notna().sum())
            seed_part = seed_summary[(seed_summary.evaluation_weather_type == domain) & (seed_summary.group == group)]
            pfp_seed_coverage[(domain, group)] = int(pd.to_numeric(seed_part.pfp_n_kg_kg, errors="coerce").notna().sum())
    pfp_pair_coverage = {
        domain: int(pd.to_numeric(paired.loc[paired.evaluation_weather_type.eq(domain), "delta_pfp_n_kg_kg"], errors="coerce").notna().sum())
        for domain in ("observed_weather", "heldout_wgen")
    }
    n_zero_counts = {
        (domain, group): int(pd.to_numeric(episodes.loc[
            (episodes.evaluation_weather_type == domain) & (episodes.group == group), "nitrogen_kg_ha"
        ], errors="coerce").eq(0).sum())
        for domain in ("observed_weather", "heldout_wgen") for group in ("control_75", "treatment_105")
    }
    lines = [
        "# YC 004_13：105 mm early-cap 配对重训练实验", "",
        "## 实验边界", "",
        "- Treatment 为 DAP≤30 累计灌溉 cap=105 mm，seed 0–7 各从头训练 100K requested steps；Control 为 004_05 canonical 75 mm seed 0–7 原模型/正式结果，不重训。",
        "- 除 early cap 外，WGEN schedule、observation、reward、动作网格、PPO、mid cap=150 mm、late/pre-DAP90=195 mm、I/N 7-day 间隔、季节 I240/N250、daily limits、DSSAT/runtime、evaluation protocol 均冻结。未修改旧配置、wrapper 或 004_05 结果；无 commit/push。",
        "- 执行记录：attempt_01–04 的 smoke checkpoint、评估产物和失败日志均保留，均未启动正式训练。attempt_01/02 暴露 episode-key join 错误；attempt_02 隔离诊断确认 I/N action trace 闭合。attempt_03 的输入哈希 gate 发现 seeds 0–2 的 control 输入归档在 004_03；attempt_04 又确认 004_05 episode 汇总表未保留 held-out runtime weather hash，但其逐日 trace 有可重算且 reconciled 的 hash。attempt_05 改用 004_05 step-level weather hash 作权威对照，并按 seed manifest 回溯 control inputs/checkpoints。相关日志见同级 attempt_01_smoke.log 至 attempt_04_smoke.log。",
        "- 正式运行审计：8 个 treatment 模型均从头完成 100080 actual steps；评估过程中的多次后处理 gate 错误均保留在原日志和 evaluation 子目录，修复后从既有 checkpoint 继续、未重训。DSSAT MgmtEvent.OUT 会累积多个 rollout block，最终提取与 Summary.OUT yield 匹配的 block；PFP_N 在 N=0 时按定义为 N/A。seed 3–7 的 observed inputs 使用 004_05/evaluation/runs 归档；W3–W7 历史 held-out hash 缺失时，仅用 W0–W2 对相同 WGEN weather seed 的共同 reconciled hash 作参照。旧正式 CSV 的投入列使用 total_irrigation/total_fertilizer。最终 treatment metrics 位于 evaluation_final（seed0–2）和 evaluation_final_retry_03（seed3–7）；control exact replay 位于 control_replay_final。",
        f"- Smoke：{manifest.get('smoke_passed')}；训练完成 seeds={manifest.get('completed_seeds', [])}；峰值 RSS={manifest.get('peak_process_tree_rss_mb', float('nan')):.1f} MiB，硬上限 {RSS_LIMIT_MB:.0f} MiB。004_05 control evaluation replay exact episodes={manifest.get('control_exact_replay_episodes', '未统计')} / 240。",
        "- 75 mm control 仍以 004_05 正式 CSV/action trace 为权威。只在逐 episode yield/I/N/reward、动作 trace 和 runtime weather hash 均复现时，才将 evaluation-only Summary.OUT 补出的 WP_ET/PFP_N 作为可比值；不匹配项的 control WP_ET/PFP_N 保留为空。",
        "", "## 配对总体结果", "",
        "“Pareto 支配”按 seed×评估域的平均三目标定义：Yield 不低、灌溉和 N 不高且至少一项严格更优。它不是事后加权总分；canonical/common reward 作为单独指标列示。下表的 ± 为 8 个 seed 间 SD。", "",
        "| Domain | Group | Yield kg/ha | Irrigation mm | N kg/ha | WP_ET | PFP_N | Common reward | Canonical reward |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        if row.evaluation_weather_type not in ("observed_weather", "heldout_wgen"):
            continue
        domain = "Observed 2014–2023" if row.evaluation_weather_type == "observed_weather" else "Held-out WGEN 1081–1100"
        lines.append(
            f"| {domain} | {row.group} | {row.yield_mean:.1f} ± {row.yield_between_seed_sd:.1f} | {row.irrigation_mean:.1f} | {row.nitrogen_mean:.1f} | {row.wp_et_mean:.3f} | {row.pfp_n_mean:.2f} | {row.common_reward_mean:.3f} | {row.canonical_reward_mean:.3f} |"
        )
    lines += [
        "", "## 8 个 paired seeds", "",
        f"- Observed：105 mm 支配 75 mm：{o_dom}/8；被 75 mm 支配：{o_bad}/8；三目标互不支配/折中：{o_mix}/8；同时增产且降低 N：{o_yln}/8。",
        f"- Held-out WGEN：105 mm 支配 75 mm：{h_dom}/8；被 75 mm 支配：{h_bad}/8；三目标互不支配/折中：{h_mix}/8；同时增产且降低 N：{h_yln}/8。",
        "- 单 seed、逐域的 delta 和 Pareto 标记见 paired_comparison.csv；不得仅凭均值或单个最好 seed 宣称胜出。",
        f"- 逐 episode 配对：Observed episode Pareto={int(observed_ep.pareto_dominates_75.sum())}/{len(observed_ep)}、增产且降 N={int(observed_ep.higher_yield_and_lower_N.sum())}/{len(observed_ep)}；Held-out WGEN episode Pareto={int(heldout_ep.pareto_dominates_75.sum())}/{len(heldout_ep)}、增产且降 N={int(heldout_ep.higher_yield_and_lower_N.sum())}/{len(heldout_ep)}。",
        "- Observed 年份级 treatment/control 对照和共同 held-out WGEN seed 对照见 paired_episode_comparison.csv，可追溯到每个 paired PPO seed。",
        "", "## 策略 archetype", "",
    ]
    for domain in ("observed_weather", "heldout_wgen"):
        sub = archetypes[archetypes.evaluation_weather_type.eq(domain)]
        label = "Observed" if domain == "observed_weather" else "Held-out WGEN"
        for phenotype, title in (("very_low_input", "very-low-input"), ("yield_collapse", "yield-collapse"), ("fixed_N240", "fixed N=240")):
            counts_by_group = sub.groupby("group")[phenotype].sum().to_dict()
            lines.append(f"- {label} {title} seed 数：75 mm={int(counts_by_group.get('control_75', 0))}/8，105 mm={int(counts_by_group.get('treatment_105', 0))}/8。")
    lines += [
        "", "## 结论", "",
        f"- 105 mm 未显示整体改善：Observed 平均 Δ（105−75）为 Yield {obs_delta['yield_kg_ha']:+.1f} kg/ha、灌溉 {obs_delta['irrigation_mm']:+.1f} mm、N {obs_delta['nitrogen_kg_ha']:+.1f} kg/ha、WP_ET {obs_delta['wp_et_kg_m3']:+.3f}、common reward {obs_delta['common_reward']:+.3f}、canonical reward {obs_delta['canonical_reward']:+.3f}；Held-out WGEN 对应为 {heldout_delta['yield_kg_ha']:+.1f} kg/ha、{heldout_delta['irrigation_mm']:+.1f} mm、{heldout_delta['nitrogen_kg_ha']:+.1f} kg/ha、{heldout_delta['wp_et_kg_m3']:+.3f}、{heldout_delta['common_reward']:+.3f}、{heldout_delta['canonical_reward']:+.3f}。两域方向一致：省 N 但平均减产、略增灌溉、WP_ET/reward 下降。",
        f"- Paired seed 层面：yield 提高 Observed {win_count(observed, 'yield_kg_ha', 'higher')}/8、WGEN {win_count(heldout, 'yield_kg_ha', 'higher')}/8；N 降低两域均 {win_count(observed, 'nitrogen_kg_ha', 'lower')}/8；灌溉降低两域均 {win_count(observed, 'irrigation_mm', 'lower')}/8；没有 seed 同时增产且减 N。三目标下 105 mm 支配数为 Observed {o_dom}/8、WGEN {h_dom}/8；WGEN 中 75 mm 支配 105 mm {h_bad}/8。",
        f"- PFP_N paired 均值上升（Observed {obs_delta['pfp_n_kg_kg']:+.2f}、WGEN {heldout_delta['pfp_n_kg_kg']:+.2f}；每域仅 {pfp_pair_coverage['observed_weather']}/8、{pfp_pair_coverage['heldout_wgen']}/8 个两侧均有效的 paired seeds）。总表按有效 seed 的等权均值；单 seed 内先对 N>0 episode 求均值。N=0 时 PFP_N 为 N/A；有效 episode 行数由 75 mm 的 {pfp_coverage[('observed_weather','control_75')]}/80、{pfp_coverage[('heldout_wgen','control_75')]}/160 降为 105 mm 的 {pfp_coverage[('observed_weather','treatment_105')]}/80、{pfp_coverage[('heldout_wgen','treatment_105')]}/160；有效 seed 均值数为 75 mm {pfp_seed_coverage[('observed_weather','control_75')]}/8、{pfp_seed_coverage[('heldout_wgen','control_75')]}/8，105 mm {pfp_seed_coverage[('observed_weather','treatment_105')]}/8、{pfp_seed_coverage[('heldout_wgen','treatment_105')]}/8。N=0 episode 增至 Observed {n_zero_counts[('observed_weather','treatment_105')]}/80（75 mm={n_zero_counts[('observed_weather','control_75')]}/80）、WGEN {n_zero_counts[('heldout_wgen','treatment_105')]}/160（75 mm={n_zero_counts[('heldout_wgen','control_75')]}/160），故不可把 PFP_N 上升单独解读为综合效率胜出。",
        "- 结构上，Observed very-low-input 与 yield-collapse 均由 75 mm 的 3/8 增至 105 mm 的 5/8；fixed N=240 均为 2/8，没有减少。当前证据更支持 105 mm 加剧低投入/减产倾向，而非稳定改善综合策略。",
        "- 这是固定训练预算下的单次 8-seed mask treatment 比较。2014–2023 与 held-out WGEN 是 004_05 的既定 evaluation domains，不等于新增独立外部验证；实验也不自动批准将 105 mm 升级为 canonical mask。",
        "", "## 可复现证据", "",
        f"- Smoke gate: {rel(OUT / 'smoke/smoke_manifest.json')}",
        f"- 运行总 manifest: {rel(OUT / 'formal_manifest.json')}",
        f"- Episode-level 数据: {rel(OUT / 'summaries/episode_metrics_all.csv')}",
        f"- 105 mm final evaluation: {rel(TREATMENT_EVAL_ROOT)}（seed 0–2），{rel(TREATMENT_EVAL_RETRY_ROOT)}（seed 3–7）",
        f"- 75 mm exact replay: {rel(CONTROL_EVAL_ROOT)}",
        f"- Seed-domain 汇总: {rel(OUT / 'summaries/seed_domain_summary.csv')}",
        f"- Paired comparison: {rel(OUT / 'summaries/paired_comparison.csv')}",
        f"- Paired episode/year-weather comparison: {rel(OUT / 'summaries/paired_episode_comparison.csv')}",
        f"- Archetypes: {rel(OUT / 'summaries/seed_archetypes.csv')}",
        f"- Checkpoint/config/input provenance: {rel(OUT / 'provenance')}",
        "", "## 图表", "",
    ]
    lines.extend(f"- {rel(ROOT / p)}" for p in figures)
    lines += ["", "未提交、未推送。", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if REPORT.exists():
        raise FileExistsError(f"Refusing to overwrite report: {REPORT}")
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def analyze() -> None:
    episodes, seed_summary, paired, overall, archetypes, paired_episode = aggregate_metrics()
    for frame, filename in (
        (episodes, "episode_metrics_all.csv"),
        (seed_summary, "seed_domain_summary.csv"),
        (paired, "paired_comparison.csv"),
        (overall, "overall_domain_summary.csv"),
        (archetypes, "seed_archetypes.csv"),
        (paired_episode, "paired_episode_comparison.csv"),
    ):
        frame.to_csv(OUT / "summaries" / filename, index=False, encoding="utf-8-sig", lineterminator="\n")
    figures = make_figures(episodes, seed_summary, paired)
    control = episodes[episodes.group.eq("control_75")]
    exact_count = int(control.get("official_replay_exact", pd.Series(False, index=control.index)).astype(str).str.lower().isin(["true", "1"]).sum())
    seed_manifests = [verify_completed_seed(seed) for seed in SEEDS]
    rss_values = [float(x.get("max_process_tree_rss_mb") or 0) for x in seed_manifests]
    evaluation_manifests = []
    for group in ("treatment_105", "control_75"):
        for seed in SEEDS:
            root = treatment_eval_root(seed) if group == "treatment_105" else CONTROL_EVAL_ROOT
            name = root / f"seed_{seed}_evaluation_manifest.json"
            evaluation_manifests.append(json.loads(name.read_text(encoding="utf-8")))
    peak = max(rss_values + [float(m.get("peak_process_tree_rss_mb") or 0) for m in evaluation_manifests])
    manifest_path = OUT / "formal_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "status": "completed", "smoke_passed": True, "completed_seeds": SEEDS,
        "evaluation_completed": True, "evaluation_episode_rows": int(len(episodes)),
        "control_exact_replay_episodes": exact_count, "control_total_replay_episodes": int(len(control)),
        "peak_process_tree_rss_mb": peak, "training_peak_rss_by_seed_mb": rss_values,
        "treatment_evaluation_roots": {str(seed): rel(treatment_eval_root(seed)) for seed in SEEDS},
        "control_evaluation_root": rel(CONTROL_EVAL_ROOT),
        "paired_comparison": rel(OUT / "summaries/paired_comparison.csv"),
        "figures": figures, "report": rel(REPORT),
        "completed_utc": pd.Timestamp.utcnow().isoformat(),
        "canonical_files_modified": False, "git_commit": None, "git_push": None,
    })
    atomic_json(manifest_path, manifest)
    provenance = {
        "prompt_sha256": sha256_file(PROMPT),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_sources": {
            rel(FROZEN / "config/canonical_yc_ppo_config.json"): sha256_file(FROZEN / "config/canonical_yc_ppo_config.json"),
            rel(FROZEN / "config/experiment_design.json"): sha256_file(FROZEN / "config/experiment_design.json"),
            rel(FROZEN / "config/training_weather_schedule_random.csv"): sha256_file(FROZEN / "config/training_weather_schedule_random.csv"),
            rel(FROZEN / "all_model_manifest_0_7.csv"): sha256_file(FROZEN / "all_model_manifest_0_7.csv"),
            rel(FROZEN / "all_evaluation_episode_level_0_7.csv"): sha256_file(FROZEN / "all_evaluation_episode_level_0_7.csv"),
            rel(FROZEN / "evaluation_step_level_all_models.csv"): sha256_file(FROZEN / "evaluation_step_level_all_models.csv"),
        },
        "treatment_checkpoints": {str(seed): sha256_file(OUT / f"training/random_weather/ppo_seed_{seed}/models/checkpoint_100000.zip") for seed in SEEDS},
        "control_checkpoints": {str(seed): sha256_file(frozen_control_model_path(seed)) for seed in SEEDS},
        "mask_override": {"early_cap_mm": EARLY_CAP_MM, "dap_end": 30, "process_local_only": True, "mid_cap_mm": 150},
        "runtime": "serial CPU; one training/evaluation process at a time; max RSS 2048 MiB",
        "control_replay_policy": "004_05 formal rows remain authoritative; replay WP_ET/PFP_N accepted only when all official metrics, action trace, and available weather hash match",
    }
    atomic_json(OUT / "provenance/provenance.json", provenance)
    write_report(episodes, seed_summary, paired, overall, archetypes, paired_episode, figures, manifest)
    completed_utc = pd.Timestamp.utcnow().isoformat()
    provenance["runner_sha256"] = sha256_file(Path(__file__).resolve())
    provenance["report_sha256"] = sha256_file(REPORT)
    atomic_json(OUT / "provenance/provenance.json", provenance)
    manifest["runner_sha256"] = provenance["runner_sha256"]
    manifest["report_sha256"] = provenance["report_sha256"]
    manifest["updated_utc"] = completed_utc
    manifest["completed_utc"] = completed_utc
    atomic_json(manifest_path, manifest)
    print(f"[004_13 complete] report={REPORT} peak_RSS={peak:.1f}MiB control_exact={exact_count}/240", flush=True)


def run_all() -> None:
    assert_new_run()
    # The smoke child owns initialization so a failed gate cannot leave a formal model process alive.
    run_child("smoke", log_path=OUT_BASE / f"{OUT.name}_smoke.log")
    smoke_manifest = json.loads((OUT / "smoke/smoke_manifest.json").read_text(encoding="utf-8"))
    if smoke_manifest.get("smoke_passed") is not True:
        raise RuntimeError("Smoke gate failed; formal training not started")
    prior_log = OUT_BASE / "attempt_01_smoke.log"
    if prior_log.is_file():
        smoke_manifest["previous_smoke_attempts"] = [
            {
                "attempt": "attempt_01", "passed": False,
                "cause": "runner episode-key join used an incorrect delimiter; no formal training started",
                "formal_training_started": False, "log": rel(prior_log),
                "outputs_preserved_under": rel(OUT_BASE / "attempt_01"),
            },
            {
                "attempt": "attempt_02", "passed": False,
                "cause": "diagnostic confirmed actions closed (I=240 mm, N=240 kg/ha); join normalization replaced the underscore inside observed_weather",
                "formal_training_started": False,
                "log": rel(OUT_BASE / "attempt_02_smoke.log"),
                "diagnostic": rel(OUT_BASE / "attempt_02/smoke/debug_evaluation/capture_diagnostic.json"),
                "outputs_preserved_under": rel(OUT_BASE / "attempt_02"),
            },
            {
                "attempt": "attempt_03", "passed": False,
                "cause": "seeds 0-2 formal observed-weather input archives are sourced from 004_03; the smoke hash check initially looked only under 004_05",
                "formal_training_started": False,
                "log": rel(OUT_BASE / "attempt_03_smoke.log"),
                "outputs_preserved_under": rel(OUT_BASE / "attempt_03"),
            },
            {
                "attempt": "attempt_04", "passed": False,
                "cause": "004_05 episode summary has no runtime weather hash; reconciled hash is available in its step-level trace",
                "formal_training_started": False,
                "log": rel(OUT_BASE / "attempt_04_smoke.log"),
                "outputs_preserved_under": rel(OUT_BASE / "attempt_04"),
            },
        ]
        atomic_json(OUT / "smoke/smoke_manifest.json", smoke_manifest)
    run_formal()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["run", "smoke", "debug-smoke", "train-one", "eval-one", "formal", "analyze"], default="run")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--group", choices=["treatment_105", "control_75"])
    args = parser.parse_args()
    if args.phase == "run":
        run_all()
    elif args.phase == "smoke":
        run_smoke()
    elif args.phase == "debug-smoke":
        debug_capture_existing_smoke()
    elif args.phase == "train-one":
        if args.seed is None:
            parser.error("--seed is required for train-one")
        train_one(args.seed)
    elif args.phase == "eval-one":
        if args.seed is None or args.group is None:
            parser.error("--seed and --group are required for eval-one")
        evaluate_group(args.group, args.seed)
    elif args.phase == "analyze":
        analyze()
    else:
        run_formal()


if __name__ == "__main__":
    main()
