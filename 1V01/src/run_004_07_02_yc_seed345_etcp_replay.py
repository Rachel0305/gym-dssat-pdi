from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "results" / "yc_random_weather_ppo" / "004_07_02_seed345_etcp_replay"
PILOT_PATH = ROOT / "results" / "yc_random_weather_ppo" / "004_03" / "run_controlled_pilot.py"
PILOT_00405_PATH = ROOT / "results" / "yc_random_weather_ppo" / "004_05" / "run_yc_weather_augmentation_multi_seed_archetype_004_05.py"
FROZEN_ROOT = ROOT / "results" / "yc_random_weather_ppo" / "004_05"
EVAL_ROOT = FROZEN_ROOT / "evaluation" / "runs" / "observed_weather" / "random_weather"
BASELINE_CSV = ROOT / "benchmark_results" / "055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000" / "tables" / "055_03_yca_five_scenario_season_summary.csv"
SEEDS = [5, 3, 4]
YEARS = list(range(2014, 2024))
SMOKE_YEAR = 2019
RSS_LIMIT_MB = 4096.0
RUNTIME_SEED = 66004
REGIME = "RANDOM_WEATHER_WGEN"
BASELINE_SCENARIOS = {
    "dssat_auto_external_n",
    "null",
    "official_extension_expert",
    "recorded_farmer_template",
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite evidence: {rel(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8", newline="\n")


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


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime():
    wrapper = import_module(PILOT_00405_PATH, "yc_00405_replay_wrapper")
    pilot = import_module(PILOT_PATH, "yc_00403_replay_pilot")
    pilot.TASK_ROOT = TASK_ROOT
    pilot.CONFIG_ROOT = FROZEN_ROOT / "config"
    pilot.TRAIN_ROOT = FROZEN_ROOT / "training"
    pilot.EVAL_ROOT = TASK_ROOT / "evaluation"
    pilot.EVAL_RUN_ROOT = TASK_ROOT / "evaluation" / "runs"
    pilot.SMOKE_ROOT = TASK_ROOT / "smoke"
    pilot.PPO_SEEDS = [3, 4, 5]
    pilot.MAX_PROCESS_TREE_RSS_MB = RSS_LIMIT_MB
    wrapper.OUT = FROZEN_ROOT
    wrapper.BASE = ROOT / "results" / "yc_random_weather_ppo" / "004_03"
    wrapper._PILOT = pilot
    return pilot


def frozen_paths(seed: int, year: int) -> tuple[Path, Path, Path, Path]:
    model = FROZEN_ROOT / "training" / "random_weather" / f"ppo_seed_{seed}" / "models" / "checkpoint_100000.zip"
    episode_csv = EVAL_ROOT / f"ppo_seed_{seed}" / "evaluation_episode_level.csv"
    step_csv = EVAL_ROOT / f"ppo_seed_{seed}" / "evaluation_step_level.csv"
    expected_input_dir = (
        EVAL_ROOT / f"ppo_seed_{seed}" / "rendered_inputs" / "YCA" / str(year)
        / f"eval_observed_weather_RANDOM_WEATHER_WGEN_{seed}_{year}"
    )
    return model, episode_csv, step_csv, expected_input_dir


def load_expected_episode(seed: int, year: int) -> dict[str, Any]:
    model, episode_path, step_path, old_input_dir = frozen_paths(seed, year)
    for path in (model, episode_path, step_path, BASELINE_CSV):
        if not path.is_file():
            raise FileNotFoundError(path)
    episodes = pd.read_csv(episode_path, keep_default_na=False)
    match = episodes[
        episodes["evaluation_weather_year"].astype(str).eq(str(year))
        & episodes["training_regime"].astype(str).eq(REGIME)
        & pd.to_numeric(episodes["ppo_seed"], errors="coerce").eq(seed)
    ]
    if len(match) != 1 or str(match.iloc[0]["status"]) != "completed":
        raise RuntimeError(f"Expected exactly one completed frozen episode for seed={seed}, year={year}")
    expected = match.iloc[0].to_dict()
    expected["model_path"] = model
    expected["step_path"] = step_path
    expected["old_input_dir"] = old_input_dir
    for name in ("CNYC" + str(year)[-2:] + "01.WTH", f"YCA_{year}_eval_observed_weather_RANDOM_WEATHER_WGEN_{seed}_{year}.jinja2"):
        if not (old_input_dir / name).is_file():
            raise FileNotFoundError(old_input_dir / name)
    return expected


def load_summary_tools():
    for path in (ROOT, ROOT / "src", ROOT / "src" / "055_yca_lowIC_site_transfer"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    helper_path = ROOT / "src" / "build_relaxed_success_five_scenario_daily_evidence_027_05.py"
    helper = import_module(helper_path, "yc_02705_summary_replay_tools")
    return helper


def selected_summary_row(snapshot: Path, final_yield: float) -> tuple[dict[str, Any], int]:
    from calculate_five_site_wue_nue_from_summary_019_10 import num, parse_summary_out

    rows = parse_summary_out(snapshot / "Summary.OUT")
    scored = []
    for index, row in enumerate(rows):
        hwam = num(row, "HWAM")
        if hwam is not None:
            scored.append((abs(hwam - final_yield), -index, row, index))
    if not scored:
        raise RuntimeError("No Summary.OUT row with numeric HWAM")
    _, _, row, index = min(scored, key=lambda item: (item[0], item[1]))
    return row, index


def snapshot_outputs(raw: Any, target: Path) -> dict[str, Any]:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot: {rel(target)}")
    source = Path(raw._tmp_folder)
    if not source.is_dir():
        raise FileNotFoundError(f"DSSAT temp directory missing: {source}")
    names = {"SUMMARY.OUT", "PLANTGRO.OUT", "SOILWAT.OUT", "WEATHER.OUT", "MGMTEVENT.OUT"}
    copied: list[dict[str, Any]] = []
    target.mkdir(parents=True)
    for item in sorted(source.iterdir()):
        if item.is_file() and item.name.upper() in names:
            shutil.copy2(item, target / item.name)
            copied.append({"name": item.name, "sha256": sha256_file(target / item.name), "bytes": (target / item.name).stat().st_size})
    missing = names.difference({row["name"].upper() for row in copied})
    if "SUMMARY.OUT" in missing or "MGMTEVENT.OUT" in missing:
        raise FileNotFoundError(f"Required DSSAT evidence missing from {source}: {sorted(missing)}")
    if not (target / "PlantGro.OUT").is_file():
        raise FileNotFoundError(f"PlantGro.OUT missing from {source}")
    return {"source_tmp_dir": str(source), "files": copied}


def check_rendered_inputs(seed: int, year: int) -> dict[str, Any]:
    old_model, _, _, old_dir = frozen_paths(seed, year)
    del old_model
    run_tag = f"eval_observed_weather_RANDOM_WEATHER_WGEN_{seed}_{year}"
    new_dir = (
        TASK_ROOT / "evaluation" / "runtime" / f"seed_{seed}" / f"YCA_{year}"
        / "rendered_inputs" / "YCA" / str(year) / run_tag
    )
    weather_name = f"CNYC{str(year)[-2:]}01.WTH"
    template_name = f"YCA_{year}_{run_tag}.jinja2"
    result = {"rendered_dir": rel(new_dir), "weather_file": weather_name, "template_file": template_name, "checks": {}}
    for name in (weather_name, template_name):
        old_file, new_file = old_dir / name, new_dir / name
        if not old_file.is_file() or not new_file.is_file():
            raise FileNotFoundError(f"Missing frozen/replayed input file: {old_file} or {new_file}")
        old_hash, new_hash = sha256_file(old_file), sha256_file(new_file)
        result["checks"][name] = {"frozen_sha256": old_hash, "replay_sha256": new_hash, "identical": old_hash == new_hash}
        if old_hash != new_hash:
            raise RuntimeError(f"Replayed observed-weather input differs from formal input: {name}")
    return result


def reconcile_existing_smoke(helper: Any) -> None:
    failed_path = TASK_ROOT / "smoke_manifest.json"
    reconciliation_path = TASK_ROOT / "smoke_reconciliation.json"
    if not failed_path.is_file():
        raise FileNotFoundError("No original smoke manifest to reconcile")
    if reconciliation_path.exists():
        raise FileExistsError(f"Refusing to overwrite reconciliation: {rel(reconciliation_path)}")
    failed = json.loads(failed_path.read_text(encoding="utf-8"))
    if failed.get("passed") is not False or "Missing frozen/replayed input file" not in str(failed.get("error", "")):
        raise RuntimeError("Only the known post-run rendered-input lookup failure can use this audit path")

    seed, year = 5, SMOKE_YEAR
    expected = load_expected_episode(seed, year)
    snapshot = TASK_ROOT / "snapshots" / f"seed_{seed}" / "YCA" / str(year)
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Completed smoke snapshot is missing: {rel(snapshot)}")
    input_provenance = check_rendered_inputs(seed, year)
    summary_row, summary_index = selected_summary_row(snapshot, float(expected["yield"]))
    metrics = helper.summary_metrics(snapshot, float(expected["yield"]))
    events = helper.parse_management_events(snapshot / "MgmtEvent.OUT")
    irrigation = float(events.loc[events["operation"].eq("irrigation"), "amount"].sum()) if not events.empty else 0.0
    nitrogen = float(events.loc[events["operation"].eq("fertilizer"), "amount"].sum()) if not events.empty else 0.0
    from calculate_five_site_wue_nue_from_summary_019_10 import num as summary_num

    hwam = summary_num(summary_row, "HWAM")
    ircm = summary_num(summary_row, "IRCM")
    nicm = summary_num(summary_row, "NICM")
    step_rows = pd.read_csv(expected["step_path"])
    step_rows = step_rows[
        step_rows["evaluation_weather_year"].astype(str).eq(str(year))
        & pd.to_numeric(step_rows["ppo_seed"], errors="coerce").eq(seed)
    ]
    irrig_actions = pd.to_numeric(step_rows["irrigation_action_mm"], errors="coerce").fillna(0)
    nitrogen_actions = pd.to_numeric(step_rows["fertilizer_action_kgN_ha"], errors="coerce").fillna(0)
    action_days = int(((irrig_actions > 0) | (nitrogen_actions > 0)).sum())
    irrig_days = int((irrig_actions > 0).sum())
    nitrogen_days = int((nitrogen_actions > 0).sum())
    signature = ";".join(
        f"{int(row.dap)}:I{float(row.irrigation_action_mm):g}/N{float(row.fertilizer_action_kgN_ha):g}"
        for row in step_rows.itertuples()
        if float(row.irrigation_action_mm) > 0 or float(row.fertilizer_action_kgN_ha) > 0
    )
    old_hash = sha256_file(failed_path)
    logs = (TASK_ROOT / "logs" / "smoke.log").read_text(encoding="utf-8", errors="replace")
    rss_values = [float(value) for value in __import__("re").findall(r"tree_mb=([0-9.]+)", logs)]
    peak_rss = max(rss_values) if rss_values else math.nan
    gates = {
        "summary_yield_within_1kg_ha": hwam is not None and abs(float(hwam) - float(expected["yield"])) <= 1.0,
        "mgmtevent_irrigation_matches_formal_and_summary": abs(irrigation - float(expected["irrigation"])) <= 1.0 and ircm is not None and abs(float(ircm) - float(expected["irrigation"])) <= 1.0,
        "mgmtevent_nitrogen_matches_formal_and_summary": abs(nitrogen - float(expected["fertilizer"])) <= 1.0 and nicm is not None and abs(float(nicm) - float(expected["fertilizer"])) <= 1.0,
        "positive_finite_etcp": math.isfinite(float(metrics["etcp_mm"])) and float(metrics["etcp_mm"]) > 0,
        "same_step_count_as_formal": len(step_rows) == int(expected["episode_days"]),
        "same_observed_weather_and_filex_hashes": all(item["identical"] for item in input_provenance["checks"].values()),
        "rss_below_4096_mib": math.isfinite(peak_rss) and peak_rss < RSS_LIMIT_MB,
    }
    if not all(gates.values()):
        payload = {
            "status": "failed", "passed": False, "source_failed_manifest_sha256": old_hash,
            "gates": gates, "completed_utc": pd.Timestamp.utcnow().isoformat(),
        }
        write_json(reconciliation_path, payload)
        raise RuntimeError(f"Existing smoke evidence did not reconcile: {gates}")

    copied_files = []
    for item in sorted(snapshot.iterdir()):
        if item.is_file():
            copied_files.append({"name": item.name, "sha256": sha256_file(item), "bytes": item.stat().st_size})
    row = {
        "seed": seed, "year": year, "checkpoint": 100000,
        "model_path": rel(expected["model_path"]), "model_sha256": sha256_file(expected["model_path"]),
        "replay_yield_kg_ha": float(hwam), "formal_yield_kg_ha": float(expected["yield"]),
        "yield_abs_error_kg_ha": abs(float(hwam) - float(expected["yield"])),
        "irrigation_mm": irrigation, "formal_irrigation_mm": float(expected["irrigation"]),
        "irrigation_event_abs_error_mm": abs(irrigation - float(expected["irrigation"])),
        "nitrogen_kg_ha": nitrogen, "formal_nitrogen_kg_ha": float(expected["fertilizer"]),
        "nitrogen_event_abs_error_kg_ha": abs(nitrogen - float(expected["fertilizer"])),
        "summary_hwam_kg_ha": float(hwam), "summary_ircm_mm": float(ircm), "summary_nicm_kg_ha": float(nicm),
        "summary_row_index_zero_based": summary_index, "etcp_mm": float(metrics["etcp_mm"]),
        "wp_et_kg_m3": float(metrics["wp_et_kg_m3"]) if math.isfinite(float(metrics["wp_et_kg_m3"])) else np.nan,
        "pfp_n_kg_kg": float(metrics["pfp_n_kg_kg"]) if math.isfinite(float(metrics["pfp_n_kg_kg"])) else np.nan,
        "canonical_reward": float(expected["reward"]),
        "common_reward": 0.158 * float(hwam) - 1.1 * irrigation - 1.58 * nitrogen,
        "steps": int(expected["episode_days"]), "management_event_count": int(len(events)),
        "management_action_days_nonzero": action_days, "irrigation_action_days": irrig_days,
        "nitrogen_action_days": nitrogen_days, "action_signature": signature,
        "snapshot_dir": rel(snapshot), "snapshot_files": json.dumps(copied_files, ensure_ascii=False, sort_keys=True),
        "rendered_input_provenance": json.dumps(input_provenance, ensure_ascii=False, sort_keys=True),
        "peak_process_tree_rss_mb": peak_rss, "status": "passed_reconciled_existing_smoke",
        "canonical_reward_source": rel(EVAL_ROOT / f"ppo_seed_{seed}" / "evaluation_episode_level.csv"),
    }
    pd.DataFrame([row]).to_csv(TASK_ROOT / "smoke_episode_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    write_json(reconciliation_path, {
        "status": "passed_by_post_run_evidence_reconciliation", "passed": True,
        "completed_utc": pd.Timestamp.utcnow().isoformat(), "seed": seed, "year": year,
        "reason": "The replay completed; the initial smoke validator searched the wrong rendered-input directory. No DSSAT episode was rerun.",
        "source_failed_manifest_sha256": old_hash, "gates": gates, "row": row,
        "input_provenance": input_provenance, "snapshot_files": copied_files,
        "peak_process_tree_rss_mb": peak_rss,
    })
    print(f"[smoke-reconciled-pass] seed=5 year=2019 ETCP={row['etcp_mm']:.2f} WP_ET={row['wp_et_kg_m3']:.4f} RSS={peak_rss:.1f} MiB", flush=True)


def run_one(pilot: Any, helper: Any, seed: int, year: int, phase: str) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    expected = load_expected_episode(seed, year)
    model_path: Path = expected["model_path"]
    run_root = TASK_ROOT / "evaluation" / "runtime" / f"seed_{seed}" / f"YCA_{year}"
    snapshot_dir = TASK_ROOT / "snapshots" / f"seed_{seed}" / "YCA" / str(year)
    if run_root.exists() or snapshot_dir.exists():
        raise FileExistsError(f"Refusing to overwrite prior replay: {rel(run_root)} or {rel(snapshot_dir)}")
    run_root.mkdir(parents=True)
    cfg, env_cfg = pilot.make_run_config(REGIME, seed, run_root)
    schedule = [{"episode_index": 1, "historical_year": year, "weather_seed": ""}]
    run_tag = f"eval_observed_weather_RANDOM_WEATHER_WGEN_{seed}"
    env = pilot.ScheduledEpisodeEnv(
        cfg, env_cfg, schedule, "HISTORICAL_WEATHER", seed, run_tag, pilot.EVAL_RUNTIME_SEED, max_cached_years=1
    )
    model = MaskablePPO.load(str(model_path), device="cpu")
    raw = None
    try:
        observation, _ = env._gym_env.reset(seed=None)
        total_reward = 0.0
        steps = 0
        while True:
            action, _ = model.predict(observation, deterministic=True, action_masks=env._gym_env.action_masks())
            observation, reward, terminated, truncated, _ = env._gym_env.step(action)
            total_reward += float(reward)
            steps += 1
            if steps % 10 == 0:
                rss = process_tree_rss_mb()
                print(f"[rss] phase={phase} seed={seed} year={year} step={steps} tree_mb={rss:.1f}", flush=True)
                if math.isfinite(rss) and rss >= RSS_LIMIT_MB:
                    raise MemoryError(f"Process-tree RSS crossed {RSS_LIMIT_MB:.0f} MiB: {rss:.1f} MiB")
            if terminated or truncated:
                break
        episode = dict(env.episode_rows[-1])
        raw = pilot.find_dssat_instance(env.current_env)
        temp_dir = Path(raw._tmp_folder)
        partial_snapshot = snapshot_dir
        copied_files = snapshot_outputs(raw, partial_snapshot)
        summary_row, summary_index = selected_summary_row(partial_snapshot, float(episode["yield"]))
        metrics = helper.summary_metrics(partial_snapshot, float(episode["yield"]))
        events = helper.parse_management_events(partial_snapshot / "MgmtEvent.OUT")
        irrigation_events = float(events.loc[events["operation"].eq("irrigation"), "amount"].sum()) if not events.empty else 0.0
        nitrogen_events = float(events.loc[events["operation"].eq("fertilizer"), "amount"].sum()) if not events.empty else 0.0
        summary_num = __import__("calculate_five_site_wue_nue_from_summary_019_10", fromlist=["num"]).num
        summary_yield = summary_num(summary_row, "HWAM")
        summary_irrigation = summary_num(summary_row, "IRCM")
        summary_nitrogen = summary_num(summary_row, "NICM")
        if summary_yield is None or abs(float(summary_yield) - float(expected["yield"])) > 1.0:
            raise RuntimeError(f"Summary.OUT yield mismatch formal endpoint: {summary_yield} vs {expected['yield']}")
        if abs(float(episode["yield"]) - float(expected["yield"])) > 1.0:
            raise RuntimeError(f"Replay yield mismatch formal endpoint: {episode['yield']} vs {expected['yield']}")
        for label, actual, formal, summary in (
            ("irrigation", irrigation_events, float(expected["irrigation"]), summary_irrigation),
            ("nitrogen", nitrogen_events, float(expected["fertilizer"]), summary_nitrogen),
        ):
            if abs(actual - formal) > 1.0 or summary is None or abs(float(summary) - formal) > 1.0:
                raise RuntimeError(f"{label} closure failed: events={actual}, formal={formal}, Summary.OUT={summary}")
        if not math.isfinite(float(metrics["etcp_mm"])) or float(metrics["etcp_mm"]) <= 0:
            raise RuntimeError(f"Summary.OUT ETCP is not finite and positive: {metrics['etcp_mm']}")
        if steps != int(expected["episode_days"]):
            raise RuntimeError(f"Step-count mismatch: replay={steps} formal={expected['episode_days']}")
        input_provenance = check_rendered_inputs(seed, year)
        step_rows = pd.read_csv(expected["step_path"])
        step_rows = step_rows[
            step_rows["evaluation_weather_year"].astype(str).eq(str(year))
            & pd.to_numeric(step_rows["ppo_seed"], errors="coerce").eq(seed)
        ]
        irrig_days = int((pd.to_numeric(step_rows["irrigation_action_mm"], errors="coerce").fillna(0) > 0).sum())
        nitrogen_days = int((pd.to_numeric(step_rows["fertilizer_action_kgN_ha"], errors="coerce").fillna(0) > 0).sum())
        action_days = int(((pd.to_numeric(step_rows["irrigation_action_mm"], errors="coerce").fillna(0) > 0) | (pd.to_numeric(step_rows["fertilizer_action_kgN_ha"], errors="coerce").fillna(0) > 0)).sum())
        signature = ";".join(
            f"{int(row.dap)}:I{float(row.irrigation_action_mm):g}/N{float(row.fertilizer_action_kgN_ha):g}"
            for row in step_rows.itertuples()
            if float(row.irrigation_action_mm) > 0 or float(row.fertilizer_action_kgN_ha) > 0
        )
        row = {
            "seed": seed,
            "year": year,
            "checkpoint": 100000,
            "model_path": rel(model_path),
            "model_sha256": sha256_file(model_path),
            "replay_yield_kg_ha": float(episode["yield"]),
            "formal_yield_kg_ha": float(expected["yield"]),
            "yield_abs_error_kg_ha": abs(float(episode["yield"]) - float(expected["yield"])),
            "irrigation_mm": irrigation_events,
            "formal_irrigation_mm": float(expected["irrigation"]),
            "irrigation_event_abs_error_mm": abs(irrigation_events - float(expected["irrigation"])),
            "nitrogen_kg_ha": nitrogen_events,
            "formal_nitrogen_kg_ha": float(expected["fertilizer"]),
            "nitrogen_event_abs_error_kg_ha": abs(nitrogen_events - float(expected["fertilizer"])),
            "summary_hwam_kg_ha": float(summary_yield),
            "summary_ircm_mm": float(summary_irrigation),
            "summary_nicm_kg_ha": float(summary_nitrogen),
            "summary_row_index_zero_based": summary_index,
            "etcp_mm": float(metrics["etcp_mm"]),
            "wp_et_kg_m3": float(metrics["wp_et_kg_m3"]) if math.isfinite(float(metrics["wp_et_kg_m3"])) else np.nan,
            "pfp_n_kg_kg": float(metrics["pfp_n_kg_kg"]) if math.isfinite(float(metrics["pfp_n_kg_kg"])) else np.nan,
            "canonical_reward": float(total_reward),
            "common_reward": 0.158 * float(episode["yield"]) - 1.1 * irrigation_events - 1.58 * nitrogen_events,
            "steps": steps,
            "management_event_count": int(len(events)),
            "management_action_days_nonzero": action_days,
            "irrigation_action_days": irrig_days,
            "nitrogen_action_days": nitrogen_days,
            "action_signature": signature,
            "snapshot_dir": rel(snapshot_dir),
            "snapshot_files": json.dumps(copied_files["files"], ensure_ascii=False, sort_keys=True),
            "rendered_input_provenance": json.dumps(input_provenance, ensure_ascii=False, sort_keys=True),
            "peak_process_tree_rss_mb": process_tree_rss_mb(),
            "status": "passed",
        }
        print(
            f"[episode-pass] seed={seed} year={year} yield={row['replay_yield_kg_ha']:.2f} "
            f"I={row['irrigation_mm']:.1f} N={row['nitrogen_kg_ha']:.1f} ETCP={row['etcp_mm']:.2f} "
            f"WP_ET={row['wp_et_kg_m3']:.4f} RSS={row['peak_process_tree_rss_mb']:.1f} MiB",
            flush=True,
        )
        return row
    except Exception:
        if raw is not None and not snapshot_dir.exists():
            try:
                snapshot_outputs(raw, snapshot_dir)
            except Exception as copy_error:
                print(f"[partial-snapshot-warning] {copy_error}", flush=True)
        raise
    finally:
        try:
            env.close()
        finally:
            del model
            gc.collect()


def baseline_table() -> pd.DataFrame:
    data = pd.read_csv(BASELINE_CSV, keep_default_na=False)
    data = data[data["scenario"].astype(str).isin(BASELINE_SCENARIOS)].copy()
    for column in (
        "year", "grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha",
        "etcp_mm", "WP_ET_kg_m3", "PFP_N_kg_kg", "unified_reward",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data.duplicated(["site", "year", "scenario"]).any():
        raise RuntimeError("Duplicate four-baseline rows in 055_03 season summary")
    for year in YEARS:
        actual = set(data.loc[pd.to_numeric(data["year"], errors="coerce").eq(year), "scenario"].astype(str))
        if actual != BASELINE_SCENARIOS:
            raise RuntimeError(f"Incomplete 055_03 baseline coverage in {year}: {actual}")
    return data


def make_comparison(rows: pd.DataFrame) -> pd.DataFrame:
    baseline = baseline_table()
    comparisons = []
    for _, candidate in rows.iterrows():
        same_year = baseline[pd.to_numeric(baseline["year"], errors="coerce").eq(int(candidate["year"]))]
        for _, reference in same_year.iterrows():
            comparisons.append({
                "seed": int(candidate["seed"]),
                "year": int(candidate["year"]),
                "baseline_scenario": str(reference["scenario"]),
                "ppo_yield_kg_ha": candidate["replay_yield_kg_ha"],
                "baseline_yield_kg_ha": reference["grain_yield_kg_ha"],
                "yield_delta_kg_ha": candidate["replay_yield_kg_ha"] - reference["grain_yield_kg_ha"],
                "ppo_irrigation_mm": candidate["irrigation_mm"],
                "baseline_irrigation_mm": reference["actual_irrigation_mm"],
                "irrigation_delta_mm": candidate["irrigation_mm"] - reference["actual_irrigation_mm"],
                "ppo_nitrogen_kg_ha": candidate["nitrogen_kg_ha"],
                "baseline_nitrogen_kg_ha": reference["actual_nitrogen_kg_ha"],
                "nitrogen_delta_kg_ha": candidate["nitrogen_kg_ha"] - reference["actual_nitrogen_kg_ha"],
                "ppo_etcp_mm": candidate["etcp_mm"],
                "baseline_etcp_mm": reference["etcp_mm"],
                "ppo_wp_et_kg_m3": candidate["wp_et_kg_m3"],
                "baseline_wp_et_kg_m3": reference["WP_ET_kg_m3"],
                "ppo_pfp_n_kg_kg": candidate["pfp_n_kg_kg"],
                "baseline_pfp_n_kg_kg": reference["PFP_N_kg_kg"],
                "ppo_common_reward": candidate["common_reward"],
                "baseline_common_reward": reference["unified_reward"],
                "common_reward_delta": candidate["common_reward"] - reference["unified_reward"],
                "ppo_canonical_reward_not_comparable_to_common_reward": candidate["canonical_reward"],
                "management_event_count": candidate["management_event_count"],
                "management_action_days_nonzero": candidate["management_action_days_nonzero"],
                "irrigation_action_days": candidate["irrigation_action_days"],
                "nitrogen_action_days": candidate["nitrogen_action_days"],
                "action_signature": candidate["action_signature"],
                "snapshot_dir": candidate["snapshot_dir"],
            })
    result = pd.DataFrame(comparisons)
    if len(result) != 120 or result.duplicated(["seed", "year", "baseline_scenario"]).any():
        raise RuntimeError(f"Unexpected pairwise comparison shape: {result.shape}")
    return result


def pareto_and_profile(rows: pd.DataFrame, comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = baseline_table()
    pareto_rows = []
    for _, candidate in rows.iterrows():
        candidates = [{
            "label": f"random_weather_seed_{int(candidate['seed'])}",
            "yield": float(candidate["replay_yield_kg_ha"]),
            "irrigation": float(candidate["irrigation_mm"]),
            "nitrogen": float(candidate["nitrogen_kg_ha"]),
        }]
        ref = baseline[pd.to_numeric(baseline["year"], errors="coerce").eq(int(candidate["year"]))]
        for _, item in ref.iterrows():
            candidates.append({
                "label": str(item["scenario"]), "yield": float(item["grain_yield_kg_ha"]),
                "irrigation": float(item["actual_irrigation_mm"]), "nitrogen": float(item["actual_nitrogen_kg_ha"]),
            })
        for item in candidates:
            dominated_by = []
            for other in candidates:
                if other is item:
                    continue
                weak = other["yield"] >= item["yield"] and other["irrigation"] <= item["irrigation"] and other["nitrogen"] <= item["nitrogen"]
                strict = other["yield"] > item["yield"] or other["irrigation"] < item["irrigation"] or other["nitrogen"] < item["nitrogen"]
                if weak and strict:
                    dominated_by.append(other["label"])
            if item["label"].startswith("random_weather_"):
                pareto_rows.append({"seed": int(candidate["seed"]), "year": int(candidate["year"]), "nondominated_among_five": not dominated_by, "dominated_by": ";".join(dominated_by)})
    flags = pd.DataFrame(pareto_rows)
    paired = comparison.merge(flags, on=["seed", "year"], how="left", validate="many_to_one")
    profiles = []
    for seed, group in rows.groupby("seed", sort=True):
        comp = comparison[comparison["seed"].eq(seed)]
        profiles.append({
            "seed": int(seed), "years": int(group["year"].nunique()),
            "mean_yield_kg_ha": group["replay_yield_kg_ha"].mean(),
            "mean_irrigation_mm": group["irrigation_mm"].mean(),
            "mean_nitrogen_kg_ha": group["nitrogen_kg_ha"].mean(),
            "mean_etcp_mm": group["etcp_mm"].mean(),
            "mean_wp_et_kg_m3": group["wp_et_kg_m3"].mean(),
            "mean_pfp_n_kg_kg": group["pfp_n_kg_kg"].mean(),
            "mean_common_reward": group["common_reward"].mean(),
            "mean_canonical_reward": group["canonical_reward"].mean(),
            "mean_management_events": group["management_event_count"].mean(),
            "mean_action_days_nonzero": group["management_action_days_nonzero"].mean(),
            "yield_best_vs_four_baselines_years": int((group.set_index("year")["replay_yield_kg_ha"] >= baseline.groupby("year")["grain_yield_kg_ha"].max()).sum()),
            "nondominated_among_five_years": int(flags.loc[flags["seed"].eq(seed), "nondominated_among_five"].sum()),
            "common_reward_best_vs_four_baselines_pairs": int((comp["common_reward_delta"] >= 0).sum()),
            "wp_et_best_vs_four_baselines_pairs": int((comp["ppo_wp_et_kg_m3"] >= comp["baseline_wp_et_kg_m3"]).sum()),
            "pfp_n_best_vs_four_baselines_pairs": int((comp["ppo_pfp_n_kg_kg"] >= comp["baseline_pfp_n_kg_kg"]).sum()),
        })
    return paired, pd.DataFrame(profiles)


def write_report(rows: pd.DataFrame, comparison: pd.DataFrame, profile: pd.DataFrame, pareto: pd.DataFrame, provenance: dict[str, Any]) -> Path:
    seed5 = rows[rows["seed"].eq(5)].sort_values("year")
    seed3 = profile[profile["seed"].eq(3)].iloc[0]
    seed4 = profile[profile["seed"].eq(4)].iloc[0]
    seed5_profile = profile[profile["seed"].eq(5)].iloc[0]
    base_by_year = baseline_table().groupby("year")["grain_yield_kg_ha"].max()
    best_yield_years = [int(year) for year, value in seed5.set_index("year")["replay_yield_kg_ha"].items() if value >= base_by_year.loc[year]]
    worse_than_yield_max = [int(year) for year, value in seed5.set_index("year")["replay_yield_kg_ha"].items() if value < base_by_year.loc[year]]
    report_path = ROOT / "docs" / "yc_random_weather_004_07_02_seed345_wp_et_replay.md"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {rel(report_path)}")
    lines = [
        "# YC Random-weather PPO seed 3/4/5 ETCP 回放与对照",
        "",
        "## 设计与证据口径",
        "",
        "- 固定 004_05 的 100K checkpoint，逐模型应用于 2014–2023 observed-weather 验证年；没有逐年换 seed、重训或修改 PPO/reward/action/DSSAT。",
        "- 先以 seed 5 / 2019 作为完整 season smoke。它通过 yield（误差不超过 1 kg/ha）、管理事件与 episode / Summary.OUT 闭合、有效 ETCP、步数和 observed 输入 SHA-256 一致性门槛后，才回放其余 seed-year。",
        "- ETCP、WP_ET、PFP_N 来自匹配最终 yield 的 Summary.OUT 行；不由日尺度表或消耗量反推 ETCP。WP_ET 使用项目既有 summary_metrics 口径。",
        "- 四基准数据取 055_03 season summary；共同 reward 采用正式 baseline unified_reward。PPO common reward 单列计算为 `0.158 × yield − 1.1 × irrigation − 1.58 × N`；PPO canonical reward 单独保留，不和 common reward 混比。",
        "- 全部 2014–2023 已经参与 seed 筛选/比较，是探索性证据，不是独立确认集。seed 5/3/4 的相互筛选仍有选择偏差。",
        f"- 所有 episode 使用 CPU 串行回放；进程树 RSS 上限 {RSS_LIMIT_MB:.0f} MiB。峰值 {rows['peak_process_tree_rss_mb'].max():.1f} MiB。",
        "- 首次 smoke 的 DSSAT episode 已完成，但校验器最初查错了新渲染目录；保留原 failed manifest 后，以现存快照、输入哈希、Summary.OUT 和 MgmtEvent.OUT 作事后核验通过，没有重跑 smoke。正式 30 行表全部写出后，原汇总脚本又在派生 Pareto 表处遇到变量名错误；本报告由独立 finalize 阶段从已审计表生成，两个失败 manifest 与日志均保留。",
        "",
        "## 三个 seed 的十年摘要",
        "",
        "| Seed | 产量 kg/ha | 灌溉 mm | N kg/ha | ETCP mm | WP_ET kg/m³ | PFP_N | common reward | canonical reward | 管理事件/季 | 非零动作日/季 | 五情景非支配年数 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, item in profile.iterrows():
        lines.append(
            f"| {int(item.seed)} | {item.mean_yield_kg_ha:.1f} | {item.mean_irrigation_mm:.1f} | {item.mean_nitrogen_kg_ha:.1f} | "
            f"{item.mean_etcp_mm:.1f} | {item.mean_wp_et_kg_m3:.3f} | {item.mean_pfp_n_kg_kg:.2f} | {item.mean_common_reward:.2f} | "
            f"{item.mean_canonical_reward:.4f} | {item.mean_management_events:.1f} | {item.mean_action_days_nonzero:.1f} | {int(item.nondominated_among_five_years)}/10 |"
        )
    lines.extend([
        "",
        "## Seed 5 逐年结果",
        "",
        "| 年份 | Yield | 灌溉 | N | ETCP | WP_ET | PFP_N | common reward | 五情景非支配 | 对四基准最高产量差值 | 管理事件 | 非零动作日 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|",
    ])
    pflags = pareto[pareto["seed"].eq(5)].drop_duplicates(["seed", "year"]).set_index("year")
    for _, item in seed5.iterrows():
        year = int(item.year)
        lines.append(
            f"| {year} | {item.replay_yield_kg_ha:.1f} | {item.irrigation_mm:.1f} | {item.nitrogen_kg_ha:.1f} | {item.etcp_mm:.1f} | "
            f"{item.wp_et_kg_m3:.3f} | {item.pfp_n_kg_kg:.2f} | {item.common_reward:.2f} | "
            f"{'是' if bool(pflags.loc[year, 'nondominated_among_five']) else '否'} | {item.replay_yield_kg_ha - base_by_year.loc[year]:+.1f} | "
            f"{int(item.management_event_count)} | {int(item.management_action_days_nonzero)} |"
        )
    lines.extend([
        "",
        "## 解释",
        "",
        f"- seed 5 达到四基准年度最高产量的年份：{', '.join(map(str, best_yield_years)) if best_yield_years else '无'}；低于年度最高基准的年份：{', '.join(map(str, worse_than_yield_max)) if worse_than_yield_max else '无'}。",
        f"- seed 5 与 seed 4 的均产差为 {seed5_profile.mean_yield_kg_ha - seed4.mean_yield_kg_ha:+.1f} kg/ha、均灌溉差为 {seed5_profile.mean_irrigation_mm - seed4.mean_irrigation_mm:+.1f} mm、均 N 差为 {seed5_profile.mean_nitrogen_kg_ha - seed4.mean_nitrogen_kg_ha:+.1f} kg/ha。",
        f"- seed 5 与 seed 3 的均产差为 {seed5_profile.mean_yield_kg_ha - seed3.mean_yield_kg_ha:+.1f} kg/ha、均灌溉差为 {seed5_profile.mean_irrigation_mm - seed3.mean_irrigation_mm:+.1f} mm、均 N 差为 {seed5_profile.mean_nitrogen_kg_ha - seed3.mean_nitrogen_kg_ha:+.1f} kg/ha。",
        f"- seed 5 在 {int(seed5_profile.nondominated_among_five_years)}/10 年位于 yield↑、灌溉↓、N↓ 的五情景部分 Pareto 前沿；其 common reward 在与四基准的 40 个 seed-year 配对中胜过或持平 {int(seed5_profile.common_reward_best_vs_four_baselines_pairs)}/40 个配对。",
        f"- seed 5 的 WP_ET 在 {int(seed5_profile.wp_et_best_vs_four_baselines_pairs)}/40 个 PPO-基准配对中不低于对应基准，PFP_N 在 {int(seed5_profile.pfp_n_best_vs_four_baselines_pairs)}/40 个配对中不低于对应基准。效率指标仍需结合产量与资源投入解释。",
        "- canonical reward 属 PPO 自身训练奖励，不等同于 common reward；跨 seed 的 canonical mean 只作描述，不能据此覆盖农业指标权衡。",
        "- 动作画像保存在逐年 CSV 的 action_signature、management_event_count、management_action_days_nonzero、irrigation_action_days、nitrogen_action_days 字段；事件以 DSSAT MgmtEvent.OUT 为准，动作天数沿用正式 step-level evaluation。",
        "",
        "## 数据与可复核文件",
        "",
        f"- 回放汇总：`{rel(TASK_ROOT / 'replay_episode_summary.csv')}`（30 行，seed 5/3/4 × 2014–2023）。",
        f"- 与四基准逐年配对：`{rel(TASK_ROOT / 'seed345_validation_wp_et_comparison.csv')}`（120 行）。",
        f"- 快照目录：`{rel(TASK_ROOT / 'snapshots')}`；每个 seed-year 包含 Summary.OUT、PlantGro.OUT、SoilWat.OUT、Weather.OUT、MgmtEvent.OUT。",
        f"- 逐文件 SHA-256 与输入一致性：`{rel(TASK_ROOT / 'provenance.json')}`。",
        f"- 运行脚本：`{rel(Path(__file__))}`；提示文件：`prompt_02/004_07_02_yc_seed5_focus_wp_et_seed3_seed4_reference_replay.md`。",
        "",
        "## 可重复性摘要",
        "",
        f"- 结果行数：{len(rows)}；唯一 seed-year：{rows[['seed', 'year']].drop_duplicates().shape[0]}；快照数：{rows['snapshot_dir'].nunique()}。",
        f"- 最终 provenance：`{json.dumps(provenance, ensure_ascii=False, sort_keys=True)}`",
        "- 未提交、未推送。",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return report_path


def run_smoke(pilot: Any, helper: Any) -> None:
    manifest_path = TASK_ROOT / "smoke_manifest.json"
    if manifest_path.exists() or (TASK_ROOT / "snapshots").exists():
        raise FileExistsError(f"Smoke refuses to reuse existing evidence: {rel(TASK_ROOT)}")
    payload: dict[str, Any] = {
        "status": "running", "passed": False, "seed": 5, "year": SMOKE_YEAR,
        "rss_limit_mb": RSS_LIMIT_MB, "started_utc": pd.Timestamp.utcnow().isoformat(),
    }
    write_json(manifest_path, payload)
    try:
        row = run_one(pilot, helper, 5, SMOKE_YEAR, "smoke")
        pd.DataFrame([row]).to_csv(TASK_ROOT / "smoke_episode_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        payload.update({
            "status": "passed", "passed": True, "completed_utc": pd.Timestamp.utcnow().isoformat(),
            "row": row, "snapshot_count": 1,
            "gate": {
                "yield_within_1kg_ha": row["yield_abs_error_kg_ha"] <= 1.0,
                "irrigation_event_summary_closed": row["irrigation_event_abs_error_mm"] <= 1.0,
                "nitrogen_event_summary_closed": row["nitrogen_event_abs_error_kg_ha"] <= 1.0,
                "positive_etcp": row["etcp_mm"] > 0,
                "exact_observed_weather_and_template_hash": all(item["identical"] for item in json.loads(row["rendered_input_provenance"])["checks"].values()),
            },
        })
        if not all(payload["gate"].values()):
            payload.update({"status": "failed", "passed": False})
            raise RuntimeError(f"Smoke gate did not pass: {payload['gate']}")
        write_json(manifest_path, payload, replace=True)
        print(f"[smoke-pass] seed=5 year={SMOKE_YEAR} ETCP={row['etcp_mm']:.2f} WP_ET={row['wp_et_kg_m3']:.4f}", flush=True)
    except Exception as exc:
        payload.update({"status": "failed", "passed": False, "completed_utc": pd.Timestamp.utcnow().isoformat(), "error": repr(exc), "traceback": traceback.format_exc()})
        write_json(manifest_path, payload, replace=True)
        raise


def run_formal(pilot: Any, helper: Any) -> None:
    smoke_manifest_path = TASK_ROOT / "smoke_manifest.json"
    if not smoke_manifest_path.is_file():
        raise FileNotFoundError("Formal replay requires a completed smoke manifest")
    smoke = json.loads(smoke_manifest_path.read_text(encoding="utf-8"))
    reconciliation_path = TASK_ROOT / "smoke_reconciliation.json"
    if smoke.get("passed") is True:
        smoke_row = smoke["row"]
        smoke_evidence = "smoke_manifest.json"
    elif reconciliation_path.is_file():
        reconciled = json.loads(reconciliation_path.read_text(encoding="utf-8"))
        if reconciled.get("passed") is not True:
            raise RuntimeError("Formal replay blocked: seed 5 / 2019 evidence reconciliation failed")
        smoke_row = reconciled["row"]
        smoke_evidence = "smoke_reconciliation.json"
    else:
        raise RuntimeError("Formal replay blocked: seed 5 / 2019 smoke did not pass and has not been reconciled")
    formal_manifest_path = TASK_ROOT / "formal_manifest.json"
    if formal_manifest_path.exists():
        raise FileExistsError(f"Formal replay refuses to overwrite prior attempt: {rel(formal_manifest_path)}")
    payload: dict[str, Any] = {
        "status": "running", "passed": False, "seeds": SEEDS, "years": YEARS,
        "rss_limit_mb": RSS_LIMIT_MB, "started_utc": pd.Timestamp.utcnow().isoformat(),
        "reused_smoke_seed_year": [5, SMOKE_YEAR], "smoke_evidence": smoke_evidence,
        "new_episode_count_expected": 29,
    }
    write_json(formal_manifest_path, payload)
    records: list[dict[str, Any]] = [smoke_row]
    try:
        for seed in SEEDS:
            for year in YEARS:
                if seed == 5 and year == SMOKE_YEAR:
                    continue
                records.append(run_one(pilot, helper, seed, year, "formal"))
        frame = pd.DataFrame(records).sort_values(["seed", "year"]).reset_index(drop=True)
        if len(frame) != 30 or frame.duplicated(["seed", "year"]).any() or set(frame["seed"].astype(int)) != set(SEEDS) or set(frame["year"].astype(int)) != set(YEARS):
            raise RuntimeError("Final replay episode coverage is incomplete or duplicated")
        if (pd.to_numeric(frame["peak_process_tree_rss_mb"], errors="coerce") >= RSS_LIMIT_MB).any():
            raise MemoryError("Recorded process-tree RSS crossed the hard stop threshold")
        comparison = make_comparison(frame)
        comparison, profile = pareto_and_profile(frame, comparison)
        summary_path = TASK_ROOT / "replay_episode_summary.csv"
        comparison_path = TASK_ROOT / "seed345_validation_wp_et_comparison.csv"
        if summary_path.exists() or comparison_path.exists():
            raise FileExistsError("Refusing to overwrite a pre-existing final results table")
        frame.to_csv(summary_path, index=False, encoding="utf-8-sig", lineterminator="\n")
        comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig", lineterminator="\n")
        profile.to_csv(TASK_ROOT / "seed_profile_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        pareto = comparison.drop_duplicates(["seed", "year"])[["seed", "year", "nondominated_among_five", "dominated_by"]].copy()
        pareto.to_csv(TASK_ROOT / "seed_year_pareto_status.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        provenance = {
            "prompt_sha256": sha256_file(ROOT / "prompt_02" / "004_07_02_yc_seed5_focus_wp_et_seed3_seed4_reference_replay.md"),
            "script_sha256": sha256_file(Path(__file__)),
            "baseline_csv_sha256": sha256_file(BASELINE_CSV),
            "model_sha256": {str(seed): sha256_file(FROZEN_ROOT / "training" / "random_weather" / f"ppo_seed_{seed}" / "models" / "checkpoint_100000.zip") for seed in SEEDS},
            "frozen_episode_csv_sha256": {str(seed): sha256_file(EVAL_ROOT / f"ppo_seed_{seed}" / "evaluation_episode_level.csv") for seed in SEEDS},
            "frozen_step_csv_sha256": {str(seed): sha256_file(EVAL_ROOT / f"ppo_seed_{seed}" / "evaluation_step_level.csv") for seed in SEEDS},
            "result_row_count": len(frame), "pairwise_row_count": len(comparison),
            "snapshot_file_count": int(sum(len(json.loads(value)) for value in frame["snapshot_files"])),
            "rss_limit_mb": RSS_LIMIT_MB, "peak_rss_mb": float(frame["peak_process_tree_rss_mb"].max()),
            "baseline_scenarios": sorted(BASELINE_SCENARIOS),
            "source_summary_metric_function": "src/build_relaxed_success_five_scenario_daily_evidence_027_05.py::summary_metrics",
        }
        write_json(TASK_ROOT / "provenance.json", provenance)
        report_path = write_report(frame, comparison, profile, pareto, provenance)
        payload.update({
            "status": "passed", "passed": True, "completed_utc": pd.Timestamp.utcnow().isoformat(),
            "episode_count": len(frame), "pairwise_comparison_count": len(comparison),
            "peak_rss_mb": provenance["peak_rss_mb"], "report_path": rel(report_path),
            "summary_path": rel(summary_path), "comparison_path": rel(comparison_path),
        })
        write_json(formal_manifest_path, payload, replace=True)
        print(f"[formal-pass] episodes={len(frame)} comparisons={len(comparison)} peak_rss={provenance['peak_rss_mb']:.1f} MiB", flush=True)
    except Exception as exc:
        payload.update({"status": "failed", "passed": False, "completed_utc": pd.Timestamp.utcnow().isoformat(), "episodes_completed": len(records), "error": repr(exc), "traceback": traceback.format_exc()})
        write_json(formal_manifest_path, payload, replace=True)
        raise


def finalize_existing_results() -> None:
    formal_manifest_path = TASK_ROOT / "formal_manifest.json"
    reconciliation_path = TASK_ROOT / "smoke_reconciliation.json"
    summary_path = TASK_ROOT / "replay_episode_summary.csv"
    comparison_path = TASK_ROOT / "seed345_validation_wp_et_comparison.csv"
    profile_path = TASK_ROOT / "seed_profile_summary.csv"
    pareto_path = TASK_ROOT / "seed_year_pareto_status.csv"
    provenance_path = TASK_ROOT / "provenance.json"
    report_path = ROOT / "docs" / "yc_random_weather_004_07_02_seed345_wp_et_replay.md"
    finalization_path = TASK_ROOT / "finalization_manifest.json"
    for path in (formal_manifest_path, reconciliation_path, summary_path, comparison_path, profile_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (pareto_path, provenance_path, report_path, finalization_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite finalized output: {rel(path)}")
    formal = json.loads(formal_manifest_path.read_text(encoding="utf-8"))
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    if formal.get("episodes_completed") != 30 or formal.get("error") != "NameError(\"name 'pareto' is not defined\")":
        raise RuntimeError("Only the verified post-processing NameError can be finalized without replay")
    if reconciliation.get("passed") is not True:
        raise RuntimeError("Smoke reconciliation did not pass")

    rows = pd.read_csv(summary_path)
    comparison = pd.read_csv(comparison_path)
    comparison["baseline_scenario"] = comparison["baseline_scenario"].fillna("null").astype(str)
    profile = pd.read_csv(profile_path)
    if len(rows) != 30 or rows.duplicated(["seed", "year"]).any():
        raise RuntimeError("Replay summary is not 30 unique seed-year rows")
    if set(rows["seed"].astype(int)) != set(SEEDS) or set(rows["year"].astype(int)) != set(YEARS):
        raise RuntimeError("Replay summary seed/year coverage differs from the registered design")
    if len(comparison) != 120 or comparison.duplicated(["seed", "year", "baseline_scenario"]).any():
        raise RuntimeError("Pairwise comparison does not contain 120 unique rows")
    if set(comparison["baseline_scenario"].astype(str)) != BASELINE_SCENARIOS:
        raise RuntimeError("Pairwise comparison does not cover the four frozen baselines")
    if len(profile) != 3:
        raise RuntimeError("Seed profile table must contain the three registered seeds")
    numeric_gates = (
        (pd.to_numeric(rows["yield_abs_error_kg_ha"], errors="coerce") <= 1.0).all(),
        (pd.to_numeric(rows["irrigation_event_abs_error_mm"], errors="coerce") <= 1.0).all(),
        (pd.to_numeric(rows["nitrogen_event_abs_error_kg_ha"], errors="coerce") <= 1.0).all(),
        (pd.to_numeric(rows["etcp_mm"], errors="coerce") > 0).all(),
        pd.to_numeric(rows["wp_et_kg_m3"], errors="coerce").notna().all(),
        (pd.to_numeric(rows["peak_process_tree_rss_mb"], errors="coerce") < RSS_LIMIT_MB).all(),
    )
    if not all(bool(value) for value in numeric_gates):
        raise RuntimeError(f"Recorded replay rows failed numerical finalization gates: {numeric_gates}")
    for _, item in rows.iterrows():
        snapshot = ROOT / str(item["snapshot_dir"])
        files = json.loads(item["snapshot_files"])
        if not snapshot.is_dir() or len(files) < 5:
            raise RuntimeError(f"Incomplete raw snapshot: {snapshot}")
        for record in files:
            path = snapshot / record["name"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"Snapshot hash mismatch: {path}")
        inputs = json.loads(item["rendered_input_provenance"])
        if not all(value.get("identical") for value in inputs["checks"].values()):
            raise RuntimeError(f"Observed input hash mismatch for {item['seed']}/{item['year']}")

    pareto = comparison.drop_duplicates(["seed", "year"])[["seed", "year", "nondominated_among_five", "dominated_by"]].copy()
    if len(pareto) != 30:
        raise RuntimeError("Pareto status table is incomplete")
    pareto.to_csv(pareto_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    provenance = {
        "prompt_sha256": sha256_file(ROOT / "prompt_02" / "004_07_02_yc_seed5_focus_wp_et_seed3_seed4_reference_replay.md"),
        "script_sha256": sha256_file(Path(__file__)),
        "baseline_csv_sha256": sha256_file(BASELINE_CSV),
        "model_sha256": {str(seed): sha256_file(FROZEN_ROOT / "training" / "random_weather" / f"ppo_seed_{seed}" / "models" / "checkpoint_100000.zip") for seed in SEEDS},
        "frozen_episode_csv_sha256": {str(seed): sha256_file(EVAL_ROOT / f"ppo_seed_{seed}" / "evaluation_episode_level.csv") for seed in SEEDS},
        "frozen_step_csv_sha256": {str(seed): sha256_file(EVAL_ROOT / f"ppo_seed_{seed}" / "evaluation_step_level.csv") for seed in SEEDS},
        "result_row_count": len(rows), "pairwise_row_count": len(comparison),
        "snapshot_file_count": int(sum(len(json.loads(value)) for value in rows["snapshot_files"])),
        "rss_limit_mb": RSS_LIMIT_MB, "peak_rss_mb": float(pd.to_numeric(rows["peak_process_tree_rss_mb"]).max()),
        "baseline_scenarios": sorted(BASELINE_SCENARIOS),
        "source_summary_metric_function": "src/build_relaxed_success_five_scenario_daily_evidence_027_05.py::summary_metrics",
        "recovery_note": "Simulation completed once; post-run validation and finalization were repaired without repeating any DSSAT episode.",
        "original_smoke_manifest_status": "failed_due_to_validator_path_bug; retained",
        "original_formal_manifest_status": "failed_after_30_episodes_due_to_postprocessing_NameError; retained",
    }
    write_json(provenance_path, provenance)
    report = write_report(rows, comparison, profile, comparison, provenance)
    write_json(finalization_path, {
        "status": "passed", "passed": True, "completed_utc": pd.Timestamp.utcnow().isoformat(),
        "replayed_again": False, "audited_episodes": len(rows), "pairwise_comparisons": len(comparison),
        "peak_rss_mb": provenance["peak_rss_mb"],
        "source_formal_manifest_sha256": sha256_file(formal_manifest_path),
        "outputs": [rel(pareto_path), rel(provenance_path), rel(report)],
    })
    print(f"[finalize-pass] episodes={len(rows)} comparisons={len(comparison)} report={rel(report)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact evaluation-only Summary.OUT/ETCP replay for YC seeds 5, 3, and 4")
    parser.add_argument("--phase", choices=["smoke", "reconcile", "formal", "finalize"], required=True)
    args = parser.parse_args()
    log_path = TASK_ROOT / "logs" / f"{args.phase}.log"
    if args.phase == "smoke":
        if TASK_ROOT.exists():
            raise FileExistsError(f"Refusing to overwrite output directory: {rel(TASK_ROOT)}")
        (TASK_ROOT / "logs").mkdir(parents=True)
    elif args.phase == "reconcile":
        if not TASK_ROOT.is_dir() or not (TASK_ROOT / "logs").is_dir():
            raise FileNotFoundError("No failed smoke output is available for reconciliation")
    elif args.phase == "finalize":
        if not (TASK_ROOT / "logs").is_dir():
            raise FileNotFoundError("Formal output/log directory is missing")
    else:
        if not (TASK_ROOT / "logs").is_dir():
            raise FileNotFoundError("Smoke output/log directory is missing")
    if log_path.exists():
        raise FileExistsError(f"Refusing to overwrite log: {rel(log_path)}")
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        original_out, original_err = sys.stdout, sys.stderr
        sys.stdout = Tee(original_out, log)
        sys.stderr = Tee(original_err, log)
        try:
            print(f"[start] phase={args.phase} task={rel(TASK_ROOT)} rss_limit={RSS_LIMIT_MB:.0f} MiB", flush=True)
            if args.phase == "smoke":
                pilot = load_runtime()
                helper = load_summary_tools()
                run_smoke(pilot, helper)
            elif args.phase == "reconcile":
                reconcile_existing_smoke(load_summary_tools())
            elif args.phase == "finalize":
                finalize_existing_results()
            else:
                pilot = load_runtime()
                helper = load_summary_tools()
                run_formal(pilot, helper)
        finally:
            sys.stdout, sys.stderr = original_out, original_err


if __name__ == "__main__":
    main()
