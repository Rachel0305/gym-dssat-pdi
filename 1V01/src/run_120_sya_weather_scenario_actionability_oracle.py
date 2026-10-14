"""Read-only SY weather-scenario actionability oracle.

This isolated diagnostic changes only a seven-day rainfall window in copied WTH
files. It does not alter source inputs, train PPO, or write formal DSSAT results.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sb3_contrib.common.maskable.utils import get_action_masks

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_all_year_direct_action_safe_ppo as direct_ppo

CONFIG = ROOT / "configs/120_00_sya_weather_scenario_actionability_oracle.json"
PROMPT = ROOT / "prompts/2026-08-15_sya_weather_scenario_actionability_oracle.md"
EXPECTED_IRRIGATION = [0.0, 15.0, 30.0, 45.0]
EXPECTED_NITROGEN = [0.0]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def scalar(value: Any, default: float = 0.0) -> float:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        return float(arr[0]) if len(arr) else default
    except Exception:
        return default


def action_grid(env) -> dict[int, tuple[float, float]]:
    return {int(idx): (float(row.get("amir", 0.0)), float(row.get("anfer", 0.0))) for idx, row in enumerate(getattr(env, "grid"))}


def reverse_grid(grid: dict[int, tuple[float, float]]) -> dict[tuple[float, float], int]:
    return {(round(i, 6), round(n, 6)): idx for idx, (i, n) in grid.items()}


def action_index(reverse: dict[tuple[float, float], int], irrigation: float, nitrogen: float) -> int:
    key = (round(float(irrigation), 6), round(float(nitrogen), 6))
    if key not in reverse:
        raise KeyError(f"reference action not on grid: {key}")
    return int(reverse[key])


def latest(env, obs, info) -> dict[str, Any]:
    return direct_ppo.latest_observation_dict(env, obs, info or {})


def reference_file(cfg: dict[str, Any], year: int) -> Path:
    return ROOT / cfg["reference_daily_root"] / f"SYA_{year}_seed0_ckpt2000_daily.csv"


def choose_target_indices(daily: pd.DataFrame, target_daps: list[int]) -> list[int]:
    daps = pd.to_numeric(daily["dap"], errors="coerce").fillna(-1).astype(int)
    out: list[int] = []
    for target in target_daps:
        candidates = daps.index[daps.ge(int(target))].tolist() or daps.index.tolist()
        if candidates:
            out.append(min(candidates, key=lambda idx: abs(int(daps.loc[idx]) - int(target))))
    return sorted(set(out))


def parse_wth(path: Path) -> tuple[list[str], dict[int, tuple[float, float, float, float]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows: dict[int, tuple[float, float, float, float]] = {}
    for line in lines:
        parts = line.split()
        if len(parts) >= 5 and re.fullmatch(r"\d{7}", parts[0]):
            try:
                rows[int(parts[0])] = tuple(float(x) for x in parts[1:5])
            except ValueError:
                continue
    return lines, rows


def write_scenario_wth(source: Path, out: Path, start_date: pd.Timestamp, mode: str, rain_mm_per_day: float) -> dict[str, Any]:
    lines, rows = parse_wth(source)
    dates = {int(start_date.year * 1000 + int(start_date.dayofyear) + offset): start_date + pd.Timedelta(days=offset) for offset in range(1, 8)}
    changed = 0
    output: list[str] = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5 and re.fullmatch(r"\d{7}", parts[0]) and int(parts[0]) in rows:
            token = int(parts[0])
            if mode == "set_window" and token in dates:
                srad, tmax, tmin, _rain = rows[token]
                # DSSAT WTH uses fixed-width numeric columns: keep the original
                # six-character field width rather than introducing extra spaces.
                output.append(f"{token:07d}{srad:6.1f}{tmax:6.1f}{tmin:6.1f}{float(rain_mm_per_day):6.1f}")
                changed += 1
                continue
        output.append(line)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(output) + "\n", encoding="utf-8")
    return {"source": rel(source), "output": rel(out), "mode": mode, "rain_mm_per_day": float(rain_mm_per_day), "changed_rows": int(changed), "window_start": (start_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "window_end": (start_date + pd.Timedelta(days=7)).strftime("%Y-%m-%d")}


def make_scenario_files(cfg: dict[str, Any], env_config: dict[str, Any], out: Path, years: list[int], daps: list[int]) -> tuple[dict[tuple[int, int, str], Path], list[dict[str, Any]]]:
    mapping: dict[tuple[int, int, str], Path] = {}
    manifest: list[dict[str, Any]] = []
    for year in years:
        planting = pd.Timestamp(direct_ppo.find_year(env_config, "SYA", int(year))["planting_date"])
        source = ppo_safe_rendering.source_weather_path("SYA", int(year))
        for dap in daps:
            current_date = planting + pd.Timedelta(days=int(dap) - 1)
            for scenario, scfg in cfg["scenarios"].items():
                target = out / "scenario_wth" / str(year) / f"dap{int(dap)}" / scenario / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if scfg["rain_mode"] == "as_recorded":
                    shutil.copyfile(source, target)
                    meta = {"source": rel(source), "output": rel(target), "mode": "as_recorded", "rain_mm_per_day": None, "changed_rows": 0, "window_start": "", "window_end": ""}
                else:
                    meta = write_scenario_wth(source, target, current_date, str(scfg["rain_mode"]), float(scfg["rain_mm_per_day"]))
                mapping[(int(year), int(dap), str(scenario))] = target
                manifest.append({"year": int(year), "target_dap": int(dap), "scenario": scenario, **meta})
    return mapping, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    task_id = str(cfg["task_id"])
    task_name = str(cfg["task_name"])
    prompt_path = ROOT / cfg["prompt"]
    issues: list[str] = []
    if not prompt_path.exists():
        issues.append("prompt_missing")
    if cfg["actions"]["irrigation_levels_mm"] != EXPECTED_IRRIGATION:
        issues.append("irrigation_grid_mismatch")
    if cfg["actions"]["nitrogen_levels_kg_ha"] != EXPECTED_NITROGEN:
        issues.append("fixed_n_grid_mismatch")
    for year in cfg["states"]["years"]:
        if not reference_file(cfg, int(year)).exists():
            issues.append(f"reference_missing_{year}")
    if args.dry_run:
        print(json.dumps({"task": f"{task_id}_{task_name}", "issues": issues, "training": False, "states": len(cfg["states"]["years"]) * len(cfg["states"]["target_daps"]), "scenarios": list(cfg["scenarios"]), "irrigation_actions": EXPECTED_IRRIGATION, "horizon_steps": cfg["states"]["horizon_steps"]}, ensure_ascii=False, indent=2))
        return 0 if not issues else 2

    out = ROOT / cfg["output_root"]
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {rel(out)}")
    out.mkdir(parents=True)

    old = {key: getattr(engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_site_names = dict(engine.base03222.SITE_NAMES)
    old_make_env = engine.base03222.base.make_env
    old_copy_weather = ppo_safe_rendering.copy_weather_to_rendered_dir
    active_weather: dict[str, Path | None] = {"path": None}

    def patched_copy_weather(station: str, year: int, template: Path, config: dict) -> Path:
        source = active_weather["path"] or ppo_safe_rendering.source_weather_path(station, year)
        rendered = template.parent / f"{ppo_safe_rendering.target_weather_stem(station, year)}.WTH"
        shutil.copyfile(source, rendered)
        return rendered

    rows: list[dict[str, Any]] = []
    try:
        engine.TASK_ID = task_id
        engine.TASK_NAME = task_name
        engine.BASE_OUT = out
        engine.BASE_DOC = prompt_path
        engine.PROMPT = prompt_path
        engine.LOWIC_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
        engine.STATION = "SYA"
        engine.SITES = ["SYA"]
        engine.BINARY_IRRIGATION_LEVELS = list(map(float, EXPECTED_IRRIGATION))
        engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
        engine.base03222.SITE_NAMES["SYA"] = "SY"
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = engine.LOWIC_INPUT_ROOT
        ppo_safe_rendering.copy_weather_to_rendered_dir = patched_copy_weather
        runtime_config = engine.load_config()
        selection = engine.base03222.build_selection(engine.base03222.load_split())
        env_config = direct_ppo.build_env_config(runtime_config, selection)
        daps = list(map(int, cfg["states"]["target_daps"]))
        years = list(map(int, cfg["states"]["years"]))
        scenario_paths, scenario_manifest = make_scenario_files(cfg, env_config, out, years, daps)
        (out / f"{task_id}_scenario_wth_manifest.json").write_text(json.dumps(scenario_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        probe = old_make_env(runtime_config, env_config, "SYA", years[0], 0, "120_grid_probe", evaluation=True)
        try:
            grid = action_grid(probe)
        finally:
            probe.close()
        reverse = reverse_grid(grid)
        candidate_indices = [idx for idx, (irrigation, nitrogen) in sorted(grid.items()) if round(nitrogen, 6) == 0.0 and irrigation in EXPECTED_IRRIGATION]
        for year in years:
            reference = pd.read_csv(reference_file(cfg, year))
            for target_idx in choose_target_indices(reference, daps):
                target_dap = int(pd.to_numeric(reference.iloc[target_idx]["dap"], errors="coerce"))
                for scenario in cfg["scenarios"]:
                    scenario_path = scenario_paths[(year, target_dap, scenario)]
                    for candidate_idx in candidate_indices:
                        active_weather["path"] = scenario_path
                        env = old_make_env(runtime_config, env_config, "SYA", year, 0, f"{task_id}_{year}_dap{target_dap}_{scenario}_a{candidate_idx}", evaluation=True)
                        try:
                            obs, info = env.reset()
                            prefix = [action_index(reverse, row.safe_action_amir, row.safe_action_anfer) for row in reference.itertuples(index=False)]
                            terminated = truncated = False
                            for step_idx in range(target_idx):
                                obs, _reward, terminated, truncated, info = env.step(prefix[step_idx] if step_idx < len(prefix) else reverse[(0.0, 0.0)])
                                if bool(terminated or truncated):
                                    break
                            before = latest(env, obs, info)
                            mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                            legal = bool(candidate_idx < len(mask) and mask[candidate_idx])
                            result: dict[str, Any] = {"status": "ok", "year": year, "target_dap": target_dap, "scenario": scenario, "candidate_action_index": candidate_idx, "candidate_irrigation_mm": grid[candidate_idx][0], "candidate_nitrogen_kg_ha": grid[candidate_idx][1], "legal_at_target": legal, "before_swfac": scalar(before.get("swfac")), "before_nstres": scalar(before.get("nstres")), "before_topwt": scalar(before.get("topwt")), "before_grnwt": scalar(before.get("grnwt")), "horizon_steps": int(cfg["states"]["horizon_steps"])}
                            if legal and not bool(terminated or truncated):
                                obs, _reward, terminated, truncated, info = env.step(candidate_idx)
                                trajectory = [latest(env, obs, info)]
                                override = 0
                                for offset in range(1, int(cfg["states"]["horizon_steps"])):
                                    ref_idx = target_idx + offset
                                    if ref_idx >= len(prefix) or bool(terminated or truncated):
                                        break
                                    continuation = prefix[ref_idx]
                                    current_mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                                    if continuation >= len(current_mask) or not current_mask[continuation]:
                                        continuation = reverse[(0.0, 0.0)]
                                        override += 1
                                    obs, _reward, terminated, truncated, info = env.step(continuation)
                                    trajectory.append(latest(env, obs, info))
                                end = trajectory[-1] if trajectory else before
                                result.update({"observed_horizon_steps": len(trajectory), "continuation_override_count": override, "end_swfac": scalar(end.get("swfac")), "end_nstres": scalar(end.get("nstres")), "end_topwt": scalar(end.get("topwt")), "end_grnwt": scalar(end.get("grnwt")), "mean_swfac_horizon": float(np.mean([scalar(row.get("swfac")) for row in trajectory])), "mean_nstres_horizon": float(np.mean([scalar(row.get("nstres")) for row in trajectory]))})
                            rows.append(result)
                        finally:
                            env.close()
    finally:
        active_weather["path"] = None
        ppo_safe_rendering.copy_weather_to_rendered_dir = old_copy_weather
        engine.base03222.base.make_env = old_make_env
        engine.base03222.SITE_NAMES.clear()
        engine.base03222.SITE_NAMES.update(old_site_names)
        for key, value in old.items():
            setattr(engine, key, value)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"{task_id}_actionability_rows.csv", index=False, encoding="utf-8-sig")
    valid = df[(df["status"] == "ok") & (df["legal_at_target"] == True)].copy()
    summary_rows: list[dict[str, Any]] = []
    for (year, target_dap, scenario), group in valid.groupby(["year", "target_dap", "scenario"], sort=True):
        summary_rows.append({"year": int(year), "target_dap": int(target_dap), "scenario": scenario, "candidate_count": int(len(group)), "end_swfac_range": float(group["end_swfac"].max() - group["end_swfac"].min()), "end_nstres_range": float(group["end_nstres"].max() - group["end_nstres"].min()), "end_topwt_range": float(group["end_topwt"].max() - group["end_topwt"].min()), "end_grnwt_range": float(group["end_grnwt"].max() - group["end_grnwt"].min()), "best_stress_irrigation_mm": float(group.loc[(group["mean_swfac_horizon"] + group["mean_nstres_horizon"]).idxmin(), "candidate_irrigation_mm"])})
    state_summary = pd.DataFrame(summary_rows)
    state_summary.to_csv(out / f"{task_id}_state_scenario_summary.csv", index=False, encoding="utf-8-sig")
    best_cross = valid.assign(stress_score=valid["mean_swfac_horizon"] + valid["mean_nstres_horizon"]).loc[valid.assign(stress_score=valid["mean_swfac_horizon"] + valid["mean_nstres_horizon"]).groupby(["year", "target_dap", "scenario"])["stress_score"].idxmin(), ["year", "target_dap", "scenario", "candidate_irrigation_mm", "candidate_nitrogen_kg_ha", "stress_score"]]
    best_cross.to_csv(out / f"{task_id}_best_stress_actions.csv", index=False, encoding="utf-8-sig")
    pivot = best_cross.pivot_table(index=["year", "target_dap"], columns="scenario", values="candidate_irrigation_mm", aggfunc="first").reset_index()
    pivot.to_csv(out / f"{task_id}_best_action_scenario_pivot.csv", index=False, encoding="utf-8-sig")
    scenario_switch = bool(len(pivot) and pivot[[c for c in cfg["scenarios"] if c in pivot.columns]].nunique(axis=1).gt(1).any())
    result = {"task": f"{task_id}_{task_name}", "status": "completed_read_only", "management_mode": "ppo_linked_irrigation_and_nitrogen", "native_dssat_automatic_irrigation_enabled": False, "n_rows": int(len(df)), "n_valid_legal_rows": int(len(valid)), "n_states": int(len(pivot)), "scenario_switch_detected": scenario_switch, "state_summary": rel(out / f"{task_id}_state_scenario_summary.csv"), "best_action_scenario_pivot": rel(out / f"{task_id}_best_action_scenario_pivot.csv"), "rows": rel(out / f"{task_id}_actionability_rows.csv"), "next_step_allowed": scenario_switch}
    (out / f"{task_id}_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
