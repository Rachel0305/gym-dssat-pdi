"""Low-cost rolling weather-action oracle for SY 2017.

This is a read-only DSSAT counterfactual diagnostic. It does not train PPO and
does not enable native DSSAT automatic irrigation. Each decision in a seven-day
window is selected by replaying prefix + candidate + reference suffix to season
end, then the selected action is carried into the next decision.
"""

from __future__ import annotations

import argparse
import json
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
import run_120_sya_weather_scenario_actionability_oracle as base120

CONFIG = ROOT / "configs/124_00_sya_2017_rolling_weather_oracle.json"
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


def latest(env, obs, info) -> dict[str, Any]:
    return direct_ppo.latest_observation_dict(env, obs, info or {})


def reference_file(cfg: dict[str, Any], year: int) -> Path:
    return ROOT / cfg["reference_daily_root"] / f"SYA_{year}_seed0_ckpt2000_daily.csv"


def action_grid(env) -> dict[int, tuple[float, float]]:
    return {int(idx): (float(row.get("amir", 0.0)), float(row.get("anfer", 0.0))) for idx, row in enumerate(getattr(env, "grid"))}


def reverse_grid(grid: dict[int, tuple[float, float]]) -> dict[tuple[float, float], int]:
    return {(round(i, 6), round(n, 6)): idx for idx, (i, n) in grid.items()}


def action_index(reverse: dict[tuple[float, float], int], irrigation: float, nitrogen: float) -> int:
    key = (round(float(irrigation), 6), round(float(nitrogen), 6))
    if key not in reverse:
        raise KeyError(f"reference action not on grid: {key}")
    return int(reverse[key])


def choose_target_indices(daily: pd.DataFrame, target_daps: list[int]) -> list[int]:
    daps = pd.to_numeric(daily["dap"], errors="coerce").fillna(-1).astype(int)
    out: list[int] = []
    for target in target_daps:
        candidates = daps.index[daps.ge(int(target))].tolist() or daps.index.tolist()
        if candidates:
            out.append(min(candidates, key=lambda idx: abs(int(daps.loc[idx]) - int(target))))
    return sorted(set(out))


def run_replay(make_env, runtime_config, env_config, year: int, run_name: str, action_prefix: list[int], reference_actions: list[int], target_idx: int, candidate_idx: int, reverse: dict[tuple[float, float], int]) -> dict[str, Any]:
    env = make_env(runtime_config, env_config, "SYA", year, 0, run_name, evaluation=True)
    try:
        obs, info = env.reset()
        terminated = truncated = False
        for step_idx in range(target_idx):
            if terminated or truncated:
                break
            action = action_prefix[step_idx] if step_idx < len(action_prefix) else reverse[(0.0, 0.0)]
            mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
            if action >= len(mask) or not bool(mask[action]):
                action = reverse[(0.0, 0.0)]
            obs, _reward, terminated, truncated, info = env.step(action)
        before = latest(env, obs, info)
        mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
        legal = bool(candidate_idx < len(mask) and mask[candidate_idx])
        overrides = 0
        if legal and not (terminated or truncated):
            obs, _reward, terminated, truncated, info = env.step(candidate_idx)
            trajectory = [latest(env, obs, info)]
            for idx in range(target_idx + 1, len(reference_actions)):
                if terminated or truncated:
                    break
                action = reference_actions[idx]
                current_mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                if action >= len(current_mask) or not bool(current_mask[action]):
                    action = reverse[(0.0, 0.0)]
                    overrides += 1
                obs, _reward, terminated, truncated, info = env.step(action)
                trajectory.append(latest(env, obs, info))
            end = trajectory[-1] if trajectory else before
        else:
            trajectory = []
            end = before
        return {"status": "ok", "legal_at_target": legal, "candidate_action_index": int(candidate_idx), "before_dap": scalar(before.get("dap")), "before_grnwt": scalar(before.get("grnwt")), "end_grnwt": scalar(end.get("grnwt")), "end_swfac": scalar(end.get("swfac")), "end_nstres": scalar(end.get("nstres")), "observed_steps": int(len(trajectory)), "continuation_override_count": int(overrides), "mismatch": 0}
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    task_id, task_name = str(cfg["task_id"]), str(cfg["task_name"])
    prompt_path = ROOT / cfg["prompt"]
    issues: list[str] = []
    if not prompt_path.exists():
        issues.append("prompt_missing")
    if cfg["actions"]["irrigation_levels_mm"] != EXPECTED_IRRIGATION:
        issues.append("irrigation_grid_mismatch")
    if cfg["actions"]["nitrogen_levels_kg_ha"] != EXPECTED_NITROGEN:
        issues.append("fixed_n_grid_mismatch")
    if cfg["observation_contract"]["weather_forecast_enabled"]:
        issues.append("forecast_must_be_false_for_oracle")
    if cfg["scope"]["native_dssat_automatic_irrigation"]:
        issues.append("native_automatic_must_be_false")
    for year in cfg["states"]["years"]:
        if not reference_file(cfg, int(year)).exists():
            issues.append(f"reference_missing_{year}")
    mode = "smoke" if args.smoke else "full"
    if args.dry_run:
        print(json.dumps({"task": f"{task_id}_{task_name}", "issues": issues, "mode": mode, "training": False, "years": cfg["states"]["years"], "target_daps": cfg["states"]["target_daps"], "scenarios": list(cfg["scenarios"]), "candidate_irrigation": EXPECTED_IRRIGATION, "rolling_window_steps": cfg["states"]["rolling_window_steps"], "native_automatic": False}, ensure_ascii=False, indent=2))
        return 0 if not issues else 2

    if args.smoke:
        years, daps, scenarios = [2017], [45], ["actual"]
    else:
        years = list(map(int, cfg["states"]["years"]))
        daps = list(map(int, cfg["states"]["target_daps"]))
        scenarios = list(cfg["scenarios"])
    out = ROOT / cfg["output_root"]
    if args.smoke:
        out = out.with_name(out.name + "_smoke")
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
        engine.TASK_ID, engine.TASK_NAME = task_id, task_name
        engine.BASE_OUT, engine.BASE_DOC, engine.PROMPT = out, prompt_path, prompt_path
        engine.LOWIC_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
        engine.STATION, engine.SITES = "SYA", ["SYA"]
        engine.BINARY_IRRIGATION_LEVELS = list(map(float, EXPECTED_IRRIGATION))
        engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
        engine.base03222.SITE_NAMES["SYA"] = "SY"
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = engine.LOWIC_INPUT_ROOT
        ppo_safe_rendering.copy_weather_to_rendered_dir = patched_copy_weather
        runtime_config = engine.load_config()
        selection = engine.base03222.build_selection(engine.base03222.load_split())
        env_config = direct_ppo.build_env_config(runtime_config, selection)
        probe = old_make_env(runtime_config, env_config, "SYA", years[0], 0, f"{task_id}_grid_probe", evaluation=True)
        try:
            grid = action_grid(probe)
        finally:
            probe.close()
        reverse = reverse_grid(grid)
        candidates = [idx for idx, (irrigation, nitrogen) in sorted(grid.items()) if round(nitrogen, 6) == 0.0 and irrigation in EXPECTED_IRRIGATION]
        manifest: list[dict[str, Any]] = []
        for year in years:
            daily = pd.read_csv(reference_file(cfg, year))
            ref_actions = [action_index(reverse, row.safe_action_amir, row.safe_action_anfer) for row in daily.itertuples(index=False)]
            target_indices = choose_target_indices(daily, daps)
            planting = pd.Timestamp(direct_ppo.find_year(env_config, "SYA", year)["planting_date"])
            source = ppo_safe_rendering.source_weather_path("SYA", year)
            for target_idx in target_indices:
                target_dap = int(pd.to_numeric(daily.iloc[target_idx]["dap"], errors="coerce"))
                current_date = planting + pd.Timedelta(days=target_dap - 1)
                scenario_paths: dict[str, Path] = {}
                for scenario in scenarios:
                    scfg = cfg["scenarios"][scenario]
                    target = out / "scenario_wth" / str(year) / f"dap{target_dap}" / scenario / source.name
                    if scfg["rain_mode"] == "as_recorded":
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, target)
                        meta = {"changed_rows": 0}
                    else:
                        meta = base120.write_scenario_wth(source, target, current_date, str(scfg["rain_mode"]), float(scfg["rain_mm_per_day"]))
                    scenario_paths[scenario] = target
                    manifest.append({"year": year, "target_dap": target_dap, "scenario": scenario, "path": rel(target), **meta})
                for scenario in scenarios:
                    active_weather["path"] = scenario_paths[scenario]
                    selected_prefix = list(ref_actions[:target_idx])
                    window_rows: list[dict[str, Any]] = []
                    for offset in range(int(cfg["states"]["rolling_window_steps"])):
                        decision_idx = target_idx + offset
                        if decision_idx >= len(ref_actions):
                            break
                        candidate_rows = []
                        for candidate_idx in candidates:
                            result = run_replay(old_make_env, runtime_config, env_config, year, f"{task_id}_{year}_dap{target_dap}_{scenario}_offset{offset}_candidate{candidate_idx}", selected_prefix, ref_actions, decision_idx, candidate_idx, reverse)
                            result.update({"year": year, "target_dap": target_dap, "scenario": scenario, "window_offset": offset, "decision_dap": int(pd.to_numeric(daily.iloc[decision_idx]["dap"], errors="coerce")), "candidate_irrigation_mm": grid[candidate_idx][0], "candidate_nitrogen_kg_ha": grid[candidate_idx][1]})
                            candidate_rows.append(result)
                        legal_rows = [row for row in candidate_rows if row["legal_at_target"]]
                        if not legal_rows:
                            raise RuntimeError(f"no legal candidate at {year} dap {target_dap} {scenario} offset {offset}")
                        best = max(legal_rows, key=lambda row: (row["end_grnwt"], -row["candidate_irrigation_mm"]))
                        selected_action = int(best["candidate_action_index"])
                        selected_prefix.append(selected_action)
                        for row in candidate_rows:
                            row["selected"] = bool(row["candidate_action_index"] == selected_action)
                            window_rows.append(row)
                    selected_end = run_replay(old_make_env, runtime_config, env_config, year, f"{task_id}_{year}_dap{target_dap}_{scenario}_selected_end", selected_prefix, ref_actions, len(ref_actions) - 1, ref_actions[-1], reverse)
                    rows.extend(window_rows)
                    selected_actions = selected_prefix[target_idx: target_idx + int(cfg["states"]["rolling_window_steps"])]
                    manifest.append({"year": year, "target_dap": target_dap, "scenario": scenario, "selected_irrigation_mm": [grid[a][0] for a in selected_actions], "selected_action_indices": selected_actions, "selected_end_grnwt": selected_end.get("end_grnwt"), "selected_end_mismatch": selected_end.get("mismatch", 0)})
    finally:
        active_weather["path"] = None
        ppo_safe_rendering.copy_weather_to_rendered_dir = old_copy_weather
        engine.base03222.base.make_env = old_make_env
        engine.base03222.SITE_NAMES.clear(); engine.base03222.SITE_NAMES.update(old_site_names)
        for key, value in old.items():
            setattr(engine, key, value)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"{task_id}_candidate_rows.csv", index=False, encoding="utf-8-sig")
    (out / f"{task_id}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    selected = df[df["selected"] == True].copy() if len(df) else df
    pivot = selected.pivot_table(index=["year", "target_dap", "scenario", "window_offset", "decision_dap"], values="candidate_irrigation_mm", aggfunc="first").reset_index() if len(selected) else pd.DataFrame()
    pivot.to_csv(out / f"{task_id}_selected_action_sequence.csv", index=False, encoding="utf-8-sig")
    weather_switch = False
    if len(pivot):
        for key, group in pivot.groupby(["year", "target_dap", "window_offset"], sort=True):
            if group["candidate_irrigation_mm"].nunique() > 1:
                weather_switch = True
    result = {"task": f"{task_id}_{task_name}", "status": "completed_read_only", "mode": mode, "management_mode": "ppo_linked_irrigation_and_nitrogen", "native_dssat_automatic_irrigation_enabled": False, "n_candidate_rows": int(len(df)), "n_selected_rows": int(len(selected)), "weather_conditioned_action_switch_detected": bool(weather_switch), "selected_actions": rel(out / f"{task_id}_selected_action_sequence.csv"), "candidate_rows": rel(out / f"{task_id}_candidate_rows.csv"), "manifest": rel(out / f"{task_id}_manifest.json"), "next_step_allowed": bool(weather_switch and not args.smoke)}
    (out / f"{task_id}_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
