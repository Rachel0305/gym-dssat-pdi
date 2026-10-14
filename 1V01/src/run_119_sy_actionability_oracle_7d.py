"""Read-only DSSAT actionability oracle for SY weather-conditioned actions."""

from __future__ import annotations

import argparse
import json
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
import forecast_engineered_observation_056_057 as forecast
from run_113_114_sy_lc_compact_forecast_smoke2k import patch_contract as patch_compact

CONFIG = ROOT / "configs/119_00_sya_actionability_oracle_7d.json"
PROMPT = ROOT / "prompts/2026-08-15_sy_actionability_oracle_7d.md"
EXPECTED_IRRIGATION = [0.0, 15.0, 30.0, 45.0]
EXPECTED_NITROGEN = [0.0, 40.0, 80.0, 120.0]


def scalar(value: Any, default: float = 0.0) -> float:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        return float(arr[0]) if len(arr) else default
    except Exception:
        return default


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def action_grid(env) -> dict[int, tuple[float, float]]:
    return {int(idx): (float(row.get("amir", 0.0)), float(row.get("anfer", 0.0))) for idx, row in enumerate(getattr(env, "grid"))}


def reverse_grid(grid: dict[int, tuple[float, float]]) -> dict[tuple[float, float], int]:
    return {(round(i, 6), round(n, 6)): idx for idx, (i, n) in grid.items()}


def action_index(reverse: dict[tuple[float, float], int], irrigation: float, nitrogen: float) -> int:
    key = (round(float(irrigation), 6), round(float(nitrogen), 6))
    if key not in reverse:
        raise KeyError(f"reference action not on grid: {key}")
    return int(reverse[key])


def compact_risk(env) -> float:
    features = dict(getattr(env, "last_norm_forecast_features", {}) or {})
    rain = float(np.clip(features.get("rain_next7_norm", 0.0), 0.0, 1.0))
    heat = float(np.clip((features.get("tmax_mean_next7_norm", 0.0) + 1.0) / 2.0, 0.0, 1.0))
    dry = float(np.clip(features.get("dry_spell_next7_flag", 0.0), 0.0, 1.0))
    return float(np.clip(0.6 * (1.0 - rain) + 0.25 * heat + 0.15 * dry, 0.0, 1.0))


def latest(env, obs, info) -> dict[str, float]:
    return forecast.direct_ppo.latest_observation_dict(env, obs, info or {})


def reference_file(cfg: dict[str, Any], year: int) -> Path:
    return ROOT / cfg["reference_daily_root"] / f"SYA_{year}_seed0_ckpt2000_daily.csv"


def choose_target_indices(daily: pd.DataFrame, target_daps: list[int]) -> list[int]:
    daps = pd.to_numeric(daily["dap"], errors="coerce").fillna(-1).astype(int)
    indices: list[int] = []
    for target in target_daps:
        candidates = daps.index[daps.ge(int(target))].tolist()
        if not candidates:
            candidates = daps.index.tolist()
        if not candidates:
            continue
        indices.append(min(candidates, key=lambda idx: abs(int(daps.loc[idx]) - int(target))))
    return sorted(set(indices))


def replay_candidate(
    make_env,
    runtime_config: dict[str, Any],
    env_config: dict[str, Any],
    year: int,
    seed: int,
    reference: pd.DataFrame,
    target_idx: int,
    candidate_idx: int,
    reverse: dict[tuple[float, float], int],
    horizon_steps: int,
    run_tag: str,
) -> dict[str, Any]:
    env = make_env(runtime_config, env_config, "SYA", year, seed, run_tag, evaluation=True)
    try:
        obs, info = env.reset()
        prefix_actions = [action_index(reverse, row.safe_action_amir, row.safe_action_anfer) for row in reference.itertuples(index=False)]
        for step_idx in range(target_idx):
            action = prefix_actions[step_idx] if step_idx < len(prefix_actions) else reverse[(0.0, 0.0)]
            obs, _reward, terminated, truncated, info = env.step(action)
            if bool(terminated or truncated):
                return {"status": "terminated_before_target", "year": year, "target_row_index": target_idx, "candidate_action": candidate_idx}
        before = latest(env, obs, info)
        mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
        legal = bool(0 <= candidate_idx < len(mask) and mask[candidate_idx])
        grid = action_grid(env)
        result: dict[str, Any] = {
            "status": "ok",
            "year": int(year),
            "target_row_index": int(target_idx),
            "target_dap": int(pd.to_numeric(reference.iloc[target_idx]["dap"], errors="coerce")),
            "forecast_risk": compact_risk(env),
            "candidate_action_index": int(candidate_idx),
            "candidate_irrigation_mm": grid[candidate_idx][0],
            "candidate_nitrogen_kg_ha": grid[candidate_idx][1],
            "legal_at_target": legal,
            "before_swfac": scalar(before.get("swfac")),
            "before_nstres": scalar(before.get("nstres")),
            "before_topwt": scalar(before.get("topwt")),
            "before_grnwt": scalar(before.get("grnwt")),
            "horizon_steps": int(horizon_steps),
        }
        if not legal:
            return result
        obs, _reward, terminated, truncated, info = env.step(candidate_idx)
        trajectory = [latest(env, obs, info)]
        continuation_override_count = 0
        for offset in range(1, horizon_steps):
            ref_idx = target_idx + offset
            if ref_idx >= len(prefix_actions) or bool(terminated or truncated):
                break
            continuation_action = prefix_actions[ref_idx]
            current_mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
            if continuation_action >= len(current_mask) or not current_mask[continuation_action]:
                continuation_override_count += 1
                continuation_action = reverse[(0.0, 0.0)]
            obs, _reward, terminated, truncated, info = env.step(continuation_action)
            trajectory.append(latest(env, obs, info))
        end = trajectory[-1] if trajectory else before
        result.update(
            {
                "continuation_override_count": int(continuation_override_count),
                "observed_horizon_steps": int(len(trajectory)),
                "end_swfac": scalar(end.get("swfac")),
                "end_nstres": scalar(end.get("nstres")),
                "end_topwt": scalar(end.get("topwt")),
                "end_grnwt": scalar(end.get("grnwt")),
                "mean_swfac_horizon": float(np.mean([scalar(row.get("swfac")) for row in trajectory])),
                "mean_nstres_horizon": float(np.mean([scalar(row.get("nstres")) for row in trajectory])),
            }
        )
        return result
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    patch_compact()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg["actions"]["irrigation_levels_mm"] != EXPECTED_IRRIGATION or cfg["actions"]["nitrogen_levels_kg_ha"] != EXPECTED_NITROGEN:
        raise ValueError("119 action grid mismatch")
    for year in cfg["states"]["years"]:
        if not reference_file(cfg, int(year)).exists():
            raise FileNotFoundError(reference_file(cfg, int(year)))
    if args.dry_run:
        print(json.dumps({"task": "119_00_sya_actionability_oracle_7d", "prompt_exists": PROMPT.exists(), "reference_years": cfg["states"]["years"], "target_daps": cfg["states"]["target_daps"], "horizon_steps": cfg["states"]["horizon_steps"], "actions": 16, "training": False, "issues": []}, ensure_ascii=False, indent=2))
        return 0

    out = ROOT / cfg["output_root"]
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {rel(out)}")
    out.mkdir(parents=True)
    old = {key: getattr(forecast.engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_names = dict(forecast.engine.base03222.SITE_NAMES)
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_make_env = forecast.engine.base03222.base.make_env
    rows: list[dict[str, Any]] = []
    try:
        forecast.engine.TASK_ID = "119_00"
        forecast.engine.TASK_NAME = str(cfg["task_name"])
        forecast.engine.BASE_OUT = out
        forecast.engine.BASE_DOC = PROMPT
        forecast.engine.PROMPT = PROMPT
        forecast.engine.LOWIC_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        forecast.engine.STATION = "SYA"
        forecast.engine.SITES = ["SYA"]
        forecast.engine.BINARY_IRRIGATION_LEVELS = list(map(float, EXPECTED_IRRIGATION))
        forecast.engine.BINARY_NITROGEN_LEVELS = list(map(float, EXPECTED_NITROGEN))
        forecast.engine.base03222.SITE_NAMES["SYA"] = "SY"
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        runtime_config = forecast.engine.load_config()
        selection = forecast.engine.base03222.build_selection(forecast.engine.base03222.load_split())
        env_config = forecast.direct_ppo.build_env_config(runtime_config, selection)
        make_env = forecast.make_forecast_env_factory(old_make_env, cfg)
        probe = make_env(runtime_config, env_config, "SYA", int(cfg["states"]["years"][0]), 0, "119_grid_probe", evaluation=True)
        try:
            grid = action_grid(probe)
        finally:
            probe.close()
        reverse = reverse_grid(grid)
        for year in map(int, cfg["states"]["years"]):
            reference = pd.read_csv(reference_file(cfg, year))
            for target_idx in choose_target_indices(reference, list(map(int, cfg["states"]["target_daps"]))):
                for candidate_idx in sorted(grid):
                    rows.append(replay_candidate(make_env, runtime_config, env_config, year, 0, reference, target_idx, candidate_idx, reverse, int(cfg["states"]["horizon_steps"]), f"119_oracle_{year}_{target_idx}_{candidate_idx}"))
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
        forecast.engine.base03222.base.make_env = old_make_env
        forecast.engine.base03222.SITE_NAMES.clear()
        forecast.engine.base03222.SITE_NAMES.update(old_names)
        for key, value in old.items():
            setattr(forecast.engine, key, value)

    results = pd.DataFrame(rows)
    results.to_csv(out / "119_00_actionability_rows.csv", index=False, encoding="utf-8-sig")
    valid = results[(results["status"] == "ok") & (results["legal_at_target"] == True)].copy()
    state_summary = valid.groupby(["year", "target_row_index", "target_dap"], as_index=False).agg(
        forecast_risk=("forecast_risk", "first"),
        legal_actions=("candidate_action_index", "count"),
        end_swfac_range=("end_swfac", lambda x: float(x.max() - x.min())),
        end_nstres_range=("end_nstres", lambda x: float(x.max() - x.min())),
        end_topwt_range=("end_topwt", lambda x: float(x.max() - x.min())),
        end_grnwt_range=("end_grnwt", lambda x: float(x.max() - x.min())),
    )
    state_summary.to_csv(out / "119_00_actionability_state_summary.csv", index=False, encoding="utf-8-sig")
    valid["stress_score"] = pd.to_numeric(valid["mean_swfac_horizon"], errors="coerce") + pd.to_numeric(valid["mean_nstres_horizon"], errors="coerce")
    best = valid.loc[valid.groupby(["year", "target_row_index"])["stress_score"].idxmin(), ["year", "target_row_index", "target_dap", "forecast_risk", "candidate_action_index", "candidate_irrigation_mm", "candidate_nitrogen_kg_ha", "stress_score"]].copy()
    best.to_csv(out / "119_00_best_stress_actions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "task": "119_00_sya_actionability_oracle_7d",
        "status": "completed_read_only",
        "n_rows": int(len(results)),
        "n_valid_legal_rows": int(len(valid)),
        "n_states": int(len(state_summary)),
        "state_summary": rel(out / "119_00_actionability_state_summary.csv"),
        "best_stress_actions": rel(out / "119_00_best_stress_actions.csv"),
        "rows": rel(out / "119_00_actionability_rows.csv"),
        "max_end_swfac_range": float(state_summary["end_swfac_range"].max()) if len(state_summary) else None,
        "max_end_nstres_range": float(state_summary["end_nstres_range"].max()) if len(state_summary) else None,
        "max_end_topwt_range": float(state_summary["end_topwt_range"].max()) if len(state_summary) else None,
        "max_end_grnwt_range": float(state_summary["end_grnwt_range"].max()) if len(state_summary) else None,
        "next_step_allowed": bool(len(state_summary) > 0 and ((state_summary["end_swfac_range"] > 0.01) | (state_summary["end_nstres_range"] > 0.01) | (state_summary["end_topwt_range"] > 1.0)).any()),
    }
    (out / "119_00_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
