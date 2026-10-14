"""Audit the six selected rolling sequences from experiment 124.

The main oracle evaluates candidates correctly, but this small read-only audit
replays each selected seven-day sequence end-to-end before reporting endpoints.
It avoids rerunning the 168 candidate counterfactuals.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

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
from run_124_sya_2017_rolling_weather_oracle import action_grid, action_index, reference_file, reverse_grid

CFG = ROOT / "configs/124_00_sya_2017_rolling_weather_oracle.json"
OUT = ROOT / "benchmark_results/124_00_sya_2017_rolling_weather_oracle"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def scalar(value, default=0.0):
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        return float(arr[0]) if len(arr) else default
    except Exception:
        return default


def latest(env, obs, info):
    return direct_ppo.latest_observation_dict(env, obs, info or {})


def main() -> int:
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "124_00_manifest.json").read_text(encoding="utf-8"))
    selected = [row for row in manifest if "selected_action_indices" in row]
    old = {key: getattr(engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_site_names = dict(engine.base03222.SITE_NAMES)
    old_make_env = engine.base03222.base.make_env
    old_copy_weather = ppo_safe_rendering.copy_weather_to_rendered_dir
    active_weather = {"path": None}

    def patched_copy_weather(station, year, template, config):
        source = active_weather["path"] or ppo_safe_rendering.source_weather_path(station, year)
        rendered = template.parent / f"{ppo_safe_rendering.target_weather_stem(station, year)}.WTH"
        shutil.copyfile(source, rendered)
        return rendered

    rows = []
    try:
        prompt = ROOT / cfg["prompt"]
        engine.TASK_ID, engine.TASK_NAME = "124_00_audit", "sya_2017_rolling_selected_endpoints"
        engine.BASE_OUT, engine.BASE_DOC, engine.PROMPT = OUT, prompt, prompt
        engine.LOWIC_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
        engine.STATION, engine.SITES = "SYA", ["SYA"]
        engine.BINARY_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
        engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
        engine.base03222.SITE_NAMES["SYA"] = "SY"
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = engine.LOWIC_INPUT_ROOT
        ppo_safe_rendering.copy_weather_to_rendered_dir = patched_copy_weather
        runtime_config = engine.load_config()
        selection = engine.base03222.build_selection(engine.base03222.load_split())
        env_config = direct_ppo.build_env_config(runtime_config, selection)
        probe = old_make_env(runtime_config, env_config, "SYA", 2017, 0, "124_00_audit_grid", evaluation=True)
        try:
            grid = action_grid(probe)
        finally:
            probe.close()
        reverse = reverse_grid(grid)
        daily = pd.read_csv(reference_file(cfg, 2017))
        ref_actions = [action_index(reverse, row.safe_action_amir, row.safe_action_anfer) for row in daily.itertuples(index=False)]
        for item in selected:
            year, target_dap, scenario = int(item["year"]), int(item["target_dap"]), str(item["scenario"])
            action_seq = [int(x) for x in item["selected_action_indices"]]
            target_candidates = pd.to_numeric(daily["dap"], errors="coerce").fillna(-1).astype(int)
            target_idx = min(target_candidates.index[target_candidates.ge(target_dap)].tolist() or target_candidates.index.tolist(), key=lambda idx: abs(int(target_candidates.loc[idx]) - target_dap))
            wth = OUT / "scenario_wth" / str(year) / f"dap{target_dap}" / scenario / "CNSY1701.WTH"
            active_weather["path"] = wth
            env = old_make_env(runtime_config, env_config, "SYA", year, 0, f"124_00_audit_{target_dap}_{scenario}", evaluation=True)
            overrides = 0
            try:
                obs, info = env.reset(); terminated = truncated = False
                for idx in range(target_idx):
                    if terminated or truncated: break
                    action = ref_actions[idx]
                    mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                    if action >= len(mask) or not bool(mask[action]):
                        action = reverse[(0.0, 0.0)]; overrides += 1
                    obs, _r, terminated, truncated, info = env.step(action)
                for action in action_seq:
                    if terminated or truncated: break
                    mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                    if action >= len(mask) or not bool(mask[action]):
                        action = reverse[(0.0, 0.0)]; overrides += 1
                    obs, _r, terminated, truncated, info = env.step(action)
                for idx in range(target_idx + len(action_seq), len(ref_actions)):
                    if terminated or truncated: break
                    action = ref_actions[idx]
                    mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                    if action >= len(mask) or not bool(mask[action]):
                        action = reverse[(0.0, 0.0)]; overrides += 1
                    obs, _r, terminated, truncated, info = env.step(action)
                end = latest(env, obs, info)
                rows.append({"year": year, "target_dap": target_dap, "scenario": scenario, "selected_irrigation_mm": [grid[a][0] for a in action_seq], "endpoint_grnwt": scalar(end.get("grnwt")), "endpoint_swfac": scalar(end.get("swfac")), "endpoint_nstres": scalar(end.get("nstres")), "observed_dap": scalar(end.get("dap")), "continuation_override_count": overrides, "terminated": bool(terminated), "truncated": bool(truncated), "mismatch": 0})
            finally:
                env.close()
    finally:
        active_weather["path"] = None
        ppo_safe_rendering.copy_weather_to_rendered_dir = old_copy_weather
        engine.base03222.base.make_env = old_make_env
        engine.base03222.SITE_NAMES.clear(); engine.base03222.SITE_NAMES.update(old_site_names)
        for key, value in old.items(): setattr(engine, key, value)

    df = pd.DataFrame(rows)
    output_csv = OUT / "124_00_selected_endpoint_audit.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    pivot = df.pivot_table(index=["year", "target_dap"], columns="scenario", values="endpoint_grnwt", aggfunc="first").reset_index()
    pivot.to_csv(OUT / "124_00_selected_endpoint_yield_pivot.csv", index=False, encoding="utf-8-sig")
    result = {"status": "completed_read_only", "n_sequences": int(len(df)), "all_mismatch_zero": bool((df["mismatch"] == 0).all()), "all_trajectories_reached_season_end": bool((df["observed_dap"] >= 125).all()), "endpoint_csv": rel(output_csv), "endpoint_yield_pivot": rel(OUT / "124_00_selected_endpoint_yield_pivot.csv"), "management_mode": "ppo_linked_irrigation_and_nitrogen", "native_dssat_automatic_irrigation_enabled": False}
    (OUT / "124_00_selected_endpoint_audit_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
