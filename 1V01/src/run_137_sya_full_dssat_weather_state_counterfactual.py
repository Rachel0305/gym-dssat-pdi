"""137_00: read-only full-DSSAT weather-state counterfactual for SY.

This runner never trains. It patches only the rendered WTH copy created for a
single DSSAT trajectory, then evaluates existing rolling-7-day checkpoints.
The forecast observation table remains on the historical control weather so
the measured action changes isolate the indirect DSSAT-state path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import forecast_engineered_observation_056_057 as forecast
import run_111_112_sy_lc_rolling7d_perfect_forecast_smoke2k as rolling


CFG = ROOT / "configs/137_00_sya_full_dssat_weather_state_counterfactual.json"
PROMPT = ROOT / "prompts/2026-08-16_sya_full_dssat_weather_state_counterfactual.md"
# The first launch stopped before DSSAT because the compact manifest lacked
# the forecast feature contract. Preserve that failed preflight directory and
# write the corrected, isolated run to a new root.
OUT = ROOT / "benchmark_results/137_00_sya_full_dssat_weather_state_counterfactual_rerun2"
SEED = 0
WEATHER_DATA_RE = re.compile(r"^\s*(\d{7})\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: Any, default: float = np.nan) -> float:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        return float(arr[0]) if len(arr) else float(default)
    except Exception:
        return float(default)


def transform_wth(path: Path, scenario: str) -> dict[str, Any]:
    """Modify a rendered copy in place and return a small provenance record."""
    original = path.read_text(encoding="utf-8", errors="replace")
    lines = original.splitlines(keepends=True)
    changed_rows = 0
    rain_before = 0.0
    rain_after = 0.0
    out_lines: list[str] = []
    for line in lines:
        match = WEATHER_DATA_RE.match(line)
        if not match:
            out_lines.append(line)
            continue
        date_text, srad_text, tmax_text, tmin_text, rain_text = match.groups()
        srad, tmax, tmin, rain = map(float, (srad_text, tmax_text, tmin_text, rain_text))
        new_rain = rain
        if scenario == "rain_half":
            new_rain = max(0.0, rain * 0.5)
        elif scenario == "rain_double":
            new_rain = max(0.0, rain * 2.0)
        elif scenario != "real":
            raise ValueError(f"unknown weather scenario: {scenario}")
        rain_before += rain
        rain_after += new_rain
        if abs(new_rain - rain) > 1e-12:
            changed_rows += 1
        # DSSAT WTH uses fixed columns: 7-char DATE followed by four 6-char
        # numeric fields. Preserve those columns; whitespace reformatting with
        # a different width silently shifts RAIN and invalidates the run.
        newline = "\n" if line.endswith("\n") else ""
        out_lines.append(f"{date_text:>7s}{srad:6.1f}{tmax:6.1f}{tmin:6.1f}{new_rain:6.1f}{newline}")
    if scenario != "real" and changed_rows == 0:
        raise RuntimeError(f"weather transform changed no rows: {path} {scenario}")
    if scenario != "real":
        path.write_text("".join(out_lines), encoding="utf-8")
    return {
        "scenario": scenario,
        "weather_path": rel(path),
        "weather_sha256": sha256_file(path),
        "changed_weather_rows": int(changed_rows),
        "rain_before_mm": float(rain_before),
        "rain_after_mm": float(rain_after),
    }


def action_map(env) -> dict[int, tuple[float, float]]:
    return {
        int(index): (float(row.get("amir", 0.0)), float(row.get("anfer", 0.0)))
        for index, row in enumerate(getattr(env, "grid"))
    }


def tuned_load_config(original_loader):
    cfg = copy.deepcopy(original_loader())
    cfg["discrete_actions"]["irrigation_levels"] = [0.0, 15.0, 30.0, 45.0]
    cfg["discrete_actions"]["nitrogen_levels"] = [0.0, 40.0, 80.0, 120.0]
    return cfg


def preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    rolling.patch_contract()
    issues: list[str] = []
    if not CFG.exists():
        issues.append(f"config missing: {rel(CFG)}")
    if not PROMPT.exists():
        issues.append(f"prompt missing: {rel(PROMPT)}")
    input_root = forecast.INPUT_PROFILES[str(cfg.get("input_profile"))]
    if not input_root.exists():
        issues.append(f"input root missing: {rel(input_root)}")
    for label, model in cfg.get("models", {}).items():
        path = ROOT / str(model)
        if not path.exists():
            issues.append(f"model missing {label}: {rel(path)}")
    for year in map(int, cfg.get("years", [])):
        try:
            path = forecast.ppo_safe_rendering.source_weather_path("SYA", year)
            if not path.exists():
                issues.append(f"weather missing {year}: {rel(path)}")
        except Exception as exc:
            issues.append(f"weather lookup failed {year}: {exc}")
    scenarios = list(cfg.get("weather_scenarios", []))
    if scenarios != ["real", "rain_half", "rain_double"]:
        issues.append(f"unexpected weather scenarios: {scenarios}")
    return {
        "task": f"{cfg.get('task_id')}_{cfg.get('task_name')}",
        "input_profile": str(cfg.get("input_profile")),
        "models": dict(cfg.get("models", {})),
        "years": list(map(int, cfg.get("years", []))),
        "weather_scenarios": scenarios,
        "external_n": bool(cfg.get("scope", {}).get("external_n", True)),
        "raw_weather_files_modified": False,
        "issues": issues,
        "next_step_allowed": not issues,
    }


def run(cfg: dict[str, Any], pf: dict[str, Any]) -> dict[str, Any]:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {rel(OUT)}")
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CFG, OUT / CFG.name)
    shutil.copy2(PROMPT, OUT / PROMPT.name)

    old_loader = forecast.engine.load_config
    old_make_env = forecast.engine.base03222.base.make_env
    old_class = forecast.engine.base03222.base.StressAwareDiscreteWrapper
    old_root = forecast.ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_i = list(forecast.engine.BINARY_IRRIGATION_LEVELS)
    old_n = list(forecast.engine.BINARY_NITROGEN_LEVELS)
    old_station = forecast.engine.STATION
    old_sites = list(forecast.engine.SITES)
    old_input = forecast.engine.LOWIC_INPUT_ROOT
    old_base_out = forecast.engine.base03222.OUT
    active = {"scenario": "real", "wth_records": []}
    all_wth_records: list[dict[str, Any]] = []

    def patched_copy_weather(station, year, template, config):
        rendered = old_copy_weather(station, year, template, config)
        record = transform_wth(rendered, str(active["scenario"]))
        active["wth_records"].append(record)
        all_wth_records.append(record)
        return rendered

    old_copy_weather = forecast.ppo_safe_rendering.copy_weather_to_rendered_dir
    rows: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []
    try:
        forecast.engine.load_config = lambda: tuned_load_config(old_loader)
        forecast.engine.BINARY_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
        forecast.engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
        forecast.engine.STATION, forecast.engine.SITES = "SYA", ["SYA"]
        forecast.engine.LOWIC_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        forecast.engine.base03222.OUT = OUT
        forecast.engine.base03222.base.StressAwareDiscreteWrapper = forecast.engine.base04036.LateIrrigationReserveMaskWrapper
        forecast.ppo_safe_rendering.MULTISITE_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        forecast.ppo_safe_rendering.copy_weather_to_rendered_dir = patched_copy_weather
        runtime_config = forecast.engine.load_config()
        selection = forecast.engine.base03222.build_selection(forecast.engine.base03222.load_split())
        env_config = forecast.direct_ppo.build_env_config(runtime_config, selection)
        # Reuse the complete, already validated 126 rolling-7d observation
        # contract for model-compatible env construction. The 137 config is
        # intentionally a compact experiment manifest, not a second model
        # contract, so it does not duplicate the 28 feature names/scales.
        base_forecast_cfg = forecast.read_json(
            ROOT / "configs/126_00_sya_originIC_rolling7d_perfect_forecast_5k10k.json"
        )
        make_env = forecast.make_forecast_env_factory(old_make_env, base_forecast_cfg)
        grid_probe = make_env(runtime_config, env_config, "SYA", int(cfg["years"][0]), SEED, "137_00_grid_probe", evaluation=True)
        try:
            grid = action_map(grid_probe)
        finally:
            grid_probe.close()

        for model_label, model_rel in cfg["models"].items():
            model_path = ROOT / str(model_rel)
            model = MaskablePPO.load(str(model_path), device="cpu")
            for year in map(int, cfg["years"]):
                for scenario in cfg["weather_scenarios"]:
                    active["scenario"] = scenario
                    active["wth_records"] = []
                    env = make_env(runtime_config, env_config, "SYA", year, SEED, f"137_00_{model_label}_{year}_{scenario}", evaluation=True)
                    total_i = 0.0
                    total_n = 0.0
                    step = 0
                    terminated = truncated = False
                    last_obs = None
                    last_info: dict[str, Any] = {}
                    try:
                        obs, info = env.reset()
                        while not terminated and not truncated and step < int(runtime_config["runtime"]["max_steps"]):
                            latest_before = forecast.direct_ppo.latest_observation_dict(env, obs, info or {})
                            dap = int(round(scalar(latest_before.get("dap"), step + 1)))
                            mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                            action_arr, _ = model.predict(obs, action_masks=mask, deterministic=True)
                            action = int(np.asarray(action_arr).reshape(-1)[0])
                            if action < 0 or action >= len(mask) or not bool(mask[action]):
                                raise RuntimeError(f"invalid model action {action} at {model_label}/{year}/{scenario}/step{step}")
                            requested_i, requested_n = grid[action]
                            obs_next, _reward, terminated, truncated, info_next = env.step(action)
                            action_info = dict(getattr(env, "last_action_info", {}) or {})
                            safe_i = scalar(action_info.get("safe_action_amir"), np.nan)
                            safe_n = scalar(action_info.get("safe_action_anfer"), np.nan)
                            total_i += 0.0 if not np.isfinite(safe_i) else safe_i
                            total_n += 0.0 if not np.isfinite(safe_n) else safe_n
                            latest_after = forecast.direct_ppo.latest_observation_dict(env, obs_next, info_next or {})
                            rows.append(
                                {
                                    "model": model_label,
                                    "year": year,
                                    "scenario": scenario,
                                    "step": step,
                                    "dap": dap,
                                    "action_index": action,
                                    "requested_i_mm": requested_i,
                                    "requested_n_kg_ha": requested_n,
                                    "action_legal": bool(mask[action]),
                                    "safe_i_mm": safe_i,
                                    "safe_n_kg_ha": safe_n,
                                    "request_safe_mismatch": bool(np.isfinite(safe_i) and np.isfinite(safe_n) and (abs(safe_i - requested_i) > 1e-6 or abs(safe_n - requested_n) > 1e-6)),
                                    "dap_after": scalar(latest_after.get("dap")),
                                    "swfac_before": scalar(latest_before.get("swfac")),
                                    "nstres_before": scalar(latest_before.get("nstres")),
                                    "grnwt_before": scalar(latest_before.get("grnwt")),
                                    "topwt_before": scalar(latest_before.get("topwt")),
                                    "swfac_after": scalar(latest_after.get("swfac")),
                                    "nstres_after": scalar(latest_after.get("nstres")),
                                    "grnwt_after": scalar(latest_after.get("grnwt")),
                                    "topwt_after": scalar(latest_after.get("topwt")),
                                    "cumulative_i_mm": total_i,
                                    "cumulative_n_kg_ha": total_n,
                                }
                            )
                            obs, info = obs_next, info_next
                            last_obs, last_info = obs_next, dict(info_next or {})
                            step += 1
                        end = forecast.direct_ppo.latest_observation_dict(env, last_obs, last_info) if last_obs is not None else {}
                        endpoints.append(
                            {
                                "model": model_label,
                                "year": year,
                                "scenario": scenario,
                                "steps": step,
                                "terminated": bool(terminated),
                                "truncated": bool(truncated),
                                "endpoint_dap": scalar(end.get("dap")),
                                "endpoint_grnwt": scalar(end.get("grnwt")),
                                "endpoint_topwt": scalar(end.get("topwt")),
                                "endpoint_swfac": scalar(end.get("swfac")),
                                "endpoint_nstres": scalar(end.get("nstres")),
                                "total_irrigation_mm": total_i,
                                "total_n_kg_ha": total_n,
                                "action_signature": "|".join(str(int(x)) for x in [r["action_index"] for r in rows if r["model"] == model_label and r["year"] == year and r["scenario"] == scenario]),
                                "all_actions_legal": all(bool(r["action_legal"]) for r in rows if r["model"] == model_label and r["year"] == year and r["scenario"] == scenario),
                                "request_safe_mismatch_count": int(sum(bool(r["request_safe_mismatch"]) for r in rows if r["model"] == model_label and r["year"] == year and r["scenario"] == scenario)),
                                "wth_records": active["wth_records"],
                            }
                        )
                    finally:
                        env.close()

        daily = pd.DataFrame(rows)
        endpoint_df = pd.DataFrame(endpoints)
        daily_path = OUT / "137_00_daily_trajectories.csv"
        endpoint_path = OUT / "137_00_endpoints.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        endpoint_df.to_csv(endpoint_path, index=False, encoding="utf-8-sig")

        base = daily[daily["scenario"].eq("real")].copy()
        comparisons: list[dict[str, Any]] = []
        for scenario in ["rain_half", "rain_double"]:
            pert = daily[daily["scenario"].eq(scenario)].copy()
            merged = base.merge(pert, on=["model", "year", "step", "dap"], suffixes=("_real", "_pert"), how="inner")
            if merged.empty:
                continue
            changed = merged["action_index_real"] != merged["action_index_pert"]
            comparisons.append(
                {
                    "scenario": scenario,
                    "model_year_pairs": int(merged[["model", "year"]].drop_duplicates().shape[0]),
                    "matched_steps": int(len(merged)),
                    "action_changed_rows": int(changed.sum()),
                    "action_change_rate": float(changed.mean()),
                    "action_change_rate_after_dap1": float(changed.loc[merged["dap"] > 1].mean()),
                    "mean_abs_swfac_delta": float(np.nanmean(np.abs(merged["swfac_after_pert"] - merged["swfac_after_real"]))),
                    "max_abs_swfac_delta": float(np.nanmax(np.abs(merged["swfac_after_pert"] - merged["swfac_after_real"]))),
                    "mean_abs_nstres_delta": float(np.nanmean(np.abs(merged["nstres_after_pert"] - merged["nstres_after_real"]))),
                    "max_abs_grnwt_delta": float(np.nanmax(np.abs(merged["grnwt_after_pert"] - merged["grnwt_after_real"]))),
                    "all_perturbed_actions_legal": bool(pert["action_legal"].all()),
                    "perturbed_request_safe_mismatch_count": int(pert["request_safe_mismatch"].sum()),
                }
            )
        comparison_df = pd.DataFrame(comparisons)
        comparison_path = OUT / "137_00_weather_state_action_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
        provenance_path = OUT / "137_00_rendered_weather_provenance.json"
        provenance_path.write_text(json.dumps(all_wth_records, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = {
            "status": "completed_read_only_full_dssat_weather_state_counterfactual",
            "preflight": pf,
            "config": rel(CFG),
            "prompt": rel(PROMPT),
            "daily_csv": rel(daily_path),
            "endpoint_csv": rel(endpoint_path),
            "comparison_csv": rel(comparison_path),
            "weather_provenance_json": rel(provenance_path),
            "models": cfg["models"],
            "years": cfg["years"],
            "weather_scenarios": cfg["weather_scenarios"],
            "forecast_observation_scope": cfg["forecast_observation_scope"],
            "training_executed": False,
            "source_weather_modified": False,
            "counterfactual_actions_executed_in_dssat": True,
            "comparison": comparisons,
            "next_step_long_training_allowed": False,
        }
        (OUT / "137_00_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        doc = ROOT / "docs/2026-08-16_sya_full_dssat_weather_state_counterfactual_execution.md"
        doc.write_text(
            "\n".join(
                [
                    "# 137_00 SY full-DSSAT 天气状态反事实",
                    "",
                    "## 实验边界",
                    "",
                    "- 只读取 126_00 已训练的 SY seed=0 2K、10K 模型；未训练、未继续训练。",
                    "- 只改变每条轨迹的 rendered DSSAT WTH 副本；源 WTH 不变。",
                    "- 未来 7 天观测仍使用真实历史天气表，目的是单独观察天气经 DSSAT 状态传递后的间接动作响应。",
                    "- 未使用 `dssat_auto_external_n`，也没有把 native automatic irrigation 作为 PPO 动作。",
                    "",
                    "## 结果",
                    "",
                    comparison_df.to_markdown(index=False) if not comparison_df.empty else "无可比较记录。",
                    "",
                    "## 输出",
                    "",
                    f"- 日轨迹：`{rel(daily_path)}`",
                    f"- 终点：`{rel(endpoint_path)}`",
                    f"- 天气状态与动作对比：`{rel(comparison_path)}`",
                    f"- 渲染 WTH provenance：`{rel(provenance_path)}`",
                    "",
                    "本轮仅用于判断 DSSAT 状态是否变化以及 PPO 是否把状态变化转为动作变化；不据此新增产量、WP_ET、PFP_N 或全面优越性结论。WP_ET 未从缺少 ET 分母的输出中推断。",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        summary["record_md"] = rel(doc)
        (OUT / "137_00_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    finally:
        forecast.ppo_safe_rendering.copy_weather_to_rendered_dir = old_copy_weather
        forecast.engine.load_config = old_loader
        forecast.engine.base03222.base.make_env = old_make_env
        forecast.engine.base03222.base.StressAwareDiscreteWrapper = old_class
        forecast.ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
        forecast.engine.BINARY_IRRIGATION_LEVELS = old_i
        forecast.engine.BINARY_NITROGEN_LEVELS = old_n
        forecast.engine.STATION, forecast.engine.SITES = old_station, old_sites
        forecast.engine.LOWIC_INPUT_ROOT = old_input
        forecast.engine.base03222.OUT = old_base_out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = forecast.read_json(CFG)
    pf = preflight(cfg)
    if args.dry_run:
        print(json.dumps({**pf, "mode": "dry_run", "output_root": rel(OUT)}, ensure_ascii=False, indent=2))
        return 0 if pf["next_step_allowed"] else 2
    if not pf["next_step_allowed"]:
        raise RuntimeError("preflight failed: " + "; ".join(pf["issues"]))
    result = run(cfg, pf)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
