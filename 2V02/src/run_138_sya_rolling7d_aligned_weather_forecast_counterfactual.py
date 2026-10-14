"""138_00: aligned rolling-perfect-forecast DSSAT counterfactual for SY.

The existing SY seed-0 rolling-forecast checkpoints are evaluated without
training.  For each weather scenario, the rendered DSSAT WTH copy and the
forecast observation table are built from the same transformed WTH.  The
forecast window is recomputed after every one-day DSSAT step.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo


CFG = ROOT / "configs/138_00_sya_rolling7d_aligned_weather_forecast_counterfactual.json"
PROMPT = ROOT / "prompts/2026-08-16_sya_rolling7d_aligned_weather_forecast_counterfactual.md"
OUT = ROOT / "benchmark_results/138_00_sya_rolling7d_aligned_weather_forecast_counterfactual_rerun2"
DOC = ROOT / "docs/2026-08-16_sya_rolling7d_aligned_weather_forecast_execution.md"
SEED = 0
WTH_RE = re.compile(r"^\s*(\d{7})\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
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


def parse_wth(path: Path, station: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = WTH_RE.match(line)
        if not match:
            continue
        date_text, srad, tmax, tmin, rain = match.groups()
        year, doy = int(date_text[:4]), int(date_text[4:])
        date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        rows.append(
            {
                "station_code": station,
                "date": date,
                "srad": float(srad),
                "tmax": float(tmax),
                "tmin": float(tmin),
                "rain": float(rain),
            }
        )
    if not rows:
        raise RuntimeError(f"No weather rows parsed from {path}")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def summary_metrics(snapshot: Path, final_yield: float, total_i: float, total_n: float) -> dict[str, Any]:
    try:
        row, score, row_index = siteppo.summary_row(snapshot, final_yield, total_i, total_n)
        etcp = siteppo.num(row, "ETCP")
        ypem = siteppo.num(row, "YPEM")
        ypnam = siteppo.num(row, "YPNAM")
        nicm = float(siteppo.num(row, "NICM") or total_n)
        wp = float(ypem) * 0.1 if ypem is not None and float(ypem) >= 0 else (final_yield / float(etcp) / 10.0 if etcp and float(etcp) > 0 else math.nan)
        pfp = float(ypnam) if ypnam is not None and float(ypnam) >= 0 else (final_yield / nicm if nicm > 0 else math.nan)
        return {
            "summary_status": "ok",
            "summary_match_score": float(score),
            "summary_row_index": int(row_index),
            "etcp_mm": float(etcp) if etcp is not None else math.nan,
            "WP_ET_kg_m3": wp,
            "PFP_N_kg_kg": pfp,
            "summary_irrigation_total": float(siteppo.num(row, "IRCM") or math.nan),
            "summary_nitrogen_total": nicm,
        }
    except Exception as exc:
        return {
            "summary_status": "unavailable",
            "summary_error": str(exc),
            "summary_match_score": math.nan,
            "summary_row_index": -1,
            "etcp_mm": math.nan,
            "WP_ET_kg_m3": math.nan,
            "PFP_N_kg_kg": math.nan,
            "summary_irrigation_total": math.nan,
            "summary_nitrogen_total": math.nan,
        }


def preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    rolling.patch_contract()
    issues: list[str] = []
    input_root = forecast.INPUT_PROFILES[str(cfg.get("input_profile"))]
    if not input_root.exists():
        issues.append(f"input_root_missing:{rel(input_root)}")
    if not CFG.exists() or not PROMPT.exists():
        issues.append("manifest_or_prompt_missing")
    expected_i = [0.0, 15.0, 30.0, 45.0]
    expected_n = [0.0, 40.0, 80.0, 120.0]
    if list(map(float, cfg["actions"]["irrigation_levels_mm"])) != expected_i:
        issues.append("irrigation_grid_mismatch")
    if list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"])) != expected_n:
        issues.append("nitrogen_grid_mismatch")
    if cfg["scope"].get("external_n") or cfg["scope"].get("native_dssat_automatic_irrigation"):
        issues.append("automatic_or_external_n_must_be_false")
    if cfg["forecast_observation_contract"].get("include_today"):
        issues.append("forecast_must_exclude_today")
    for label, model in cfg.get("models", {}).items():
        if not (ROOT / str(model)).exists():
            issues.append(f"model_missing:{label}")
    for year in map(int, cfg.get("years", [])):
        path = forecast.ppo_safe_rendering.source_weather_path("SYA", year)
        if not path.exists():
            issues.append(f"source_weather_missing:{year}")
    return {
        "task": f"{cfg['task_id']}_{cfg['task_name']}",
        "models": dict(cfg["models"]),
        "years": list(map(int, cfg["years"])),
        "weather_scenarios": list(cfg["weather_scenarios"]),
        "forecast_mode": cfg["forecast_observation_contract"]["mode"],
        "external_n": bool(cfg["scope"]["external_n"]),
        "native_automatic_irrigation": bool(cfg["scope"]["native_dssat_automatic_irrigation"]),
        "issues": issues,
        "next_step_allowed": not issues,
    }


def run(cfg: dict[str, Any], pf: dict[str, Any], smoke: bool) -> dict[str, Any]:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {rel(OUT)}")
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CFG, OUT / CFG.name)
    shutil.copy2(PROMPT, OUT / PROMPT.name)

    model_items = list(cfg["models"].items())
    years = list(map(int, cfg["years"]))
    scenarios = list(cfg["weather_scenarios"])
    if smoke:
        model_items, years = model_items[:1], years[:1]

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
    old_copy_weather = forecast.ppo_safe_rendering.copy_weather_to_rendered_dir
    old_weather_table = forecast._weather_table
    old_raw_features = forecast.raw_forecast_features_for_date
    active: dict[str, Any] = {"scenario": "real", "year": None, "run_tag": "", "weather": None}
    weather_records: list[dict[str, Any]] = []
    forecast_records: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []

    def aligned_transform(path: Path, scenario: str) -> dict[str, Any]:
        record = rolling_base_transform(path, scenario)
        try:
            year = int(path.parent.parent.name)
        except Exception as exc:
            raise RuntimeError(f"Cannot infer year from rendered WTH path {path}") from exc
        active["year"] = year
        active["run_tag"] = path.parent.name
        active["weather"] = parse_wth(path, "SYA")
        source = forecast.ppo_safe_rendering.source_weather_path("SYA", year)
        record["source_weather_path"] = rel(source)
        record["source_weather_sha256"] = sha256(source)
        record["rendered_weather_sha256"] = sha256(path)
        record["forecast_weather_rain_total_mm"] = float(active["weather"]["rain"].sum())
        weather_records.append(record)
        return record

    rolling_base_transform = None

    def patched_copy_weather(station, year, template, config):
        rendered = old_copy_weather(station, year, template, config)
        aligned_transform(rendered, str(active["scenario"]))
        return rendered

    def patched_weather_table(config, station):
        if str(station) == "SYA" and active.get("weather") is not None:
            return active["weather"].copy()
        return old_weather_table(config, station)

    def patched_raw_features(weather, date, scales):
        values = old_raw_features(weather, date, scales)
        forecast_records.append(
            {
                "scenario": active.get("scenario"),
                "year": active.get("year"),
                "run_tag": active.get("run_tag"),
                "forecast_date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "future_window_start": (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "future_window_end": (pd.Timestamp(date) + pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
                **values,
            }
        )
        return values

    try:
        rolling.patch_contract()
        rolling_base_transform = getattr(__import__("run_137_sya_full_dssat_weather_state_counterfactual", fromlist=["transform_wth"]), "transform_wth")
        forecast.engine.load_config = lambda: tuned_load_config(old_loader)
        forecast.engine.BINARY_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
        forecast.engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
        forecast.engine.STATION, forecast.engine.SITES = "SYA", ["SYA"]
        forecast.engine.LOWIC_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        forecast.engine.base03222.OUT = OUT
        forecast.engine.base03222.base.StressAwareDiscreteWrapper = forecast.engine.base04036.LateIrrigationReserveMaskWrapper
        forecast.ppo_safe_rendering.MULTISITE_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        forecast.ppo_safe_rendering.copy_weather_to_rendered_dir = patched_copy_weather
        forecast._weather_table = patched_weather_table
        forecast.raw_forecast_features_for_date = patched_raw_features
        runtime_config = forecast.engine.load_config()
        selection = forecast.engine.base03222.build_selection(forecast.engine.base03222.load_split())
        env_config = forecast.direct_ppo.build_env_config(runtime_config, selection)
        base_forecast_cfg = forecast.read_json(ROOT / "configs/126_00_sya_originIC_rolling7d_perfect_forecast_5k10k.json")
        make_env = forecast.make_forecast_env_factory(old_make_env, base_forecast_cfg)
        probe = make_env(runtime_config, env_config, "SYA", years[0], SEED, "138_00_grid_probe", evaluation=True)
        try:
            grid = action_map(probe)
        finally:
            probe.close()

        for model_label, model_rel in model_items:
            model = MaskablePPO.load(str(ROOT / model_rel), device="cpu")
            for year in years:
                for scenario in scenarios:
                    active["scenario"] = scenario
                    active["weather"] = None
                    run_tag = f"138_00_{model_label}_{year}_{scenario}"
                    env = make_env(runtime_config, env_config, "SYA", year, SEED, run_tag, evaluation=True)
                    total_i = 0.0
                    total_n = 0.0
                    step = 0
                    terminated = truncated = False
                    last_obs = None
                    last_info: dict[str, Any] = {}
                    try:
                        obs, info = env.reset()
                        while not terminated and not truncated and step < int(runtime_config["runtime"]["max_steps"]):
                            before = forecast.direct_ppo.latest_observation_dict(env, obs, info or {})
                            dap = int(round(scalar(before.get("dap"), step + 1)))
                            mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                            action_arr, _ = model.predict(obs, action_masks=mask, deterministic=True)
                            action = int(np.asarray(action_arr).reshape(-1)[0])
                            if action < 0 or action >= len(mask) or not bool(mask[action]):
                                raise RuntimeError(f"invalid action {action} at {run_tag}/step{step}")
                            requested_i, requested_n = grid[action]
                            forecast_info = dict(getattr(env, "last_action_info", {}) or {})
                            obs_next, reward, terminated, truncated, info_next = env.step(action)
                            action_info = dict(getattr(env, "last_action_info", {}) or {})
                            safe_i = scalar(action_info.get("safe_action_amir"), np.nan)
                            safe_n = scalar(action_info.get("safe_action_anfer"), np.nan)
                            total_i += 0.0 if not np.isfinite(safe_i) else safe_i
                            total_n += 0.0 if not np.isfinite(safe_n) else safe_n
                            after = forecast.direct_ppo.latest_observation_dict(env, obs_next, info_next or {})
                            row: dict[str, Any] = {
                                "model": model_label,
                                "year": year,
                                "scenario": scenario,
                                "run_tag": run_tag,
                                "step": step,
                                "dap": dap,
                                "action_index": action,
                                "requested_i_mm": requested_i,
                                "requested_n_kg_ha": requested_n,
                                "action_legal": bool(mask[action]),
                                "safe_i_mm": safe_i,
                                "safe_n_kg_ha": safe_n,
                                "request_safe_mismatch": bool(np.isfinite(safe_i) and np.isfinite(safe_n) and (abs(safe_i - requested_i) > 1e-6 or abs(safe_n - requested_n) > 1e-6)),
                                "reward": float(reward),
                                "forecast_observation_date": forecast_info.get("forecast_observation_date", ""),
                                "forecast_window_start": forecast_info.get("forecast_observation_date", ""),
                                "swfac": scalar(after.get("swfac")),
                                "nstres": scalar(after.get("nstres")),
                                "grnwt": scalar(after.get("grnwt")),
                                "topwt": scalar(after.get("topwt")),
                                "cumulative_i_mm": total_i,
                                "cumulative_n_kg_ha": total_n,
                            }
                            for key, value in forecast_info.items():
                                if str(key).startswith("forecast_raw_") or str(key).startswith("forecast_obs_"):
                                    row[str(key)] = value
                            daily_rows.append(row)
                            obs, info = obs_next, info_next
                            last_obs, last_info = obs_next, dict(info_next or {})
                            step += 1
                        end = forecast.direct_ppo.latest_observation_dict(env, last_obs, last_info) if last_obs is not None else {}
                        final_yield = scalar(end.get("grnwt"), np.nan)
                        snapshot_rel = ""
                        metrics = {"summary_status": "unavailable", "WP_ET_kg_m3": math.nan, "PFP_N_kg_kg": math.nan, "etcp_mm": math.nan}
                        snapshot_dir = OUT / "snapshots" / model_label / str(year) / scenario
                        try:
                            tmp = siteppo.snapshot_from_env(env)
                            if snapshot_dir.exists():
                                raise FileExistsError(f"snapshot already exists: {snapshot_dir}")
                            snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copytree(tmp, snapshot_dir)
                            snapshot_rel = rel(snapshot_dir)
                            metrics = summary_metrics(snapshot_dir, final_yield, total_i, total_n)
                        except Exception as exc:
                            metrics["summary_error"] = str(exc)
                        trajectory = [r for r in daily_rows if r["run_tag"] == run_tag]
                        endpoint_rows.append(
                            {
                                "model": model_label,
                                "year": year,
                                "scenario": scenario,
                                "steps": step,
                                "terminated": bool(terminated),
                                "truncated": bool(truncated),
                                "endpoint_grnwt": final_yield,
                                "total_irrigation_mm": total_i,
                                "total_n_kg_ha": total_n,
                                "action_signature": "|".join(str(int(r["action_index"])) for r in trajectory),
                                "distinct_actions": int(len(set(int(r["action_index"]) for r in trajectory))),
                                "all_actions_legal": bool(all(r["action_legal"] for r in trajectory)),
                                "request_safe_mismatch_count": int(sum(bool(r["request_safe_mismatch"]) for r in trajectory)),
                                "snapshot_path": snapshot_rel,
                                **metrics,
                            }
                        )
                    finally:
                        env.close()

        daily = pd.DataFrame(daily_rows)
        endpoints = pd.DataFrame(endpoint_rows)
        forecast_audit = pd.DataFrame(forecast_records)
        daily_path = OUT / "138_00_daily_trajectories.csv"
        endpoint_path = OUT / "138_00_endpoints.csv"
        forecast_path = OUT / "138_00_forecast_window_audit.csv"
        weather_path = OUT / "138_00_weather_provenance.json"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        endpoints.to_csv(endpoint_path, index=False, encoding="utf-8-sig")
        forecast_audit.to_csv(forecast_path, index=False, encoding="utf-8-sig")
        weather_path.write_text(json.dumps(weather_records, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")

        comparisons: list[dict[str, Any]] = []
        if not daily.empty:
            base = daily[daily["scenario"].eq("real")]
            for scenario in ["rain_half", "rain_double"]:
                pert = daily[daily["scenario"].eq(scenario)]
                merged = base.merge(pert, on=["model", "year", "step", "dap"], suffixes=("_real", "_pert"), how="inner")
                if merged.empty:
                    continue
                changed = merged["action_index_real"] != merged["action_index_pert"]
                forecast_delta = np.abs(pd.to_numeric(merged.get("forecast_raw_rain_d1_mm_pert"), errors="coerce") - pd.to_numeric(merged.get("forecast_raw_rain_d1_mm_real"), errors="coerce"))
                comparisons.append(
                    {
                        "scenario": scenario,
                        "model_year_pairs": int(merged[["model", "year"]].drop_duplicates().shape[0]),
                        "matched_steps": int(len(merged)),
                        "forecast_d1_rain_mean_abs_delta_mm": float(np.nanmean(forecast_delta)),
                        "action_changed_rows": int(changed.sum()),
                        "action_change_rate": float(changed.mean()),
                        "action_change_rate_after_dap1": float(changed.loc[merged["dap"] > 1].mean()),
                        "mean_abs_swfac_delta": float(np.nanmean(np.abs(pd.to_numeric(merged["swfac_pert"], errors="coerce") - pd.to_numeric(merged["swfac_real"], errors="coerce")))),
                        "max_abs_grnwt_delta": float(np.nanmax(np.abs(pd.to_numeric(merged["grnwt_pert"], errors="coerce") - pd.to_numeric(merged["grnwt_real"], errors="coerce")))),
                        "all_perturbed_actions_legal": bool(pert["action_legal"].all()),
                        "perturbed_request_safe_mismatch_count": int(pert["request_safe_mismatch"].sum()),
                    }
                )

        summary = {
            "status": "completed_read_only_aligned_rolling_forecast_counterfactual",
            "mode": "smoke" if smoke else "full",
            "preflight": pf,
            "config": rel(CFG),
            "prompt": rel(PROMPT),
            "daily_csv": rel(daily_path),
            "endpoint_csv": rel(endpoint_path),
            "forecast_window_audit_csv": rel(forecast_path),
            "weather_provenance_json": rel(weather_path),
            "forecast_recomputed_each_step": True,
            "future_window_excludes_today": True,
            "forecast_and_dssat_weather_same_rendered_wth": True,
            "source_weather_modified": False,
            "training_executed": False,
            "native_dssat_automatic_irrigation_enabled": False,
            "external_n_enabled": False,
            "comparisons": comparisons,
            "next_step_full_run_allowed": bool(smoke and not daily.empty and all(x["all_perturbed_actions_legal"] and x["perturbed_request_safe_mismatch_count"] == 0 for x in comparisons)),
        }
        (OUT / "138_00_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
        DOC.write_text(
            "\n".join(
                [
                    "# 138_00 SY 滚动 7 天完美天气预报执行记录",
                    "",
                    "本轮只读评估把 DSSAT rendered WTH 与 forecast observation 绑定到同一情景，并在每个 DSSAT 日步重新计算 date+1 到 date+7 的天气窗口。没有训练，没有 native automatic irrigation，也没有 `dssat_auto_external_n`。",
                    "",
                    "## 对照结果",
                    "",
                    pd.DataFrame(comparisons).to_markdown(index=False) if comparisons else "没有形成可比较记录。",
                    "",
                    "## 输出",
                    "",
                    f"- 日轨迹：`{rel(daily_path)}`",
                    f"- 终点指标：`{rel(endpoint_path)}`",
                    f"- 滚动窗口审计：`{rel(forecast_path)}`",
                    f"- WTH provenance：`{rel(weather_path)}`",
                    f"- 汇总：`{rel(OUT / '138_00_summary.json')}`",
                    "",
                    "WP_ET 只有在 Summary.OUT/ETCP 有效时才报告；缺失时保持 unavailable，不从不完整的逐日 CSV 推算。",
                ]
            ),
            encoding="utf-8",
        )
        summary["record_md"] = rel(DOC)
        (OUT / "138_00_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
        return summary
    finally:
        forecast.ppo_safe_rendering.copy_weather_to_rendered_dir = old_copy_weather
        forecast._weather_table = old_weather_table
        forecast.raw_forecast_features_for_date = old_raw_features
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    pf = preflight(cfg)
    if args.dry_run:
        print(json.dumps({**pf, "mode": "dry_run", "output_root": rel(OUT)}, ensure_ascii=False, indent=2))
        return 0 if pf["next_step_allowed"] else 2
    if not args.smoke and not args.full:
        raise SystemExit("choose --smoke or --full; no implicit long run")
    if not pf["next_step_allowed"]:
        raise RuntimeError("preflight failed: " + "; ".join(pf["issues"]))
    result = run(cfg, pf, smoke=bool(args.smoke))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
