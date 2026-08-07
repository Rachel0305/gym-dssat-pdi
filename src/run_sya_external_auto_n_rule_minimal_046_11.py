"""046_11: minimal external auto-N for SYA originIC.

This keeps DSSAT native automatic irrigation, but uses the gym-DSSAT action
channel for nitrogen because DSSAT automatic fertilizer is not operational in
this workflow.  The nitrogen rule is intentionally minimal:

    apply nitrogen_dose_kg_ha whenever pre-action NSTRES >= threshold

There is no DAP cutoff, no minimum interval, and no seasonal nitrogen cap.
Outputs keep the 046_04 CSV filenames so existing 046_05/046_06 reporting code
can consume this branch by selecting its output_suffix.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_sya_external_auto_n_rule_046_04 as base04604


DEFAULT_CONFIG = ROOT / "configs" / "046_11_sya_originIC_external_auto_n_rule_nstd050_minimal.json"
PROMPT = ROOT / "prompts" / "046_11_sya_external_auto_n_rule_minimal.md"
SCENARIO = base04604.SCENARIO


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if str(cfg.get("station_code")) != "SYA":
        raise ValueError("046_11 only supports SYA")
    if str(cfg.get("input_profile")) not in base04604.INPUT_PROFILES:
        raise ValueError("input_profile must be originIC or lowIC")
    rule = cfg.get("external_auto_n_rule", {})
    for key in ["nitrogen_stress_threshold", "nitrogen_dose_kg_ha"]:
        if key not in rule:
            raise ValueError(f"external_auto_n_rule missing {key}")
    forbidden = ["min_days_between_nitrogen", "season_nitrogen_cap_kg_ha", "fertilization_last_dap"]
    present = [key for key in forbidden if key in rule]
    if present:
        raise ValueError(f"046_11 minimal auto-N must not include removed constraints: {present}")
    return cfg


def output_root(cfg: dict[str, Any], run_id: str = "") -> Path:
    return base04604.output_root(cfg, run_id)


def evaluate_year(
    run_config: dict[str, Any],
    env_config: dict[str, Any],
    year: int,
    rule: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    env, env_args, rendered_text = base04604.make_env_auto_irrigation_external_n(
        env_config,
        year,
        f"SYA_{year}_046_11_external_auto_n_minimal",
    )
    rows: list[dict[str, Any]] = []
    total_requested_n = 0.0
    event_count = 0
    planting = pd.Timestamp(base04604.direct_ppo.find_year(env_config, "SYA", int(year))["planting_date"])
    weather = base04604.direct_ppo.weather_for_daily(run_config)
    threshold = float(rule["nitrogen_stress_threshold"])
    dose = float(rule["nitrogen_dose_kg_ha"])
    try:
        obs, info = env.reset()
        done, steps = False, 0
        while not done and steps < 260:
            latest_pre = base04604.baseline.latest_observation_dict(env, obs, info)
            dap = int(round(float(base04604.baseline.scalar(latest_pre.get("dap", steps + 1), steps + 1))))
            nstres_pre = float(base04604.baseline.scalar(latest_pre.get("nstres"), 0.0))
            amount = dose if nstres_pre >= threshold else 0.0
            action = base04604.baseline.step_scheduled_action(env, {"amir": 0.0, "anfer": amount})
            obs, reward, terminated, truncated, info = env.step(action)
            if amount > 0:
                total_requested_n += amount
                event_count += 1
            latest = base04604.baseline.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            wrow = base04604.baseline.weather_row(weather, "SYA", date)
            rows.append({
                "station_code": "SYA",
                "site": "SY",
                "year": int(year),
                "scenario": SCENARIO,
                "date": date.strftime("%Y-%m-%d"),
                "dap": dap,
                "rain": base04604.baseline.scalar(wrow.get("rain"), np.nan),
                "tmax": base04604.baseline.scalar(wrow.get("tmax"), np.nan),
                "tmin": base04604.baseline.scalar(wrow.get("tmin"), np.nan),
                "srad": base04604.baseline.scalar(wrow.get("srad"), np.nan),
                "grnwt": base04604.baseline.scalar(latest.get("grnwt")),
                "topwt": base04604.baseline.scalar(latest.get("topwt")),
                "swfac": base04604.baseline.scalar(latest.get("swfac")),
                "nstres": base04604.baseline.scalar(latest.get("nstres")),
                "nstres_pre_action": nstres_pre,
                "nitrogen_requested_kg_ha": amount,
                "irrigation_requested_mm": 0.0,
                "external_action_note": "minimal_NSTRES_rule_external_N;DSSAT_native_auto_irrigation",
                "reward": float(reward),
                "done": bool(terminated or truncated),
            })
            done = bool(terminated or truncated)
            steps += 1

        daily = pd.DataFrame(rows)
        if daily.empty or not bool(daily["done"].iloc[-1]):
            raise RuntimeError(f"SYA{year} did not finish within 260 steps")
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        final_b = float(pd.to_numeric(daily["topwt"], errors="coerce").iloc[-1])
        snapshot_tmp = base04604.baseline.siteppo.snapshot_from_env(env)
        snapshot = base04604.baseline.OUT / "snapshots" / "SYA" / str(year) / SCENARIO
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(snapshot_tmp, snapshot)
        metrics = base04604.baseline.metrics_from_snapshot(snapshot, final_y)
        summary_n = float(metrics["actual_nitrogen_kg_ha"])
        mgmt_event_n = base04604.executed_nitrogen_from_mgmt_event(snapshot)
        actual_n = mgmt_event_n if mgmt_event_n > 1e-9 else summary_n
        pfp_n = final_y / actual_n if actual_n > 1e-9 else np.nan
        n_closure_gap = actual_n - total_requested_n
        summary = {
            "station_code": "SYA",
            "site": "SY",
            "year": int(year),
            "scenario": SCENARIO,
            "grain_yield_kg_ha": final_y,
            "biomass_kg_ha": final_b,
            "actual_irrigation_mm": float(metrics["actual_irrigation_mm"]),
            "actual_nitrogen_kg_ha": actual_n,
            "summary_nitrogen_kg_ha": summary_n,
            "mgmt_event_nitrogen_kg_ha": mgmt_event_n,
            "requested_nitrogen_kg_ha": total_requested_n,
            "external_n_event_count": event_count,
            "external_n_closure_gap_kg_ha": n_closure_gap,
            "external_n_closure_status": "ok" if abs(n_closure_gap) <= 1.0 else "requested_vs_mgmt_mismatch",
            "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
            "PFP_N_kg_kg": pfp_n,
            "max_water_stress": float(pd.to_numeric(daily["swfac"], errors="coerce").max()),
            "max_nitrogen_stress": float(pd.to_numeric(daily["nstres"], errors="coerce").max()),
            "input_profile": str(env_config.get("input_profile", "")),
            "snapshot_path": rel(snapshot),
            "minimal_rule_threshold": threshold,
            "minimal_rule_dose_kg_ha": dose,
        }
        rendered = {
            "management_mode_A_L_present": " 1 MA              R     A     L     R     M" in rendered_text,
            "rendered_template_path": rel(Path(env_args["fileX_template_path"])),
        }
        return daily, summary, rendered
    finally:
        env.close()


def run(cfg_path: Path, dry_run: bool, years_override: list[int] | None = None, run_id: str = "") -> dict[str, Any]:
    cfg = read_config(cfg_path)
    if years_override:
        cfg = json.loads(json.dumps(cfg))
        cfg["scope"]["validation_years"] = years_override
    input_root = base04604.INPUT_PROFILES[str(cfg["input_profile"])]
    years = list(map(int, cfg["scope"]["validation_years"]))
    rule = dict(cfg["external_auto_n_rule"])
    out = output_root(cfg, run_id)
    pf = {
        "input_profile": cfg["input_profile"],
        "resolved_input_root": rel(input_root),
        "years": years,
        "rule": rule,
        "removed_constraints": ["DAP cutoff", "minimum N interval", "seasonal N cap"],
        "next_step_allowed": input_root.exists(),
        "output_root": rel(out),
    }
    if dry_run:
        return {"task": "046_11_sya_external_auto_n_rule_minimal", "mode": "dry_run", **pf}
    if not pf["next_step_allowed"]:
        raise RuntimeError("input root missing")
    for sub in ["configs", "evaluation", "snapshots"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, out / "configs" / cfg_path.name)
    if PROMPT.exists():
        shutil.copy2(PROMPT, out / "configs" / PROMPT.name)
    selected = base04604.selected_rows(years)
    run_config, env_config = base04604.build_configs(out, selected)
    env_config["input_profile"] = str(cfg["input_profile"])
    base04604.direct_ppo.write_yaml(env_config, out / "configs" / "046_11_resolved_env_config.yaml")

    originals = (
        base04604.baseline.MULTISITE_INPUT_ROOT,
        base04604.baseline.OUT,
        base04604.baseline.TASK_ID,
        base04604.baseline.MAX_STEPS,
        base04604.ppo_safe_rendering.MULTISITE_INPUT_ROOT,
    )
    base04604.baseline.MULTISITE_INPUT_ROOT = input_root
    base04604.baseline.OUT = out
    base04604.baseline.TASK_ID = "046_11"
    base04604.baseline.MAX_STEPS = 260
    base04604.ppo_safe_rendering.MULTISITE_INPUT_ROOT = input_root
    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    rendered_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    started = time.time()
    try:
        for year in years:
            print(f"[046_11] SYA{year} {SCENARIO}", flush=True)
            try:
                daily, summary, rendered = evaluate_year(run_config, env_config, year, rule)
                daily_frames.append(daily)
                summary_rows.append(summary)
                rendered_rows.append({"year": year, **rendered})
            except Exception as exc:
                failure_rows.append({"year": year, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]})
    finally:
        (
            base04604.baseline.MULTISITE_INPUT_ROOT,
            base04604.baseline.OUT,
            base04604.baseline.TASK_ID,
            base04604.baseline.MAX_STEPS,
            base04604.ppo_safe_rendering.MULTISITE_INPUT_ROOT,
        ) = originals

    summary = pd.DataFrame(summary_rows)
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    rendered_df = pd.DataFrame(rendered_rows)
    failures = pd.DataFrame(failure_rows)
    summary.to_csv(out / "evaluation" / "046_04_external_auto_n_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(out / "evaluation" / "046_04_external_auto_n_daily.csv", index=False, encoding="utf-8-sig")
    rendered_df.to_csv(out / "evaluation" / "046_11_rendered_management_mode_audit.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(out / "evaluation" / "046_11_failures.csv", index=False, encoding="utf-8-sig")
    record = ROOT / "docs" / f"046_11_sya_{cfg['input_profile']}_external_auto_n_rule_minimal_record.md"
    record.write_text("\n".join([
        f"# 046_11 SYA {cfg['input_profile']} minimal external auto-N record",
        "",
        "- Irrigation: DSSAT native automatic irrigation.",
        "- Nitrogen: external gym-DSSAT action channel.",
        f"- Rule: apply {rule['nitrogen_dose_kg_ha']} kg/ha whenever pre-action NSTRES >= {rule['nitrogen_stress_threshold']}.",
        "- Removed constraints: no DAP cutoff, no minimum interval, no seasonal N cap.",
        f"- Successful years: {len(summary)}; failed years: {len(failures)}; elapsed: {time.time() - started:.1f} s.",
        f"- Summary CSV: `{rel(out / 'evaluation' / '046_04_external_auto_n_summary.csv')}`.",
    ]) + "\n", encoding="utf-8")
    result = {
        "task": "046_11_sya_external_auto_n_rule_minimal",
        "output_root": rel(out),
        "record_md": rel(record),
        "successful_runs": int(len(summary)),
        "failed_runs": int(len(failures)),
    }
    (out / "046_11_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--years", type=str, default="", help="Optional smoke years, e.g. 2014; empty means all validation years.")
    parser.add_argument("--run-id", type=str, default="", help="Optional timestamp suffix to avoid overwriting earlier outputs.")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    years = [int(x.strip()) for x in args.years.split(",") if x.strip()] or None
    print(json.dumps(run(path, args.dry_run, years, args.run_id), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
