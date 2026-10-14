"""054_01: HLA lowIC minimal external auto-N baseline.

Irrigation uses DSSAT native automatic management. Nitrogen uses the external
gym-DSSAT action channel because DSSAT automatic fertilizer is not operational
in this workflow.
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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline
from build_relaxed_success_five_scenario_daily_evidence_027_05 import parse_management_events
from ppo_action_safety import normalize_action


DEFAULT_CONFIG = ROOT / "configs" / "054_01_hla_lowIC_external_auto_n_rule_nstd050_minimal.json"
PROMPT = ROOT / "prompts" / "054_hla_lowIC_site_transfer_expanded_action_ppo_and_auto.md"
INPUT_PROFILES = {
    "originIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013",
    "lowIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual",
}
SCENARIO = "dssat_auto_irrigation_external_n_rule"
STATION = "HLA"
SITE = "HLA"
MAX_STEPS = 260


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("station_code") != STATION or cfg.get("site") != SITE:
        raise ValueError("054_01 is registered for station_code=HLA and site=HLA.")
    if cfg.get("input_profile") not in INPUT_PROFILES:
        raise ValueError(f"input_profile must be one of {list(INPUT_PROFILES)}.")
    rule = cfg.get("external_auto_n_rule", {})
    for key in ["nitrogen_stress_threshold", "nitrogen_dose_kg_ha"]:
        if key not in rule:
            raise ValueError(f"external_auto_n_rule missing {key}")
    forbidden = ["min_days_between_nitrogen", "season_nitrogen_cap_kg_ha", "fertilization_last_dap"]
    present = [key for key in forbidden if key in rule]
    if present:
        raise ValueError(f"054_01 minimal auto-N must not include removed constraints: {present}")
    return cfg


def run_suffix(run_id: str = "") -> str:
    run_id = str(run_id).strip()
    return f"_run_{run_id}" if run_id else ""


def output_root(cfg: dict[str, Any], run_id: str = "") -> Path:
    suffix = str(cfg.get("external_auto_n_rule", {}).get("output_suffix", "")).strip()
    name = f"{cfg['task_id']}_{cfg['task_name']}"
    if suffix and suffix not in name:
        name = f"{name}_{suffix}"
    return ROOT / "benchmark_results" / f"{name}{run_suffix(run_id)}"


def selected_rows(years: list[int]) -> pd.DataFrame:
    split = pd.read_csv(baseline.SPLIT_CSV, keep_default_na=False)
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    out = split[split["station_code"].astype(str).eq(STATION) & split["year"].isin(years)].copy()
    if sorted(out["year"].astype(int).tolist()) != sorted(years):
        raise RuntimeError("Some HLA validation years are missing from the split registry.")
    return out.sort_values("year").reset_index(drop=True)


def build_configs(out: Path, selected: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    run_config = direct_ppo.load_yaml(baseline.BASE_CONFIG)
    run_config = json.loads(json.dumps(run_config))
    run_config["seed"] = 0
    run_config["paths"]["output_root"] = rel(out)
    run_config["runtime"]["max_steps"] = MAX_STEPS
    env_config = baseline.build_env_config(run_config, selected)
    env_config["paths"]["output_root"] = rel(out)
    env_config["runtime"]["max_steps"] = MAX_STEPS
    env_config["seed"] = 0
    return run_config, env_config


def executed_nitrogen_from_mgmt_event(snapshot: Path) -> float:
    events = parse_management_events(snapshot / "MgmtEvent.OUT")
    if events.empty:
        return 0.0
    fertilizer = events[events["operation"].astype(str).str.lower().str.startswith("fert")]
    return float(pd.to_numeric(fertilizer["amount"], errors="coerce").fillna(0.0).sum())


def make_env_auto_irrigation_external_n(env_config: dict[str, Any], year: int, run_tag: str):
    baseline.ensure_project_on_path()
    import gym
    from sb3_wrapper import GymDssatWrapper

    info = direct_ppo.find_year(env_config, STATION, int(year))
    env_args = ppo_safe_rendering.build_env_args(
        station=STATION,
        year=int(year),
        planting_date=info["planting_date"],
        seed=0,
        config=env_config,
        run_tag=run_tag,
        evaluation=True,
        mode=env_config.get("runtime", {}).get("mode", "all"),
    )
    template = Path(env_args["fileX_template_path"])
    text = template.read_text(encoding="utf-8", errors="replace")
    text = baseline.set_auto_treatment_one(text, int(year))
    text = baseline.set_management_for_treatment(text, 1, "A", "L")
    template.write_text(text, encoding="utf-8")
    return GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped), env_args, text


def step_scheduled_action(env: Any, action_real: dict[str, float]) -> np.ndarray:
    return normalize_action(env.formator.action_names, env.formator.action_space_dict, action_real)


def evaluate_year(run_config: dict[str, Any], env_config: dict[str, Any], year: int, rule: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    env, env_args, rendered_text = make_env_auto_irrigation_external_n(env_config, year, f"HLA_{year}_054_01_external_auto_n")
    rows: list[dict[str, Any]] = []
    total_requested_n = 0.0
    event_count = 0
    planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, int(year))["planting_date"])
    weather = direct_ppo.weather_for_daily(run_config)
    threshold = float(rule["nitrogen_stress_threshold"])
    dose = float(rule["nitrogen_dose_kg_ha"])
    try:
        obs, info = env.reset()
        done, steps = False, 0
        while not done and steps < MAX_STEPS:
            latest_pre = baseline.latest_observation_dict(env, obs, info)
            dap = int(round(float(baseline.scalar(latest_pre.get("dap", steps + 1), steps + 1))))
            nstres_pre = float(baseline.scalar(latest_pre.get("nstres"), 0.0))
            amount = dose if nstres_pre >= threshold else 0.0
            action = step_scheduled_action(env, {"amir": 0.0, "anfer": amount})
            obs, reward, terminated, truncated, info = env.step(action)
            if amount > 0:
                total_requested_n += amount
                event_count += 1
            latest = baseline.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            wrow = baseline.weather_row(weather, STATION, date)
            rows.append(
                {
                    "station_code": STATION,
                    "site": SITE,
                    "year": int(year),
                    "scenario": SCENARIO,
                    "date": date.strftime("%Y-%m-%d"),
                    "dap": dap,
                    "rain": baseline.scalar(wrow.get("rain"), np.nan),
                    "tmax": baseline.scalar(wrow.get("tmax"), np.nan),
                    "tmin": baseline.scalar(wrow.get("tmin"), np.nan),
                    "srad": baseline.scalar(wrow.get("srad"), np.nan),
                    "grnwt": baseline.scalar(latest.get("grnwt")),
                    "topwt": baseline.scalar(latest.get("topwt")),
                    "swfac": baseline.scalar(latest.get("swfac")),
                    "nstres": baseline.scalar(latest.get("nstres")),
                    "nstres_pre_action": nstres_pre,
                    "nitrogen_requested_kg_ha": amount,
                    "irrigation_requested_mm": 0.0,
                    "external_action_note": "minimal_NSTRES_rule_external_N;DSSAT_native_auto_irrigation",
                    "reward": float(reward),
                    "done": bool(terminated or truncated),
                }
            )
            done = bool(terminated or truncated)
            steps += 1
        daily = pd.DataFrame(rows)
        if daily.empty or not bool(daily["done"].iloc[-1]):
            raise RuntimeError(f"HLA{year} did not finish within {MAX_STEPS} steps")
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        final_b = float(pd.to_numeric(daily["topwt"], errors="coerce").iloc[-1])
        snapshot_tmp = baseline.siteppo.snapshot_from_env(env)
        snapshot = baseline.OUT / "snapshots" / STATION / str(year) / SCENARIO
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(snapshot_tmp, snapshot)
        metrics = baseline.metrics_from_snapshot(snapshot, final_y)
        summary_n = float(metrics["actual_nitrogen_kg_ha"])
        mgmt_event_n = executed_nitrogen_from_mgmt_event(snapshot)
        actual_n = mgmt_event_n if mgmt_event_n > 1e-9 else summary_n
        pfp_n = final_y / actual_n if actual_n > 1e-9 else np.nan
        n_closure_gap = actual_n - total_requested_n
        summary = {
            "station_code": STATION,
            "site": SITE,
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


def write_record(out: Path, cfg: dict[str, Any], summary: pd.DataFrame, failures: pd.DataFrame, elapsed: float) -> Path:
    record = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
    rule = cfg["external_auto_n_rule"]
    lines = [
        f"# {cfg['task_id']} HLA lowIC minimal external auto-N record",
        "",
        "- Irrigation: DSSAT native automatic irrigation.",
        "- Nitrogen: external gym-DSSAT action channel.",
        f"- Rule: apply {rule['nitrogen_dose_kg_ha']} kg/ha whenever pre-action NSTRES >= {rule['nitrogen_stress_threshold']}.",
        "- Removed constraints: no DAP cutoff, no minimum interval, no seasonal N cap.",
        f"- Successful years: {len(summary)}; failed years: {len(failures)}; elapsed: {elapsed:.1f} s.",
        f"- Summary CSV: `{rel(out / 'evaluation' / '054_01_external_auto_n_summary.csv')}`.",
    ]
    if not summary.empty:
        means = summary[["grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]].apply(pd.to_numeric, errors="coerce").mean()
        lines.extend(["", "## Means", ""])
        for key, value in means.items():
            lines.append(f"- `{key}`: `{value:.4f}`")
    record.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return record


def run(cfg_path: Path, dry_run: bool, years_override: list[int] | None = None, run_id: str = "") -> dict[str, Any]:
    cfg = read_config(cfg_path)
    if years_override:
        cfg = json.loads(json.dumps(cfg))
        cfg["scope"]["validation_years"] = years_override
    input_root = INPUT_PROFILES[str(cfg["input_profile"])]
    years = list(map(int, cfg["scope"]["validation_years"]))
    rule = dict(cfg["external_auto_n_rule"])
    out = output_root(cfg, run_id)
    pf = {
        "task": "054_01_hla_external_auto_n_rule_minimal",
        "input_profile": cfg["input_profile"],
        "resolved_input_root": rel(input_root),
        "years": years,
        "rule": rule,
        "removed_constraints": ["DAP cutoff", "minimum N interval", "seasonal N cap"],
        "output_root": rel(out),
        "next_step_allowed": input_root.exists() and (input_root / "HL" / "CNHL0701_corrected_IC123.MZX").exists(),
    }
    if dry_run:
        return {"mode": "dry_run", **pf}
    if not pf["next_step_allowed"]:
        raise RuntimeError("preflight failed: HLA input root or source MZX missing")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {rel(out)}")
    for sub in ["configs", "evaluation", "snapshots"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, out / "configs" / cfg_path.name)
    if PROMPT.exists():
        shutil.copy2(PROMPT, out / "configs" / PROMPT.name)
    selected = selected_rows(years)
    run_config, env_config = build_configs(out, selected)
    env_config["input_profile"] = str(cfg["input_profile"])
    direct_ppo.write_yaml(env_config, out / "configs" / "054_01_resolved_env_config.yaml")

    originals = (baseline.MULTISITE_INPUT_ROOT, baseline.OUT, baseline.TASK_ID, baseline.MAX_STEPS, ppo_safe_rendering.MULTISITE_INPUT_ROOT)
    baseline.MULTISITE_INPUT_ROOT = input_root
    baseline.OUT = out
    baseline.TASK_ID = "054_01"
    baseline.MAX_STEPS = MAX_STEPS
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = input_root
    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    rendered_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    started = time.time()
    try:
        for year in years:
            print(f"[054_01] HLA{year} {SCENARIO}", flush=True)
            try:
                daily, summary, rendered = evaluate_year(run_config, env_config, year, rule)
                daily_frames.append(daily)
                summary_rows.append(summary)
                rendered_rows.append({"year": year, **rendered})
            except Exception as exc:
                failure_rows.append({"year": year, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]})
    finally:
        baseline.MULTISITE_INPUT_ROOT, baseline.OUT, baseline.TASK_ID, baseline.MAX_STEPS, ppo_safe_rendering.MULTISITE_INPUT_ROOT = originals

    summary = pd.DataFrame(summary_rows)
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    rendered_df = pd.DataFrame(rendered_rows)
    failures = pd.DataFrame(failure_rows)
    summary.to_csv(out / "evaluation" / "054_01_external_auto_n_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(out / "evaluation" / "054_01_external_auto_n_daily.csv", index=False, encoding="utf-8-sig")
    rendered_df.to_csv(out / "evaluation" / "054_01_rendered_management_mode_audit.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(out / "evaluation" / "054_01_failures.csv", index=False, encoding="utf-8-sig")
    record = write_record(out, cfg, summary, failures, time.time() - started)
    result = {
        **pf,
        "mode": "evaluate_auto_baseline",
        "record_md": rel(record),
        "successful_runs": int(len(summary)),
        "failed_runs": int(len(failures)),
    }
    (out / "054_01_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--years", type=str, default="")
    parser.add_argument("--run-id", type=str, default="")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    years = [int(x.strip()) for x in args.years.split(",") if x.strip()] or None
    print(json.dumps(run(cfg_path, args.dry_run, years, args.run_id), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
